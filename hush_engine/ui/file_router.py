#!/usr/bin/env python3
"""
File Router - Routes dropped files to appropriate scrubber
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import mimetypes

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocr import VisionOCR
from detectors import PIIDetector, TableDetector, ContextAwarePIIDetector
from detectors.face_detector import FaceDetector
from detectors.qr_detector import QRDetector
from detectors.barcode_detector import BarcodeDetector

# Spatial filtering for precision improvement
try:
    from detectors.spatial_filter import apply_spatial_filtering, create_spatial_context
    SPATIAL_FILTER_AVAILABLE = True
except ImportError:
    SPATIAL_FILTER_AVAILABLE = False
    apply_spatial_filtering = None
    create_spatial_context = None
from anonymizers import ImageAnonymizer, SpreadsheetAnonymizer
from pdf import PDFProcessor
from image_optimizer import optimize_image

# Import locale manager for locale-aware PII detection
try:
    from locale_manager import get_locale_manager
    LOCALE_MANAGER_AVAILABLE = True
except ImportError:
    LOCALE_MANAGER_AVAILABLE = False
from PIL import Image
import pandas as pd
import tempfile
import os
import stat
from dataclasses import dataclass

# Maximum number of parallel workers for PDF page processing
# Limit to avoid memory pressure (each page is ~10-20MB at 400 DPI)
MAX_PDF_WORKERS = min(4, os.cpu_count() or 2)


# =============================================================================
# TEXT REGION MERGING: Merge adjacent OCR regions for better PII detection
# =============================================================================

@dataclass
class MergedTextRegion:
    """A merged text region combining adjacent OCR detections."""
    text: str
    detections: list  # Original TextDetection objects
    char_offsets: list  # (start, end) offset in merged text for each detection


def merge_adjacent_detections(
    detections: list,
    horizontal_threshold: float = 50.0,
    vertical_threshold: float = 20.0
) -> List[MergedTextRegion]:
    """
    Merge horizontally adjacent text detections into logical groups.

    This helps detect PII patterns that span multiple OCR regions, like:
    - "808921738 RT0001" (business number + program ID)
    - "John Smith" (first name + last name as separate detections)
    """
    if not detections:
        return []

    # Sort by y (top to bottom), then x (left to right)
    sorted_detections = sorted(detections, key=lambda d: (d.bbox[1], d.bbox[0]))

    merged_regions = []
    current_group = [sorted_detections[0]]

    for detection in sorted_detections[1:]:
        last_detection = current_group[-1]

        # Check if on same line (vertical alignment)
        last_y_center = (last_detection.bbox[1] + last_detection.bbox[3]) / 2
        curr_y_center = (detection.bbox[1] + detection.bbox[3]) / 2
        vertical_diff = abs(curr_y_center - last_y_center)

        # Check horizontal gap
        horizontal_gap = detection.bbox[0] - last_detection.bbox[2]

        # Merge if on same line and close enough horizontally
        if vertical_diff <= vertical_threshold and 0 <= horizontal_gap <= horizontal_threshold:
            current_group.append(detection)
        else:
            merged_regions.append(_create_merged_region(current_group))
            current_group = [detection]

    if current_group:
        merged_regions.append(_create_merged_region(current_group))

    return merged_regions


def _create_merged_region(detections: list) -> MergedTextRegion:
    """Create a MergedTextRegion from a list of adjacent detections."""
    texts = []
    char_offsets = []
    current_offset = 0

    for detection in detections:
        start = current_offset
        end = start + len(detection.text)
        char_offsets.append((start, end))
        texts.append(detection.text)
        current_offset = end + 1  # +1 for space separator

    return MergedTextRegion(
        text=" ".join(texts),
        detections=detections,
        char_offsets=char_offsets
    )


def get_bbox_for_entity(
    entity_start: int,
    entity_end: int,
    merged_region: MergedTextRegion
) -> Tuple[float, float, float, float]:
    """Calculate bounding box for an entity within a merged text region."""
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    for i, (det_start, det_end) in enumerate(merged_region.char_offsets):
        detection = merged_region.detections[i]

        # Check if this detection overlaps with the entity
        if det_start < entity_end and det_end > entity_start:
            local_start = max(0, entity_start - det_start)
            local_end = min(len(detection.text), entity_end - det_start)

            if detection.char_boxes and len(detection.char_boxes) >= local_end:
                for j in range(local_start, local_end):
                    if j < len(detection.char_boxes) and detection.char_boxes[j]:
                        box = detection.char_boxes[j]
                        min_x = min(min_x, box[0])
                        min_y = min(min_y, box[1])
                        max_x = max(max_x, box[2])
                        max_y = max(max_y, box[3])
            else:
                min_x = min(min_x, detection.bbox[0])
                min_y = min(min_y, detection.bbox[1])
                max_x = max(max_x, detection.bbox[2])
                max_y = max(max_y, detection.bbox[3])

    if min_x == float('inf'):
        return merged_region.detections[0].bbox

    return (min_x, min_y, max_x, max_y)


# =============================================================================
# SECURITY: Secure Temp File Handling
# =============================================================================

def get_secure_temp_dir() -> Path:
    """
    Get or create a secure temp directory for Hush.

    Uses ~/.hush/tmp with restrictive permissions (0o700).
    This prevents other users from reading sensitive image data during processing.
    """
    temp_dir = Path.home() / ".hush" / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Set restrictive permissions (owner only)
    try:
        os.chmod(temp_dir, stat.S_IRWXU)  # 0o700
    except OSError:
        pass  # May fail if directory was just created by another process

    return temp_dir


def create_secure_temp_file(suffix: str = '.png') -> str:
    """
    Create a secure temporary file with restrictive permissions.

    Args:
        suffix: File extension (e.g., '.png', '.jpg')

    Returns:
        Path to the temporary file
    """
    temp_dir = get_secure_temp_dir()
    fd, temp_path = tempfile.mkstemp(suffix=suffix, dir=temp_dir)

    # Set restrictive permissions (owner only)
    os.chmod(temp_path, stat.S_IRUSR | stat.S_IWUSR)  # 0o600
    os.close(fd)

    return temp_path


class FileRouter:
    """
    Routes files to appropriate scrubbing engine based on file type
    """

    def __init__(self, output_dir: str = None):
        """
        Initialize the file router

        Args:
            output_dir: Directory to save scrubbed files (default: same as input)
        """
        self.output_dir = Path(output_dir) if output_dir else None

        # Initialize engines (lazy loading for performance)
        self._ocr = None
        self._detector = None
        self._face_detector = None
        self._qr_detector = None
        self._barcode_detector = None
        self._image_anonymizer = None
        self._spreadsheet_anonymizer = None
        self._pdf_processor = None
        self._table_detector = None
        self._context_aware_detector = None

        # File type mapping
        self.image_extensions = {'.png', '.jpg', '.jpeg', '.heic'}
        self.spreadsheet_extensions = {'.csv', '.xlsx', '.xls'}
        self.pdf_extensions = {'.pdf'}

    @property
    def ocr(self):
        """Lazy-load OCR engine"""
        if self._ocr is None:
            self._ocr = VisionOCR(recognition_level="accurate")
        return self._ocr

    @property
    def detector(self):
        """Lazy-load PII detector"""
        if self._detector is None:
            self._detector = PIIDetector()
        return self._detector

    @property
    def face_detector(self):
        """Lazy-load face detector"""
        if self._face_detector is None:
            self._face_detector = FaceDetector()
        return self._face_detector

    @property
    def qr_detector(self):
        """Lazy-load QR code detector"""
        if self._qr_detector is None:
            self._qr_detector = QRDetector()
        return self._qr_detector

    @property
    def barcode_detector(self):
        """Lazy-load barcode detector"""
        if self._barcode_detector is None:
            self._barcode_detector = BarcodeDetector()
        return self._barcode_detector

    @property
    def image_anonymizer(self):
        """Lazy-load image anonymizer"""
        if self._image_anonymizer is None:
            self._image_anonymizer = ImageAnonymizer(method="black_bar")
        return self._image_anonymizer

    @property
    def spreadsheet_anonymizer(self):
        """Lazy-load spreadsheet anonymizer"""
        if self._spreadsheet_anonymizer is None:
            self._spreadsheet_anonymizer = SpreadsheetAnonymizer()
        return self._spreadsheet_anonymizer

    @property
    def pdf_processor(self):
        """Lazy-load PDF processor (400 DPI for accurate OCR detection)"""
        if self._pdf_processor is None:
            self._pdf_processor = PDFProcessor(dpi=400)  # Use 400 DPI for accurate OCR detection
        return self._pdf_processor

    @property
    def table_detector(self):
        """Lazy-load table structure detector"""
        if self._table_detector is None:
            self._table_detector = TableDetector()
        return self._table_detector

    def get_user_locales(self) -> list:
        """
        Get the user's locale preferences for PII detection.

        Returns:
            List of ISO locale codes (e.g., ["en-US"]) or None for auto-detect.
        """
        if not LOCALE_MANAGER_AVAILABLE:
            return None

        try:
            locale_mgr = get_locale_manager()
            locale_info = locale_mgr.get_locale()
            current_locale = locale_info.get("current", {}).get("locale")

            if current_locale and current_locale != "auto":
                # Convert locale code format (en_US -> en-US)
                iso_locale = current_locale.replace("_", "-")
                return [iso_locale]

            # Auto mode - return None to let detector auto-detect
            return None
        except Exception:
            return None

    @property
    def context_aware_detector(self):
        """Lazy-load context-aware PII detector"""
        if self._context_aware_detector is None:
            self._context_aware_detector = ContextAwarePIIDetector(
                self.detector, self.table_detector
            )
        return self._context_aware_detector

    @property
    def preview_pdf_processor(self):
        """Lazy-load PDF processor for preview images (150 DPI, JPG format for speed)"""
        if not hasattr(self, '_preview_pdf_processor'):
            self._preview_pdf_processor = PDFProcessor(dpi=150)
        return self._preview_pdf_processor



    @property
    def output_pdf_processor(self):
        """Lazy-load PDF processor for final output (400 DPI for print quality)"""
        if not hasattr(self, '_output_pdf_processor'):
            self._output_pdf_processor = PDFProcessor(dpi=400)
        return self._output_pdf_processor

    def detect_file_type(self, file_path: str) -> str:
        """
        Detect file type

        Args:
            file_path: Path to file

        Returns:
            "image", "spreadsheet", "pdf", or "unknown"
        """
        ext = Path(file_path).suffix.lower()

        if ext in self.image_extensions:
            return "image"
        elif ext in self.spreadsheet_extensions:
            return "spreadsheet"
        elif ext in self.pdf_extensions:
            return "pdf"
        else:
            return "unknown"

    def get_output_path(self, input_path: str) -> str:
        """
        Generate output file path

        Args:
            input_path: Input file path

        Returns:
            Output file path with _scrubbed suffix
        """
        input_file = Path(input_path)

        if self.output_dir:
            output_file = self.output_dir / f"{input_file.stem}_scrubbed{input_file.suffix}"
        else:
            output_file = input_file.parent / f"{input_file.stem}_scrubbed{input_file.suffix}"

        return str(output_file)

    def warmup(self) -> None:
        """
        Preload the PII detector (and optionally OCR) so the first analysis
        doesn't block for 30–60 seconds loading the spaCy model.
        """
        sys.stderr.write("FileRouter: warming up PII detector...\n")
        sys.stderr.flush()
        _ = self.detector
        self.detector.analyze_text("warmup")
        sys.stderr.write("FileRouter: warmup done\n")
        sys.stderr.flush()

    def detect_pii_image(self, input_path: str, detect_faces: bool = True) -> Dict[str, Any]:
        """
        Detect PII in an image without saving

        Args:
            input_path: Path to image
            detect_faces: Whether to detect faces (default True)

        Returns:
            Dictionary with 'detections' (PII) and 'all_text_blocks' (all OCR results)
        """
        # Extract text
        ocr_detections = self.ocr.extract_text(input_path)

        # Convert all OCR detections to text blocks
        all_text_blocks = [{
            'text': detection.text,
            'bbox': detection.bbox
        } for detection in ocr_detections]

        # Merge adjacent text regions for better PII detection
        # This helps detect patterns that span multiple OCR regions (e.g., "808921738 RT0001")
        merged_regions = merge_adjacent_detections(ocr_detections)

        # Detect PII in merged text regions using user's locale preferences
        pii_detections = []
        user_locales = self.get_user_locales()
        for merged_region in merged_regions:
            entities = self.detector.analyze_text(
                merged_region.text,
                locales=user_locales,
                auto_detect_locale=(user_locales is None)
            )

            for entity in entities:
                # Calculate bounding box spanning the relevant original detections
                entity_bbox = get_bbox_for_entity(
                    entity_start=entity.start,
                    entity_end=entity.end,
                    merged_region=merged_region
                )
                pii_detections.append({
                    'entity_type': entity.entity_type,
                    'text': entity.text,
                    'confidence': entity.confidence,
                    'bbox': entity_bbox
                })

        # Detect faces if enabled
        if detect_faces:
            try:
                from PIL import Image
                img = Image.open(input_path)
                face_detections = self.face_detector.detect_faces(img)
                for face in face_detections:
                    # Convert bbox tuple to array format [x1, y1, x2, y2] expected by frontend
                    x, y, w, h = face.bbox
                    pii_detections.append({
                        'entity_type': 'FACE',
                        'text': '[Face]',
                        'confidence': face.confidence,
                        'bbox': [float(x), float(y), float(x + w), float(y + h)]
                    })
                print(f"Detected {len(face_detections)} face(s)", file=sys.stderr)
            except Exception as e:
                print(f"Face detection error: {e}", file=sys.stderr)

        # Detect QR codes
        try:
            from PIL import Image
            img = Image.open(input_path)
            qr_detections = self.qr_detector.detect_qr_codes(img)
            for qr in qr_detections:
                # Convert bbox tuple to array format [x1, y1, x2, y2] expected by frontend
                x, y, w, h = qr.bbox
                pii_detections.append({
                    'entity_type': 'QR_CODE',
                    'text': qr.data if qr.data else '[QR Code]',
                    'confidence': qr.confidence,
                    'bbox': [float(x), float(y), float(x + w), float(y + h)]
                })
            if qr_detections:
                print(f"Detected {len(qr_detections)} QR code(s)", file=sys.stderr)
        except Exception as e:
            print(f"QR code detection error: {e}", file=sys.stderr)

        # Detect barcodes
        try:
            from PIL import Image
            img = Image.open(input_path)
            barcode_detections = self.barcode_detector.detect_barcodes(img)
            for barcode in barcode_detections:
                # Convert bbox tuple to array format [x1, y1, x2, y2] expected by frontend
                x, y, w, h = barcode.bbox
                pii_detections.append({
                    'entity_type': 'BARCODE',
                    'text': f"[{barcode.barcode_type}] {barcode.data}" if barcode.data else f'[{barcode.barcode_type}]',
                    'confidence': barcode.confidence,
                    'bbox': [float(x), float(y), float(x + w), float(y + h)]
                })
            if barcode_detections:
                print(f"Detected {len(barcode_detections)} barcode(s)", file=sys.stderr)
        except Exception as e:
            print(f"Barcode detection error: {e}", file=sys.stderr)

        # Apply spatial filtering for improved precision
        if SPATIAL_FILTER_AVAILABLE and pii_detections and all_text_blocks:
            try:
                # Get image dimensions for spatial context
                from PIL import Image
                img = Image.open(input_path)
                page_width, page_height = img.size

                # Create spatial context with all text detections
                spatial_context = create_spatial_context(
                    page_width=float(page_width),
                    page_height=float(page_height),
                    text_blocks=all_text_blocks
                )

                # Apply spatial filtering to text-based PII detections only
                # (Face, QR, Barcode don't need form label filtering)
                text_pii = [d for d in pii_detections if d['entity_type'] not in ('FACE', 'QR_CODE', 'BARCODE')]
                other_pii = [d for d in pii_detections if d['entity_type'] in ('FACE', 'QR_CODE', 'BARCODE')]

                filtered_text_pii = apply_spatial_filtering(text_pii, spatial_context)
                pii_detections = filtered_text_pii + other_pii
            except Exception as e:
                print(f"Spatial filtering error (continuing without): {e}", file=sys.stderr)

        return {
            'detections': pii_detections,
            'all_text_blocks': all_text_blocks
        }

    def _process_pdf_page(
        self,
        page_num: int,
        page_image: "Image.Image",
        user_locales: Optional[List[str]],
        detect_faces: bool,
        detect_qr: bool,
        detect_barcodes: bool
    ) -> Tuple[int, List[Dict], List[Dict], Tuple[float, float]]:
        """
        Process a single PDF page for PII detection.

        This method is designed to be called in parallel from ThreadPoolExecutor.
        Uses in-memory OCR to avoid disk I/O overhead.

        Args:
            page_num: 1-indexed page number
            page_image: PIL Image of the page
            user_locales: User's locale preferences for detection
            detect_faces: Whether to detect faces
            detect_qr: Whether to detect QR codes
            detect_barcodes: Whether to detect barcodes

        Returns:
            Tuple of (page_num, page_detections, page_text_blocks, page_dimensions)
        """
        page_dimensions = (float(page_image.width), float(page_image.height))
        page_detections = []
        page_text_blocks = []

        # Extract text from this page using in-memory OCR (no temp file I/O)
        ocr_detections = self.ocr.extract_text_from_image(page_image)

        # Convert OCR detections to text blocks with page number
        for detection in ocr_detections:
            block = {
                'text': detection.text,
                'bbox': detection.bbox,
                'page': page_num
            }
            page_text_blocks.append(block)

        # Merge adjacent text regions for better PII detection
        merged_regions = merge_adjacent_detections(ocr_detections)

        # Detect PII in this page (initial pass)
        for merged_region in merged_regions:
            entities = self.detector.analyze_text(
                merged_region.text,
                locales=user_locales,
                auto_detect_locale=(user_locales is None)
            )

            for entity in entities:
                # Calculate bounding box spanning the relevant original detections
                entity_bbox = get_bbox_for_entity(
                    entity_start=entity.start,
                    entity_end=entity.end,
                    merged_region=merged_region
                )
                page_detections.append({
                    'entity_type': entity.entity_type,
                    'text': entity.text,
                    'confidence': entity.confidence,
                    'bbox': entity_bbox,
                    'page': page_num
                })

        # Apply context-aware detection (table structure + header context)
        try:
            # Enhance detections using table context
            page_detections = self.context_aware_detector.detect_with_context(
                page_text_blocks, page_detections
            )
            # Fill in missing detections based on header context
            page_detections = self.context_aware_detector.fill_missing_detections(
                page_text_blocks, page_detections
            )
            # Add page number to any new detections
            for det in page_detections:
                if 'page' not in det:
                    det['page'] = page_num
        except Exception as e:
            print(f"Page {page_num}: Context-aware detection error: {e}", file=sys.stderr)

        # Detect faces in this page if enabled
        if detect_faces:
            try:
                face_detections = self.face_detector.detect_faces(page_image)
                for face in face_detections:
                    x, y, w, h = face.bbox
                    page_detections.append({
                        'entity_type': 'FACE',
                        'text': '[Face]',
                        'confidence': face.confidence,
                        'bbox': [float(x), float(y), float(x + w), float(y + h)],
                        'page': page_num
                    })
                if face_detections:
                    print(f"Page {page_num}: Detected {len(face_detections)} face(s)", file=sys.stderr)
            except Exception as e:
                print(f"Page {page_num}: Face detection error: {e}", file=sys.stderr)

        # Detect QR codes in this page if enabled
        if detect_qr:
            try:
                qr_detections = self.qr_detector.detect_qr_codes(page_image)
                for qr in qr_detections:
                    x, y, w, h = qr.bbox
                    page_detections.append({
                        'entity_type': 'QR_CODE',
                        'text': qr.data if qr.data else '[QR Code]',
                        'confidence': qr.confidence,
                        'bbox': [float(x), float(y), float(x + w), float(y + h)],
                        'page': page_num
                    })
                if qr_detections:
                    print(f"Page {page_num}: Detected {len(qr_detections)} QR code(s)", file=sys.stderr)
            except Exception as e:
                print(f"Page {page_num}: QR code detection error: {e}", file=sys.stderr)

        # Detect barcodes in this page if enabled
        if detect_barcodes:
            try:
                barcode_detections = self.barcode_detector.detect_barcodes(page_image)
                for barcode in barcode_detections:
                    x, y, w, h = barcode.bbox
                    page_detections.append({
                        'entity_type': 'BARCODE',
                        'text': f"[{barcode.barcode_type}] {barcode.data}" if barcode.data else f'[{barcode.barcode_type}]',
                        'confidence': barcode.confidence,
                        'bbox': [float(x), float(y), float(x + w), float(y + h)],
                        'page': page_num
                    })
                if barcode_detections:
                    print(f"Page {page_num}: Detected {len(barcode_detections)} barcode(s)", file=sys.stderr)
            except Exception as e:
                print(f"Page {page_num}: Barcode detection error: {e}", file=sys.stderr)

        print(f"Page {page_num}: Found {len(page_detections)} total detections", file=sys.stderr)

        return (page_num, page_detections, page_text_blocks, page_dimensions)

    def detect_pii_pdf(
        self,
        input_path: str,
        detect_faces: bool = True,
        detect_qr: bool = True,
        detect_barcodes: bool = True
    ) -> Dict[str, Any]:
        """
        Detect PII in a PDF without saving.

        Uses parallel processing across pages for improved performance on multi-core systems.

        Args:
            input_path: Path to PDF file
            detect_faces: Whether to detect faces (default True)
            detect_qr: Whether to detect QR codes (default True)
            detect_barcodes: Whether to detect barcodes (default True)

        Returns:
            Dictionary with 'detections' (PII with page numbers), 'all_text_blocks', and 'total_pages'
        """
        # Get user's locale preferences for detection
        user_locales = self.get_user_locales()

        # Convert PDF to images
        page_images = self.pdf_processor.pdf_to_images(input_path)
        total_pages = len(page_images)

        print(f"Processing {total_pages} page(s) from PDF using {MAX_PDF_WORKERS} workers", file=sys.stderr)

        all_detections = []
        all_text_blocks = []
        page_dimensions = {}  # Track actual page dimensions for spatial filtering

        # Process pages in parallel for better performance
        with ThreadPoolExecutor(max_workers=MAX_PDF_WORKERS) as executor:
            # Submit all page processing tasks
            futures = {
                executor.submit(
                    self._process_pdf_page,
                    page_num,
                    page_image,
                    user_locales,
                    detect_faces,
                    detect_qr,
                    detect_barcodes
                ): page_num
                for page_num, page_image in enumerate(page_images, start=1)
            }

            # Collect results as they complete
            for future in as_completed(futures):
                page_num = futures[future]
                try:
                    result_page_num, page_dets, page_blocks, dims = future.result()
                    all_detections.extend(page_dets)
                    all_text_blocks.extend(page_blocks)
                    page_dimensions[result_page_num] = dims
                except Exception as e:
                    print(f"Page {page_num} processing failed: {e}", file=sys.stderr)

        # Apply spatial filtering for improved precision (per page)
        if SPATIAL_FILTER_AVAILABLE and all_detections and all_text_blocks:
            try:
                # Group detections and text blocks by page
                from collections import defaultdict
                detections_by_page = defaultdict(list)
                text_blocks_by_page = defaultdict(list)

                for det in all_detections:
                    page = det.get('page', 1)
                    detections_by_page[page].append(det)

                for block in all_text_blocks:
                    page = block.get('page', 1)
                    text_blocks_by_page[page].append(block)

                # Apply spatial filtering per page
                filtered_all = []
                for page_num in sorted(detections_by_page.keys()):
                    page_dets = detections_by_page[page_num]
                    page_blocks = text_blocks_by_page.get(page_num, [])

                    if not page_blocks:
                        filtered_all.extend(page_dets)
                        continue

                    # Get actual page dimensions (in render pixels, typically 400 DPI)
                    page_width, page_height = page_dimensions.get(page_num, (612.0, 792.0))

                    # Create spatial context with actual render DPI for proper threshold scaling
                    spatial_context = create_spatial_context(
                        page_width=page_width,
                        page_height=page_height,
                        text_blocks=page_blocks,
                        source_dpi=float(self.pdf_processor.dpi)  # Typically 400 DPI
                    )

                    # Apply spatial filtering to text-based PII only
                    text_pii = [d for d in page_dets if d['entity_type'] not in ('FACE', 'QR_CODE', 'BARCODE')]
                    other_pii = [d for d in page_dets if d['entity_type'] in ('FACE', 'QR_CODE', 'BARCODE')]

                    filtered_text_pii = apply_spatial_filtering(text_pii, spatial_context)
                    filtered_all.extend(filtered_text_pii)
                    filtered_all.extend(other_pii)

                all_detections = filtered_all
            except Exception as e:
                print(f"Spatial filtering error (continuing without): {e}", file=sys.stderr)

        return {
            'detections': all_detections,
            'all_text_blocks': all_text_blocks,
            'total_pages': total_pages
        }

    def get_pdf_page_image(self, input_path: str, page_number: int, optimize: bool = False) -> Dict[str, str]:
        """
        Get a specific page from a PDF as a JPG image (optimized for preview)

        Args:
            input_path: Path to PDF file
            page_number: Page number (1-indexed)
            optimize: If True, compress output using MozJPEG

        Returns:
            Dictionary with 'image_path' pointing to temporary JPG file
        """
        # Convert just the requested page using preview processor (150 DPI for speed)
        page_images = self.preview_pdf_processor.pdf_to_images(input_path, first_page=page_number, last_page=page_number)

        if not page_images:
            raise ValueError(f"Could not extract page {page_number} from PDF")

        page_image = page_images[0]

        # SECURITY: Save to secure temporary file with restrictive permissions
        temp_path = create_secure_temp_file(suffix='.jpg')
        page_image.save(temp_path, 'JPEG', quality=85)

        # Optionally optimize the preview image
        if optimize:
            optimize_image(temp_path)

        return {'image_path': temp_path}

    def save_scrubbed_image(
        self,
        input_path: str,
        output_path: str,
        detections: List[Dict[str, Any]],
        selected_indices: List[int],
        optimize: bool = False
    ):
        """
        Save scrubbed image with only selected detections redacted

        Args:
            input_path: Input image path
            output_path: Output image path
            detections: All detected PII items
            selected_indices: Indices of items to scrub
            optimize: If True, compress output using Zopfli/MozJPEG
        """
        # Get bboxes for selected items only
        selected_bboxes = [detections[i]['bbox'] for i in selected_indices]

        # Load and redact
        img = Image.open(input_path)
        if selected_bboxes:
            scrubbed = self.image_anonymizer.redact_regions(img, selected_bboxes)
        else:
            scrubbed = img

        scrubbed.save(output_path)
        print(f"Saved scrubbed image to: {output_path}", file=sys.stderr)

        # Optionally optimize the output image
        if optimize:
            optimize_image(output_path)

    def save_scrubbed_pdf(
        self,
        input_path: str,
        output_path: str,
        detections: List[Dict[str, Any]],
        selected_indices: List[int],
        optimize: bool = False
    ):
        """
        Save scrubbed PDF with only selected detections redacted

        Args:
            input_path: Input PDF path
            output_path: Output PDF path
            detections: All detected PII items (with 'page' field)
            selected_indices: Indices of items to scrub
            optimize: If True, compress page images before combining into PDF
        """
        # Convert PDF to images at same DPI as detection (400) to ensure bbox coordinates align
        page_images = self.pdf_processor.pdf_to_images(input_path)
        total_pages = len(page_images)
        
        print(f"Processing {total_pages} page(s) for redaction", file=sys.stderr)
        
        # Group selected detections by page
        selected_by_page = {}
        for idx in selected_indices:
            detection = detections[idx]
            page_num = detection.get('page', 1)
            
            if page_num not in selected_by_page:
                selected_by_page[page_num] = []
            
            selected_by_page[page_num].append(detection['bbox'])
        
        # Process each page
        scrubbed_pages = []
        for page_num in range(1, total_pages + 1):
            page_image = page_images[page_num - 1]  # 0-indexed

            # Get bboxes for this page
            page_bboxes = selected_by_page.get(page_num, [])

            if page_bboxes:
                # SECURITY: Save to secure temp file for processing
                temp_path = create_secure_temp_file(suffix='.jpg')
                page_image.save(temp_path)

                try:
                    # Load and redact
                    img = Image.open(temp_path)
                    scrubbed = self.image_anonymizer.redact_regions(img, page_bboxes)

                    # Optionally optimize the page image
                    if optimize:
                        opt_temp = create_secure_temp_file(suffix='.jpg')
                        scrubbed.save(opt_temp, 'JPEG', quality=85)
                        optimize_image(opt_temp)
                        scrubbed = Image.open(opt_temp)
                        os.unlink(opt_temp)

                    scrubbed_pages.append(scrubbed)
                    print(f"Page {page_num}: Redacted {len(page_bboxes)} region(s)", file=sys.stderr)
                finally:
                    os.unlink(temp_path)
            else:
                # No redactions on this page, but still optimize if requested
                if optimize:
                    opt_temp = create_secure_temp_file(suffix='.jpg')
                    page_image.save(opt_temp, 'JPEG', quality=85)
                    optimize_image(opt_temp)
                    page_image = Image.open(opt_temp)
                    os.unlink(opt_temp)

                scrubbed_pages.append(page_image)
                print(f"Page {page_num}: No redactions", file=sys.stderr)

        # Convert scrubbed images to PDF
        self.output_pdf_processor.images_to_pdf(scrubbed_pages, output_path)
        print(f"Saved scrubbed PDF to: {output_path}", file=sys.stderr)

    def scrub_image(self, input_path: str) -> Dict[str, Any]:
        """
        Scrub an image file

        Args:
            input_path: Path to image

        Returns:
            Dictionary with results
        """
        print(f"Scrubbing image: {input_path}")

        # Get user's locale preferences
        user_locales = self.get_user_locales()

        # Extract text
        detections = self.ocr.extract_text(input_path)
        print(f"  → Detected {len(detections)} text regions")

        # Merge adjacent text regions for better PII detection
        merged_regions = merge_adjacent_detections(detections)

        # Detect PII
        pii_regions = []
        pii_count = 0

        for merged_region in merged_regions:
            entities = self.detector.analyze_text(
                merged_region.text,
                locales=user_locales,
                auto_detect_locale=(user_locales is None)
            )
            if entities:
                for entity in entities:
                    entity_bbox = get_bbox_for_entity(
                        entity_start=entity.start,
                        entity_end=entity.end,
                        merged_region=merged_region
                    )
                    pii_regions.append(entity_bbox)
                    pii_count += 1
                    print(f"  → Found {entity.entity_type}: '{entity.text}'")

        print(f"  → Total: {len(pii_regions)} regions with PII")

        # Redact
        output_path = self.get_output_path(input_path)

        if pii_regions:
            img = Image.open(input_path)
            scrubbed = self.image_anonymizer.redact_regions(img, pii_regions)
            scrubbed.save(output_path)
            print(f"  → Saved to: {output_path}")
        else:
            # No PII found, just copy
            Image.open(input_path).save(output_path)
            print(f"  → No PII detected, copied to: {output_path}")

        return {
            'input': input_path,
            'output': output_path,
            'type': 'image',
            'text_regions': len(detections),
            'pii_regions': len(pii_regions),
            'pii_count': pii_count
        }

    def detect_pii_spreadsheet(self, input_path: str) -> List[Dict[str, Any]]:
        """
        Detect PII in a spreadsheet without saving

        Args:
            input_path: Path to spreadsheet

        Returns:
            List of detection dictionaries (one per column with PII)
        """
        print(f"Analyzing spreadsheet: {input_path}")

        # Get user's locale preferences (None for spreadsheets = use all patterns)
        # Spreadsheets often contain international data, so we don't restrict by locale
        user_locales = self.get_user_locales()

        # Load file
        ext = Path(input_path).suffix.lower()
        if ext == '.csv':
            df = pd.read_csv(input_path)
        else:
            df = pd.read_excel(input_path, engine='openpyxl')

        print(f"  → Loaded {df.shape[0]} rows, {df.shape[1]} columns")

        # Analyze columns
        pii_detections = []

        for column in df.columns:
            non_null = df[column].dropna()
            if len(non_null) == 0:
                continue

            # Sample
            sample = non_null.sample(n=min(20, len(non_null)), random_state=42)
            sample_text = " | ".join(str(val) for val in sample)

            # Detect (use locales=None for international spreadsheet data)
            entities = self.detector.analyze_text(sample_text, locales=user_locales)
            high_conf = [e for e in entities if e.confidence >= 0.5]

            if high_conf:
                entity_types = [e.entity_type for e in high_conf]
                primary_type = max(set(entity_types), key=entity_types.count)

                pii_detections.append({
                    'entity_type': primary_type,
                    'text': f"Column: {column}",
                    'confidence': sum(e.confidence for e in high_conf) / len(high_conf),
                    'column': column,
                    'entity_types': list(set(entity_types)),
                    'bbox': None
                })
                print(f"  → Column '{column}': {primary_type}")

        print(f"  → Total: {len(pii_detections)} columns with PII")
        return pii_detections

    def save_scrubbed_spreadsheet(
        self,
        input_path: str,
        output_path: str,
        detections: List[Dict[str, Any]],
        selected_indices: List[int]
    ):
        """
        Save scrubbed spreadsheet with only selected columns anonymized

        Args:
            input_path: Input spreadsheet path
            output_path: Output spreadsheet path
            detections: All detected PII columns
            selected_indices: Indices of columns to scrub
        """
        # Load file
        ext = Path(input_path).suffix.lower()
        if ext == '.csv':
            df = pd.read_csv(input_path)
        else:
            df = pd.read_excel(input_path, engine='openpyxl')

        # Build entity map for selected columns only
        entity_map = {}
        for idx in selected_indices:
            detection = detections[idx]
            entity_map[detection['column']] = detection['entity_types']

        # Anonymize
        if entity_map:
            scrubbed_df = self.spreadsheet_anonymizer.anonymize_dataframe(df, entity_map)
        else:
            scrubbed_df = df.copy()

        # Save
        if ext == '.csv':
            scrubbed_df.to_csv(output_path, index=False)
        else:
            scrubbed_df.to_excel(output_path, index=False, engine='openpyxl')

        print(f"Saved scrubbed spreadsheet to: {output_path}")

    def scrub_spreadsheet(self, input_path: str) -> Dict[str, Any]:
        """
        Scrub a spreadsheet file

        Args:
            input_path: Path to spreadsheet

        Returns:
            Dictionary with results
        """
        print(f"Scrubbing spreadsheet: {input_path}")

        # Get user's locale preferences (spreadsheets often have international data)
        user_locales = self.get_user_locales()

        # Load file
        ext = Path(input_path).suffix.lower()
        if ext == '.csv':
            df = pd.read_csv(input_path)
        else:  # xlsx, xls
            df = pd.read_excel(input_path, engine='openpyxl')

        print(f"  → Loaded {df.shape[0]} rows, {df.shape[1]} columns")

        # Analyze columns
        column_analysis = {}

        for column in df.columns:
            non_null = df[column].dropna()
            if len(non_null) == 0:
                continue

            # Sample
            sample = non_null.sample(n=min(20, len(non_null)), random_state=42)
            sample_text = " | ".join(str(val) for val in sample)

            # Detect (use user's locale preference)
            entities = self.detector.analyze_text(sample_text, locales=user_locales)
            high_conf = [e for e in entities if e.confidence >= 0.5]

            if high_conf:
                entity_types = [e.entity_type for e in high_conf]
                primary_type = max(set(entity_types), key=entity_types.count)

                column_analysis[column] = {
                    'primary_type': primary_type,
                    'entity_types': list(set(entity_types))
                }
                print(f"  → Column '{column}': {primary_type}")

        # Build entity map
        entity_map = {}
        for column, info in column_analysis.items():
            entity_map[column] = info['entity_types']

        # Anonymize
        if entity_map:
            scrubbed_df = self.spreadsheet_anonymizer.anonymize_dataframe(df, entity_map)
        else:
            scrubbed_df = df.copy()

        # Save
        output_path = self.get_output_path(input_path)

        if ext == '.csv':
            scrubbed_df.to_csv(output_path, index=False)
        else:
            scrubbed_df.to_excel(output_path, index=False, engine='openpyxl')

        print(f"  → Scrubbed {len(column_analysis)} columns")
        print(f"  → Saved to: {output_path}")

        return {
            'input': input_path,
            'output': output_path,
            'type': 'spreadsheet',
            'rows': df.shape[0],
            'columns': df.shape[1],
            'pii_columns': len(column_analysis),
            'column_analysis': column_analysis
        }

    def scrub_file(self, file_path: str) -> Dict[str, Any]:
        """
        Automatically detect file type and scrub

        Args:
            file_path: Path to file

        Returns:
            Dictionary with results
        """
        file_type = self.detect_file_type(file_path)

        if file_type == "image":
            return self.scrub_image(file_path)
        elif file_type == "spreadsheet":
            return self.scrub_spreadsheet(file_path)
        else:
            raise ValueError(f"Unsupported file type: {Path(file_path).suffix}")


def main():
    """Test the file router"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python file_router.py <file_path>")
        return 1

    router = FileRouter()
    result = router.scrub_file(sys.argv[1])

    print("\nResult:")
    for key, value in result.items():
        if key != 'column_analysis':  # Skip detailed column analysis
            print(f"  {key}: {value}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
