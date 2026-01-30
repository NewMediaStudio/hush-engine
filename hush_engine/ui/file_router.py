#!/usr/bin/env python3
"""
File Router - Routes dropped files to appropriate scrubber
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import mimetypes

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocr import VisionOCR
from detectors import PIIDetector
from anonymizers import ImageAnonymizer, SpreadsheetAnonymizer
from pdf import PDFProcessor
from PIL import Image
import pandas as pd
import tempfile
import os


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
        self._image_anonymizer = None
        self._spreadsheet_anonymizer = None
        self._pdf_processor = None

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
        """Lazy-load PDF processor"""
        if self._pdf_processor is None:
            self._pdf_processor = PDFProcessor(dpi=300)
        return self._pdf_processor

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

    def detect_pii_image(self, input_path: str) -> Dict[str, Any]:
        """
        Detect PII in an image without saving

        Args:
            input_path: Path to image

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

        # Detect PII
        pii_detections = []
        for detection in ocr_detections:
            entities = self.detector.analyze_text(detection.text)
            
            for entity in entities:
                pii_detections.append({
                    'entity_type': entity.entity_type,
                    'text': entity.text,
                    'confidence': entity.confidence,
                    'bbox': detection.bbox
                })

        return {
            'detections': pii_detections,
            'all_text_blocks': all_text_blocks
        }

    def detect_pii_pdf(self, input_path: str) -> Dict[str, Any]:
        """
        Detect PII in a PDF without saving

        Args:
            input_path: Path to PDF file

        Returns:
            Dictionary with 'detections' (PII with page numbers), 'all_text_blocks', and 'total_pages'
        """
        # Convert PDF to images
        page_images = self.pdf_processor.pdf_to_images(input_path)
        total_pages = len(page_images)
        
        print(f"Processing {total_pages} page(s) from PDF", file=sys.stderr)
        
        all_detections = []
        all_text_blocks = []
        
        # Process each page
        for page_num, page_image in enumerate(page_images, start=1):
            # Save page image temporarily for OCR processing
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                temp_path = tmp_file.name
                page_image.save(temp_path)
            
            try:
                # Extract text from this page
                ocr_detections = self.ocr.extract_text(temp_path)
                
                # Convert OCR detections to text blocks with page number
                for detection in ocr_detections:
                    all_text_blocks.append({
                        'text': detection.text,
                        'bbox': detection.bbox,
                        'page': page_num
                    })
                
                # Detect PII in this page
                for detection in ocr_detections:
                    entities = self.detector.analyze_text(detection.text)
                    
                    for entity in entities:
                        all_detections.append({
                            'entity_type': entity.entity_type,
                            'text': entity.text,
                            'confidence': entity.confidence,
                            'bbox': detection.bbox,
                            'page': page_num
                        })
                
                print(f"Page {page_num}: Found {len([d for d in all_detections if d['page'] == page_num])} PII detections", file=sys.stderr)
                
            finally:
                # Clean up temporary file
                os.unlink(temp_path)
        
        return {
            'detections': all_detections,
            'all_text_blocks': all_text_blocks,
            'total_pages': total_pages
        }
    
    def get_pdf_page_image(self, input_path: str, page_number: int) -> Dict[str, str]:
        """
        Get a specific page from a PDF as a PNG image
        
        Args:
            input_path: Path to PDF file
            page_number: Page number (1-indexed)
            
        Returns:
            Dictionary with 'image_path' pointing to temporary PNG file
        """
        # Convert just the requested page
        page_images = self.pdf_processor.pdf_to_images(input_path, first_page=page_number, last_page=page_number)
        
        if not page_images:
            raise ValueError(f"Could not extract page {page_number} from PDF")
        
        page_image = page_images[0]
        
        # Save to temporary file
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        page_image.save(temp_path)
        
        return {'image_path': temp_path}

    def save_scrubbed_image(
        self,
        input_path: str,
        output_path: str,
        detections: List[Dict[str, Any]],
        selected_indices: List[int]
    ):
        """
        Save scrubbed image with only selected detections redacted

        Args:
            input_path: Input image path
            output_path: Output image path
            detections: All detected PII items
            selected_indices: Indices of items to scrub
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
        print(f"Saved scrubbed image to: {output_path}")

    def save_scrubbed_pdf(
        self,
        input_path: str,
        output_path: str,
        detections: List[Dict[str, Any]],
        selected_indices: List[int]
    ):
        """
        Save scrubbed PDF with only selected detections redacted

        Args:
            input_path: Input PDF path
            output_path: Output PDF path
            detections: All detected PII items (with 'page' field)
            selected_indices: Indices of items to scrub
        """
        # Convert PDF to images
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
                # Save to temp file for processing
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    temp_path = tmp_file.name
                    page_image.save(temp_path)
                
                try:
                    # Load and redact
                    img = Image.open(temp_path)
                    scrubbed = self.image_anonymizer.redact_regions(img, page_bboxes)
                    scrubbed_pages.append(scrubbed)
                    print(f"Page {page_num}: Redacted {len(page_bboxes)} region(s)", file=sys.stderr)
                finally:
                    os.unlink(temp_path)
            else:
                # No redactions on this page
                scrubbed_pages.append(page_image)
                print(f"Page {page_num}: No redactions", file=sys.stderr)
        
        # Convert scrubbed images to PDF
        self.pdf_processor.images_to_pdf(scrubbed_pages, output_path)
        print(f"Saved scrubbed PDF to: {output_path}")

    def scrub_image(self, input_path: str) -> Dict[str, Any]:
        """
        Scrub an image file

        Args:
            input_path: Path to image

        Returns:
            Dictionary with results
        """
        print(f"Scrubbing image: {input_path}")

        # Extract text
        detections = self.ocr.extract_text(input_path)
        print(f"  → Detected {len(detections)} text regions")

        # Detect PII
        pii_regions = []
        pii_count = 0

        for detection in detections:
            entities = self.detector.analyze_text(detection.text)
            if entities:
                pii_regions.append(detection.bbox)
                pii_count += len(entities)
                for entity in entities:
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

            # Detect
            entities = self.detector.analyze_text(sample_text)
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

            # Detect
            entities = self.detector.analyze_text(sample_text)
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
