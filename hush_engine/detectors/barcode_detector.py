"""
Barcode Detection using Apple Vision Framework (macOS)

Detects various barcode formats in images for redaction using VNDetectBarcodesRequest.
Falls back to pyzbar if Vision framework is unavailable.

Supports: Code 128, Code 39, Code 93, EAN-13, EAN-8, UPC-A, UPC-E,
          ITF, PDF417, DataMatrix, Aztec, and more.

License: MIT
"""

import io
import logging
import sys
from dataclasses import dataclass
from typing import List, Optional

from PIL import Image

logger = logging.getLogger(__name__)

# Try Vision framework first (macOS native, zero-install)
VISION_AVAILABLE = False
try:
    import Vision
    from Cocoa import NSData
    from Quartz import CIImage
    VISION_AVAILABLE = True
except ImportError:
    pass


@dataclass
class BarcodeDetection:
    """Represents a detected barcode"""
    bbox: tuple  # (x, y, width, height) in pixels
    confidence: float
    data: str  # Decoded content
    barcode_type: str  # Type of barcode (CODE128, EAN13, etc.)
    entity_type: str = "BARCODE"


# Map Vision symbology constants to human-readable names
_VISION_SYMBOLOGY_NAMES = {}


def _get_barcode_symbologies():
    """Get the list of barcode symbologies to detect (excludes QR which is handled separately)."""
    if not VISION_AVAILABLE:
        return []

    symbologies = []
    # Map each symbology and build the name lookup
    _symbology_map = {
        "VNBarcodeSymbologyCode128": "CODE128",
        "VNBarcodeSymbologyCode39": "CODE39",
        "VNBarcodeSymbologyCode39Checksum": "CODE39",
        "VNBarcodeSymbologyCode39FullASCII": "CODE39",
        "VNBarcodeSymbologyCode39FullASCIIChecksum": "CODE39",
        "VNBarcodeSymbologyCode93": "CODE93",
        "VNBarcodeSymbologyCode93i": "CODE93",
        "VNBarcodeSymbologyEAN13": "EAN13",
        "VNBarcodeSymbologyEAN8": "EAN8",
        "VNBarcodeSymbologyUPCE": "UPCE",
        "VNBarcodeSymbologyITF14": "ITF",
        "VNBarcodeSymbologyI2of5": "I25",
        "VNBarcodeSymbologyI2of5Checksum": "I25",
        "VNBarcodeSymbologyPDF417": "PDF417",
        "VNBarcodeSymbologyDataMatrix": "DATAMATRIX",
        "VNBarcodeSymbologyAztec": "AZTEC",
        "VNBarcodeSymbologyCodabar": "CODABAR",
        "VNBarcodeSymbologyGS1DataBar": "DATABAR",
        "VNBarcodeSymbologyGS1DataBarExpanded": "DATABAR_EXP",
        "VNBarcodeSymbologyGS1DataBarLimited": "DATABAR",
    }

    for attr_name, human_name in _symbology_map.items():
        sym = getattr(Vision, attr_name, None)
        if sym is not None:
            symbologies.append(sym)
            _VISION_SYMBOLOGY_NAMES[sym] = human_name

    return symbologies


def _vision_bbox_to_pixel_xywh(vision_box, img_width: int, img_height: int) -> tuple:
    """Convert Vision normalized CGRect to pixel (x, y, w, h) with top-left origin."""
    nx = vision_box.origin.x
    ny = vision_box.origin.y
    nw = vision_box.size.width
    nh = vision_box.size.height

    x = int(nx * img_width)
    w = int(nw * img_width)
    h = int(nh * img_height)
    y = int((1.0 - ny - nh) * img_height)

    return (x, y, w, h)


def _pil_to_ciimage(image: Image.Image):
    """Convert PIL Image to CIImage in memory."""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    png_bytes = buffer.getvalue()

    ns_data = NSData.dataWithBytes_length_(png_bytes, len(png_bytes))
    ci_image = CIImage.imageWithData_(ns_data)
    if ci_image is None:
        raise ValueError("Failed to create CIImage from PIL Image")
    return ci_image


class VisionBarcodeDetector:
    """
    Detects barcodes in images using Apple Vision Framework.

    Uses VNDetectBarcodesRequest with linear barcode symbologies (excludes QR codes
    which are handled by qr_detector.py).
    """

    def __init__(self):
        self._symbologies = _get_barcode_symbologies() if VISION_AVAILABLE else []

    def detect_barcodes(
        self,
        image: Image.Image,
        expand_bbox: float = 0.1
    ) -> List[BarcodeDetection]:
        """
        Detect barcodes in an image.

        Args:
            image: PIL Image to analyze
            expand_bbox: Fraction to expand bounding box (default 10%)

        Returns:
            List of BarcodeDetection objects with bounding boxes
        """
        if not VISION_AVAILABLE or not self._symbologies:
            return []

        img_width, img_height = image.size
        ci_image = _pil_to_ciimage(image)

        # Check CIImage extent for resolution differences
        ci_extent = ci_image.extent()
        ci_width = ci_extent.size.width
        ci_height = ci_extent.size.height
        if abs(ci_width - img_width) > 1 or abs(ci_height - img_height) > 1:
            img_width = int(ci_width)
            img_height = int(ci_height)

        request = Vision.VNDetectBarcodesRequest.alloc().init()
        request.setSymbologies_(self._symbologies)

        handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(ci_image, None)
        success, error = handler.performRequests_error_([request], None)

        if not success:
            sys.stderr.write(f"[BarcodeDetector] Vision barcode detection failed: {error}\n")
            return []

        results = request.results()
        if not results:
            return []

        detections = []
        for observation in results:
            bbox = _vision_bbox_to_pixel_xywh(
                observation.boundingBox(), img_width, img_height
            )
            x, y, w, h = bbox

            # Expand bbox slightly
            pad_x = int(w * expand_bbox)
            pad_y = int(h * expand_bbox)
            x = max(0, x - pad_x)
            y = max(0, y - pad_y)
            w = min(img_width - x, w + 2 * pad_x)
            h = min(img_height - y, h + 2 * pad_y)

            # Get decoded data
            data = ""
            try:
                payload = observation.payloadStringValue()
                if payload:
                    data = str(payload)
            except Exception:
                pass

            # Get barcode type name
            symbology = observation.symbology()
            barcode_type = _VISION_SYMBOLOGY_NAMES.get(symbology, str(symbology))

            confidence = float(observation.confidence())

            detections.append(BarcodeDetection(
                bbox=(x, y, w, h),
                confidence=confidence,
                data=data,
                barcode_type=barcode_type
            ))

        return detections

    def detect_from_file(self, image_path: str) -> List[BarcodeDetection]:
        """Detect barcodes in an image file."""
        image = Image.open(image_path)
        return self.detect_barcodes(image)


# Provide original class name for compatibility
BarcodeDetector = VisionBarcodeDetector


# Singleton instance for reuse
_detector_instance: Optional[VisionBarcodeDetector] = None


def get_barcode_detector():
    """Get or create barcode detector singleton.

    Uses Vision framework on macOS, falls back to pyzbar.
    """
    global _detector_instance
    if _detector_instance is None:
        if VISION_AVAILABLE:
            _detector_instance = VisionBarcodeDetector()
        else:
            try:
                from hush_engine.detectors.barcode_detector_pyzbar import BarcodeDetector as PyzbarBarcodeDetector
                _detector_instance = PyzbarBarcodeDetector()
                sys.stderr.write("[BarcodeDetector] Using pyzbar fallback\n")
            except ImportError:
                sys.stderr.write("[BarcodeDetector] No barcode detection backend available\n")
                _detector_instance = VisionBarcodeDetector()
    return _detector_instance
