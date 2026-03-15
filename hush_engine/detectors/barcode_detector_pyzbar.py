"""
Barcode Detection using pyzbar

Detects various barcode formats in images for redaction.
Supports Code 128, Code 39, EAN-13, EAN-8, UPC-A, UPC-E, and more.

License: MIT (pyzbar)
"""

import logging
from typing import List, Optional
from dataclasses import dataclass
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

try:
    from pyzbar import pyzbar
    from pyzbar.pyzbar import ZBarSymbol
    PYZBAR_AVAILABLE = True
except ImportError:
    PYZBAR_AVAILABLE = False
    logger.debug("[BarcodeDetector] pyzbar not installed - barcode detection disabled")


@dataclass
class BarcodeDetection:
    """Represents a detected barcode"""
    bbox: tuple  # (x, y, width, height) in pixels
    confidence: float
    data: str  # Decoded content
    barcode_type: str  # Type of barcode (CODE128, EAN13, etc.)
    entity_type: str = "BARCODE"


class BarcodeDetector:
    """
    Detects barcodes in images using pyzbar.

    Supports multiple barcode formats:
    - Code 128, Code 39, Code 93
    - EAN-13, EAN-8
    - UPC-A, UPC-E
    - ISBN-10, ISBN-13
    - ITF (Interleaved 2 of 5)
    - PDF417
    - DataMatrix
    """

    def __init__(self):
        """Initialize barcode detector."""
        if not PYZBAR_AVAILABLE:
            sys.stderr.write("[BarcodeDetector] pyzbar not available\n")

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
        if not PYZBAR_AVAILABLE:
            return []

        # Convert PIL Image to format pyzbar can read
        img_array = np.array(image)

        # Get image dimensions
        img_height, img_width = img_array.shape[:2]

        detections = []

        try:
            # Decode all barcodes (excluding QR codes which are handled separately)
            # ZBarSymbol types for linear barcodes only
            barcode_types = [
                ZBarSymbol.CODE128,
                ZBarSymbol.CODE39,
                ZBarSymbol.CODE93,
                ZBarSymbol.EAN13,
                ZBarSymbol.EAN8,
                ZBarSymbol.UPCA,
                ZBarSymbol.UPCE,
                ZBarSymbol.I25,  # Interleaved 2 of 5
                ZBarSymbol.PDF417,
                ZBarSymbol.DATABAR,
                ZBarSymbol.DATABAR_EXP,
            ]

            decoded_objects = pyzbar.decode(img_array, symbols=barcode_types)

            for obj in decoded_objects:
                # Get bounding box
                rect = obj.rect
                x, y, w, h = rect.left, rect.top, rect.width, rect.height

                # Expand bbox slightly
                pad_x = int(w * expand_bbox)
                pad_y = int(h * expand_bbox)

                x = max(0, x - pad_x)
                y = max(0, y - pad_y)
                w = min(img_width - x, w + 2 * pad_x)
                h = min(img_height - y, h + 2 * pad_y)

                # Get decoded data
                try:
                    data = obj.data.decode('utf-8')
                except (UnicodeDecodeError, AttributeError):
                    data = str(obj.data)

                # Get barcode type
                barcode_type = obj.type

                detections.append(BarcodeDetection(
                    bbox=(x, y, w, h),
                    confidence=0.95,  # High confidence for decoded barcodes
                    data=data,
                    barcode_type=barcode_type
                ))

        except Exception as e:
            sys.stderr.write(f"[BarcodeDetector] Error detecting barcodes: {e}\n")

        return detections

    def detect_from_file(self, image_path: str) -> List[BarcodeDetection]:
        """
        Detect barcodes in an image file.

        Args:
            image_path: Path to image file

        Returns:
            List of BarcodeDetection objects
        """
        image = Image.open(image_path)
        return self.detect_barcodes(image)


# Singleton instance for reuse
_detector_instance: Optional[BarcodeDetector] = None


def get_barcode_detector() -> BarcodeDetector:
    """Get or create barcode detector singleton."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = BarcodeDetector()
    return _detector_instance
