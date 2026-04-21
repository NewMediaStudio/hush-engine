"""
QR Code Detection using Apple Vision Framework (macOS)

Detects QR codes in images for redaction using VNDetectBarcodesRequest.
Falls back to OpenCV QRCodeDetector if Vision framework is unavailable.

License: MIT
"""

import io
import sys
from dataclasses import dataclass
from typing import List, Optional

from PIL import Image

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
class QRDetection:
    """Represents a detected QR code or barcode"""
    bbox: tuple  # (x, y, width, height) in pixels
    confidence: float
    data: str  # Decoded content (if available)
    entity_type: str = "QR_CODE"


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


class VisionQRDetector:
    """
    Detects QR codes in images using Apple Vision Framework.

    Uses VNDetectBarcodesRequest with QR symbology which provides:
    - Hardware-accelerated detection on Apple Silicon
    - Native QR code decoding (payload extraction)
    - Multi-QR detection in a single pass
    """

    def detect_qr_codes(
        self,
        image: Image.Image,
        expand_bbox: float = 0.1
    ) -> List[QRDetection]:
        """
        Detect QR codes in an image.

        Args:
            image: PIL Image to analyze
            expand_bbox: Fraction to expand bounding box (default 10%)

        Returns:
            List of QRDetection objects with bounding boxes
        """
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
        # Filter to QR codes only (barcodes handled by barcode_detector.py)
        request.setSymbologies_([Vision.VNBarcodeSymbologyQR])

        handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(ci_image, None)
        success, error = handler.performRequests_error_([request], None)

        if not success:
            sys.stderr.write(f"[QRDetector] Vision QR detection failed: {error}\n")
            return []

        results = request.results()
        if not results:
            return []

        detections = []
        for observation in results:
            # Get bounding box in pixel coordinates
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

            confidence = float(observation.confidence())

            detections.append(QRDetection(
                bbox=(x, y, w, h),
                confidence=confidence,
                data=data
            ))

        return detections

    def detect_from_file(self, image_path: str) -> List[QRDetection]:
        """Detect QR codes in an image file."""
        image = Image.open(image_path)
        return self.detect_qr_codes(image)


# Provide the original class name for compatibility
QRDetector = VisionQRDetector


# Singleton instance for reuse
_detector_instance: Optional[VisionQRDetector] = None


def get_qr_detector():
    """Get or create QR detector singleton.

    Uses Vision framework on macOS, falls back to OpenCV.
    """
    global _detector_instance
    if _detector_instance is None:
        if VISION_AVAILABLE:
            _detector_instance = VisionQRDetector()
        else:
            try:
                from hush_engine.detectors.qr_detector_cv import QRDetector as CVQRDetector
                _detector_instance = CVQRDetector()
                sys.stderr.write("[QRDetector] Using OpenCV fallback\n")
            except ImportError:
                sys.stderr.write("[QRDetector] No QR detection backend available\n")
                _detector_instance = VisionQRDetector()  # Will fail gracefully
    return _detector_instance
