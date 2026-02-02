"""
QR Code Detection using OpenCV

Detects QR codes and barcodes in images for redaction.
Uses OpenCV's built-in QR code detector.

License: Apache 2.0 (OpenCV)
"""

import sys
from typing import List, Optional
from dataclasses import dataclass
import cv2
import numpy as np
from PIL import Image


@dataclass
class QRDetection:
    """Represents a detected QR code or barcode"""
    bbox: tuple  # (x, y, width, height) in pixels
    confidence: float
    data: str  # Decoded content (if available)
    entity_type: str = "QR_CODE"


class QRDetector:
    """
    Detects QR codes and barcodes in images using OpenCV.
    """

    def __init__(self):
        """Initialize QR code detector."""
        self.qr_detector = cv2.QRCodeDetector()

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
        # Convert PIL Image to OpenCV format
        img_array = np.array(image)

        # Handle different image modes
        if len(img_array.shape) == 2:
            # Grayscale - convert to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:
            # RGBA - convert to BGR
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        else:
            # RGB - convert to BGR
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        img_height, img_width = img_bgr.shape[:2]
        detections = []

        # Try to detect and decode QR codes
        try:
            # detectAndDecodeMulti returns: retval, decoded_info, points, straight_qrcode
            retval, decoded_info, points, _ = self.qr_detector.detectAndDecodeMulti(img_bgr)

            if retval and points is not None:
                for i, pts in enumerate(points):
                    if pts is not None and len(pts) >= 4:
                        # Get bounding box from corner points
                        pts = pts.astype(int)
                        x_coords = pts[:, 0]
                        y_coords = pts[:, 1]

                        x_min = int(np.min(x_coords))
                        y_min = int(np.min(y_coords))
                        x_max = int(np.max(x_coords))
                        y_max = int(np.max(y_coords))

                        w = x_max - x_min
                        h = y_max - y_min

                        # Expand bbox slightly
                        pad_x = int(w * expand_bbox)
                        pad_y = int(h * expand_bbox)

                        x_min = max(0, x_min - pad_x)
                        y_min = max(0, y_min - pad_y)
                        w = min(img_width - x_min, w + 2 * pad_x)
                        h = min(img_height - y_min, h + 2 * pad_y)

                        # Get decoded data if available
                        data = decoded_info[i] if decoded_info and i < len(decoded_info) else ""

                        detections.append(QRDetection(
                            bbox=(x_min, y_min, w, h),
                            confidence=0.95,  # High confidence for detected QR codes
                            data=data
                        ))

        except Exception as e:
            sys.stderr.write(f"[QRDetector] Error detecting QR codes: {e}\n")

        # Also try basic QR detection for single codes (fallback)
        if not detections:
            try:
                data, points, _ = self.qr_detector.detectAndDecode(img_bgr)
                if points is not None and len(points) > 0:
                    pts = points[0].astype(int)
                    x_coords = pts[:, 0]
                    y_coords = pts[:, 1]

                    x_min = int(np.min(x_coords))
                    y_min = int(np.min(y_coords))
                    x_max = int(np.max(x_coords))
                    y_max = int(np.max(y_coords))

                    w = x_max - x_min
                    h = y_max - y_min

                    # Expand bbox slightly
                    pad_x = int(w * expand_bbox)
                    pad_y = int(h * expand_bbox)

                    x_min = max(0, x_min - pad_x)
                    y_min = max(0, y_min - pad_y)
                    w = min(img_width - x_min, w + 2 * pad_x)
                    h = min(img_height - y_min, h + 2 * pad_y)

                    detections.append(QRDetection(
                        bbox=(x_min, y_min, w, h),
                        confidence=0.95,
                        data=data if data else ""
                    ))
            except Exception as e:
                sys.stderr.write(f"[QRDetector] Fallback detection error: {e}\n")

        return detections

    def detect_from_file(self, image_path: str) -> List[QRDetection]:
        """
        Detect QR codes in an image file.

        Args:
            image_path: Path to image file

        Returns:
            List of QRDetection objects
        """
        image = Image.open(image_path)
        return self.detect_qr_codes(image)


# Singleton instance for reuse
_detector_instance: Optional[QRDetector] = None


def get_qr_detector() -> QRDetector:
    """Get or create QR detector singleton."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = QRDetector()
    return _detector_instance
