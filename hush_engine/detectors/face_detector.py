"""
Face Detection using Apple Vision Framework (macOS)

Uses VNDetectFaceRectanglesRequest for hardware-accelerated face detection.
Falls back to OpenCV Haar Cascades if Vision framework is unavailable.

License: MIT
"""

import io
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from PIL import Image

# Try Vision framework first (macOS native, zero-install)
VISION_AVAILABLE = False
try:
    import Vision
    from Quartz import CIImage
    from Cocoa import NSData
    VISION_AVAILABLE = True
except ImportError:
    pass


@dataclass
class FaceDetection:
    """Represents a detected face"""
    bbox: tuple  # (x, y, width, height) in pixels
    confidence: float
    entity_type: str = "FACE"


def _vision_normalized_to_pixel_xywh(
    vision_box,
    img_width: int,
    img_height: int
) -> tuple:
    """
    Convert Vision normalized bbox (bottom-left origin) to pixel (x, y, w, h)
    with top-left origin, matching the OpenCV convention used throughout the engine.

    Args:
        vision_box: CGRect with origin.x, origin.y, size.width, size.height (0-1, bottom-left)
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        (x, y, width, height) in pixels with top-left origin
    """
    nx = vision_box.origin.x
    ny = vision_box.origin.y
    nw = vision_box.size.width
    nh = vision_box.size.height

    # Denormalize
    x = int(nx * img_width)
    w = int(nw * img_width)
    h = int(nh * img_height)
    # Flip Y-axis: Vision origin is bottom-left, PIL/OpenCV is top-left
    y = int((1.0 - ny - nh) * img_height)

    return (x, y, w, h)


def _pil_to_ciimage(image: Image.Image):
    """Convert PIL Image to CIImage in memory (same pattern as vision_ocr.py)."""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    png_bytes = buffer.getvalue()

    ns_data = NSData.dataWithBytes_length_(png_bytes, len(png_bytes))
    ci_image = CIImage.imageWithData_(ns_data)
    if ci_image is None:
        raise ValueError("Failed to create CIImage from PIL Image")

    return ci_image


class VisionFaceDetector:
    """
    Detects faces in images using Apple Vision Framework.

    Uses VNDetectFaceRectanglesRequest which provides:
    - Hardware-accelerated detection on Apple Silicon
    - Real confidence scores (unlike Haar's hardcoded values)
    - Frontal and profile face detection in a single pass
    """

    def __init__(self, min_confidence: float = 0.5):
        self.min_confidence = min_confidence

    def _expand_bbox_for_head_shoulders(
        self,
        bbox: tuple,
        img_width: int,
        img_height: int,
        top_expand: float = 0.25,
        side_expand: float = 0.25,
        bottom_expand: float = 0.5
    ) -> tuple:
        """
        Expand a face bounding box to include full head and partial neck/shoulders.

        Args:
            bbox: (x, y, w, h) of detected face
            img_width: Image width for clamping
            img_height: Image height for clamping
            top_expand: Fraction to expand upward (for hair/forehead) - 25%
            side_expand: Fraction to expand sideways (for ears) - 25%
            bottom_expand: Fraction to expand downward (for neck) - 50%

        Returns:
            Expanded (x, y, w, h) tuple clamped to image boundaries
        """
        x, y, w, h = bbox

        top_pad = int(h * top_expand)
        side_pad = int(w * side_expand)
        bottom_pad = int(h * bottom_expand)

        new_x = max(0, x - side_pad)
        new_y = max(0, y - top_pad)
        new_w = min(img_width - new_x, w + 2 * side_pad)
        new_h = min(img_height - new_y, h + top_pad + bottom_pad)

        return (new_x, new_y, new_w, new_h)

    def detect_faces(
        self,
        image: Image.Image,
        include_profiles: bool = True,
        scale_factor: float = 1.1,
        min_size: tuple = (30, 30)
    ) -> List[FaceDetection]:
        """
        Detect faces in an image using Vision framework.

        Args:
            image: PIL Image to analyze
            include_profiles: Ignored (Vision detects all orientations natively)
            scale_factor: Ignored (Vision handles multi-scale internally)
            min_size: Minimum face size in pixels (width, height)

        Returns:
            List of FaceDetection objects with bounding boxes
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

        request = Vision.VNDetectFaceRectanglesRequest.alloc().init()
        handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(ci_image, None)

        success, error = handler.performRequests_error_([request], None)
        if not success:
            sys.stderr.write(f"[FaceDetector] Vision face detection failed: {error}\n")
            return []

        results = request.results()
        if not results:
            return []

        detections = []
        for observation in results:
            confidence = float(observation.confidence())
            if confidence < self.min_confidence:
                continue

            # Convert normalized bbox to pixel (x, y, w, h)
            bbox = _vision_normalized_to_pixel_xywh(
                observation.boundingBox(), img_width, img_height
            )

            # Filter by minimum size
            if bbox[2] < min_size[0] or bbox[3] < min_size[1]:
                continue

            # Expand bbox to include head and shoulders
            expanded_bbox = self._expand_bbox_for_head_shoulders(
                bbox, img_width, img_height
            )

            detections.append(FaceDetection(
                bbox=expanded_bbox,
                confidence=confidence
            ))

        return detections

    def detect_from_file(self, image_path: str) -> List[FaceDetection]:
        """Detect faces in an image file."""
        image = Image.open(image_path)
        return self.detect_faces(image)


# Also provide the OpenCV fallback class name for compatibility
FaceDetector = VisionFaceDetector


# Singleton instance for reuse
_detector_instance: Optional[VisionFaceDetector] = None


def get_face_detector(min_confidence: float = 0.5):
    """Get or create face detector singleton.

    Uses Vision framework on macOS, falls back to OpenCV Haar Cascades.
    """
    global _detector_instance
    if _detector_instance is None:
        if VISION_AVAILABLE:
            _detector_instance = VisionFaceDetector(min_confidence)
        else:
            try:
                from hush_engine.detectors.face_detector_cv import FaceDetector as CVFaceDetector
                _detector_instance = CVFaceDetector(min_confidence)
                sys.stderr.write("[FaceDetector] Using OpenCV fallback\n")
            except ImportError:
                sys.stderr.write("[FaceDetector] No face detection backend available\n")
                _detector_instance = VisionFaceDetector(min_confidence)  # Will fail gracefully
    return _detector_instance


def detect_faces_in_image(image_path: str, min_confidence: float = 0.5) -> List[Dict[str, Any]]:
    """
    Convenience function to detect faces and return as dictionaries.

    Args:
        image_path: Path to image file
        min_confidence: Minimum confidence threshold

    Returns:
        List of detection dictionaries with bbox, confidence, entity_type
    """
    detector = get_face_detector(min_confidence)
    detections = detector.detect_from_file(image_path)

    return [
        {
            'bbox': detection.bbox,
            'confidence': detection.confidence,
            'entity_type': detection.entity_type,
            'text': '[FACE]'
        }
        for detection in detections
    ]
