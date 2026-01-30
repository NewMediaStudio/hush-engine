"""
Apple Vision Framework wrapper for OCR
Uses hardware-accelerated text recognition on Apple Silicon
"""

from dataclasses import dataclass
from typing import List, Tuple
from PIL import Image
import os

try:
    import Vision
    from Quartz import CIImage, CIContext
    from Cocoa import NSURL
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    print("Warning: PyObjC Vision framework not available. OCR will not work.")


@dataclass
class TextDetection:
    """Represents detected text with its bounding box"""
    text: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in PIL coordinates


class VisionOCR:
    """
    Wrapper around Apple's Vision Framework for OCR

    Uses VNRecognizeTextRequest for hardware-accelerated text recognition.
    Handles coordinate transformation from Vision (normalized, bottom-left origin)
    to PIL (pixel-based, top-left origin).
    """

    def __init__(self, recognition_level: str = "accurate"):
        """
        Initialize the OCR engine

        Args:
            recognition_level: "fast" or "accurate" (default: accurate)
        """
        if not VISION_AVAILABLE:
            raise RuntimeError("Vision framework not available. Install pyobjc-framework-Vision.")

        self.recognition_level = recognition_level

        # Map recognition level to Vision constants
        if recognition_level == "fast":
            self.vision_level = Vision.VNRequestTextRecognitionLevelFast
        else:  # "accurate"
            self.vision_level = Vision.VNRequestTextRecognitionLevelAccurate

    def extract_text(self, image_path: str) -> List[TextDetection]:
        """
        Extract text from an image

        Args:
            image_path: Path to the image file

        Returns:
            List of TextDetection objects with text and coordinates
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Get image dimensions using PIL for coordinate transformation
        pil_image = Image.open(image_path)
        image_width, image_height = pil_image.size

        # Load image with Vision framework
        image_url = NSURL.fileURLWithPath_(image_path)

        # Create a CIImage from the file
        ci_image = CIImage.imageWithContentsOfURL_(image_url)
        if ci_image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Check CIImage extent to see if Vision is using a different resolution
        ci_extent = ci_image.extent()
        ci_width = ci_extent.size.width
        ci_height = ci_extent.size.height
        
        # If they don't match, Vision is using different dimensions - update our transform
        if abs(ci_width - image_width) > 1 or abs(ci_height - image_height) > 1:
            image_width = int(ci_width)
            image_height = int(ci_height)

        # Create the text recognition request
        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(self.vision_level)
        request.setUsesLanguageCorrection_(True)

        # Create request handler and perform the request
        handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(ci_image, None)
        success = handler.performRequests_error_([request], None)

        if not success:
            raise RuntimeError(f"Vision OCR failed on image: {image_path}")

        # Extract results
        results = request.results()
        if not results:
            return []  # No text detected

        detections = []
        for observation in results:
            # Get the top candidate (highest confidence)
            top_candidates = observation.topCandidates_(1)
            if not top_candidates or len(top_candidates) == 0:
                continue

            candidate = top_candidates[0]
            # Convert PyObjC unicode to native Python string for Spacy compatibility
            text = str(candidate.string())
            confidence = float(candidate.confidence())

            # Get bounding box
            bbox = observation.boundingBox()

            # Vision bounding box is (x, y, width, height) normalized (0-1)
            # with origin at bottom-left
            vision_box = (
                bbox.origin.x,
                bbox.origin.y,
                bbox.size.width,
                bbox.size.height
            )

            # Transform to PIL coordinates (x1, y1, x2, y2) in pixels
            pil_bbox = self.vision_to_pil_coords(
                vision_box,
                image_height,
                image_width
            )

            detections.append(TextDetection(
                text=text,
                confidence=confidence,
                bbox=pil_bbox
            ))

        return detections

    @staticmethod
    def vision_to_pil_coords(
        vision_box: Tuple[float, float, float, float],
        image_height: int,
        image_width: int
    ) -> Tuple[float, float, float, float]:
        """
        Transform Vision coordinates to PIL coordinates

        Vision uses normalized coords (0-1) with origin at bottom-left.
        PIL uses pixel coords with origin at top-left.

        Args:
            vision_box: (x, y, width, height) in Vision format
            image_height: Height of image in pixels
            image_width: Width of image in pixels

        Returns:
            (x1, y1, x2, y2) in PIL format
        """
        x, y, width, height = vision_box

        # Denormalize to pixels
        x_pixel = x * image_width
        width_pixel = width * image_width
        height_pixel = height * image_height

        # Flip Y-axis (account for bbox height!)
        y_pixel = (1 - y - height) * image_height

        return (
            x_pixel,
            y_pixel,
            x_pixel + width_pixel,
            y_pixel + height_pixel
        )
