"""
Apple Vision Framework wrapper for OCR
Uses hardware-accelerated text recognition on Apple Silicon
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
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
    char_boxes: Optional[List[Tuple[float, float, float, float]]] = None  # Per-character bounding boxes (optional)


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

            # Get bounding box for the entire text block
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

            # Extract character-level bounding boxes for precise entity localization
            char_boxes = []
            try:
                for i in range(len(text)):
                    # Get bounding box for each character
                    char_range = (i, 1)  # (location, length)
                    char_bbox = observation.boundingBoxForRange_error_(char_range, None)[0]
                    if char_bbox:
                        char_vision_box = (
                            char_bbox.origin.x,
                            char_bbox.origin.y,
                            char_bbox.size.width,
                            char_bbox.size.height
                        )
                        char_pil_bbox = self.vision_to_pil_coords(
                            char_vision_box,
                            image_height,
                            image_width
                        )
                        char_boxes.append(char_pil_bbox)
                    else:
                        # If character bbox is not available, use None
                        char_boxes.append(None)
            except Exception:
                # If character-level boxes are not available, leave as empty list
                char_boxes = []

            detections.append(TextDetection(
                text=text,
                confidence=confidence,
                bbox=pil_bbox,
                char_boxes=char_boxes if char_boxes else None
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

    @staticmethod
    def calculate_substring_bbox(
        text: str,
        char_boxes: Optional[List[Optional[Tuple[float, float, float, float]]]],
        start: int,
        end: int,
        fallback_bbox: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float, float]:
        """
        Calculate precise bounding box for a substring based on character positions.

        Args:
            text: The full text string
            char_boxes: List of character-level bounding boxes (x1, y1, x2, y2)
            start: Start index of substring in text
            end: End index of substring in text
            fallback_bbox: Full text bounding box to use as fallback

        Returns:
            (x1, y1, x2, y2) bounding box for the substring
        """
        # Validate indices
        if start < 0 or end > len(text) or start >= end:
            return fallback_bbox

        # If no character boxes available, estimate proportionally
        if not char_boxes or len(char_boxes) != len(text):
            return VisionOCR._estimate_substring_bbox(
                text, start, end, fallback_bbox
            )

        # Collect valid character boxes for the substring
        substring_boxes = []
        for i in range(start, end):
            if i < len(char_boxes) and char_boxes[i] is not None:
                substring_boxes.append(char_boxes[i])

        # If we have valid character boxes, compute their union
        if substring_boxes:
            x1 = min(box[0] for box in substring_boxes)
            y1 = min(box[1] for box in substring_boxes)
            x2 = max(box[2] for box in substring_boxes)
            y2 = max(box[3] for box in substring_boxes)
            return (x1, y1, x2, y2)

        # Fallback to estimation if character boxes are incomplete
        return VisionOCR._estimate_substring_bbox(
            text, start, end, fallback_bbox
        )

    @staticmethod
    def _estimate_substring_bbox(
        text: str,
        start: int,
        end: int,
        full_bbox: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float, float]:
        """
        Estimate substring bounding box using proportional division.
        
        Assumes uniform character spacing (less accurate but works as fallback).

        Args:
            text: The full text string
            start: Start index of substring
            end: End index of substring
            full_bbox: Bounding box for the full text (x1, y1, x2, y2)

        Returns:
            Estimated (x1, y1, x2, y2) for the substring
        """
        if not text or start >= end:
            return full_bbox

        x1, y1, x2, y2 = full_bbox
        text_width = x2 - x1
        text_len = len(text)

        # Calculate proportional positions (simple linear interpolation)
        char_width = text_width / text_len if text_len > 0 else 0
        
        # Estimate x coordinates based on character positions
        sub_x1 = x1 + (start * char_width)
        sub_x2 = x1 + (end * char_width)

        # Y coordinates stay the same (same line)
        return (sub_x1, y1, sub_x2, y2)
