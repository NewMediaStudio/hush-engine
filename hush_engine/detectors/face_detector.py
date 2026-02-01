"""
Face Detection using OpenCV Haar Cascades

Lightweight face detection without heavy ML dependencies.
Uses pre-trained Haar cascade classifier from OpenCV.

License: Apache 2.0 (OpenCV)
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import cv2
import numpy as np
from PIL import Image


@dataclass
class FaceDetection:
    """Represents a detected face"""
    bbox: tuple  # (x, y, width, height) in pixels
    confidence: float
    entity_type: str = "FACE"


class FaceDetector:
    """
    Detects faces in images using OpenCV Haar Cascades.

    Uses frontal face detection by default. Can also detect profile faces.
    """

    def __init__(self, min_confidence: float = 0.5):
        """
        Initialize face detector.

        Args:
            min_confidence: Minimum confidence threshold (0.0 - 1.0)
                           Maps to minNeighbors parameter in cascade classifier
        """
        self.min_confidence = min_confidence

        # Load Haar cascade classifiers
        # These are bundled with OpenCV
        cascade_path = cv2.data.haarcascades

        self.face_cascade = cv2.CascadeClassifier(
            cascade_path + 'haarcascade_frontalface_default.xml'
        )
        self.profile_cascade = cv2.CascadeClassifier(
            cascade_path + 'haarcascade_profileface.xml'
        )

        if self.face_cascade.empty():
            sys.stderr.write("[FaceDetector] Warning: Could not load frontal face cascade\n")
        if self.profile_cascade.empty():
            sys.stderr.write("[FaceDetector] Warning: Could not load profile face cascade\n")

    def detect_faces(
        self,
        image: Image.Image,
        include_profiles: bool = True,
        scale_factor: float = 1.1,
        min_size: tuple = (30, 30)
    ) -> List[FaceDetection]:
        """
        Detect faces in an image.

        Args:
            image: PIL Image to analyze
            include_profiles: Also detect profile (side) faces
            scale_factor: Image scale factor for multi-scale detection
            min_size: Minimum face size in pixels (width, height)

        Returns:
            List of FaceDetection objects with bounding boxes
        """
        # Convert PIL Image to OpenCV format (BGR)
        img_array = np.array(image)

        # Handle different image modes
        if len(img_array.shape) == 2:
            # Grayscale
            gray = img_array
        elif img_array.shape[2] == 4:
            # RGBA - convert to RGB then grayscale
            rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        else:
            # RGB
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Map confidence threshold to minNeighbors
        # Higher confidence = more neighbors required = fewer false positives
        # confidence 0.3 -> minNeighbors 3
        # confidence 0.5 -> minNeighbors 5
        # confidence 0.8 -> minNeighbors 8
        min_neighbors = max(3, int(self.min_confidence * 10))

        detections = []
        seen_bboxes = set()

        # Detect frontal faces
        if not self.face_cascade.empty():
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=min_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            for (x, y, w, h) in faces:
                bbox = (int(x), int(y), int(w), int(h))
                bbox_key = f"{x},{y},{w},{h}"
                if bbox_key not in seen_bboxes:
                    seen_bboxes.add(bbox_key)
                    # Confidence is approximated - Haar cascades don't provide exact confidence
                    # We use a fixed high confidence since minNeighbors filtering already applied
                    detections.append(FaceDetection(
                        bbox=bbox,
                        confidence=0.85
                    ))

        # Detect profile faces
        if include_profiles and not self.profile_cascade.empty():
            profiles = self.profile_cascade.detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=min_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            for (x, y, w, h) in profiles:
                bbox = (int(x), int(y), int(w), int(h))
                bbox_key = f"{x},{y},{w},{h}"
                # Avoid duplicates (frontal might overlap with profile)
                if bbox_key not in seen_bboxes and not self._is_overlapping(bbox, seen_bboxes):
                    seen_bboxes.add(bbox_key)
                    detections.append(FaceDetection(
                        bbox=bbox,
                        confidence=0.80  # Slightly lower for profiles
                    ))

            # Also check flipped image for profiles facing the other direction
            gray_flipped = cv2.flip(gray, 1)  # Horizontal flip
            profiles_flipped = self.profile_cascade.detectMultiScale(
                gray_flipped,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=min_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            img_width = gray.shape[1]
            for (x, y, w, h) in profiles_flipped:
                # Flip x coordinate back
                x_orig = img_width - x - w
                bbox = (int(x_orig), int(y), int(w), int(h))
                bbox_key = f"{x_orig},{y},{w},{h}"
                if bbox_key not in seen_bboxes and not self._is_overlapping(bbox, seen_bboxes):
                    seen_bboxes.add(bbox_key)
                    detections.append(FaceDetection(
                        bbox=bbox,
                        confidence=0.80
                    ))

        return detections

    def _is_overlapping(self, bbox: tuple, seen_bboxes: set, threshold: float = 0.5) -> bool:
        """
        Check if bbox significantly overlaps with any existing bbox.

        Args:
            bbox: (x, y, w, h) tuple
            seen_bboxes: Set of "x,y,w,h" strings
            threshold: IoU threshold for considering overlap

        Returns:
            True if overlapping with existing detection
        """
        x1, y1, w1, h1 = bbox

        for seen_key in seen_bboxes:
            x2, y2, w2, h2 = map(int, seen_key.split(','))

            # Calculate intersection
            ix1 = max(x1, x2)
            iy1 = max(y1, y2)
            ix2 = min(x1 + w1, x2 + w2)
            iy2 = min(y1 + h1, y2 + h2)

            if ix2 > ix1 and iy2 > iy1:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                area1 = w1 * h1
                area2 = w2 * h2
                union = area1 + area2 - intersection
                iou = intersection / union if union > 0 else 0

                if iou > threshold:
                    return True

        return False

    def detect_from_file(self, image_path: str) -> List[FaceDetection]:
        """
        Detect faces in an image file.

        Args:
            image_path: Path to image file

        Returns:
            List of FaceDetection objects
        """
        image = Image.open(image_path)
        return self.detect_faces(image)


# Singleton instance for reuse
_detector_instance: Optional[FaceDetector] = None


def get_face_detector(min_confidence: float = 0.5) -> FaceDetector:
    """Get or create face detector singleton."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = FaceDetector(min_confidence)
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
