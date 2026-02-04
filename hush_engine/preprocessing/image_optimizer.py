"""
Image preprocessing for improved OCR accuracy.

Implements adaptive thresholding, scaling, binarization, and noise removal
to optimize images before Apple Vision Framework OCR processing.
"""

import tempfile
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


class OCRPreprocessor:
    """
    Preprocessor for optimizing images before OCR.

    Uses OpenCV to apply adaptive thresholding, scaling, binarization,
    and noise removal to improve character recognition accuracy.
    """

    def __init__(
        self,
        target_dpi: int = 300,
        enable_adaptive_threshold: bool = True,
        enable_noise_removal: bool = True,
        enable_deskew: bool = False,  # Experimental
    ):
        """
        Initialize the preprocessor.

        Args:
            target_dpi: Target DPI for scaling (300 is optimal for OCR)
            enable_adaptive_threshold: Use adaptive thresholding for uneven lighting
            enable_noise_removal: Remove salt-and-pepper noise
            enable_deskew: Attempt to correct image skew (experimental)
        """
        if not OPENCV_AVAILABLE:
            raise RuntimeError("OpenCV not available. Install opencv-python.")

        self.target_dpi = target_dpi
        self.enable_adaptive_threshold = enable_adaptive_threshold
        self.enable_noise_removal = enable_noise_removal
        self.enable_deskew = enable_deskew

    def process(self, image_path: str, output_path: Optional[str] = None) -> str:
        """
        Process an image for optimal OCR.

        Args:
            image_path: Path to input image
            output_path: Optional path for output (creates temp file if None)

        Returns:
            Path to the processed image
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Apply preprocessing pipeline
        processed = self._preprocess_pipeline(img)

        # Save to output path or temp file
        if output_path is None:
            suffix = Path(image_path).suffix or '.png'
            fd, output_path = tempfile.mkstemp(suffix=suffix)
            import os
            os.close(fd)

        cv2.imwrite(output_path, processed)
        return output_path

    def _preprocess_pipeline(self, img: np.ndarray) -> np.ndarray:
        """
        Apply the full preprocessing pipeline.

        Pipeline order:
        1. Scale to target DPI (if needed)
        2. Convert to grayscale
        3. Apply adaptive thresholding (optional)
        4. Remove noise (optional)
        5. Deskew (optional, experimental)
        """
        # Step 1: Scale if image is small (assume 72 DPI source if small)
        img = self._scale_to_target_dpi(img)

        # Step 2: Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Step 3: Apply adaptive thresholding
        if self.enable_adaptive_threshold:
            # Adaptive threshold handles uneven lighting in scanned documents
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=11,  # Size of local region
                C=2  # Constant subtracted from mean
            )
        else:
            # Standard Otsu binarization
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Step 4: Noise removal
        if self.enable_noise_removal:
            # Morphological opening removes small noise particles
            kernel = np.ones((1, 1), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

            # Optional: median blur for salt-and-pepper noise
            binary = cv2.medianBlur(binary, 3)

        # Step 5: Deskew (experimental)
        if self.enable_deskew:
            binary = self._deskew(binary)

        return binary

    def _scale_to_target_dpi(self, img: np.ndarray) -> np.ndarray:
        """
        Scale image to target DPI.

        Assumes source images below 200 pixels height are low-DPI
        and need upscaling for better OCR.
        """
        height, width = img.shape[:2]

        # Heuristic: if image is small, upscale
        # Assume 72 DPI source for small images
        if height < 500 or width < 500:
            # Scale factor to reach ~300 DPI equivalent
            scale_factor = max(2.0, self.target_dpi / 72)
            img = cv2.resize(
                img, None,
                fx=scale_factor,
                fy=scale_factor,
                interpolation=cv2.INTER_CUBIC
            )
        elif height > 4000 or width > 4000:
            # Downscale very large images to prevent memory issues
            scale_factor = min(4000 / height, 4000 / width)
            img = cv2.resize(
                img, None,
                fx=scale_factor,
                fy=scale_factor,
                interpolation=cv2.INTER_AREA
            )

        return img

    def _deskew(self, img: np.ndarray) -> np.ndarray:
        """
        Attempt to correct image skew.

        Uses Hough line detection to find dominant angle and rotate.
        """
        # Find edges
        edges = cv2.Canny(img, 50, 150, apertureSize=3)

        # Detect lines
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10
        )

        if lines is None or len(lines) < 5:
            return img  # Not enough lines to determine skew

        # Calculate angles
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:  # Avoid division by zero
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                # Only consider near-horizontal lines
                if abs(angle) < 45:
                    angles.append(angle)

        if not angles:
            return img

        # Use median angle to avoid outliers
        median_angle = np.median(angles)

        # Only rotate if skew is significant (> 0.5 degrees)
        if abs(median_angle) < 0.5:
            return img

        # Rotate image
        height, width = img.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(
            img, rotation_matrix, (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        return rotated


def preprocess_for_ocr(
    image_path: str,
    output_path: Optional[str] = None,
    target_dpi: int = 300,
    adaptive_threshold: bool = True,
    noise_removal: bool = True,
) -> str:
    """
    Convenience function to preprocess an image for OCR.

    Args:
        image_path: Path to input image
        output_path: Optional path for output (creates temp file if None)
        target_dpi: Target DPI for scaling
        adaptive_threshold: Use adaptive thresholding
        noise_removal: Remove noise

    Returns:
        Path to the processed image
    """
    if not OPENCV_AVAILABLE:
        # Return original path if OpenCV not available
        return image_path

    preprocessor = OCRPreprocessor(
        target_dpi=target_dpi,
        enable_adaptive_threshold=adaptive_threshold,
        enable_noise_removal=noise_removal,
    )

    return preprocessor.process(image_path, output_path)
