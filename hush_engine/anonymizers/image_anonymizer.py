"""
Image anonymization - Redact or blur sensitive regions
"""

from PIL import Image, ImageDraw, ImageFilter
from typing import List, Tuple


class ImageAnonymizer:
    """
    Anonymizes images by redacting or blurring detected text regions
    """

    def __init__(self, method: str = "black_bar"):
        """
        Initialize anonymizer

        Args:
            method: "black_bar" or "blur"
        """
        self.method = method

    def redact_regions(
        self,
        image: Image.Image,
        bboxes: List[Tuple[float, float, float, float]]
    ) -> Image.Image:
        """
        Redact specified regions in an image

        Args:
            image: PIL Image object
            bboxes: List of bounding boxes (x1, y1, x2, y2)

        Returns:
            Anonymized image
        """
        img_copy = image.copy()

        if self.method == "black_bar":
            return self._black_bar_redaction(img_copy, bboxes)
        elif self.method == "blur":
            return self._blur_redaction(img_copy, bboxes)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _black_bar_redaction(
        self,
        image: Image.Image,
        bboxes: List[Tuple[float, float, float, float]]
    ) -> Image.Image:
        """Draw black rectangles over regions"""
        draw = ImageDraw.Draw(image)

        for bbox in bboxes:
            draw.rectangle(bbox, fill="black")

        return image

    def _blur_redaction(
        self,
        image: Image.Image,
        bboxes: List[Tuple[float, float, float, float]]
    ) -> Image.Image:
        """Apply Gaussian blur to regions"""
        # Create a mask for the regions to blur
        mask = Image.new("L", image.size, 0)
        mask_draw = ImageDraw.Draw(mask)

        for bbox in bboxes:
            mask_draw.rectangle(bbox, fill=255)

        # Blur the entire image
        blurred = image.filter(ImageFilter.GaussianBlur(radius=20))

        # Composite: use blurred version only where mask is white
        result = Image.composite(blurred, image, mask)

        return result
