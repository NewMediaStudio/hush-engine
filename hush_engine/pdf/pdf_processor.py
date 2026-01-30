"""
PDF Processor - Convert PDFs to images and back for secure redaction
"""

import sys
from pathlib import Path
from typing import List
from PIL import Image

try:
    from pdf2image import convert_from_path
except ImportError:
    print("Error: pdf2image not installed. Run: pip install pdf2image", file=sys.stderr)
    print("Note: pdf2image requires Poppler. On macOS: brew install poppler", file=sys.stderr)
    raise

try:
    import img2pdf
except ImportError:
    print("Error: img2pdf not installed. Run: pip install img2pdf", file=sys.stderr)
    raise


class PDFProcessor:
    """
    Handles PDF to image conversion and back.
    Uses rasterization approach for secure redaction - destroys text layer.
    """

    def __init__(self, dpi: int = 400):
        """
        Initialize PDF processor

        Args:
            dpi: Resolution for rasterization (400 for OCR accuracy)
                 Higher = better quality but larger files
        """
        self.dpi = dpi

    def pdf_to_images(self, pdf_path: str, first_page: int = None, last_page: int = None) -> List[Image.Image]:
        """
        Convert PDF pages to PIL Images using pdf2image

        Args:
            pdf_path: Path to PDF file
            first_page: First page to convert (1-indexed, optional)
            last_page: Last page to convert (1-indexed, optional)

        Returns:
            List of PIL Images, one per page

        Raises:
            Exception: If PDF cannot be converted (corrupted, encrypted, etc.)
        """
        try:
            # Convert PDF to images at specified DPI
            # pdf2image returns RGB images by default
            images = convert_from_path(
                pdf_path,
                dpi=self.dpi,
                fmt='png',  # Internal format
                thread_count=2,  # Use 2 threads for faster processing
                first_page=first_page,
                last_page=last_page
            )

            print(f"Converted PDF to {len(images)} page(s) at {self.dpi} DPI", file=sys.stderr)
            return images

        except Exception as e:
            print(f"Error converting PDF to images: {e}", file=sys.stderr)
            raise

    def images_to_pdf(self, images: List[Image.Image], output_path: str) -> None:
        """
        Convert PIL Images to single PDF using img2pdf

        Args:
            images: List of PIL Images (one per page)
            output_path: Path where PDF should be saved

        Raises:
            Exception: If images cannot be converted to PDF
        """
        try:
            # img2pdf requires image bytes, so we need to save images temporarily
            # or convert them to bytes
            import io

            image_bytes_list = []

            for i, img in enumerate(images):
                # Convert to RGB if necessary (PDF doesn't support RGBA)
                if img.mode == 'RGBA':
                    # Create white background
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')

                # Convert to bytes
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='PNG')
                image_bytes_list.append(img_bytes.getvalue())

            # Create PDF from image bytes
            pdf_bytes = img2pdf.convert(image_bytes_list)

            # Write to file
            with open(output_path, 'wb') as f:
                f.write(pdf_bytes)

            print(f"Created PDF with {len(images)} page(s): {output_path}", file=sys.stderr)

        except Exception as e:
            print(f"Error converting images to PDF: {e}", file=sys.stderr)
            raise

    def get_page_count(self, pdf_path: str) -> int:
        """
        Get the number of pages in a PDF without fully converting it

        Args:
            pdf_path: Path to PDF file

        Returns:
            Number of pages in the PDF
        """
        try:
            # Use pdf2image's built-in page count
            from pdf2image import pdfinfo_from_path
            info = pdfinfo_from_path(pdf_path)
            return info.get('Pages', 0)
        except Exception as e:
            print(f"Error getting PDF page count: {e}", file=sys.stderr)
            # Fallback: convert and count (less efficient)
            try:
                images = self.pdf_to_images(pdf_path)
                return len(images)
            except:
                return 0
