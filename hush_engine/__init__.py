"""
Hush Engine - Local-first PII detection

Open-source PII detection engine using Presidio and Apple Vision OCR.
"""

try:
    from hush_engine.detection_config import VERSION
except ImportError:
    from .detection_config import VERSION
__version__ = VERSION

from .ui.file_router import FileRouter
from .detectors.pii_detector import PIIDetector
from .ocr.vision_ocr import VisionOCR
from .pdf.pdf_processor import PDFProcessor

__all__ = ["FileRouter", "PIIDetector", "VisionOCR", "PDFProcessor"]
