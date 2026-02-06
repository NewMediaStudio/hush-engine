"""
Hush Engine - Local-first PII detection

Open-source PII detection engine using Presidio and Apple Vision OCR.
"""

try:
    from hush_engine.detection_config import VERSION
except ImportError:
    from .detection_config import VERSION
__version__ = VERSION

# Lazy imports to avoid triggering PyObjC Vision warnings on non-macOS
# or when Vision framework is not needed
_lazy_imports = {
    "FileRouter": ".ui.file_router",
    "PIIDetector": ".detectors.pii_detector",
    "VisionOCR": ".ocr.vision_ocr",
    "PDFProcessor": ".pdf.pdf_processor",
}


def __getattr__(name):
    """Lazy import for heavy modules."""
    if name in _lazy_imports:
        import importlib
        module = importlib.import_module(_lazy_imports[name], __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["FileRouter", "PIIDetector", "VisionOCR", "PDFProcessor"]
