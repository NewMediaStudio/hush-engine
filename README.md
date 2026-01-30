# Hush Engine

**Local-first PII detection engine using Presidio and Apple Vision OCR**

Hush Engine is an open-source Python library for detecting personally identifiable information (PII) in images, PDFs, and spreadsheets. It uses Microsoft Presidio for text-based PII detection and Apple Vision OCR for extracting text from images.

## Features

- **Multi-format support**: Images (PNG, JPEG, HEIC), PDFs, Spreadsheets (Excel, CSV)
- **Comprehensive PII detection**: Names, emails, phone numbers, SSN, credit cards, API keys, crypto wallets, IBAN, and more
- **Apple Vision OCR**: Native macOS optical character recognition
- **Privacy-first**: All processing happens locally, no data leaves your machine
- **Extensible**: Easy to add custom PII recognizers

## Installation

```bash
pip install hush-engine
```

### Additional Setup

The engine requires the spaCy language model:

```bash
python -m spacy download en_core_web_lg
```

For PDF processing, you'll need Poppler:

```bash
brew install poppler  # macOS
```

## Quick Start

### Detect PII in an Image

```python
from hush_engine import FileRouter

# Initialize the router
router = FileRouter()

# Process an image
result = router.detect_pii_image("screenshot.png")

# Print detections
print(f"Found {len(result['detections'])} PII detections")
for detection in result['detections']:
    print(f"- {detection['entity_type']}: {detection['score']:.2f} confidence")
```

### Process a PDF

```python
from hush_engine import FileRouter

router = FileRouter()

# Process multi-page PDF
result = router.detect_pii_pdf("document.pdf")

print(f"Processed {result['total_pages']} pages")
print(f"Found {len(result['detections'])} PII detections")
```

### Direct API Usage

```python
from hush_engine import PIIDetector, VisionOCR

# OCR text from image
ocr = VisionOCR()
text_blocks = ocr.extract_text("image.png")

# Detect PII in text
detector = PIIDetector()
detections = detector.analyze_text("John Doe's email is john@example.com")

for detection in detections:
    print(f"{detection.entity_type}: {detection.start}-{detection.end}")
```

## Supported PII Types

- **Personal**: Names (PERSON), Email addresses, Phone numbers, Dates of birth
- **Financial**: Credit card numbers, IBAN, Crypto wallet addresses
- **Government**: SSN, Passport numbers, Driver's license numbers
- **Technical**: API keys, AWS keys, IP addresses, URLs
- **Location**: Street addresses, ZIP codes

## Architecture

### Core Components

- **FileRouter**: High-level API for processing different file types
- **PIIDetector**: Text-based PII detection using Presidio
- **VisionOCR**: Apple Vision-powered OCR
- **PDFProcessor**: PDF to image conversion
- **ImageAnonymizer**: Apply red censor bars to detected areas
- **SpreadsheetAnonymizer**: Redact PII in Excel/CSV files

## Custom PII Recognizers

Add custom detection patterns:

```python
from hush_engine import PIIDetector
from presidio_analyzer import Pattern, PatternRecognizer

# Create custom recognizer
custom_recognizer = PatternRecognizer(
    supported_entity="CUSTOM_ID",
    patterns=[Pattern("custom pattern", r"[A-Z]{3}-\d{6}", 0.8)]
)

# Add to detector
detector = PIIDetector()
detector.analyzer.registry.add_recognizer(custom_recognizer)
```

## API Reference

### FileRouter

Main entry point for file processing.

```python
router = FileRouter()

# Image processing
result = router.detect_pii_image(image_path: str) -> dict

# PDF processing  
result = router.detect_pii_pdf(pdf_path: str) -> dict

# Spreadsheet processing
result = router.detect_pii_spreadsheet(file_path: str) -> dict

# Anonymize and save
router.anonymize_and_save(
    input_path: str,
    output_path: str,
    detections: list,
    file_type: str
)
```

### PIIDetector

Core PII detection engine.

```python
detector = PIIDetector()

# Analyze text
detections = detector.analyze_text(
    text: str,
    language: str = "en"
) -> list[RecognizerResult]

# Get confidence threshold for entity type
threshold = detector.get_threshold(entity_type: str) -> float
```

### VisionOCR

Apple Vision OCR wrapper.

```python
ocr = VisionOCR()

# Extract text blocks with positions
text_blocks = ocr.extract_text(image_path: str) -> list[dict]

# Each text block contains:
# - text: str
# - x, y, width, height: float (normalized 0-1)
```

## Configuration

Detection thresholds can be customized in `detection_config.py`:

```python
CONFIDENCE_THRESHOLDS = {
    "PERSON": 0.6,
    "EMAIL_ADDRESS": 0.7,
    "CREDIT_CARD": 0.8,
    # ... customize per entity type
}
```

## Platform Requirements

- **macOS 10.15+**: Required for Apple Vision OCR
- **Python 3.10+**: Modern Python features used throughout

## Development

### Running Tests

```bash
python -m pytest tests/
```

### Building from Source

```bash
git clone https://github.com/NewMediaStudio/hush-engine.git
cd hush-engine
pip install -e .
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Related Projects

- **Hush** (macOS app): Native SwiftUI wrapper with beautiful UI (proprietary)
- **Presidio**: Microsoft's PII detection framework
- **spaCy**: Industrial-strength NLP

## Security

For security issues, please email security@newmediastudio.com instead of using the issue tracker.

## Roadmap

- [ ] Windows/Linux support (alternative OCR engines)
- [ ] Pre-trained custom models for specific domains
- [ ] Custom local model training
- [ ] Batch processing optimizations
- [ ] Video frame processing

## Acknowledgments

Built on top of:
- Microsoft Presidio
- Apple Vision Framework
- spaCy NLP
