# Hush Engine

**Local-first PII detection engine using Presidio and Apple Vision OCR**

Hush Engine is an open-source Python library for detecting personally identifiable information (PII) in images, PDFs, and spreadsheets. It uses Microsoft Presidio for text-based PII detection and Apple Vision OCR for extracting text from images.

## Features

### Core Detection
- **Multi-format support**: Images (PNG, JPEG, HEIC), PDFs, Spreadsheets (Excel, CSV)
- **Comprehensive PII detection**: Names, emails, phone numbers, SSN, credit cards, API keys, crypto wallets, IBAN, age, and more
- **Apple Vision OCR**: Native macOS optical character recognition with 400 DPI processing
- **Privacy-first**: All processing happens locally, no data leaves your machine

### Advanced NER
- **LightGBM NER classifiers**: Fast, lightweight token classification (5-10x faster, ~10MB models)
- **Multi-NER cascade for names**: Combines spaCy, Flair, Transformers (BERT), GLiNER, and curated names database for high-recall person detection (89% recall on ai4privacy)
- **Medical NER**: Disease and drug detection using Fast Data Science libraries (MIT, zero dependencies)
- **Company NER**: Dictionary-based company name detection (100% F1 on golden set)
- **Address parsing**: libpostal integration for 99.45% accuracy, 94% recall on golden set

### International Support
- **International validation**: 116 IBAN countries, 150+ phone number patterns, 35+ national ID formats
- **Cities database**: ~800 major world cities and towns for improved LOCATION detection
- **Countries database**: Complete country name and demonym recognition
- **Locale-aware detection**: User-configurable locale preferences with automatic document locale detection

### Validation & Precision
- **Checksum validation**: Luhn, Verhoeff, Mod-11, Mod-97 algorithms for ID validation
- **Spatial filtering**: Form label detection and zone penalties for precision
- **Table detection**: Context-aware PII detection in structured data (headers boost confidence)

### Additional Features
- **Face detection**: OpenCV Haar cascade face detection in images
- **Names database**: Curated 7,200+ names across 53 locales for fast lookup
- **Library management**: Optional libraries (phonenumbers, spaCy, etc.) can be enabled/disabled
- **Extensible**: Easy to add custom PII recognizers

## Installation

```bash
pip install hush-engine
```

### Required Setup

The engine requires the spaCy language model:

```bash
python -m spacy download en_core_web_lg
```

For PDF processing, install Poppler:

```bash
brew install poppler  # macOS
```

### Optional: High-Accuracy Address Detection

For best address detection accuracy (99.45%), install libpostal:

```bash
brew install libpostal
pip install postal
```

### Optional: Additional NER Models

The engine automatically downloads NER models on first use. For offline setup:

```bash
# Flair NER model (~420MB)
python -c "from flair.models import SequenceTagger; SequenceTagger.load('ner')"

# GLiNER PII model (~500MB)
python -c "from gliner import GLiNER; GLiNER.from_pretrained('urchade/gliner_multi_pii-v1')"
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

| Category | Entity Types | Notes |
|----------|-------------|-------|
| **Personal** | PERSON, EMAIL_ADDRESS, PHONE_NUMBER, DATE_TIME, AGE | Multi-NER cascade for names (89% recall) |
| **Financial** | CREDIT_CARD, IBAN_CODE, FINANCIAL (SWIFT/BIC), crypto wallets | Luhn/Mod-97 validated |
| **Government** | NATIONAL_ID (SSN, passport, driver's license) | 35+ countries via python-stdnum |
| **Medical** | MEDICAL (diagnoses, medications, ICD-10, lab results) | Fast Data Science NER |
| **Technical** | CREDENTIAL (API keys, tokens), IP_ADDRESS, URL | AWS, Stripe, GitHub tokens |
| **Network** | NETWORK (MAC, IMEI, UUID, cookies, device IDs) | Device identifiers |
| **Location** | ADDRESS (addresses, cities, countries, coordinates) | 800+ cities, countries databases, libpostal |
| **Biometric** | BIOMETRIC, FACE | Fingerprint IDs, facial recognition, OpenCV |
| **Demographics** | GENDER, AGE | "25 years old", "Age: 45" |
| **Organization** | COMPANY, ORGANIZATION | Dictionary + NER based |
| **Vehicle** | VEHICLE (VIN, license plates) | VIN validation |
| **Generic** | ID (customer ID, employee ID, generic IDs) | Pattern-based |

See [docs/PII_REFERENCE.md](docs/PII_REFERENCE.md) for detailed entity documentation with regulatory context (HIPAA, GDPR, CCPA).

## Architecture

### Core Components

| Component | Purpose |
|-----------|---------|
| **FileRouter** | High-level API for processing different file types |
| **PIIDetector** | Text-based PII detection using Presidio with 50+ custom recognizers |
| **PersonRecognizer** | Multi-NER cascade: spaCy, Flair, Transformers, GLiNER |
| **TableDetector** | Context-aware detection for structured data (spreadsheets, tables) |
| **VisionOCR** | Apple Vision-powered OCR (400 DPI) |
| **PDFProcessor** | PDF to image conversion (400 DPI for accuracy) |
| **ImageAnonymizer** | Apply censor bars to detected areas |
| **SpreadsheetAnonymizer** | Redact PII in Excel/CSV files |
| **FaceDetector** | OpenCV Haar cascade face detection |
| **AddressVerifier** | LightGBM-based address validation with street type/number checks |
| **CompanyVerifier** | Corporate suffix detection with S&P 500 and international company database |
| **CredentialEntropy** | Shannon entropy analysis for API keys, tokens, and secrets |
| **HeuristicVerifier** | Zero-dependency form label filtering and confidence adjustment |
| **Validators** | Industry-standard validation using python-stdnum and phonenumbers |
| **DetectionConfig** | Runtime configuration with threshold adjustment |

### NER Model Cascade

The PersonRecognizer uses a tiered approach balancing speed and accuracy:

**Lightweight (default, fast)**:
1. **LightGBM NER** - Fast token classifiers (~10MB, 5-10x faster than transformers)
2. **spaCy** (`en_core_web_lg`) - Fast, reliable baseline
3. **name-dataset** - Dictionary lookup from name-dataset library
4. **NamesDatabase** - Curated 7,200+ names across 53 locales

**Heavyweight (optional, high accuracy)**:
5. **Flair** (`ner`) - High accuracy sequence labeling
6. **Transformers** (`dslim/bert-base-NER`) - BERT-based NER
7. **GLiNER** (`urchade/gliner_multi_pii-v1`) - Zero-shot PII detection

Install heavyweight models: `pip install hush-engine[accurate]`

Models can be enabled/disabled via `DetectionConfig.set_enabled_integration()`.

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

### Detection Thresholds

Thresholds are stored in `~/.hush/detection_config.json` and can be customized:

```python
from hush_engine import DetectionConfig

config = DetectionConfig()

# Get/set thresholds
config.set_threshold("PERSON", 0.6)
config.set_threshold("EMAIL_ADDRESS", 0.5)

# Enable/disable entity types
config.set_enabled_entity("FACE", False)
```

### Integration Toggles

Enable/disable NER backends for performance tuning:

```python
config = DetectionConfig()

# Disable heavyweight models for faster processing
config.set_enabled_integration("flair", False)
config.set_enabled_integration("transformers", False)

# Enable LLM verification (Apple Silicon only)
config.set_enabled_integration("mlx_verifier", True)
```

Available integrations: `lgbm_ner`, `spacy`, `flair`, `transformers`, `gliner`, `name_dataset`, `libpostal`, `urlextract`, `phonenumbers`

## Platform Requirements

- **macOS 10.15+**: Required for Apple Vision OCR
- **Python 3.10+**: Modern Python features used throughout

## Development

### Running Tests

```bash
python -m pytest tests/
```

### Benchmarking

Run accuracy benchmarks with the synthetic golden test set:

```bash
# Quick test (100 samples)
python3.10 tests/benchmark_accuracy.py --samples 100 --no-pdf

# Full benchmark on synthetic golden set
python3.10 tests/benchmark_accuracy.py --golden --no-pdf

# Start the benchmark dashboard
python3.10 tests/benchmark_server.py
# Then open http://localhost:8000
```

### Performance

**Synthetic Golden Set** (1000 samples, 2522 entities): **F1 99.7%**
- Precision: 99.5% | Recall: 100%
- Perfect (100% F1): EMAIL, DATE_TIME, CREDIT_CARD, AGE, PHONE, NATIONAL_ID, COMPANY, PERSON
- Near-perfect: ADDRESS 98.7%

**ai4privacy benchmark** (1000 samples from 3000, ~5000 entities): **F1 94.6%**
- Precision: 96.8% | Recall: 92.6%

| Entity | F1 | Precision | Recall |
|--------|-----|-----------|--------|
| EMAIL | 100% | 100% | 100% |
| DATE_TIME | 99.6% | 100% | 99.2% |
| CREDIT_CARD | ~100% | 100% | ~100% |
| IP_ADDRESS | 96.3% | 100% | 92.9% |
| NATIONAL_ID | 94.7% | 100% | 90.0% |
| PHONE | 94.5% | 100% | 89.5% |
| ADDRESS | 93.4% | 97.7% | 89.4% |
| PERSON | 89.4% | 89.6% | 89.2% |

**Key capabilities:**
- Cross-type recall: detections under wrong label still count (valid for redaction)
- Handles international formats: 150+ phone patterns, 35+ national ID formats
- Context-aware: SSN/ID keywords boost NATIONAL_ID confidence
- Name detection: Multi-NER cascade with 7,200+ name database achieving 89% recall on diverse international names

### Training LightGBM Models

Train or retrain the NER classifiers using synthetic data or the ai4privacy dataset:

```bash
# Train all entity types with synthetic data
python3.10 tools/train_lgbm_ner.py --all --samples 5000

# Train with SMOTE class balancing + noise augmentation
python3.10 tools/train_lgbm_ner.py --all --smote --augment --samples 10000

# Train with ai4privacy/pii-masking-300k external dataset
python3.10 tools/train_lgbm_ner.py --all --ai4privacy --augment --samples 10000

# SVM classifier alternative
python3.10 tools/train_lgbm_ner.py --entity-type PERSON --classifier svm
```

See the training README for dataset integration details.

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

For security issues, please email studio@newmediastudio.com instead of using the issue tracker.

## Roadmap

### Completed (v1.5.0)
- [x] Expanded names database: 7,200+ names across 53 locales (up from 1,400)
- [x] NamesDatabase lookup in PersonRecognizer cascade (89% PERSON recall, up from 71%)
- [x] Expanded cities database: 800+ world cities/towns including UK coverage
- [x] UK compound place name patterns (Newcastle-under-Lyme, Stoke-on-Trent, etc.)
- [x] Secondary address patterns (Block, Basement, Office, Level, Wing, etc.)
- [x] Phone mixed-separator patterns (4-3-4, dot-dash formats)
- [x] UK postcode detection through OCR garbage filter
- [x] Address verifier with LightGBM classifier
- [x] Company verifier with S&P 500 database and corporate suffix detection
- [x] Credential entropy analyzer (Shannon entropy for secrets/tokens)
- [x] Heuristic verifier for form label filtering (zero dependencies)
- [x] LightGBM model preloader (resolves OpenMP/spaCy conflict on macOS)
- [x] OCR noise filter for detected text
- [x] SVM classifier option for NER training
- [x] SMOTE class balancing and noise augmentation for training pipeline
- [x] ai4privacy/pii-masking-300k dataset integration (225K+ annotated samples)
- [x] Threshold calibration tool with precision-recall analysis
- [x] Benchmark dashboard with multi-dataset support and run history
- [x] Person recognizer cascade with tuned model weights and USERNAME detection

### Completed (v1.4.0)
- [x] LightGBM NER classifiers (5-10x faster, ~10MB models)
- [x] New entity types (BIOMETRIC, CREDENTIAL, ID, NATIONAL_ID, NETWORK, VEHICLE)
- [x] Precision improvements (spatial filtering, negative gazetteers)
- [x] IVW calibration from feedback data

### Completed (v1.3.0)
- [x] International PII validation (116 IBAN countries, 150+ phone patterns)
- [x] Medical/biomedical NER (Fast Data Science libraries)
- [x] Face detection in images (OpenCV Haar cascade)
- [x] Table/structured data detection
- [x] Age detection ("25 years old", "Age: 45")
- [x] Library management (enable/disable optional libraries)
- [x] Locale configuration (30+ locales supported)
- [x] Multi-NER cascade for PERSON
- [x] Cities database and countries database (complete country/demonym recognition)
- [x] libpostal address parsing (99.45% accuracy)

### Planned
- [ ] Windows/Linux support (alternative OCR engines)
- [ ] Batch processing optimizations
- [ ] GDPR Article 9 special categories (racial origin, political opinions, religious beliefs)

## Acknowledgments

Built on top of:
- [Microsoft Presidio](https://github.com/microsoft/presidio) - PII detection framework
- [Apple Vision Framework](https://developer.apple.com/documentation/vision) - OCR
- [spaCy](https://spacy.io/) - Industrial-strength NLP
- [Flair](https://github.com/flairNLP/flair) - State-of-the-art NER
- [GLiNER](https://github.com/urchade/GLiNER) - Zero-shot NER
- [Fast Data Science](https://fastdatascience.com/) - Medical NER
- [libpostal](https://github.com/openvenues/libpostal) - Address parsing
- [python-stdnum](https://github.com/arthurdejong/python-stdnum) - ID validation
