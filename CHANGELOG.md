# Changelog

All notable changes to hush-engine will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-02-02

### Added
- **International PII Validation**
  - IBAN validation for 116 countries (php-iban registry)
  - Phone number validation for 150+ countries (ariankoochak patterns + phonenumbers library)
  - National ID validation for 35+ countries via python-stdnum
  - Checksum algorithms: Luhn, Verhoeff, Mod-11, Mod-97

- **Locale-Aware Detection**
  - Automatic document locale detection from content patterns
  - Confidence boosting for locale-specific entity types
  - Support for 30+ locales (ISO codes)

- **Table Detection**
  - Context-aware PII detection for structured data
  - Header-based confidence boosting (e.g., "SSN" column boosts SSN detection)
  - Spreadsheet column analysis

- **Medical NER**
  - Biomedical entity recognition using scispaCy
  - ICD-10 code detection
  - Medication, diagnosis, and lab result detection

- **Face Detection**
  - OpenCV Haar cascade face detection in images
  - Automatic face region identification for redaction

- **New Validators Module**
  - `validate_iban()` - ISO 13616 IBAN validation
  - `validate_bic()` - BIC/SWIFT code validation
  - `validate_phone()` - International phone validation
  - `validate_credit_card()` - Luhn checksum validation
  - `validate_national_id()` - Country-specific ID validation
  - `validate_south_african_id()` - South African ID validation

### Changed
- Detection engine now uses validation libraries instead of pattern-only matching
- Improved false positive filtering for credit card vs national ID detection

### Fixed
- IBAN detection no longer produces duplicates (unified to IBAN_CODE entity type)
- National ID no longer falsely matches credit card numbers

## [1.0.3] - 2026-01-30

### Fixed
- **Critical:** Fixed PDF export bar misalignment by using correct DPI processor
  - Export now uses 400 DPI (matching detection) instead of 150 DPI (preview)
  - Redaction bars in exported PDFs now align perfectly with detected text
  - Root cause: coordinate mismatch (400 DPI bboxes applied to 150 DPI images = 2.67x error)
- Preserved DPI metadata when saving temporary PDF page images for OCR processing
  - Ensures consistent detection across single-page and multi-page PDFs

### Changed
- `save_scrubbed_pdf()` now uses `pdf_processor` (400 DPI) instead of `preview_pdf_processor` (150 DPI)
- Exported PDFs are now higher quality (400 DPI, suitable for printing)

## [1.0.2] - 2026-01-30

### Added
- International street address recognition with 5 new pattern recognizers
  - Numbered street addresses (e.g., "12 Crane Ave", "221B Baker Street")
  - Street names without numbers (e.g., "Baker Street")  
  - European street formats (Rue, Via, Calle, Avenida, etc.)
  - PO Box addresses
  - Unit/apartment addresses
- Support for 20+ North American street types (US/Canada)
- Support for 15+ UK/Irish street types
- Support for Australian/NZ street types (Parade, Esplanade, Circuit)
- Support for European street prefixes (French, Italian, Spanish, German, Dutch, Portuguese)
- 7 new address-related terms added to denylist (apartment, unit, suite, floor, level, building, po box)

### Changed
- Increased default PDF processing DPI from 300 to 400 for better OCR accuracy on stylized text
- PDF detection quality now matches PNG detection quality (100% parity)
- Updated docstrings to reflect 400 DPI for OCR accuracy

### Fixed
- PDF OCR now correctly detects large stylized text, logos, and decorative fonts
- Detection boxes align correctly with 400 DPI→150 DPI preview scaling

### Performance
- Detection time increased by ~30-50% (acceptable tradeoff for 100% accuracy)
- Preview performance unchanged (still 150 DPI for display)
- PDF files with stylized text now detected completely vs ~82% at 300 DPI

## [1.0.0] - 2026-01-29

### Added
- Initial release of hush-engine
- PII detection using Microsoft Presidio
- Apple Vision Framework OCR integration
- Image anonymization (black bars, blur)
- Spreadsheet anonymization (synthetic data)
- PDF processing support
- Canadian address detection (full addresses, postal codes, provinces)
- File routing for images, PDFs, spreadsheets
- RPC server for inter-process communication

[1.1.0]: https://github.com/NewMediaStudio/hush-engine/compare/v1.0.3...v1.1.0
[1.0.3]: https://github.com/NewMediaStudio/hush-engine/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/NewMediaStudio/hush-engine/compare/v1.0.0...v1.0.2
[1.0.0]: https://github.com/NewMediaStudio/hush-engine/releases/tag/v1.0.0
