# Changelog

All notable changes to hush-engine will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] - 2026-02-04

### Added
- **Cities database** - ~500 major world cities for improved LOCATION detection
  - US major metros, state capitals, and top 50 cities
  - Canadian, European, Asian, African, South American, and Oceanian cities
  - City lookup with country and population data
- **Countries database** - Complete country name recognition
  - Official names, common names, and demonyms
  - ISO country codes mapping
- **Text preprocessing module** - New preprocessing pipeline
  - Text normalization for improved detection consistency
  - OCR text cleanup and standardization

### Changed
- **PERSON detection improvements** - Recall improved to 74%
  - Enhanced multi-NER cascade with better name matching
  - Improved title + name detection (Dr., Mr., Mrs., etc.)
  - Better handling of names with middle initials
- **ADDRESS/LOCATION detection** - Recall improved to 65%
  - Cities database integration for context-aware detection
  - Improved international address format recognition
  - Better street name and postal code matching
- **OCR bounding box padding** - More accurate text region extraction
  - Reduced edge clipping for better text capture
  - Improved alignment for redaction

### Performance
- PERSON recall: 74% (up from previous baseline)
- ADDRESS recall: 65% (up from previous baseline)
- Overall detection accuracy improved across all entity types

## [1.2.0] - 2026-02-02

### Added
- **AGE detection** - Detects age mentions in various formats
  - Patterns: "25 years old", "Age: 45", "aged 30", "32-year-old"
  - Contextual detection with entity type `AGE`
- **SWIFT/BIC code labels** - Improved financial entity detection
  - Better labeling for SWIFT codes in `FINANCIAL` entity type
  - BIC code pattern recognition
- **Currency detection improvements** - Enhanced financial patterns
  - Currency with spaces after symbol ($100, €50)
  - International currency formats (INR, GBP, EUR, USD)
- **Body part medical terms** - Expanded medical NER
  - Anatomical terms for medical document processing
- **Title + name detection** - Better person recognition
  - Professional titles (Dr., Prof., Rev., etc.)
  - Honorifics combined with names
- **Training infrastructure** - New feedback analysis system
  - `tools/feedback_analyzer.py` for analyzing user feedback
  - Automated recommendations for detection improvements
  - Claude-actionable JSON output for iterative improvements
- **Benchmark system** - Accuracy testing framework
  - `tests/benchmark_accuracy.py` for measuring detection accuracy
  - Historical benchmark tracking
  - Ground truth caching

### Changed
- Detection thresholds tuned based on feedback analysis
- Improved false positive filtering across all entity types

## [1.1.1] - 2026-02-02

### Added
- **Multi-name company pattern** - Detects "Name, Name and Name" format (e.g., "Nguyen, Turner and Mcgee")
  - Covers 41% of company names in training data that use this pattern
  - Score: 0.85 (high confidence)

### Changed
- **LOCATION filtering improvements**
  - Added minimum 4-character length requirement to filter short false positives ("in", "as", "WY")
  - Increased confidence threshold from 0.60 to 0.65
  - Added blocklist for common short phrases ("claimed as", "delay in", "lakhs in")

- **COMPANY filtering improvements**
  - Reduced hyphenated company pattern score from 0.65 to 0.55 to reduce false positives
  - Added blocklist for hyphenated adjectives ("cross-verified", "high-value", "tax-related")
  - Added maximum length check and phrase filtering
  - Allows dual PERSON/COMPANY detection for ambiguous patterns like "Jackson-Guzman"

- **PERSON filtering improvements**
  - Added US cities commonly confused with names (Austin, Jackson, Madison, Houston, etc.)
  - Added last names that appear in company names (Hill, Coleman, Phillips, etc.)
  - Added credit card brand names (Visa, Mastercard, Maestro) to blocklist
  - Preserved detection of hyphenated surnames (can be both person and company)

- **DATE_TIME filtering**
  - Added filtering for fiscal year phrases ("fiscal year ending", "year ended")
  - Added filtering for standalone month names
  - Increased confidence threshold to 0.75

- **FINANCIAL filtering**
  - Added filtering for plain currency amounts ($125,000, INR 2 Lakhs)
  - Added filtering for amounts with currency codes (USD 100, EUR 500)

### Performance
- Text-based detection F1 improved from 62.7% to 76.8%
- Precision improved from 49.0% to 65.8% (+16.8%)
- Recall improved from 87.2% to 92.2% (+5.0%)
- False positives reduced by 48% (233 → 120)

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
  - Biomedical entity recognition using Fast Data Science libraries
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

[1.3.0]: https://github.com/NewMediaStudio/hush-engine/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/NewMediaStudio/hush-engine/compare/v1.1.1...v1.2.0
[1.1.1]: https://github.com/NewMediaStudio/hush-engine/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/NewMediaStudio/hush-engine/compare/v1.0.3...v1.1.0
[1.0.3]: https://github.com/NewMediaStudio/hush-engine/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/NewMediaStudio/hush-engine/compare/v1.0.0...v1.0.2
[1.0.0]: https://github.com/NewMediaStudio/hush-engine/releases/tag/v1.0.0
