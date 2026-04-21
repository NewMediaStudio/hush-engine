# Changelog

All notable changes to hush-engine will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.9.0] - 2026-04-21

### Added

- **Kaggle PII Detection 2024 benchmark support** via `tools/create_kaggle_golden.py`
  - 1,000-sample golden set builder (all PII docs + non-PII for false-positive testing)
  - Achieved **F1 93.0%** (P=94.5%, R=91.6%) on Kaggle essays with retrained LightGBM
- **LLM comparison benchmark** — Hush vs local LLMs (Llama, Mistral, Qwen, Phi, Gemma) and cloud models (Claude, Gemini)
  - Results saved to `tests/benchmark_history/llm_comparison_results.json`
  - Live progress tracking integrated with the benchmark dashboard
- **NamesDatabase expansion** — +277 curated names for South Asian, Arabic, African, Hispanic, and European coverage (7,233 → 7,510 total)
- **Essay-context PERSON patterns** — "According to {Name}", "{Name}'s research", "{Name} (2024)", first-line author detection
- **Financial PII patterns**
  - Salary/compensation: `$128k/yr`, `Salary: $75,000/year`, `per hour`, `per month`, `per week`, `per day`
  - Masked account numbers: `****7823`, `Acct: ****1234`
  - Labeled balances: `Balance: $14,208.43`, `Amount: $500.00`
- **Driver's license pattern** — `K420-8891-5537` (letter + dash-separated digit groups)
- **Credit card expiry pattern** — `EXPIRES 09/28`, `EXP: 12/27`, `Valid Thru MM/YY`
- **OCR fragment reconstruction** — credit card numbers and expiry dates split across multiple OCR blocks (common on card graphics) are now reassembled and detected
- **ALL CAPS name detection** — PersonRecognizer converts uppercase text to Title Case for NER; FP filter checks names database before rejecting multi-word uppercase PERSON
- **Bootstrap confidence intervals tool** — `tools/bootstrap_ci.py` reports 95% CIs on F1/precision/recall with 5,000 resamples
- **Held-out test set generator** — `tools/generate_holdout_set.py` for deterministic non-overlapping evaluation slices
- **Custom dataset training** — `tools/train_lgbm_ner.py --custom-dataset <path>` supports retraining on user-provided JSON datasets
- **Test samples** — 12 HTML mockups (banking, chat, IDs, medical, email, SaaS) with PDF/PNG exports

### Changed

- **LightGBM PERSON classifier retrained** on Kaggle essay data — PERSON F1 improved from 88.9% → 93.9% on essay-style text
- **spaCy auto-enabled in balanced mode** when installed (no effect if not present)
- **Long-text FP filters** — detect_pii applies stricter filtering for texts >500 chars (essays, articles) to cut FPs on common words, sentence-boundary artifacts, and ALL CAPS headers
- **Benchmark scoring** — `calculate_metrics()` no longer counts plausible-looking real names as false positives when evaluating on datasets with incomplete GT labels

### Fixed

- `$24,831.50` dollar amounts no longer mistakenly filtered as OCR artifacts (currency symbol now exempts from repeated-punctuation heuristic)
- USERNAME false positives on hyphenated English words (`mind-mapping`, `start-up`) in essays
- ID digit range extended to catch 11- and 12-digit student IDs (`762035863358`)

### Open source release

- Removed internal `CLAUDE.md` (AI assistant instructions)
- Removed committed training analysis data (replaced by gitignored `training/`)
- Moved `names-dataset` (GPL-3.0) to optional `[names]` extra to keep base install MIT-compatible
- Added `SECURITY.md`, `CODE_OF_CONDUCT.md`, GitHub issue/PR templates, and CI workflow
- Added ruff + pytest configuration to `pyproject.toml`

## [1.4.0] - 2026-02-06

### Added
- **LightGBM NER classifiers** - Fast, lightweight token classification
  - 5-10x faster than transformer-based models
  - ~10MB total model size (vs 1GB+ for BERT/GLiNER)
  - Classifiers for PERSON, LOCATION, ORGANIZATION, DATE_TIME
  - Feature extraction based on token context, POS tags, and character patterns
  - `tools/train_lgbm_ner.py` for training custom classifiers
- **New entity types** for comprehensive PII coverage
  - `BIOMETRIC` - Fingerprints, facial recognition, iris scans
  - `CREDENTIAL` - Passwords, PINs, API keys (consolidates AWS_ACCESS_KEY, STRIPE_KEY)
  - `ID` - Customer ID, Employee ID, generic identifiers
  - `NATIONAL_ID` - SSN, passport, driver's license (consolidates country-specific IDs)
  - `NETWORK` - MAC addresses, device IDs, cookies, IMEI
  - `VEHICLE` - VIN, license plates
- **Precision improvement features**
  - Spatial filtering for form label detection and zone penalties
  - Negative gazetteer for common word false positive filtering
  - Version string disambiguation for IP addresses
  - Optional IVW calibration from feedback data

### Changed
- **NER model defaults** - Optimized for speed/accuracy balance
  - LightGBM NER now enabled by default
  - Heavy models (Flair, Transformers, GLiNER) disabled by default
  - Install with `pip install hush-engine[accurate]` for high-accuracy mode

### Removed
- **LLM verifier** - Removed MLX-based LLM verification
  - Replaced by more efficient LightGBM classifiers
  - Removes ~1GB model download requirement

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
- **Body part medical terms** - Expanded pattern-based medical detection
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

- **Medical NER** (now optional: `pip install hush-engine[medical]`)
  - Pattern-based detection ships by default (blood types, ICD-10, conditions, medications)
  - Optional: Fast Data Science NER for broader disease/drug coverage

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

[1.4.0]: https://github.com/NewMediaStudio/hush-engine/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/NewMediaStudio/hush-engine/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/NewMediaStudio/hush-engine/compare/v1.1.1...v1.2.0
[1.1.1]: https://github.com/NewMediaStudio/hush-engine/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/NewMediaStudio/hush-engine/compare/v1.0.3...v1.1.0
[1.0.3]: https://github.com/NewMediaStudio/hush-engine/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/NewMediaStudio/hush-engine/compare/v1.0.0...v1.0.2
[1.0.0]: https://github.com/NewMediaStudio/hush-engine/releases/tag/v1.0.0
