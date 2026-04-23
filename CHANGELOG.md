# Changelog

All notable changes to hush-engine will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.11.1] - 2026-04-23

### Changed

- README expansion for the OpenAI Privacy Filter add-on section:
  full link set to the OpenAI announcement, the HuggingFace model page,
  the GitHub source, and the model card PDF. Acknowledgments and Related
  sections updated to credit Privacy Filter as an optional backend.
  License-compatibility note clarifies that Apache-2.0 links cleanly
  against Hush's AGPL-3.0 license.
- CI: PyPI publishing now runs through a Trusted Publishing workflow
  (`.github/workflows/publish.yml`). Future releases ship on `v*` tag
  push, no API tokens stored anywhere.

Docs-only patch. No code changes from 1.11.0.

## [1.11.0] - 2026-04-23

### Added

- **OpenAI Privacy Filter integration** (Apache-2.0, bidirectional token
  classifier, 1.5B total parameters with 50M active). Ships as an opt-in
  add-on backend. Install with `pip install hush-engine[privacy-filter]`
  and toggle on via `DetectionConfig.set_enabled_integration(
  "openai_privacy_filter", True)`. Two gating modes:
  - **candidate** (default): Privacy Filter votes in the PERSON cascade
    alongside LightGBM, spaCy, Flair, Transformers, and contributes spans
    for EMAIL, PHONE, URL, LOCATION, DATE_TIME, FINANCIAL, CREDENTIAL.
  - **authoritative**: Privacy Filter's PERSON verdict short-circuits the
    cascade, verifiers skipped.
  Set `HUSH_PRIVACY_FILTER_MODEL=/path/to/local/dir` to load weights from
  disk instead of HuggingFace Hub.
- `PrivacyFilterRecognizer` in `hush_engine.detectors.privacy_filter_recognizer`
  maps the 7 non-PERSON OpenAI labels into the existing Presidio entity
  taxonomy.
- `tests/benchmark_accuracy.py` gains `--with-privacy-filter` and
  `--privacy-filter-ablation` flags. The ablation runs baseline and
  with-PF passes back-to-back on the same sampled rows and prints a
  side-by-side comparison with per-entity deltas.
- `tests/benchmark_llm_comparison.py` gains `openai-privacy-filter` as a
  model row, driven through `run_privacy_filter()` (uses the HF pipeline
  directly, no Ollama or LLM prompt plumbing).
- **Release privacy gates** behind a new `HUSH_AUDIT` environment flag
  (default off):
  - `~/.hush/audit.log` writer attaches a `NullHandler` when
    `HUSH_AUDIT` is unset. Release builds produce no audit log.
  - `ingestTrainingFeedback` RPC method disappears from the allow-list
    when `HUSH_AUDIT` is unset, and its handler self-gates as
    defense-in-depth.
  - When `HUSH_AUDIT=1` is set (internal debugging), audit lines carry a
    10-char SHA-256 prefix of the path instead of `.name`, `src`, `dst`.
- `FileRouter.__init__` now sweeps stale `tmp*` files out of
  `~/.hush/tmp` on startup. Every `create_secure_temp_file` caller sits
  in a `try/finally` unlink block, so preview JPEGs no longer accumulate
  between runs.

### Changed

- `pyproject.toml` adds a new `[privacy-filter]` extra (`transformers>=4.40.0`,
  `torch>=2.0.0`). The `[full]` meta-extra now includes it.
- `DEFAULT_INTEGRATIONS` in `detection_config.py` gains two keys:
  `openai_privacy_filter` and `openai_privacy_filter_authoritative`,
  both default `False`.
- `PersonRecognizer.__init__` accepts `use_privacy_filter` and
  `privacy_filter_authoritative` kwargs. `get_person_recognizer()`
  factory passes them through.
- Audit-log comments in `rpc_server.py` walked back from the old
  claim that logging filenames was safe. Filenames carry PII (e.g.
  `jane_doe_passport.jpg`), so they no longer touch disk even when
  `HUSH_AUDIT=1`.

### Fixed

- `tests/benchmark_accuracy.py` raised a hard error on missing
  `benchmark_template.html` even under `--no-pdf`, when the template
  is never used. The check now skips when `--no-pdf` is set, so the
  new ablation mode runs cleanly without a PDF converter installed.

## [1.10.2] - 2026-04-21

### Fixed

- README logo path on PyPI: switched from a repo-relative path
  (`assets/hush-engine-logo.png`) to an absolute GitHub raw URL so the
  logo renders on https://pypi.org/project/hush-engine/ as well as on
  GitHub.

## [1.10.1] - 2026-04-21

### Fixed

- Bare sibling imports in `scrub_image.py`, `scrub_spreadsheet.py`,
  `analyze_feedback.py`, `rpc_server.py`, and `ui/file_router.py` raised
  `ModuleNotFoundError` when the package was imported normally. Converted
  to proper relative imports (`from .anonymizers`, `from ..anonymizers`).
  Users on 1.10.0 who called `from hush_engine.scrub_spreadsheet import ...`
  hit this immediately.
- Missing `import sys` in `image_optimizer.py` and
  `barcode_detector_pyzbar.py` — error paths using `sys.stderr.write`
  would crash on exception instead of logging.
- `error_response` was referenced but never defined in the JSON-decode
  error handler of `rpc_server.py`, causing a `NameError` when invalid
  JSON was received.

### Added

- Hush Engine logo at the top of the README (`assets/hush-engine-logo.png`).
- Credits for Valentine Makhouleen (https://new-media.ca) and New Media
  Studio (https://wearenewmedia.com/) in README, LICENSE, and
  COMMERCIAL-LICENSING.md.
- CI: install `libomp` on macOS runners so LightGBM loads at import.

### Changed

- Benchmark dashboard: rotating metallic H cube replaces the bee logo.

## [1.10.0] - 2026-04-21

### Changed

- **License: MIT → AGPL-3.0.** Hush Engine is now dual-licensed. Open source use
  falls under AGPL-3.0. Proprietary and closed-source commercial use requires a
  commercial license; see [COMMERCIAL-LICENSING.md](COMMERCIAL-LICENSING.md).
- Version 1.9.0 (MIT) was yanked from PyPI before any significant uptake. Users
  already on 1.9.0 retain their MIT grant for that release under its original terms.

### Why

The MIT release invited unrestricted commercial use, including by surveillance
and mass-data-collection products that the project explicitly aims to be used
against. AGPL-3.0 with a commercial option preserves open source values for
researchers, non-commercial users, and AGPL-compatible projects while giving us
a conversation with commercial integrators about how Hush will be used.

## [1.9.0] - 2026-04-21 [YANKED]

> This release was published to PyPI under MIT and yanked the same day as part
> of the license change to AGPL-3.0. See 1.10.0 for the current release.



### Added

- Kaggle PII Detection 2024 benchmark. `tools/create_kaggle_golden.py` builds a 1,000-sample set (945 PII + 55 non-PII). Hush scores F1 93.0% (P=94.5%, R=91.6%) after LightGBM retraining.
- LLM comparison benchmark (`tests/benchmark_llm_comparison.py`) for Ollama (Llama, Mistral, Qwen, Phi, Gemma) and API (Claude, Gemini). Live progress feeds the dashboard.
- Curated NamesDatabase expanded by 277 names covering South Asian, Arabic, African, Hispanic, and European locales (7,233 → 7,510).
- Essay-context PERSON patterns: "According to {Name}", "{Name}'s research", "{Name} (2024)", first-line author detection.
- Financial PII patterns: salary (`$128k/yr`, `per month`, `per week`, `per day`, `per hour`), masked accounts (`****7823`), labeled balances (`Balance: $14,208.43`).
- Driver's license pattern `K420-8891-5537`.
- Credit card expiry pattern (`EXPIRES 09/28`, `EXP: 12/27`).
- OCR fragment reconstruction for credit card numbers and expiry dates split across separate OCR blocks.
- ALL CAPS name detection (PersonRecognizer converts uppercase to Title Case for NER; FP filter consults the names database before rejecting).
- Bootstrap 95% confidence intervals: `tools/bootstrap_ci.py` (5,000 resamples by default).
- Held-out test set generator: `tools/generate_holdout_set.py`.
- Custom training datasets: `tools/train_lgbm_ner.py --custom-dataset <path>`.
- 12 HTML sample mockups (banking, chat, IDs, medical, email, SaaS) with PDF and PNG exports under `tests/data/samples/`.

### Changed

- LightGBM PERSON classifier retrained with Kaggle essay data. PERSON F1 on essays: 88.9% → 93.9%.
- spaCy auto-enabled in balanced mode when installed.
- Long-text FP filters in `detect_pii()` for text > 500 chars (cuts common-word, sentence-boundary, and ALL-CAPS-header false positives).
- `calculate_metrics()` stops counting plausible-name detections as FPs when the dataset has incomplete GT labels.

### Fixed

- `$24,831.50` dollar amounts passed through the OCR-artifact filter that previously rejected them for having three distinct punctuation characters.
- USERNAME false positives on hyphenated English words (`mind-mapping`, `start-up`).
- ID digit range extended from `{6,10}` to `{6,12}` for 11- and 12-digit student IDs.

### Open source release

- Removed `CLAUDE.md` (AI assistant instructions).
- Removed committed `training/` analysis data; folder is now gitignored.
- Moved `names-dataset` (GPL-3.0) to the optional `[names]` extra. Base install is MIT-clean.
- Added `SECURITY.md`, `CODE_OF_CONDUCT.md`, GitHub issue and PR templates, and a CI workflow.
- Added ruff and pytest configuration to `pyproject.toml`.

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
