# Changelog

All notable changes to hush-engine will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[1.0.2]: https://github.com/NewMediaStudio/hush-engine/compare/v1.0.0...v1.0.2
[1.0.0]: https://github.com/NewMediaStudio/hush-engine/releases/tag/v1.0.0
