#!/usr/bin/env python3
"""
PII Detection Accuracy Test

Tests the engine's ability to correctly detect and classify PII from PDF documents.
Compares detected entity types against expected column mappings.
Target: 99.9% accuracy
"""

import sys
from pathlib import Path
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "hush_engine"))

from ui.file_router import FileRouter


# Expected entity type mappings for each column in fake_pii_data.csv
EXPECTED_COLUMNS = {
    "Full Name": ["PERSON"],
    "Email": ["EMAIL_ADDRESS"],
    "Phone": ["PHONE_NUMBER"],
    "Street Address": ["LOCATION"],
    "City": ["LOCATION"],
    "State": ["LOCATION"],
    "ZIP": ["LOCATION"],
    "Country": ["LOCATION", "NRP"],  # Can be detected as either
    "Date of Birth": ["DATE_TIME"],
    "SSN": ["SSN", "US_SSN"],  # Presidio uses "SSN" by default
    "Passport Number": ["PASSPORT"],
    "Drivers License": ["DRIVERS_LICENSE"],
    "Credit Card": ["CREDIT_CARD"],
    "IBAN": ["IBAN_CODE"],
    "SWIFT BIC": ["FINANCIAL"],  # SWIFT codes
    "Bank Account": ["BANK_NUMBER", "DRIVERS_LICENSE"],  # May be detected as numbers
    "IP Address": ["IP_ADDRESS"],
    "GPS Coordinates": ["COORDINATES"],
    "Medical Info": ["MEDICAL"],
    "Gender": ["GENDER"],
    "VIN": ["VEHICLE_ID"],
    "MAC Address": ["DEVICE_ID"],
    "IMEI": ["DEVICE_ID"],
    "UUID": ["DEVICE_ID"],
    "URL": ["URL"],
    "Company": ["COMPANY", "PERSON"],  # Company names sometimes detected as PERSON
    "Nationality": ["NRP"],
    "AWS Access Key": ["AWS_ACCESS_KEY"],
    "UK NHS": ["UK_NHS", "DATE_TIME"],  # Format similar to dates
    "Currency Amount": ["FINANCIAL", "LOCATION"],  # $ amounts or location context
}

# Number of data rows (excluding header)
EXPECTED_DATA_ROWS = 48


def test_pdf_detection(pdf_path: str, verbose: bool = True) -> dict:
    """
    Test PII detection accuracy on a PDF file.

    Args:
        pdf_path: Path to PDF file
        verbose: Print detailed results

    Returns:
        Dictionary with accuracy metrics
    """
    print(f"\n{'='*60}")
    print(f"Testing: {Path(pdf_path).name}")
    print(f"{'='*60}\n")

    router = FileRouter()

    print("Processing PDF... (this may take a minute)")
    result = router.detect_pii_pdf(pdf_path, detect_faces=False)

    detections = result['detections']
    total_pages = result['total_pages']

    print(f"\nTotal pages: {total_pages}")
    print(f"Total detections: {len(detections)}")

    # Count detections by entity type
    by_type = defaultdict(list)
    for d in detections:
        by_type[d['entity_type']].append(d)

    print(f"\nDetections by type:")
    for entity_type, items in sorted(by_type.items(), key=lambda x: -len(x[1])):
        print(f"  {entity_type}: {len(items)}")

    # Calculate expected detections
    # 30 columns * 48 data rows = 1440 cells (minus header which shouldn't be detected as PII)
    num_columns = len(EXPECTED_COLUMNS)
    expected_total = num_columns * EXPECTED_DATA_ROWS

    print(f"\n--- Accuracy Analysis ---")
    print(f"Expected columns: {num_columns}")
    print(f"Expected data rows: {EXPECTED_DATA_ROWS}")
    print(f"Expected total PII cells: {expected_total}")
    print(f"Actual detections: {len(detections)}")

    # Detection rate (are we finding enough?)
    detection_rate = len(detections) / expected_total if expected_total > 0 else 0
    print(f"\nDetection rate: {detection_rate:.1%} ({len(detections)}/{expected_total})")

    # For accuracy, we need to check if detected types match expected types
    # This is harder without knowing which detection corresponds to which column
    # We can estimate by checking if we have the right proportion of each type

    type_expectations = defaultdict(int)
    for col, expected_types in EXPECTED_COLUMNS.items():
        # Use first expected type as primary
        type_expectations[expected_types[0]] += EXPECTED_DATA_ROWS

    print(f"\nExpected type distribution:")
    for entity_type, count in sorted(type_expectations.items(), key=lambda x: -x[1]):
        actual = len(by_type.get(entity_type, []))
        match_rate = actual / count if count > 0 else 0
        status = "✓" if match_rate >= 0.8 else "⚠" if match_rate >= 0.5 else "✗"
        print(f"  {status} {entity_type}: expected {count}, got {actual} ({match_rate:.0%})")

    # Calculate overall accuracy estimate
    # Count detections that match expected types
    matched = 0
    for entity_type, items in by_type.items():
        # Check if this type is expected
        expected_count = type_expectations.get(entity_type, 0)
        matched += min(len(items), expected_count)

    type_accuracy = matched / expected_total if expected_total > 0 else 0

    # Alternative metric: Coverage (what % of expected types have sufficient detections)
    types_covered = 0
    types_partial = 0
    types_missing = 0
    for entity_type, expected_count in type_expectations.items():
        actual = len(by_type.get(entity_type, []))
        if actual >= expected_count * 0.8:  # 80%+ coverage
            types_covered += 1
        elif actual >= expected_count * 0.5:  # 50-80% coverage
            types_partial += 1
        else:
            types_missing += 1

    coverage_rate = types_covered / len(type_expectations) if type_expectations else 0

    print(f"\n{'='*60}")
    print(f"TYPE ACCURACY: {type_accuracy:.1%} (matched types / expected)")
    print(f"COVERAGE RATE: {coverage_rate:.1%} ({types_covered}/{len(type_expectations)} types at 80%+)")
    print(f"  - Full coverage (80%+): {types_covered}")
    print(f"  - Partial (50-80%): {types_partial}")
    print(f"  - Low (<50%): {types_missing}")
    print(f"TARGET: 99.9% type accuracy")
    print(f"GAP: {99.9 - type_accuracy*100:.1f}%")
    print(f"{'='*60}")

    # Unique text values detected (better measure of coverage)
    unique_texts = set(d['text'] for d in detections)
    unique_coverage = len(unique_texts) / expected_total if expected_total > 0 else 0
    print(f"\nUNIQUE TEXTS DETECTED: {len(unique_texts)} ({unique_coverage:.1%} of {expected_total} expected cells)")

    return {
        'total_pages': total_pages,
        'total_detections': len(detections),
        'expected_detections': expected_total,
        'detection_rate': detection_rate,
        'type_accuracy': type_accuracy,
        'coverage_rate': coverage_rate,
        'unique_texts': len(unique_texts),
        'by_type': dict(by_type),
        'type_expectations': dict(type_expectations)
    }


def main():
    """Run accuracy tests on PDF test files."""
    test_dir = Path(__file__).parent / "data"

    # Test both PDF files
    pdf_files = [
        test_dir / "fake_pii_data_onepage.pdf",
        test_dir / "fake_pii_data_multipage.pdf"
    ]

    results = {}
    for pdf_path in pdf_files:
        if pdf_path.exists():
            results[pdf_path.name] = test_pdf_detection(str(pdf_path))
        else:
            print(f"Warning: {pdf_path} not found")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"  Detection rate: {metrics['detection_rate']:.1%}")
        print(f"  Type accuracy: {metrics['type_accuracy']:.1%}")
        print(f"  Coverage rate: {metrics['coverage_rate']:.1%}")
        print(f"  Unique texts: {metrics['unique_texts']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
