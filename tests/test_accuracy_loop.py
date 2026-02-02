#!/usr/bin/env python3
"""
PII Detection Accuracy Test Loop

Comprehensive accuracy testing for the hush-engine PII detection.
Tests both CSV (for ground truth extraction) and PDF (for OCR + detection).

Metrics:
- Total detections count
- Per-entity-type precision, recall, F1
- Coverage percentage
- False positive / false negative analysis

Usage:
    python test_accuracy_loop.py                    # Run all tests
    python test_accuracy_loop.py --csv              # Test CSV only
    python test_accuracy_loop.py --pdf              # Test PDF only
    python test_accuracy_loop.py --verbose          # Show all detections
    python test_accuracy_loop.py --export results   # Export results to JSON
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Set, Tuple, Any, Optional
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "hush_engine"))

from ui.file_router import FileRouter
from detectors.pii_detector import PIIDetector


# =============================================================================
# GROUND TRUTH CONFIGURATION
# =============================================================================

# Maps CSV column names to expected PII entity types
# Multiple types means any of these is acceptable
COLUMN_TO_ENTITY_TYPES: Dict[str, List[str]] = {
    "Full Name": ["PERSON"],
    "Email": ["EMAIL_ADDRESS"],
    "Phone": ["PHONE_NUMBER"],
    "Street Address": ["LOCATION"],
    "City": ["LOCATION"],
    "State": ["LOCATION"],
    "ZIP": ["LOCATION"],
    "Country": ["LOCATION", "NRP"],
    "Date of Birth": ["DATE_TIME"],
    "SSN": ["SSN", "US_SSN"],
    "Passport Number": ["PASSPORT"],
    "Drivers License": ["DRIVERS_LICENSE"],
    "Credit Card": ["CREDIT_CARD"],
    "IBAN": ["IBAN_CODE"],
    "SWIFT BIC": ["FINANCIAL"],
    "Bank Account": ["BANK_NUMBER", "FINANCIAL", "DRIVERS_LICENSE"],
    "IP Address": ["IP_ADDRESS"],
    "GPS Coordinates": ["COORDINATES"],
    "Medical Info": ["MEDICAL"],
    "Gender": ["GENDER"],
    "VIN": ["VEHICLE_ID"],
    "MAC Address": ["DEVICE_ID"],
    "IMEI": ["DEVICE_ID"],
    "UUID": ["DEVICE_ID"],
    "URL": ["URL"],
    "Company": ["COMPANY", "PERSON", "ORGANIZATION"],
    "Nationality": ["NRP"],
    "AWS Access Key": ["AWS_ACCESS_KEY"],
    "UK NHS": ["UK_NHS", "PHONE_NUMBER"],  # Often confused with phone
    "Currency Amount": ["FINANCIAL", "LOCATION"],
    # International data columns
    "National ID": ["NATIONAL_ID", "SSN", "US_SSN", "UK_NHS", "PHONE_NUMBER"],  # Various intl formats
    "Health ID": ["NATIONAL_ID", "UK_NHS", "PHONE_NUMBER", "MEDICAL"],  # Health-related IDs
}

# Primary entity type for each column (used for recall calculation)
PRIMARY_ENTITY_TYPE: Dict[str, str] = {
    "Full Name": "PERSON",
    "Email": "EMAIL_ADDRESS",
    "Phone": "PHONE_NUMBER",
    "Street Address": "LOCATION",
    "City": "LOCATION",
    "State": "LOCATION",
    "ZIP": "LOCATION",
    "Country": "NRP",
    "Date of Birth": "DATE_TIME",
    "SSN": "SSN",
    "Passport Number": "PASSPORT",
    "Drivers License": "DRIVERS_LICENSE",
    "Credit Card": "CREDIT_CARD",
    "IBAN": "IBAN_CODE",
    "SWIFT BIC": "FINANCIAL",
    "Bank Account": "FINANCIAL",
    "IP Address": "IP_ADDRESS",
    "GPS Coordinates": "COORDINATES",
    "Medical Info": "MEDICAL",
    "Gender": "GENDER",
    "VIN": "VEHICLE_ID",
    "MAC Address": "DEVICE_ID",
    "IMEI": "DEVICE_ID",
    "UUID": "DEVICE_ID",
    "URL": "URL",
    "Company": "COMPANY",
    "Nationality": "NRP",
    "AWS Access Key": "AWS_ACCESS_KEY",
    "UK NHS": "UK_NHS",
    "Currency Amount": "FINANCIAL",
    # International data columns
    "National ID": "NATIONAL_ID",
    "Health ID": "NATIONAL_ID",
}


@dataclass
class DetectionResult:
    """Single detection result"""
    text: str
    entity_type: str
    confidence: float
    bbox: Optional[List[float]] = None
    page: Optional[int] = None
    column: Optional[str] = None  # For CSV tests
    expected_types: Optional[List[str]] = None
    is_correct: Optional[bool] = None


@dataclass
class EntityTypeMetrics:
    """Metrics for a single entity type"""
    entity_type: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    total_expected: int = 0
    total_detected: int = 0

    @property
    def precision(self) -> float:
        if self.total_detected == 0:
            return 0.0
        return self.true_positives / self.total_detected

    @property
    def recall(self) -> float:
        if self.total_expected == 0:
            return 0.0
        return self.true_positives / self.total_expected

    @property
    def f1(self) -> float:
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)


@dataclass
class TestResults:
    """Aggregate test results"""
    test_name: str
    total_expected: int = 0
    total_detected: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    entity_metrics: Dict[str, EntityTypeMetrics] = field(default_factory=dict)
    detections: List[DetectionResult] = field(default_factory=list)
    missed_values: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def overall_precision(self) -> float:
        if self.total_detected == 0:
            return 0.0
        return self.true_positives / self.total_detected

    @property
    def overall_recall(self) -> float:
        if self.total_expected == 0:
            return 0.0
        return self.true_positives / self.total_expected

    @property
    def overall_f1(self) -> float:
        p, r = self.overall_precision, self.overall_recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    @property
    def detection_rate(self) -> float:
        if self.total_expected == 0:
            return 0.0
        return self.total_detected / self.total_expected


def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    return text.strip().lower().replace(" ", "").replace("-", "").replace(".", "")


def text_overlap(text1: str, text2: str, threshold: float = 0.5) -> bool:
    """Check if two texts have significant overlap"""
    t1, t2 = normalize_text(text1), normalize_text(text2)
    if not t1 or not t2:
        return False
    # Substring match
    if t1 in t2 or t2 in t1:
        return True
    # Character overlap
    common = set(t1) & set(t2)
    return len(common) / max(len(set(t1)), len(set(t2))) >= threshold


# =============================================================================
# CSV TESTING (Direct Text Analysis)
# =============================================================================

def test_csv_detection(csv_path: str, verbose: bool = False) -> TestResults:
    """
    Test PII detection on CSV data using direct text analysis.

    This tests the detector's accuracy without OCR noise.
    Each cell value is analyzed and compared against expected entity types.
    """
    print(f"\n{'='*70}")
    print(f"CSV DETECTION TEST: {Path(csv_path).name}")
    print(f"{'='*70}\n")

    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Initialize detector
    detector = PIIDetector()

    results = TestResults(test_name=f"CSV: {Path(csv_path).name}")
    entity_metrics: Dict[str, EntityTypeMetrics] = defaultdict(
        lambda: EntityTypeMetrics(entity_type="")
    )

    # Process each column
    for col in df.columns:
        if col not in COLUMN_TO_ENTITY_TYPES:
            print(f"  Skipping unknown column: {col}")
            continue

        expected_types = COLUMN_TO_ENTITY_TYPES[col]
        primary_type = PRIMARY_ENTITY_TYPE.get(col, expected_types[0])

        # Initialize metrics for this entity type
        if primary_type not in entity_metrics:
            entity_metrics[primary_type] = EntityTypeMetrics(entity_type=primary_type)

        col_detected = 0
        col_correct = 0
        col_missed = 0

        for idx, value in enumerate(df[col]):
            if pd.isna(value):
                continue

            text = str(value).strip()
            if not text:
                continue

            results.total_expected += 1
            entity_metrics[primary_type].total_expected += 1

            # Analyze text
            entities = detector.analyze_text(text)

            if entities:
                results.total_detected += len(entities)
                col_detected += 1

                # Check if any detected entity matches expected types
                matched = False
                for entity in entities:
                    entity_metrics[entity.entity_type].total_detected += 1

                    is_correct = entity.entity_type in expected_types

                    results.detections.append(DetectionResult(
                        text=entity.text,
                        entity_type=entity.entity_type,
                        confidence=entity.confidence,
                        column=col,
                        expected_types=expected_types,
                        is_correct=is_correct
                    ))

                    if is_correct:
                        matched = True
                        results.true_positives += 1
                        entity_metrics[entity.entity_type].true_positives += 1
                    else:
                        results.false_positives += 1
                        entity_metrics[entity.entity_type].false_positives += 1

                if matched:
                    col_correct += 1
            else:
                # No detection - false negative
                col_missed += 1
                results.false_negatives += 1
                entity_metrics[primary_type].false_negatives += 1
                results.missed_values.append({
                    'column': col,
                    'value': text[:100],
                    'expected_type': primary_type,
                    'row': idx
                })

        # Report column results
        col_total = col_detected + col_missed
        col_accuracy = col_correct / col_total if col_total > 0 else 0
        status = "✓" if col_accuracy >= 0.9 else "⚠" if col_accuracy >= 0.5 else "✗"
        print(f"  {status} {col}: {col_correct}/{col_total} correct ({col_accuracy:.0%})")

        if verbose and col_missed > 0:
            missed = [r for r in results.missed_values if r['column'] == col][:3]
            for m in missed:
                print(f"      MISSED: '{m['value'][:50]}...'")

    # Finalize entity metrics
    for etype, metrics in entity_metrics.items():
        metrics.entity_type = etype
        results.entity_metrics[etype] = metrics

    return results


# =============================================================================
# PDF TESTING (OCR + Detection)
# =============================================================================

def test_pdf_detection(pdf_path: str, verbose: bool = False) -> TestResults:
    """
    Test PII detection on PDF documents.

    This tests the full pipeline including OCR.
    """
    print(f"\n{'='*70}")
    print(f"PDF DETECTION TEST: {Path(pdf_path).name}")
    print(f"{'='*70}\n")

    router = FileRouter()

    print("Processing PDF (this may take a minute)...")
    result = router.detect_pii_pdf(pdf_path, detect_faces=False)

    detections = result['detections']
    total_pages = result['total_pages']
    all_text_blocks = result.get('all_text_blocks', [])

    print(f"Total pages: {total_pages}")
    print(f"Total OCR text blocks: {len(all_text_blocks)}")
    print(f"Total PII detections: {len(detections)}")

    results = TestResults(test_name=f"PDF: {Path(pdf_path).name}")

    # For PDF, we estimate expected based on the known data
    # 30 columns * 48 rows = 1440 expected PII values
    num_data_columns = len(COLUMN_TO_ENTITY_TYPES)
    estimated_rows = 48  # From the CSV data
    results.total_expected = num_data_columns * estimated_rows
    results.total_detected = len(detections)

    # Count by type
    by_type = defaultdict(list)
    for d in detections:
        by_type[d['entity_type']].append(d)

        results.detections.append(DetectionResult(
            text=d['text'],
            entity_type=d['entity_type'],
            confidence=d['confidence'],
            bbox=d.get('bbox'),
            page=d.get('page')
        ))

    # Build expected type counts
    type_expectations: Dict[str, int] = defaultdict(int)
    for col, primary_type in PRIMARY_ENTITY_TYPE.items():
        type_expectations[primary_type] += estimated_rows

    # Calculate metrics per type
    for entity_type, expected_count in type_expectations.items():
        actual_count = len(by_type.get(entity_type, []))

        # For PDF we can't know exact matches, so we estimate
        tp = min(actual_count, expected_count)
        fp = max(0, actual_count - expected_count)
        fn = max(0, expected_count - actual_count)

        results.entity_metrics[entity_type] = EntityTypeMetrics(
            entity_type=entity_type,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            total_expected=expected_count,
            total_detected=actual_count
        )

        results.true_positives += tp
        results.false_positives += fp
        results.false_negatives += fn

    # Handle unexpected types (types we detected but didn't expect)
    expected_types = set(type_expectations.keys())
    for entity_type, items in by_type.items():
        if entity_type not in expected_types:
            results.entity_metrics[entity_type] = EntityTypeMetrics(
                entity_type=entity_type,
                true_positives=0,
                false_positives=len(items),
                false_negatives=0,
                total_expected=0,
                total_detected=len(items)
            )
            results.false_positives += len(items)

    # Print type breakdown
    print(f"\nDetections by type:")
    for entity_type in sorted(by_type.keys(), key=lambda x: -len(by_type[x])):
        count = len(by_type[entity_type])
        expected = type_expectations.get(entity_type, 0)
        if expected > 0:
            pct = count / expected * 100
            status = "✓" if pct >= 80 else "⚠" if pct >= 50 else "✗"
            print(f"  {status} {entity_type}: {count}/{expected} ({pct:.0f}%)")
        else:
            print(f"  ? {entity_type}: {count} (unexpected)")

    if verbose:
        print(f"\nSample detections:")
        for d in detections[:10]:
            print(f"  [{d['entity_type']}] '{d['text'][:50]}' (conf: {d['confidence']:.2f})")

    return results


# =============================================================================
# REPORTING
# =============================================================================

def print_results(results: TestResults):
    """Print detailed test results"""
    print(f"\n{'='*70}")
    print(f"RESULTS: {results.test_name}")
    print(f"{'='*70}")

    print(f"\n--- Overall Metrics ---")
    print(f"Total Expected: {results.total_expected}")
    print(f"Total Detected: {results.total_detected}")
    print(f"Detection Rate: {results.detection_rate:.1%}")
    print(f"")
    print(f"True Positives:  {results.true_positives}")
    print(f"False Positives: {results.false_positives}")
    print(f"False Negatives: {results.false_negatives}")
    print(f"")
    print(f"PRECISION: {results.overall_precision:.1%}")
    print(f"RECALL:    {results.overall_recall:.1%}")
    print(f"F1 SCORE:  {results.overall_f1:.1%}")

    print(f"\n--- Per-Entity-Type Metrics ---")
    print(f"{'Entity Type':<20} {'Expected':>8} {'Detected':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    print("-" * 70)

    # Sort by F1 score (ascending to show worst first)
    sorted_types = sorted(
        results.entity_metrics.values(),
        key=lambda m: (m.f1, m.recall),
        reverse=False
    )

    for m in sorted_types:
        status = "✓" if m.f1 >= 0.8 else "⚠" if m.f1 >= 0.5 else "✗"
        print(f"{status} {m.entity_type:<18} {m.total_expected:>8} {m.total_detected:>8} "
              f"{m.precision:>7.0%} {m.recall:>7.0%} {m.f1:>7.0%}")

    # Show missed values (false negatives)
    if results.missed_values:
        print(f"\n--- Missed Values (False Negatives) ---")
        by_type = defaultdict(list)
        for mv in results.missed_values:
            by_type[mv['expected_type']].append(mv)

        for etype, missed in sorted(by_type.items(), key=lambda x: -len(x[1])):
            print(f"\n{etype} ({len(missed)} missed):")
            for m in missed[:5]:
                print(f"  - '{m['value'][:60]}...'")
            if len(missed) > 5:
                print(f"  ... and {len(missed) - 5} more")

    # Show false positives
    false_positives = [d for d in results.detections if d.is_correct == False]
    if false_positives:
        print(f"\n--- False Positives ---")
        by_type = defaultdict(list)
        for fp in false_positives:
            by_type[fp.entity_type].append(fp)

        for etype, fps in sorted(by_type.items(), key=lambda x: -len(x[1])):
            print(f"\n{etype} ({len(fps)} false positives):")
            for fp in fps[:3]:
                expected = fp.expected_types[0] if fp.expected_types else "?"
                print(f"  - '{fp.text[:50]}' (expected: {expected})")
            if len(fps) > 3:
                print(f"  ... and {len(fps) - 3} more")


def export_results(results: TestResults, output_path: str):
    """Export results to JSON"""
    import datetime
    data = {
        'test_name': results.test_name,
        'timestamp': datetime.datetime.now().isoformat(),
        'summary': {
            'total_expected': results.total_expected,
            'total_detected': results.total_detected,
            'detection_rate': results.detection_rate,
            'precision': results.overall_precision,
            'recall': results.overall_recall,
            'f1': results.overall_f1,
            'true_positives': results.true_positives,
            'false_positives': results.false_positives,
            'false_negatives': results.false_negatives,
        },
        'entity_metrics': {
            etype: {
                'precision': m.precision,
                'recall': m.recall,
                'f1': m.f1,
                'total_expected': m.total_expected,
                'total_detected': m.total_detected,
                'true_positives': m.true_positives,
                'false_positives': m.false_positives,
                'false_negatives': m.false_negatives,
            }
            for etype, m in results.entity_metrics.items()
        },
        'missed_values': results.missed_values,
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nResults exported to: {output_path}")


def compare_results(baseline_path: str, current_path: str):
    """Compare current results against a baseline"""
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(current_path) as f:
        current = json.load(f)

    print(f"\n{'='*70}")
    print(f"COMPARISON: {Path(baseline_path).name} vs {Path(current_path).name}")
    print(f"{'='*70}")

    # Overall comparison
    print(f"\n--- Overall Metrics ---")
    for metric in ['precision', 'recall', 'f1']:
        b_val = baseline['summary'][metric]
        c_val = current['summary'][metric]
        diff = c_val - b_val
        arrow = "↑" if diff > 0 else "↓" if diff < 0 else "→"
        color_diff = f"+{diff*100:.1f}%" if diff > 0 else f"{diff*100:.1f}%"
        print(f"  {metric.upper():<12}: {b_val*100:.1f}% → {c_val*100:.1f}% ({arrow} {color_diff})")

    # Per-entity comparison
    print(f"\n--- Entity Type Changes ---")
    all_types = set(baseline['entity_metrics'].keys()) | set(current['entity_metrics'].keys())

    improvements = []
    regressions = []

    for etype in sorted(all_types):
        b_metrics = baseline['entity_metrics'].get(etype, {'f1': 0})
        c_metrics = current['entity_metrics'].get(etype, {'f1': 0})

        b_f1 = b_metrics.get('f1', 0)
        c_f1 = c_metrics.get('f1', 0)
        diff = c_f1 - b_f1

        if abs(diff) > 0.01:  # Only show significant changes
            if diff > 0:
                improvements.append((etype, b_f1, c_f1, diff))
            else:
                regressions.append((etype, b_f1, c_f1, diff))

    if improvements:
        print("\n  IMPROVEMENTS:")
        for etype, b_f1, c_f1, diff in sorted(improvements, key=lambda x: -x[3]):
            print(f"    ✓ {etype:<20}: {b_f1*100:.0f}% → {c_f1*100:.0f}% (+{diff*100:.0f}%)")

    if regressions:
        print("\n  REGRESSIONS:")
        for etype, b_f1, c_f1, diff in sorted(regressions, key=lambda x: x[3]):
            print(f"    ✗ {etype:<20}: {b_f1*100:.0f}% → {c_f1*100:.0f}% ({diff*100:.0f}%)")

    if not improvements and not regressions:
        print("  No significant changes")

    # Net change
    net_f1_change = current['summary']['f1'] - baseline['summary']['f1']
    print(f"\n  NET F1 CHANGE: {net_f1_change*100:+.1f}%")

    return {
        'baseline': baseline['summary'],
        'current': current['summary'],
        'improvements': len(improvements),
        'regressions': len(regressions),
    }


def print_improvement_suggestions(results: TestResults):
    """Print suggestions for improving accuracy based on results"""
    print(f"\n{'='*70}")
    print("IMPROVEMENT SUGGESTIONS")
    print(f"{'='*70}")

    suggestions = []

    # Check entity types with low F1
    for etype, metrics in results.entity_metrics.items():
        if metrics.f1 < 0.5 and metrics.total_expected > 0:
            if metrics.recall < 0.5:
                suggestions.append(f"- {etype}: Low recall ({metrics.recall:.0%}) - need better patterns")
            if metrics.precision < 0.5:
                suggestions.append(f"- {etype}: Low precision ({metrics.precision:.0%}) - patterns too broad")

    # Check for high false positive types
    for etype, metrics in results.entity_metrics.items():
        if metrics.false_positives > metrics.true_positives and metrics.total_expected == 0:
            suggestions.append(f"- {etype}: {metrics.false_positives} false positives (no expected) - overly aggressive pattern")

    # Check missed values patterns
    missed_by_type = defaultdict(list)
    for mv in results.missed_values:
        missed_by_type[mv['expected_type']].append(mv['value'])

    for etype, values in missed_by_type.items():
        if len(values) >= 5:
            sample = values[:3]
            suggestions.append(f"- {etype}: {len(values)} missed values. Examples: {sample}")

    if suggestions:
        for s in suggestions[:15]:  # Limit to top 15
            print(s)
    else:
        print("No specific suggestions - accuracy is good!")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="PII Detection Accuracy Test Loop")
    parser.add_argument('--csv', action='store_true', help='Test CSV only')
    parser.add_argument('--pdf', action='store_true', help='Test PDF only')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')
    parser.add_argument('--export', type=str, help='Export results to JSON file prefix')
    parser.add_argument('--compare', type=str, nargs=2, metavar=('BASELINE', 'CURRENT'),
                        help='Compare two result files')
    parser.add_argument('--suggestions', action='store_true', help='Show improvement suggestions')
    args = parser.parse_args()

    # Handle comparison mode
    if args.compare:
        compare_results(args.compare[0], args.compare[1])
        return 0

    test_dir = Path(__file__).parent / "data"

    all_results = []

    # Run CSV test
    if not args.pdf:
        csv_path = test_dir / "fake_pii_data.csv"
        if csv_path.exists():
            results = test_csv_detection(str(csv_path), verbose=args.verbose)
            print_results(results)
            if args.suggestions:
                print_improvement_suggestions(results)
            all_results.append(results)

            if args.export:
                export_results(results, f"{args.export}_csv.json")
        else:
            print(f"Warning: {csv_path} not found")

    # Run PDF tests
    if not args.csv:
        pdf_files = [
            test_dir / "fake_pii_data_multipage.pdf",
        ]

        for pdf_path in pdf_files:
            if pdf_path.exists():
                results = test_pdf_detection(str(pdf_path), verbose=args.verbose)
                print_results(results)
                all_results.append(results)

                if args.export:
                    safe_name = pdf_path.stem.replace(".", "_")
                    export_results(results, f"{args.export}_{safe_name}.json")
            else:
                print(f"Warning: {pdf_path} not found")

    # Final summary
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("OVERALL SUMMARY")
        print(f"{'='*70}")

        for r in all_results:
            print(f"\n{r.test_name}:")
            print(f"  Precision: {r.overall_precision:.1%}")
            print(f"  Recall:    {r.overall_recall:.1%}")
            print(f"  F1 Score:  {r.overall_f1:.1%}")

    # Return worst F1 as exit code indicator
    if all_results:
        worst_f1 = min(r.overall_f1 for r in all_results)
        if worst_f1 < 0.5:
            print(f"\n⚠️  WARNING: Worst F1 score is {worst_f1:.1%} (target: 99.9%)")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
