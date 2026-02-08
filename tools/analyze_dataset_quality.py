#!/usr/bin/env python3
"""
Dataset Quality Analyzer

Calculates quality metrics for each dataset and their combinations.
Outputs results as JSON and markdown table.

Usage:
    python3 tools/analyze_dataset_quality.py
"""

import json
import re
import csv
import ast
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Set, Tuple

# Paths
BASE_DIR = Path(__file__).parent.parent
TESTS_DIR = BASE_DIR / "tests" / "data" / "training"
OUTPUT_PATH = Path("/private/tmp/claude-501/-Users-valentine-Documents-GitHub-hush-engine/7813f052-f613-448e-b6a8-5fd516d4dde8/scratchpad/dataset_matrix.json")

# Entity type mapping to Hush Engine types
ENTITY_TYPE_MAP = {
    # Person
    'NAME_STUDENT': 'PERSON',
    'first_name': 'PERSON',
    'last_name': 'PERSON',
    'user_name': 'PERSON',
    'PERSON': 'PERSON',

    # Address
    'STREET_ADDRESS': 'ADDRESS',
    'street_address': 'ADDRESS',
    'ADDRESS': 'ADDRESS',
    'city': 'ADDRESS',
    'state': 'ADDRESS',
    'postcode': 'ADDRESS',
    'country': 'ADDRESS',
    'county': 'ADDRESS',

    # Phone
    'PHONE_NUM': 'PHONE',
    'phone_number': 'PHONE',
    'PHONE': 'PHONE',
    'fax_number': 'PHONE',

    # Email
    'EMAIL': 'EMAIL',
    'email': 'EMAIL',

    # National ID
    'ssn': 'NATIONAL_ID',
    'SSN': 'NATIONAL_ID',
    'national_id': 'NATIONAL_ID',
    'passport': 'NATIONAL_ID',
    'passport_number': 'NATIONAL_ID',
    'drivers_license': 'NATIONAL_ID',
    'driver_license': 'NATIONAL_ID',
    'certificate_license_number': 'NATIONAL_ID',
    'tax_id': 'NATIONAL_ID',

    # Date/Time
    'date': 'DATE_TIME',
    'date_time': 'DATE_TIME',
    'date_of_birth': 'DATE_TIME',
    'time': 'DATE_TIME',
    'DATE': 'DATE_TIME',

    # Credit Card
    'credit_debit_card': 'CREDIT_CARD',
    'CREDIT_CARD': 'CREDIT_CARD',
    'cvv': 'CREDIT_CARD',

    # Company/Organization
    'company_name': 'COMPANY',
    'ORG': 'COMPANY',
    'organization': 'COMPANY',
    'org': 'COMPANY',

    # Age
    'age': 'AGE',
    'AGE': 'AGE',

    # URL
    'url': 'URL',
    'URL': 'URL',
    'URL_PERSONAL': 'URL',

    # Other types (not in main mapping but present in datasets)
    'ipv4': 'IP_ADDRESS',
    'ipv6': 'IP_ADDRESS',
    'coordinate': 'COORDINATES',
    'gender': 'GENDER',
    'occupation': 'OTHER',
    'blood_type': 'MEDICAL',
    'mac_address': 'NETWORK',
    'swift_bic': 'FINANCIAL',
    'bank_routing_number': 'FINANCIAL',
    'account_number': 'FINANCIAL',
    'password': 'CREDENTIAL',
    'pin': 'CREDENTIAL',
    'api_key': 'CREDENTIAL',
    'customer_id': 'ID',
    'employee_id': 'ID',
    'license_plate': 'VEHICLE',
    'vehicle_identifier': 'VEHICLE',
    'employment_status': 'OTHER',
    'device_identifier': 'NETWORK',
    'http_cookie': 'NETWORK',
    'health_plan_beneficiary_number': 'MEDICAL',
    'medical_record_number': 'MEDICAL',
    'biometric_identifier': 'BIOMETRIC',
}

# Corruption patterns to detect masked/corrupted values
CORRUPTION_PATTERNS = [
    re.compile(r'^x{3,}$', re.IGNORECASE),           # xxx, xxxx, XXXX
    re.compile(r'^\*{3,}$'),                          # ***, ****
    re.compile(r'^#{3,}$'),                           # ###, ####
    re.compile(r'^\[REDACTED\]$', re.IGNORECASE),    # [REDACTED]
    re.compile(r'^\[MASKED\]$', re.IGNORECASE),      # [MASKED]
    re.compile(r'^N/?A$', re.IGNORECASE),            # N/A, NA
    re.compile(r'^\d{3}-\*{2}-\*{4}$'),              # SSN masked: 123-**-****
    re.compile(r'^\*{4}\s?\*{4}\s?\*{4}\s?\d{4}$'),  # Credit card masked
]

# Case corruption patterns
def has_case_corruption(text: str) -> bool:
    """Check if text has corrupted casing (random caps in middle of words)."""
    if not text or len(text) < 3:
        return False

    # Check for unusual capitalization patterns
    words = text.split()
    for word in words:
        if len(word) < 2:
            continue
        # Check for lowercase letter followed by uppercase in middle of word
        for i in range(1, len(word) - 1):
            if word[i].isupper() and word[i-1].islower() and word[i+1].islower():
                return True
    return False


def is_corrupted_value(text: str) -> bool:
    """Check if a value appears to be masked or corrupted."""
    if not text or not isinstance(text, str):
        return False

    text = text.strip()

    # Check against corruption patterns
    for pattern in CORRUPTION_PATTERNS:
        if pattern.match(text):
            return True

    # Check for case corruption
    if has_case_corruption(text):
        return True

    # Check for excessive placeholder characters
    if len(text) > 3:
        x_ratio = text.lower().count('x') / len(text)
        star_ratio = text.count('*') / len(text)
        if x_ratio > 0.6 or star_ratio > 0.6:
            return True

    return False


class DatasetAnalyzer:
    """Analyzes dataset quality metrics."""

    def __init__(self):
        self.datasets = {}

    def load_csv(self, path: Path) -> Dict[str, Any]:
        """Load and analyze CSV dataset."""
        samples = 0
        total_entities = 0
        entity_types = defaultdict(int)
        corrupted_count = 0
        total_values = 0
        raw_entity_types = set()

        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                samples += 1

                # Check labels column for entity types
                labels_str = row.get('labels', '[]')
                try:
                    labels = ast.literal_eval(labels_str)
                    for label in labels:
                        if label != 'O' and label.startswith(('B-', 'I-')):
                            raw_type = label[2:]  # Remove B- or I- prefix
                            raw_entity_types.add(raw_type)
                            if label.startswith('B-'):  # Only count entity starts
                                total_entities += 1
                                mapped_type = ENTITY_TYPE_MAP.get(raw_type, 'OTHER')
                                entity_types[mapped_type] += 1
                except (ValueError, SyntaxError):
                    pass

                # Check ground truth columns for corruption
                for col in ['name', 'email', 'phone', 'address', 'url']:
                    if col in row and row[col]:
                        total_values += 1
                        if is_corrupted_value(row[col]):
                            corrupted_count += 1

        corruption_pct = (corrupted_count / total_values * 100) if total_values > 0 else 0
        entity_density = total_entities / samples if samples > 0 else 0

        return {
            'format': 'CSV',
            'path': str(path),
            'total_samples': samples,
            'total_entities': total_entities,
            'entity_types_covered': dict(entity_types),
            'raw_entity_types': sorted(list(raw_entity_types)),
            'corruption_percentage': round(corruption_pct, 2),
            'entity_density': round(entity_density, 3),
            'corrupted_values': corrupted_count,
            'total_values_checked': total_values,
        }

    def load_parquet(self, path: Path) -> Dict[str, Any]:
        """Load and analyze Parquet dataset."""
        try:
            import pyarrow.parquet as pq
        except ImportError:
            return {'error': 'pyarrow not installed'}

        df = pq.read_table(path).to_pandas()

        samples = len(df)
        total_entities = 0
        entity_types = defaultdict(int)
        corrupted_count = 0
        total_values = 0
        raw_entity_types = set()

        for _, row in df.iterrows():
            text = row.get('text', '')
            spans_raw = row.get('spans', '[]')

            # Parse spans
            if isinstance(spans_raw, str):
                try:
                    spans = ast.literal_eval(spans_raw)
                except (ValueError, SyntaxError):
                    spans = []
            else:
                spans = spans_raw if spans_raw else []

            for span in spans:
                label = span.get('label', '')
                raw_entity_types.add(label)

                mapped_type = ENTITY_TYPE_MAP.get(label)
                if mapped_type:
                    total_entities += 1
                    entity_types[mapped_type] += 1

                    # Check for corruption in span text
                    span_text = span.get('text', '')
                    if not span_text and 'start' in span and 'end' in span:
                        span_text = text[span['start']:span['end']]

                    if span_text:
                        total_values += 1
                        if is_corrupted_value(span_text):
                            corrupted_count += 1

        corruption_pct = (corrupted_count / total_values * 100) if total_values > 0 else 0
        entity_density = total_entities / samples if samples > 0 else 0

        return {
            'format': 'Parquet',
            'path': str(path),
            'total_samples': samples,
            'total_entities': total_entities,
            'entity_types_covered': dict(entity_types),
            'raw_entity_types': sorted(list(raw_entity_types)),
            'corruption_percentage': round(corruption_pct, 2),
            'entity_density': round(entity_density, 3),
            'corrupted_values': corrupted_count,
            'total_values_checked': total_values,
        }

    def load_arrow(self, path: Path) -> Dict[str, Any]:
        """Load and analyze Arrow dataset."""
        try:
            import pyarrow as pa
        except ImportError:
            return {'error': 'pyarrow not installed'}

        # Try streaming format first
        try:
            with pa.memory_map(str(path), 'r') as source:
                reader = pa.ipc.open_stream(source)
                df = reader.read_all().to_pandas()
        except Exception:
            try:
                with pa.memory_map(str(path), 'r') as source:
                    reader = pa.ipc.open_file(source)
                    df = reader.read_all().to_pandas()
            except Exception as e:
                return {'error': f'Failed to read Arrow file: {e}'}

        samples = len(df)
        total_entities = 0
        entity_types = defaultdict(int)
        corrupted_count = 0
        total_values = 0
        raw_entity_types = set()

        for _, row in df.iterrows():
            text = row.get('text', '')
            spans_raw = row.get('spans', [])

            # Handle different span formats
            if spans_raw is None:
                spans = []
            elif isinstance(spans_raw, str):
                try:
                    spans = ast.literal_eval(spans_raw)
                except (ValueError, SyntaxError):
                    spans = []
            else:
                try:
                    spans = list(spans_raw) if len(spans_raw) > 0 else []
                except (TypeError, ValueError):
                    spans = []

            for span in spans:
                label = span.get('label', '')
                raw_entity_types.add(label)

                mapped_type = ENTITY_TYPE_MAP.get(label)
                if mapped_type:
                    total_entities += 1
                    entity_types[mapped_type] += 1

                    # Extract text from positions
                    start = span.get('start', 0)
                    end = span.get('end', 0)
                    span_text = text[start:end] if start < end <= len(text) else ''

                    if span_text:
                        total_values += 1
                        if is_corrupted_value(span_text):
                            corrupted_count += 1

        corruption_pct = (corrupted_count / total_values * 100) if total_values > 0 else 0
        entity_density = total_entities / samples if samples > 0 else 0

        return {
            'format': 'Arrow',
            'path': str(path),
            'total_samples': samples,
            'total_entities': total_entities,
            'entity_types_covered': dict(entity_types),
            'raw_entity_types': sorted(list(raw_entity_types)),
            'corruption_percentage': round(corruption_pct, 2),
            'entity_density': round(entity_density, 3),
            'corrupted_values': corrupted_count,
            'total_values_checked': total_values,
        }

    def analyze_all(self) -> Dict[str, Any]:
        """Analyze all datasets and compute combined metrics."""
        results = {
            'individual_datasets': {},
            'combinations': {},
        }

        # Load individual datasets
        csv_path = TESTS_DIR / 'pii_dataset_1.csv'
        parquet_path = TESTS_DIR / 'pii_dataset_2.parquet'
        arrow_path = TESTS_DIR / 'pii_dataset_3.arrow'

        if csv_path.exists():
            print(f"Analyzing CSV: {csv_path}")
            results['individual_datasets']['csv'] = self.load_csv(csv_path)

        if parquet_path.exists():
            print(f"Analyzing Parquet: {parquet_path}")
            results['individual_datasets']['parquet'] = self.load_parquet(parquet_path)

        if arrow_path.exists():
            print(f"Analyzing Arrow: {arrow_path}")
            results['individual_datasets']['arrow'] = self.load_arrow(arrow_path)

        # Compute combinations
        datasets = results['individual_datasets']

        combinations = [
            ('arrow_csv', ['arrow', 'csv']),
            ('arrow_parquet', ['arrow', 'parquet']),
            ('csv_parquet', ['csv', 'parquet']),
            ('all_three', ['arrow', 'csv', 'parquet']),
        ]

        for combo_name, combo_keys in combinations:
            combo_data = self._combine_datasets([datasets.get(k, {}) for k in combo_keys if k in datasets])
            if combo_data:
                results['combinations'][combo_name] = combo_data

        return results

    def _combine_datasets(self, datasets: List[Dict]) -> Dict[str, Any]:
        """Combine metrics from multiple datasets."""
        if not datasets:
            return {}

        total_samples = sum(d.get('total_samples', 0) for d in datasets)
        total_entities = sum(d.get('total_entities', 0) for d in datasets)
        corrupted_values = sum(d.get('corrupted_values', 0) for d in datasets)
        total_values = sum(d.get('total_values_checked', 0) for d in datasets)

        # Merge entity types
        combined_types = defaultdict(int)
        all_raw_types = set()
        formats = []

        for d in datasets:
            formats.append(d.get('format', 'Unknown'))
            for etype, count in d.get('entity_types_covered', {}).items():
                combined_types[etype] += count
            for rt in d.get('raw_entity_types', []):
                all_raw_types.add(rt)

        corruption_pct = (corrupted_values / total_values * 100) if total_values > 0 else 0
        entity_density = total_entities / total_samples if total_samples > 0 else 0

        return {
            'formats': formats,
            'total_samples': total_samples,
            'total_entities': total_entities,
            'entity_types_covered': dict(combined_types),
            'unique_entity_types_count': len(combined_types),
            'raw_entity_types': sorted(list(all_raw_types)),
            'corruption_percentage': round(corruption_pct, 2),
            'entity_density': round(entity_density, 3),
            'corrupted_values': corrupted_values,
            'total_values_checked': total_values,
        }

    def print_markdown_table(self, results: Dict[str, Any]) -> str:
        """Generate markdown table from results."""
        lines = []
        lines.append("# Dataset Quality Analysis\n")

        # Individual datasets table
        lines.append("## Individual Datasets\n")
        lines.append("| Dataset | Format | Samples | Entities | Entity Types | Corruption % | Density |")
        lines.append("|---------|--------|---------|----------|--------------|--------------|---------|")

        for name, data in results['individual_datasets'].items():
            if 'error' in data:
                lines.append(f"| {name} | ERROR | - | - | - | - | - |")
            else:
                entity_types = len(data.get('entity_types_covered', {}))
                lines.append(f"| {name} | {data.get('format', 'N/A')} | {data.get('total_samples', 0):,} | {data.get('total_entities', 0):,} | {entity_types} | {data.get('corruption_percentage', 0):.2f}% | {data.get('entity_density', 0):.3f} |")

        # Entity type breakdown
        lines.append("\n## Entity Types by Dataset\n")

        # Collect all entity types
        all_types = set()
        for data in results['individual_datasets'].values():
            if 'entity_types_covered' in data:
                all_types.update(data['entity_types_covered'].keys())

        all_types = sorted(all_types)

        # Header row
        header = "| Entity Type | " + " | ".join(results['individual_datasets'].keys()) + " |"
        sep = "|-------------|" + "|".join(["-------"] * len(results['individual_datasets'])) + "|"
        lines.append(header)
        lines.append(sep)

        for etype in all_types:
            row = f"| {etype} |"
            for name, data in results['individual_datasets'].items():
                count = data.get('entity_types_covered', {}).get(etype, 0)
                row += f" {count:,} |"
            lines.append(row)

        # Combinations table
        lines.append("\n## Dataset Combinations\n")
        lines.append("| Combination | Samples | Entities | Entity Types | Corruption % | Density |")
        lines.append("|-------------|---------|----------|--------------|--------------|---------|")

        for name, data in results['combinations'].items():
            entity_types = data.get('unique_entity_types_count', 0)
            lines.append(f"| {name} | {data.get('total_samples', 0):,} | {data.get('total_entities', 0):,} | {entity_types} | {data.get('corruption_percentage', 0):.2f}% | {data.get('entity_density', 0):.3f} |")

        # Raw entity types by dataset
        lines.append("\n## Raw Entity Types (before mapping)\n")
        for name, data in results['individual_datasets'].items():
            if 'raw_entity_types' in data:
                lines.append(f"\n### {name} ({data.get('format', 'N/A')})")
                lines.append(f"```\n{', '.join(data['raw_entity_types'])}\n```")

        return "\n".join(lines)


def main():
    """Main entry point."""
    analyzer = DatasetAnalyzer()
    results = analyzer.analyze_all()

    # Save JSON
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved JSON to: {OUTPUT_PATH}")

    # Print markdown table
    markdown = analyzer.print_markdown_table(results)
    print("\n" + markdown)

    return results


if __name__ == '__main__':
    main()
