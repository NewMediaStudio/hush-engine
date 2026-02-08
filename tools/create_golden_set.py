#!/usr/bin/env python3
"""
Golden Test Set Creator for Hush Engine

Creates a fixed, balanced test set for consistent benchmark comparisons.
Uses Faker to generate clean, well-formatted synthetic PII data.

DEPRECATED: This script is replaced by tools/create_synthetic_golden.py
which uses Faker for clean, consistent ground truth.

Usage:
    python tools/create_synthetic_golden.py --samples 1000  # Recommended
    python tools/create_golden_set.py --samples 500         # Legacy (CSV only)

Output:
    tests/data/golden_test_set.json - Fixed test set with ground truth
"""

import json
import random
import argparse
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Entity type mapping from Arrow dataset labels to benchmark types
ARROW_LABEL_MAP = {
    'PERSON': 'PERSON',
    'EMAIL': 'EMAIL',
    'PHONE': 'PHONE',
    'ADDRESS': 'ADDRESS',
    'SSN': 'NATIONAL_ID',
    'CREDIT_CARD': 'CREDIT_CARD',
    'AGE': 'AGE',
    'DATE': 'DATE_TIME',
    'ORG': 'COMPANY',
    'ORGANIZATION': 'COMPANY',
}

# Legacy CSV field mapping
CSV_FIELD_MAP = {
    'name': 'PERSON',
    'email': 'EMAIL',
    'phone': 'PHONE',
    'address': 'ADDRESS',
    'url': 'URL',
}

# Target entity distribution for balanced sampling
# These are the primary entity types we want to benchmark
TARGET_ENTITIES = {
    'PERSON': 100,       # Person names - core entity
    'ADDRESS': 80,       # Street addresses - challenging
    'EMAIL': 60,         # Email addresses - usually easy
    'PHONE': 60,         # Phone numbers - format variations
    'COMPANY': 50,       # Company/organization names
    'CREDIT_CARD': 30,   # Credit card numbers
    'NATIONAL_ID': 30,   # SSN and national IDs
    'DATE_TIME': 30,     # Dates and times
    'MIXED': 60,         # Records with 3+ entity types (challenging)
}


def load_csv_dataset(path, max_rows=None):
    """Load legacy CSV dataset."""
    import csv
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            # Convert to standardized format
            text = row.get('text', '')
            ground_truth = {}
            for field, entity_type in CSV_FIELD_MAP.items():
                value = row.get(field, '').strip()
                if value:
                    ground_truth[entity_type] = [value]
            rows.append({
                'text': text,
                'ground_truth': ground_truth,
                'source': 'csv',
            })
    return rows


def load_arrow_dataset(path, max_rows=None):
    """Load Arrow dataset with span-based ground truth."""
    try:
        import pyarrow as pa
    except ImportError:
        print("Error: pyarrow required. Run: pip install pyarrow")
        return []

    with pa.memory_map(str(path), 'r') as source:
        reader = pa.ipc.open_stream(source)
        df = reader.read_all().to_pandas()

    if max_rows:
        df = df.head(max_rows)

    rows = []
    for _, row in df.iterrows():
        text = row.get('text', '')
        spans = row.get('spans', [])

        # Handle various span formats
        if spans is None:
            spans = []
        elif hasattr(spans, 'tolist'):
            spans = spans.tolist()

        ground_truth = defaultdict(list)
        for span in spans:
            if isinstance(span, dict):
                label = span.get('label', '')
                entity_type = ARROW_LABEL_MAP.get(label)
                if entity_type:
                    start = span.get('start', 0)
                    end = span.get('end', 0)
                    span_text = text[start:end] if start < end <= len(text) else ''
                    if span_text:
                        ground_truth[entity_type].append(span_text)

        rows.append({
            'text': text,
            'ground_truth': dict(ground_truth),
            'source': 'arrow',
        })

    return rows


def load_jsonl_dataset(path, max_rows=None):
    """Load JSONL dataset."""
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = record.get('text', '')
            # Handle pre-processed ground truth
            if 'ground_truth' in record:
                ground_truth = record['ground_truth']
            else:
                # Parse spans if available
                spans = record.get('spans', [])
                ground_truth = defaultdict(list)
                for span in spans:
                    label = span.get('label', '')
                    entity_type = ARROW_LABEL_MAP.get(label)
                    if entity_type:
                        span_text = span.get('text', '')
                        if not span_text and 'start' in span and 'end' in span:
                            span_text = text[span['start']:span['end']]
                        if span_text:
                            ground_truth[entity_type].append(span_text)
                ground_truth = dict(ground_truth)

            rows.append({
                'text': text,
                'ground_truth': ground_truth,
                'source': 'jsonl',
            })

    return rows


def load_all_datasets(training_dir):
    """Load all available datasets."""
    all_rows = []

    # CSV dataset (legacy format)
    csv_path = training_dir / 'pii_dataset_1.csv'
    if csv_path.exists():
        print(f"Loading {csv_path.name}...")
        rows = load_csv_dataset(csv_path)
        print(f"  Loaded {len(rows)} rows")
        all_rows.extend(rows)

    # Arrow dataset (span format)
    arrow_path = training_dir / 'pii_dataset_3.arrow'
    if arrow_path.exists():
        print(f"Loading {arrow_path.name}...")
        rows = load_arrow_dataset(arrow_path)
        print(f"  Loaded {len(rows)} rows")
        all_rows.extend(rows)

    return all_rows


def categorize_row(row):
    """Categorize a row by its primary entity types."""
    gt = row.get('ground_truth', {})
    entity_types = set(gt.keys())

    # Count how many different entity types
    num_types = len(entity_types)

    # Categorize as MIXED if 3+ entity types
    if num_types >= 3:
        return 'MIXED'

    # Otherwise, categorize by primary entity type
    priority = ['PERSON', 'ADDRESS', 'EMAIL', 'PHONE', 'COMPANY',
                'CREDIT_CARD', 'NATIONAL_ID', 'DATE_TIME']
    for entity_type in priority:
        if entity_type in entity_types:
            return entity_type

    return None  # No relevant entities


def calculate_difficulty(row):
    """Calculate difficulty score for a row (higher = harder)."""
    text = row.get('text', '')
    gt = row.get('ground_truth', {})

    score = 0

    # More entity types = harder
    score += len(gt) * 2

    # Longer text = harder (more context to search)
    score += len(text) // 200

    # Multi-word names are harder
    for name in gt.get('PERSON', []):
        if len(name.split()) > 2:
            score += 1

    # International phone formats are harder
    for phone in gt.get('PHONE', []):
        if phone.startswith('+') or not phone.replace(' ', '').replace('-', '').replace('(', '').replace(')', '').isdigit():
            score += 1

    # Complex addresses are harder
    for addr in gt.get('ADDRESS', []):
        if ',' in addr or len(addr.split()) > 5:
            score += 1

    return score


def select_balanced_sample(all_rows, target_samples):
    """Select a balanced sample across entity types and difficulty levels."""
    # Group rows by category
    by_category = defaultdict(list)
    for row in all_rows:
        category = categorize_row(row)
        if category:
            row['difficulty'] = calculate_difficulty(row)
            by_category[category].append(row)

    print(f"\nDataset distribution:")
    for cat, rows in sorted(by_category.items()):
        print(f"  {cat}: {len(rows)} rows")

    # Calculate samples per category
    total_target = sum(TARGET_ENTITIES.values())
    samples_per_category = {}
    for cat, target in TARGET_ENTITIES.items():
        # Scale targets to requested sample size
        samples_per_category[cat] = int(target_samples * target / total_target)

    # Adjust to hit exact target
    diff = target_samples - sum(samples_per_category.values())
    if diff > 0:
        # Add to largest categories
        for cat in sorted(samples_per_category.keys(),
                         key=lambda x: samples_per_category[x], reverse=True):
            if diff <= 0:
                break
            samples_per_category[cat] += 1
            diff -= 1

    print(f"\nTarget samples per category (total={target_samples}):")
    for cat, count in sorted(samples_per_category.items()):
        available = len(by_category.get(cat, []))
        print(f"  {cat}: {count} (available: {available})")

    # Sample from each category
    selected = []
    for category, target_count in samples_per_category.items():
        available = by_category.get(category, [])
        if not available:
            print(f"  Warning: No samples available for {category}")
            continue

        # Sort by difficulty to get a mix
        available.sort(key=lambda x: x['difficulty'])

        # Take samples from different difficulty tiers
        n = min(target_count, len(available))
        if n <= len(available):
            # Sample evenly across difficulty spectrum
            step = len(available) / n if n > 0 else 1
            indices = [int(i * step) for i in range(n)]
            for idx in indices:
                if idx < len(available):
                    selected.append(available[idx])
        else:
            selected.extend(available[:n])

    print(f"\nSelected {len(selected)} samples")

    # Shuffle to mix categories
    random.shuffle(selected)

    return selected


def create_golden_set(training_dir, output_path, target_samples=500, seed=42):
    """Create the golden test set."""
    random.seed(seed)

    print(f"Creating golden test set with {target_samples} samples...")
    print(f"Random seed: {seed}")
    print()

    # Load all datasets
    all_rows = load_all_datasets(training_dir)
    print(f"\nTotal rows loaded: {len(all_rows)}")

    # Filter rows with ground truth
    rows_with_gt = [r for r in all_rows if r.get('ground_truth')]
    print(f"Rows with ground truth: {len(rows_with_gt)}")

    # Select balanced sample
    selected = select_balanced_sample(rows_with_gt, target_samples)

    # Prepare output format
    golden_set = {
        'version': '1.0',
        'created': datetime.now().isoformat(),
        'seed': seed,
        'total_samples': len(selected),
        'composition': {},
        'samples': []
    }

    # Calculate composition stats
    composition = defaultdict(lambda: {'count': 0, 'entity_count': 0})
    for row in selected:
        gt = row.get('ground_truth', {})
        for entity_type, values in gt.items():
            composition[entity_type]['count'] += 1
            composition[entity_type]['entity_count'] += len(values)

    golden_set['composition'] = {
        k: {'samples_with_type': v['count'], 'total_entities': v['entity_count']}
        for k, v in sorted(composition.items())
    }

    # Add samples (remove internal fields)
    for i, row in enumerate(selected):
        sample = {
            'id': i,
            'text': row['text'],
            'ground_truth': row['ground_truth'],
            'source': row.get('source', 'unknown'),
        }
        golden_set['samples'].append(sample)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(golden_set, f, indent=2, ensure_ascii=False)

    print(f"\nGolden set saved to: {output_path}")
    print(f"Total samples: {golden_set['total_samples']}")

    return golden_set


def show_stats(golden_path):
    """Show statistics for existing golden set."""
    if not golden_path.exists():
        print(f"Golden set not found: {golden_path}")
        return

    with open(golden_path, 'r') as f:
        golden = json.load(f)

    print(f"Golden Test Set Statistics")
    print(f"=" * 50)
    print(f"Version: {golden.get('version', 'unknown')}")
    print(f"Created: {golden.get('created', 'unknown')}")
    print(f"Seed: {golden.get('seed', 'unknown')}")
    print(f"Total samples: {golden.get('total_samples', len(golden.get('samples', [])))}")
    print()

    print("Entity Type Distribution:")
    print("-" * 50)
    composition = golden.get('composition', {})
    for entity_type, stats in sorted(composition.items()):
        samples = stats.get('samples_with_type', 0)
        entities = stats.get('total_entities', 0)
        print(f"  {entity_type:15} {samples:4} samples, {entities:4} entities")

    # Show sample distribution by source
    sources = defaultdict(int)
    for sample in golden.get('samples', []):
        sources[sample.get('source', 'unknown')] += 1

    print()
    print("Source Distribution:")
    print("-" * 50)
    for source, count in sorted(sources.items()):
        pct = 100 * count / len(golden.get('samples', [1]))
        print(f"  {source:15} {count:4} samples ({pct:.1f}%)")

    # Show some sample text lengths
    texts = [s['text'] for s in golden.get('samples', [])]
    if texts:
        lengths = [len(t) for t in texts]
        print()
        print("Text Length Statistics:")
        print("-" * 50)
        print(f"  Min: {min(lengths)} chars")
        print(f"  Max: {max(lengths)} chars")
        print(f"  Avg: {sum(lengths) / len(lengths):.0f} chars")


def validate_golden_set(golden_path):
    """Validate the golden set for completeness and consistency."""
    if not golden_path.exists():
        print(f"Golden set not found: {golden_path}")
        return False

    with open(golden_path, 'r') as f:
        golden = json.load(f)

    issues = []

    # Check required fields
    if 'samples' not in golden:
        issues.append("Missing 'samples' field")

    if 'version' not in golden:
        issues.append("Missing 'version' field")

    # Validate samples
    samples = golden.get('samples', [])
    for i, sample in enumerate(samples):
        if 'text' not in sample:
            issues.append(f"Sample {i}: missing 'text' field")
        if 'ground_truth' not in sample:
            issues.append(f"Sample {i}: missing 'ground_truth' field")
        if not sample.get('text', '').strip():
            issues.append(f"Sample {i}: empty text")

    # Check for duplicates
    texts = [s.get('text', '') for s in samples]
    unique_texts = set(texts)
    if len(unique_texts) < len(texts):
        issues.append(f"Found {len(texts) - len(unique_texts)} duplicate texts")

    # Report results
    if issues:
        print("Validation FAILED:")
        for issue in issues[:20]:  # Limit output
            print(f"  - {issue}")
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more issues")
        return False
    else:
        print("Validation PASSED")
        print(f"  {len(samples)} samples validated")
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Create a fixed golden test set for Hush Engine benchmarks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tools/create_golden_set.py                    # Create 500-sample set
    python tools/create_golden_set.py --samples 1000     # Create 1000-sample set
    python tools/create_golden_set.py --stats            # Show current stats
    python tools/create_golden_set.py --validate         # Validate existing set
"""
    )
    parser.add_argument('--samples', type=int, default=500,
                       help='Number of samples in golden set (default: 500)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--stats', action='store_true',
                       help='Show statistics for existing golden set')
    parser.add_argument('--validate', action='store_true',
                       help='Validate existing golden set')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path (default: tests/data/golden_test_set.json)')

    args = parser.parse_args()

    # Determine paths
    base_dir = Path(__file__).parent.parent
    training_dir = base_dir / 'tests' / 'data' / 'training'

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = base_dir / 'tests' / 'data' / 'golden_test_set.json'

    if args.stats:
        show_stats(output_path)
        return 0

    if args.validate:
        success = validate_golden_set(output_path)
        return 0 if success else 1

    # Create the golden set
    create_golden_set(
        training_dir=training_dir,
        output_path=output_path,
        target_samples=args.samples,
        seed=args.seed
    )

    # Show stats
    print()
    show_stats(output_path)

    # Validate
    print()
    validate_golden_set(output_path)

    return 0


if __name__ == '__main__':
    sys.exit(main())
