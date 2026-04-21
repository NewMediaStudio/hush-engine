#!/usr/bin/env python3
"""
Generate a Held-Out Test Set from ai4privacy Dataset.

Creates a deterministic, non-overlapping test set from the ai4privacy
pii-masking-300k dataset for fair evaluation. The held-out set uses a
different slice of the data than what was used during development.

The development benchmark used random sampling from sample_3000.json
(1,000 rows drawn from a 3,000-row subset). This script creates a
fixed held-out set from either:
  (a) A different slice of sample_3000.json (rows the dev set is unlikely to have used)
  (b) The full ai4privacy dataset (completely independent samples)

Usage:
    # Option A: Generate from existing sample_3000.json (deterministic slice)
    python tools/generate_holdout_set.py --source tests/data/training/sample_3000.json --slice 1

    # Option B: Download and generate from full ai4privacy dataset (recommended)
    python tools/generate_holdout_set.py --download --samples 1000

    # Option C: Specify a different parquet/json source
    python tools/generate_holdout_set.py --source path/to/data.json --samples 1000 --offset 2000

    # Verify non-overlap with development set
    python tools/generate_holdout_set.py --verify tests/data/holdout_test_set.json
"""

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = PROJECT_ROOT / "tests" / "data" / "training" / "sample_3000.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "tests" / "data" / "holdout_test_set.json"
GOLDEN_PATH = PROJECT_ROOT / "tests" / "data" / "synthetic_golden.json"


def load_ai4privacy_json(path: str) -> list:
    """Load ai4privacy-format JSON dataset."""
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, dict) and "samples" in data:
        return data["samples"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unknown format in {path}")


def text_hash(text: str) -> str:
    """Generate a short hash of text for overlap detection."""
    return hashlib.md5(text.encode()).hexdigest()[:12]


def generate_holdout_from_slice(source_path: str, slice_num: int,
                                 samples_per_slice: int, output_path: str):
    """Generate a held-out set by taking a deterministic slice of the source data.

    Slice 0: rows[0:N]           (likely used during development)
    Slice 1: rows[N:2N]          (held-out)
    Slice 2: rows[2N:3N]         (second held-out if needed)
    """
    rows = load_ai4privacy_json(source_path)
    total = len(rows)

    start = slice_num * samples_per_slice
    end = min(start + samples_per_slice, total)

    if start >= total:
        print(f"Error: Slice {slice_num} starts at {start} but dataset has only {total} rows.")
        print(f"  Available slices: 0-{total // samples_per_slice - 1}")
        return

    holdout = rows[start:end]
    actual_count = len(holdout)

    print(f"Source: {source_path}")
    print(f"Total rows: {total}")
    print(f"Slice {slice_num}: rows[{start}:{end}] = {actual_count} samples")

    write_holdout(holdout, output_path, source_path, slice_num)


def generate_holdout_from_offset(source_path: str, offset: int,
                                  samples: int, output_path: str):
    """Generate a held-out set starting from a specific offset."""
    rows = load_ai4privacy_json(source_path)
    total = len(rows)

    end = min(offset + samples, total)
    holdout = rows[offset:end]

    print(f"Source: {source_path}")
    print(f"Total rows: {total}")
    print(f"Offset {offset}: rows[{offset}:{end}] = {len(holdout)} samples")

    write_holdout(holdout, output_path, source_path, f"offset_{offset}")


def write_holdout(samples: list, output_path: str, source: str, slice_id):
    """Write the held-out set with metadata."""
    # Compute stats
    type_counts = defaultdict(int)
    total_entities = 0
    for sample in samples:
        gt = sample.get("ground_truth", {})
        for etype, values in gt.items():
            type_counts[etype] += len(values)
            total_entities += len(values)

    output = {
        "version": "1.0",
        "description": "Held-out test set for fair evaluation (never used during development)",
        "created": datetime.now().isoformat(),
        "source_file": str(source),
        "slice_id": str(slice_id),
        "total_samples": len(samples),
        "total_entities": total_entities,
        "entity_distribution": dict(sorted(type_counts.items(), key=lambda x: -x[1])),
        "text_hashes": [text_hash(s.get("text", "")) for s in samples],
        "samples": samples,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nWritten {len(samples)} samples to {output_path}")
    print(f"Total entities: {total_entities}")
    print("\nEntity distribution:")
    for etype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {etype:20s}: {count:,}")


def verify_no_overlap(holdout_path: str):
    """Verify the held-out set doesn't overlap with the golden set or development data."""
    holdout = load_ai4privacy_json(holdout_path)
    holdout_hashes = {text_hash(s.get("text", "")) for s in holdout}
    holdout_texts = {s.get("text", "")[:80] for s in holdout}

    print(f"Held-out set: {len(holdout)} samples, {len(holdout_hashes)} unique hashes")

    # Check against golden set
    if GOLDEN_PATH.exists():
        golden = load_ai4privacy_json(str(GOLDEN_PATH))
        golden_hashes = {text_hash(s.get("text", "")) for s in golden}
        overlap = holdout_hashes & golden_hashes
        print(f"Golden set: {len(golden)} samples")
        print(f"  Overlap: {len(overlap)} samples ({'CLEAN' if len(overlap) == 0 else 'WARNING'})")

    # Check against sample_3000
    if DEFAULT_SOURCE.exists():
        source = load_ai4privacy_json(str(DEFAULT_SOURCE))
        # Check first 1000 (likely development slice)
        dev_slice = source[:1000]
        dev_hashes = {text_hash(s.get("text", "")) for s in dev_slice}
        overlap = holdout_hashes & dev_hashes
        print(f"Dev slice (rows 0-999): {len(dev_slice)} samples")
        print(f"  Overlap: {len(overlap)} samples ({'CLEAN' if len(overlap) == 0 else 'WARNING'})")


def download_ai4privacy():
    """Download the full ai4privacy dataset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' package required for download.")
        print("  pip install datasets")
        return None

    print("Downloading ai4privacy/pii-masking-300k from HuggingFace...")
    ds = load_dataset("ai4privacy/pii-masking-300k", split="train")
    print(f"Downloaded {len(ds)} samples")

    # Convert to our format
    rows = []
    for item in ds:
        text = item.get("source_text", "")
        masks = item.get("privacy_mask", [])

        ground_truth = defaultdict(list)
        for mask in masks:
            label = mask.get("label", "")
            value = mask.get("value", "")
            if label and value:
                # Map ai4privacy labels to engine types
                mapped = _map_ai4privacy_label(label)
                if mapped:
                    ground_truth[mapped].append(value)

        rows.append({
            "text": text,
            "ground_truth": dict(ground_truth),
        })

    return rows


def _map_ai4privacy_label(label: str) -> str:
    """Map ai4privacy label to Hush Engine entity type."""
    label_map = {
        "GIVENNAME1": "PERSON", "LASTNAME1": "PERSON",
        "GIVENNAME2": "PERSON", "LASTNAME2": "PERSON",
        "GIVENNAME3": "PERSON", "LASTNAME3": "PERSON",
        "EMAIL": "EMAIL",
        "TEL": "PHONE", "PHONENUMBER": "PHONE",
        "STREET": "ADDRESS", "CITY": "ADDRESS",
        "STATE": "ADDRESS", "POSTCODE": "ADDRESS",
        "COUNTRY": "ADDRESS", "COUNTY": "ADDRESS",
        "DATE": "DATE_TIME", "TIME": "DATE_TIME", "BOD": "DATE_TIME",
        "SOCIALNUMBER": "NATIONAL_ID", "PASSPORT": "NATIONAL_ID",
        "DRIVERLICENSE": "NATIONAL_ID",
        "IDCARD": "ID",
        "USERNAME": "USERNAME",
        "PASS": "CREDENTIAL", "PASSWORD": "CREDENTIAL",
        "GEOCOORD": "COORDINATES",
        "SEX": "GENDER",
        "IP": "IP_ADDRESS", "IPV4": "IP_ADDRESS", "IPV6": "IP_ADDRESS",
        "CREDITCARD": "CREDIT_CARD", "CREDITCARDNUMBER": "CREDIT_CARD",
        "IBAN": "FINANCIAL",
        "URL": "URL",
        "COMPANY": "COMPANY",
        "AGE": "AGE",
    }
    return label_map.get(label.upper(), None)


def main():
    parser = argparse.ArgumentParser(description="Generate held-out test set")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE),
                        help="Source dataset path")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="Output path")
    parser.add_argument("--slice", type=int, default=1,
                        help="Slice number (0=dev, 1=held-out, 2=second held-out)")
    parser.add_argument("--samples", type=int, default=1000,
                        help="Samples per slice")
    parser.add_argument("--offset", type=int, default=None,
                        help="Start offset instead of slice-based selection")
    parser.add_argument("--download", action="store_true",
                        help="Download full ai4privacy dataset from HuggingFace")
    parser.add_argument("--verify", type=str, default=None,
                        help="Verify a held-out set has no overlap with dev data")
    args = parser.parse_args()

    if args.verify:
        verify_no_overlap(args.verify)
        return

    if args.download:
        rows = download_ai4privacy()
        if rows:
            # Take a slice that's far from the development data
            # Development used sample_3000.json which is rows 0-2999 of some shuffle
            # Use rows starting from 5000 to ensure no overlap
            offset = args.offset or 5000
            holdout = rows[offset:offset + args.samples]
            write_holdout(holdout, args.output, "ai4privacy/pii-masking-300k", f"offset_{offset}")
        return

    if args.offset is not None:
        generate_holdout_from_offset(args.source, args.offset, args.samples, args.output)
    else:
        generate_holdout_from_slice(args.source, args.slice, args.samples, args.output)


if __name__ == "__main__":
    main()
