#!/usr/bin/env python3
"""
Create a Golden 1,000-Sample Kaggle Test Set for Hush Engine and LLM Benchmarking.

Takes the Kaggle "PII Detection 2024" competition dataset (BIO-tagged student essays)
and produces a fixed, deterministic 1,000-sample golden set:
  - All 945 PII-containing documents (complete coverage)
  - 55 non-PII documents (for false positive measurement)
  - Deterministic selection and shuffle (seed=42)

The output format is compatible with benchmark_accuracy.py, benchmark_llm_comparison.py,
and bootstrap_ci.py.

Usage:
    python tools/create_kaggle_golden.py
    python tools/create_kaggle_golden.py --stats              # Preview without writing
    python tools/create_kaggle_golden.py --validate           # Validate existing set
    python tools/create_kaggle_golden.py --output custom.json # Custom output path

Input:  tests/data/pii-detection-removal-from-educational-data/train.json
Output: tests/data/kaggle_golden_1000.json
"""

import json
import random
import hashlib
import argparse
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "tests" / "data" / "pii-detection-removal-from-educational-data" / "train.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "tests" / "data" / "kaggle_golden_1000.json"

# Kaggle entity types -> Hush Engine entity types
ENTITY_MAP = {
    "NAME_STUDENT": "PERSON",
    "EMAIL": "EMAIL",
    "PHONE_NUM": "PHONE",
    "URL_PERSONAL": "URL",
    "STREET_ADDRESS": "ADDRESS",
    "ID_NUM": "ID",
    "USERNAME": "USERNAME",
}


def reconstruct_text(tokens: list, trailing_whitespace: list) -> str:
    """Reconstruct full text from tokens and trailing_whitespace arrays."""
    parts = []
    for token, ws in zip(tokens, trailing_whitespace):
        parts.append(token)
        if ws:
            parts.append(" ")
    return "".join(parts)


def extract_entities_from_bio(tokens: list, labels: list, trailing_whitespace: list) -> dict:
    """Extract entity spans from BIO-tagged tokens.

    Returns dict mapping Hush Engine entity type -> list of entity text strings.
    """
    entities = defaultdict(list)
    current_entity = None
    current_tokens = []

    for i, (token, label) in enumerate(zip(tokens, labels)):
        if label.startswith("B-"):
            # Save previous entity
            if current_entity and current_tokens:
                entity_text = reconstruct_text(
                    current_tokens,
                    trailing_whitespace[i - len(current_tokens):i]
                ).strip()
                if entity_text:
                    hush_type = ENTITY_MAP.get(current_entity)
                    if hush_type:
                        entities[hush_type].append(entity_text)

            current_entity = label[2:]
            current_tokens = [token]

        elif label.startswith("I-"):
            tag = label[2:]
            if tag == current_entity:
                current_tokens.append(token)
            else:
                # Mismatched I- tag
                if current_entity and current_tokens:
                    entity_text = reconstruct_text(
                        current_tokens,
                        trailing_whitespace[i - len(current_tokens):i]
                    ).strip()
                    if entity_text:
                        hush_type = ENTITY_MAP.get(current_entity)
                        if hush_type:
                            entities[hush_type].append(entity_text)
                current_entity = tag
                current_tokens = [token]

        else:  # "O" label
            if current_entity and current_tokens:
                entity_text = reconstruct_text(
                    current_tokens,
                    trailing_whitespace[i - len(current_tokens):i]
                ).strip()
                if entity_text:
                    hush_type = ENTITY_MAP.get(current_entity)
                    if hush_type:
                        entities[hush_type].append(entity_text)
                current_entity = None
                current_tokens = []

    # Handle last entity
    if current_entity and current_tokens:
        entity_text = reconstruct_text(
            current_tokens,
            trailing_whitespace[len(tokens) - len(current_tokens):]
        ).strip()
        if entity_text:
            hush_type = ENTITY_MAP.get(current_entity)
            if hush_type:
                entities[hush_type].append(entity_text)

    return dict(entities)


def text_hash(text: str) -> str:
    """Short hash for overlap detection."""
    return hashlib.md5(text.encode()).hexdigest()[:12]


def create_kaggle_golden(input_path: str, output_path: str,
                          target_samples: int = 1000, seed: int = 42,
                          stats_only: bool = False):
    """Create the golden Kaggle test set."""
    rng = random.Random(seed)

    print(f"Loading {input_path}...")
    with open(input_path) as f:
        kaggle_data = json.load(f)
    print(f"Loaded {len(kaggle_data)} documents")

    # Convert all documents
    pii_docs = []
    non_pii_docs = []

    for i, doc in enumerate(kaggle_data):
        tokens = doc["tokens"]
        labels = doc.get("labels", [])
        trailing_ws = doc.get("trailing_whitespace", [True] * len(tokens))

        if not labels:
            continue

        full_text = doc.get("full_text") or reconstruct_text(tokens, trailing_ws)
        ground_truth = extract_entities_from_bio(tokens, labels, trailing_ws)

        sample = {
            "text": full_text,
            "ground_truth": ground_truth,
            "id": doc.get("document", i),
            "source": "kaggle_pii_2024",
        }

        if ground_truth:
            pii_docs.append(sample)
        else:
            non_pii_docs.append(sample)

    print(f"  PII documents: {len(pii_docs)}")
    print(f"  Non-PII documents: {len(non_pii_docs)}")

    # Build the golden set: all PII docs + fill remainder with non-PII
    non_pii_needed = max(0, target_samples - len(pii_docs))
    if non_pii_needed > len(non_pii_docs):
        print(f"  Warning: requested {non_pii_needed} non-PII docs but only {len(non_pii_docs)} available")
        non_pii_needed = len(non_pii_docs)

    # Select non-PII docs spread across the dataset (not just first N)
    rng.shuffle(non_pii_docs)
    selected_non_pii = non_pii_docs[:non_pii_needed]
    # Mark non-PII docs for stats
    for doc in selected_non_pii:
        doc["has_pii"] = False
    for doc in pii_docs:
        doc["has_pii"] = True

    golden = pii_docs + selected_non_pii

    # Deterministic shuffle
    rng.shuffle(golden)

    # Compute stats
    type_counts = defaultdict(int)
    total_entities = 0
    text_lengths = []
    for sample in golden:
        for etype, values in sample["ground_truth"].items():
            type_counts[etype] += len(values)
            total_entities += len(values)
        text_lengths.append(len(sample["text"]))

    print(f"\nGolden Set Composition ({len(golden)} samples):")
    print(f"  PII documents: {len(pii_docs)}")
    print(f"  Non-PII documents: {non_pii_needed} (false positive testing)")
    print(f"  Total entities: {total_entities}")
    print(f"\n  Entity distribution:")
    for etype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {etype:20s}: {count:,}")

    print(f"\n  Text length: min={min(text_lengths)}, max={max(text_lengths)}, "
          f"avg={sum(text_lengths)//len(text_lengths)}")

    if stats_only:
        return

    # Remove internal has_pii field before writing
    for sample in golden:
        sample.pop("has_pii", None)

    output = {
        "version": "1.0",
        "description": "Golden 1,000-sample Kaggle PII Detection 2024 test set",
        "source": "https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data",
        "created": datetime.now().isoformat(),
        "seed": seed,
        "total_samples": len(golden),
        "pii_samples": len(pii_docs),
        "non_pii_samples": non_pii_needed,
        "total_entities": total_entities,
        "entity_distribution": dict(sorted(type_counts.items(), key=lambda x: -x[1])),
        "text_hashes": [text_hash(s["text"]) for s in golden],
        "samples": golden,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"\nWritten to {output_path} ({size_mb:.1f} MB)")


def validate(path: str):
    """Validate an existing golden set."""
    with open(path) as f:
        data = json.load(f)

    samples = data.get("samples", [])
    issues = []

    if not samples:
        issues.append("No samples found")

    # Check required fields
    for i, s in enumerate(samples):
        if not s.get("text", "").strip():
            issues.append(f"Sample {i}: empty text")
        if "ground_truth" not in s:
            issues.append(f"Sample {i}: missing ground_truth")

    # Check for duplicates
    texts = [s.get("text", "") for s in samples]
    dupes = len(texts) - len(set(texts))
    if dupes:
        issues.append(f"{dupes} duplicate texts")

    # Check entity distribution matches header
    type_counts = defaultdict(int)
    for s in samples:
        for etype, vals in s.get("ground_truth", {}).items():
            type_counts[etype] += len(vals)

    header_dist = data.get("entity_distribution", {})
    for etype, count in type_counts.items():
        if header_dist.get(etype) != count:
            issues.append(f"{etype}: header says {header_dist.get(etype)}, actual {count}")

    if issues:
        print("Validation FAILED:")
        for issue in issues[:20]:
            print(f"  - {issue}")
        return False

    pii_count = sum(1 for s in samples if s.get("ground_truth"))
    non_pii_count = len(samples) - pii_count
    print(f"Validation PASSED")
    print(f"  {len(samples)} samples ({pii_count} PII, {non_pii_count} non-PII)")
    print(f"  {sum(type_counts.values())} total entities across {len(type_counts)} types")
    print(f"  {len(set(texts))} unique texts (0 duplicates)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Create golden 1,000-sample Kaggle test set")
    parser.add_argument("--input", default=str(DEFAULT_INPUT),
                        help="Path to Kaggle train.json")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="Output path (default: tests/data/kaggle_golden_1000.json)")
    parser.add_argument("--samples", type=int, default=1000,
                        help="Target sample count (default: 1000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--stats", action="store_true",
                        help="Preview stats without writing")
    parser.add_argument("--validate", action="store_true",
                        help="Validate existing golden set")
    args = parser.parse_args()

    if args.validate:
        if not Path(args.output).exists():
            print(f"File not found: {args.output}")
            sys.exit(1)
        success = validate(args.output)
        sys.exit(0 if success else 1)

    if not Path(args.input).exists():
        print(f"Kaggle train.json not found: {args.input}")
        print("Download from: https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data")
        sys.exit(1)

    create_kaggle_golden(
        input_path=args.input,
        output_path=args.output,
        target_samples=args.samples,
        seed=args.seed,
        stats_only=args.stats,
    )


if __name__ == "__main__":
    main()
