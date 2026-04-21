#!/usr/bin/env python3
"""
Kaggle PII Detection 2024 Dataset Adapter for Hush Engine Benchmarking.

Converts the Kaggle "PII Data Detection" competition dataset (BIO token-level
tags on student essays) into the format expected by Hush Engine's benchmark
scripts.

Dataset: https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data
Format: JSON with tokens, trailing_whitespace, and BIO labels per document.

Entity type mapping:
    NAME_STUDENT  -> PERSON
    EMAIL         -> EMAIL
    PHONE_NUM     -> PHONE
    URL_PERSONAL  -> URL
    STREET_ADDRESS -> ADDRESS
    ID_NUM        -> ID
    USERNAME      -> USERNAME

Usage:
    # Step 1: Download train.json from Kaggle and place in tests/data/
    # Step 2: Convert to Hush Engine format
    python tools/kaggle_pii_adapter.py --input tests/data/kaggle_train.json --output tests/data/kaggle_pii.json

    # Step 3: Run benchmark on converted data
    python tests/benchmark_accuracy.py --datasets kaggle_pii.json --samples 500

    # Or run LLM comparison
    python tests/benchmark_llm_comparison.py --datasets kaggle_pii.json --samples 500

Options:
    --max-samples N     Limit to N documents (default: all)
    --only-pii          Only include documents that contain at least one PII entity
    --stats             Print entity type distribution and exit
"""

import argparse
import json
from collections import defaultdict

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
    """Reconstruct full_text from tokens and trailing_whitespace arrays."""
    parts = []
    for token, has_ws in zip(tokens, trailing_whitespace):
        parts.append(token)
        if has_ws:
            parts.append(" ")
    return "".join(parts)


def extract_entities_from_bio(tokens: list, labels: list, trailing_whitespace: list) -> dict:
    """Extract entity spans from BIO-tagged tokens.

    Returns:
        dict mapping Hush Engine entity type -> list of entity text strings
    """
    entities = defaultdict(list)
    current_entity = None
    current_tokens = []

    for i, (token, label) in enumerate(zip(tokens, labels)):
        if label.startswith("B-"):
            # Save previous entity if any
            if current_entity and current_tokens:
                entity_text = reconstruct_text(
                    current_tokens,
                    trailing_whitespace[i - len(current_tokens):i]
                ).strip()
                if entity_text:
                    hush_type = ENTITY_MAP.get(current_entity)
                    if hush_type:
                        entities[hush_type].append(entity_text)

            # Start new entity
            current_entity = label[2:]  # Remove "B-" prefix
            current_tokens = [token]

        elif label.startswith("I-"):
            tag = label[2:]
            if tag == current_entity:
                current_tokens.append(token)
            else:
                # Mismatched I- tag, save current and start new
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


def convert_kaggle_to_hush(input_path: str, output_path: str,
                            max_samples: int = None, only_pii: bool = False,
                            stats_only: bool = False):
    """Convert Kaggle PII dataset to Hush Engine benchmark format."""

    print(f"Loading {input_path}...")
    with open(input_path) as f:
        kaggle_data = json.load(f)

    print(f"Loaded {len(kaggle_data)} documents")

    # Collect stats
    type_counts = defaultdict(int)
    docs_with_pii = 0

    samples = []
    for i, doc in enumerate(kaggle_data):
        if max_samples and len(samples) >= max_samples:
            break

        tokens = doc["tokens"]
        labels = doc.get("labels", [])
        trailing_ws = doc.get("trailing_whitespace", [True] * len(tokens))

        if not labels:
            continue

        # Reconstruct full text
        full_text = doc.get("full_text") or reconstruct_text(tokens, trailing_ws)

        # Extract entities from BIO tags
        ground_truth = extract_entities_from_bio(tokens, labels, trailing_ws)

        # Track stats
        for etype, values in ground_truth.items():
            type_counts[etype] += len(values)

        has_pii = bool(ground_truth)
        if has_pii:
            docs_with_pii += 1

        if only_pii and not has_pii:
            continue

        samples.append({
            "text": full_text,
            "ground_truth": ground_truth,
            "id": doc.get("document", i),
            "source": "kaggle_pii_2024",
        })

    # Print stats
    total_docs = min(len(kaggle_data), max_samples or len(kaggle_data))
    print("\nDataset Statistics:")
    print(f"  Total documents processed: {total_docs}")
    print(f"  Documents with PII: {docs_with_pii} ({docs_with_pii/total_docs*100:.1f}%)")
    print(f"  Documents converted: {len(samples)}")
    print("\n  Entity distribution:")
    for etype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {etype:20s}: {count:,}")
    print(f"    {'TOTAL':20s}: {sum(type_counts.values()):,}")

    if stats_only:
        return

    # Write output
    output = {
        "version": "1.0",
        "description": "Kaggle PII Detection 2024 - converted for Hush Engine benchmark",
        "source": "https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data",
        "total_samples": len(samples),
        "total_entities": sum(type_counts.values()),
        "entity_distribution": dict(type_counts),
        "samples": samples,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nWritten {len(samples)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert Kaggle PII dataset for Hush Engine")
    parser.add_argument("--input", required=True, help="Path to Kaggle train.json")
    parser.add_argument("--output", default="tests/data/kaggle_pii.json",
                        help="Output path (default: tests/data/kaggle_pii.json)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit to N documents")
    parser.add_argument("--only-pii", action="store_true",
                        help="Only include documents containing PII")
    parser.add_argument("--stats", action="store_true",
                        help="Print stats and exit without writing")
    args = parser.parse_args()

    convert_kaggle_to_hush(
        input_path=args.input,
        output_path=args.output,
        max_samples=args.max_samples,
        only_pii=args.only_pii,
        stats_only=args.stats,
    )


if __name__ == "__main__":
    main()
