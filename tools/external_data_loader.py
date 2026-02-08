#!/usr/bin/env python3
"""
External Data Loader for LightGBM NER Training

Loads training data from external sources (e.g., ai4privacy dataset)
and converts them to the format expected by train_lgbm_ner.py.

This module provides a unified interface for loading training data from:
1. Converted ai4privacy JSON files
2. BIO-format TSV files
3. Other external datasets

Usage with train_lgbm_ner.py:
    from external_data_loader import load_external_training_data

    # Load external data
    external_samples = load_external_training_data(
        entity_type="PERSON",
        data_dir=Path("tests/data/ai4privacy")
    )

    # Combine with synthetic data
    all_samples = synthetic_samples + external_samples
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


def load_ai4privacy_json(
    entity_type: str,
    data_dir: Path,
    max_samples: Optional[int] = None
) -> List[Tuple[str, List[Tuple[int, int, str]]]]:
    """
    Load training data from ai4privacy JSON export.

    The JSON format is:
    {
        "entity_type": "PERSON",
        "samples": [
            {
                "text": "Hello John Smith",
                "entities": [[6, 16, "PERSON"]],
                "source_id": "12345",
                "language": "English"
            },
            ...
        ]
    }

    Args:
        entity_type: The entity type to load (e.g., "PERSON")
        data_dir: Directory containing the JSON files
        max_samples: Maximum number of samples to load (None = all)

    Returns:
        List of (text, entities) tuples matching train_lgbm_ner.py format
    """
    json_file = data_dir / f"{entity_type.lower()}_training_data.json"

    if not json_file.exists():
        logger.warning(f"No data file found: {json_file}")
        return []

    logger.info(f"Loading {entity_type} data from {json_file}")

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []
    for item in data.get("samples", []):
        text = item.get("text", "")
        entities = item.get("entities", [])

        # Convert list format to tuple format
        entity_tuples = [(start, end, label) for start, end, label in entities]
        samples.append((text, entity_tuples))

        if max_samples and len(samples) >= max_samples:
            break

    logger.info(f"Loaded {len(samples)} samples for {entity_type}")
    return samples


def load_bio_format(
    filepath: Path,
    target_entity_type: Optional[str] = None,
    max_samples: Optional[int] = None
) -> List[Tuple[str, List[Tuple[int, int, str]]]]:
    """
    Load training data from BIO-format TSV file.

    TSV format:
        token\tstart\tend\ttag\tsample_id
        John\t6\t10\tB-PERSON\t12345
        Smith\t11\t16\tI-PERSON\t12345

        Hello\t0\t5\tO\t12346
        ...

    Args:
        filepath: Path to the BIO TSV file
        target_entity_type: If specified, only load this entity type
        max_samples: Maximum number of samples to load

    Returns:
        List of (text, entities) tuples
    """
    if not filepath.exists():
        logger.warning(f"No BIO file found: {filepath}")
        return []

    logger.info(f"Loading BIO data from {filepath}")

    samples = []
    current_sample_id = None
    current_tokens = []
    current_entities = []

    with open(filepath, 'r', encoding='utf-8') as f:
        # Skip header
        next(f, None)

        for line in f:
            line = line.strip()

            # Blank line = end of sample
            if not line:
                if current_tokens:
                    # Reconstruct text from tokens
                    text = _reconstruct_text(current_tokens)
                    samples.append((text, current_entities))
                    current_tokens = []
                    current_entities = []
                    current_sample_id = None

                    if max_samples and len(samples) >= max_samples:
                        break
                continue

            parts = line.split('\t')
            if len(parts) < 4:
                continue

            token, start, end, tag = parts[0], int(parts[1]), int(parts[2]), parts[3]
            sample_id = parts[4] if len(parts) > 4 else None

            # New sample
            if sample_id != current_sample_id and current_tokens:
                text = _reconstruct_text(current_tokens)
                samples.append((text, current_entities))
                current_tokens = []
                current_entities = []

            current_sample_id = sample_id
            current_tokens.append((token, start, end))

            # Extract entity info from BIO tag
            if tag.startswith("B-"):
                entity_type = tag[2:]
                if target_entity_type is None or entity_type == target_entity_type:
                    current_entities.append([start, end, entity_type, True])  # True = is beginning
            elif tag.startswith("I-"):
                entity_type = tag[2:]
                if current_entities and current_entities[-1][2] == entity_type:
                    # Extend the previous entity
                    current_entities[-1][1] = end

    # Handle last sample
    if current_tokens:
        text = _reconstruct_text(current_tokens)
        samples.append((text, current_entities))

    # Convert entity lists to tuples
    result = []
    for text, entities in samples:
        entity_tuples = [(start, end, label) for start, end, label, *_ in entities]
        result.append((text, entity_tuples))

    logger.info(f"Loaded {len(result)} samples from BIO format")
    return result


def _reconstruct_text(tokens: List[Tuple[str, int, int]]) -> str:
    """
    Reconstruct text from tokenized form.

    Args:
        tokens: List of (token, start, end) tuples

    Returns:
        Reconstructed text with proper spacing
    """
    if not tokens:
        return ""

    # Find the maximum end position
    max_end = max(end for _, _, end in tokens)

    # Build text character by character
    text = [' '] * max_end
    for token, start, end in tokens:
        for i, char in enumerate(token):
            if start + i < max_end:
                text[start + i] = char

    return ''.join(text).strip()


def load_external_training_data(
    entity_type: str,
    data_dir: Path,
    format: str = "auto",
    max_samples: Optional[int] = None
) -> List[Tuple[str, List[Tuple[int, int, str]]]]:
    """
    Load training data from external sources.

    This is the main entry point for loading external data.

    Args:
        entity_type: The entity type to load
        data_dir: Directory containing the training data
        format: Data format ("json", "bio", or "auto")
        max_samples: Maximum samples to load

    Returns:
        List of (text, entities) tuples compatible with train_lgbm_ner.py
    """
    if not data_dir.exists():
        logger.warning(f"Data directory does not exist: {data_dir}")
        return []

    samples = []

    if format == "auto" or format == "json":
        json_samples = load_ai4privacy_json(entity_type, data_dir, max_samples)
        samples.extend(json_samples)

    if format == "auto" or format == "bio":
        bio_file = data_dir / f"bio_format_{entity_type.lower()}.tsv"
        if bio_file.exists():
            remaining = (max_samples - len(samples)) if max_samples else None
            bio_samples = load_bio_format(bio_file, entity_type, remaining)
            samples.extend(bio_samples)

    return samples


def get_available_entity_types(data_dir: Path) -> List[str]:
    """
    List entity types available in the data directory.

    Args:
        data_dir: Directory containing training data

    Returns:
        List of entity type names
    """
    if not data_dir.exists():
        return []

    entity_types = set()

    # Check JSON files
    for json_file in data_dir.glob("*_training_data.json"):
        # Extract entity type from filename
        name = json_file.stem.replace("_training_data", "")
        entity_types.add(name.upper())

    # Check BIO files
    for bio_file in data_dir.glob("bio_format_*.tsv"):
        name = bio_file.stem.replace("bio_format_", "")
        entity_types.add(name.upper())

    return sorted(entity_types)


def get_sample_count(entity_type: str, data_dir: Path) -> int:
    """
    Get the number of samples available for an entity type.

    Args:
        entity_type: The entity type
        data_dir: Directory containing training data

    Returns:
        Number of samples, or 0 if no data
    """
    json_file = data_dir / f"{entity_type.lower()}_training_data.json"

    if json_file.exists():
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("sample_count", len(data.get("samples", [])))

    return 0


def combine_training_data(
    synthetic_samples: List[Tuple[str, List[Tuple[int, int, str]]]],
    external_samples: List[Tuple[str, List[Tuple[int, int, str]]]],
    synthetic_ratio: float = 0.5
) -> List[Tuple[str, List[Tuple[int, int, str]]]]:
    """
    Combine synthetic and external training data.

    Args:
        synthetic_samples: Samples from SyntheticDataGenerator
        external_samples: Samples from external sources
        synthetic_ratio: Target ratio of synthetic samples (0.0-1.0)

    Returns:
        Combined samples with specified ratio
    """
    import random

    if not external_samples:
        return synthetic_samples

    if not synthetic_samples:
        return external_samples

    # Calculate target counts
    total_target = len(synthetic_samples) + len(external_samples)
    synthetic_target = int(total_target * synthetic_ratio)
    external_target = total_target - synthetic_target

    # Sample appropriately
    synthetic_subset = random.sample(
        synthetic_samples,
        min(len(synthetic_samples), synthetic_target)
    )
    external_subset = random.sample(
        external_samples,
        min(len(external_samples), external_target)
    )

    combined = synthetic_subset + external_subset
    random.shuffle(combined)

    logger.info(f"Combined {len(synthetic_subset)} synthetic + {len(external_subset)} external = {len(combined)} total samples")
    return combined


if __name__ == "__main__":
    # Test loading
    import argparse

    parser = argparse.ArgumentParser(description="Test external data loading")
    parser.add_argument("--data-dir", type=Path, required=True, help="Data directory")
    parser.add_argument("--entity-type", type=str, default="PERSON", help="Entity type to load")
    parser.add_argument("--max-samples", type=int, default=10, help="Max samples to show")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # List available types
    available = get_available_entity_types(args.data_dir)
    print(f"\nAvailable entity types: {available}")

    # Load samples
    samples = load_external_training_data(
        entity_type=args.entity_type,
        data_dir=args.data_dir,
        max_samples=args.max_samples
    )

    print(f"\nLoaded {len(samples)} samples for {args.entity_type}")
    print("\nFirst few samples:")
    for i, (text, entities) in enumerate(samples[:5]):
        print(f"\n--- Sample {i+1} ---")
        print(f"Text: {text[:100]}...")
        print(f"Entities: {entities}")
