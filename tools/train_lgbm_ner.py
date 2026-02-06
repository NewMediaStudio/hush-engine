#!/usr/bin/env python3
"""
Training Script for LightGBM NER Models

Generates synthetic training data using Faker and public datasets,
then trains LightGBM classifiers for each entity type.

PRIVACY NOTICE:
- Training data is 100% synthetic (Faker) or from public datasets
- NO real user data is ever used
- NO user feedback files are read for training
- Models are trained locally, nothing is uploaded

Usage:
    python tools/train_lgbm_ner.py --entity-type PERSON --samples 10000
    python tools/train_lgbm_ner.py --all --samples 5000
"""

import argparse
import logging
import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Set
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check dependencies
try:
    import numpy as np
    import lightgbm as lgb
    from faker import Faker
except ImportError as e:
    logger.error(f"Missing dependency: {e}")
    logger.error("Install with: pip install lightgbm faker numpy")
    sys.exit(1)

from hush_engine.detectors.feature_extractor import (
    extract_features_with_context,
    features_to_matrix,
    tokenize,
    FEATURE_NAMES,
)

# Output directory for trained models
MODEL_DIR = Path(__file__).parent.parent / "hush_engine" / "models" / "lgbm"

# Supported entity types
ENTITY_TYPES = ["PERSON", "LOCATION", "ORGANIZATION", "DATE_TIME"]


class SyntheticDataGenerator:
    """
    Generates synthetic training data for NER using Faker.

    All data is synthetic - no real PII is used.
    """

    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducibility."""
        self.fake = Faker()
        Faker.seed(seed)
        random.seed(seed)

        # Load additional data from our databases
        self._load_databases()

    def _load_databases(self):
        """Load supplementary data from our databases."""
        try:
            from hush_engine.data.names_database import NamesDatabase
            db = NamesDatabase()
            self.first_names = list(db._first_names)[:1000]
            self.last_names = list(db._last_names)[:1000]
        except Exception:
            self.first_names = [self.fake.first_name() for _ in range(500)]
            self.last_names = [self.fake.last_name() for _ in range(500)]

        try:
            from hush_engine.data.cities_database import CitiesDatabase
            db = CitiesDatabase()
            self.cities = list(db._all_cities)[:500]
        except Exception:
            self.cities = [self.fake.city() for _ in range(200)]

    def generate_person_samples(self, n: int) -> List[Tuple[str, List[Tuple[int, int, str]]]]:
        """
        Generate sentences containing person names.

        Args:
            n: Number of samples to generate

        Returns:
            List of (sentence, [(start, end, 'PERSON'), ...]) tuples
        """
        samples = []
        templates = [
            "My name is {name}.",
            "{name} works at the company.",
            "Please contact {name} for more information.",
            "Dr. {name} will see you now.",
            "The report was written by {name}.",
            "{name} and {name2} attended the meeting.",
            "Dear {name}, thank you for your inquiry.",
            "Signed, {name}",
            "Patient: {name}",
            "The CEO, {name}, announced the news.",
            "{name} sent an email yesterday.",
            "According to {name}, the project is complete.",
            "Hello {name}, welcome back!",
            "Meet {name}, our new team member.",
        ]

        for _ in range(n):
            template = random.choice(templates)
            first = random.choice(self.first_names).title()
            last = random.choice(self.last_names).title()
            name = f"{first} {last}"

            if "{name2}" in template:
                first2 = random.choice(self.first_names).title()
                last2 = random.choice(self.last_names).title()
                name2 = f"{first2} {last2}"
                sentence = template.format(name=name, name2=name2)

                # Find positions of both names
                entities = []
                pos1 = sentence.find(name)
                if pos1 >= 0:
                    entities.append((pos1, pos1 + len(name), "PERSON"))
                pos2 = sentence.find(name2)
                if pos2 >= 0:
                    entities.append((pos2, pos2 + len(name2), "PERSON"))
            else:
                sentence = template.format(name=name)
                pos = sentence.find(name)
                entities = [(pos, pos + len(name), "PERSON")] if pos >= 0 else []

            samples.append((sentence, entities))

        return samples

    def generate_location_samples(self, n: int) -> List[Tuple[str, List[Tuple[int, int, str]]]]:
        """Generate sentences containing location names."""
        samples = []
        templates = [
            "I live in {city}.",
            "The conference will be held in {city}.",
            "She moved to {city} last year.",
            "{city} is a beautiful city.",
            "Our office is located in {city}.",
            "The flight departs from {city}.",
            "Weather in {city} is sunny today.",
            "I visited {city} during my vacation.",
            "The company has branches in {city} and {city2}.",
            "Address: 123 Main St, {city}",
        ]

        for _ in range(n):
            template = random.choice(templates)
            city = random.choice(self.cities).title()

            if "{city2}" in template:
                city2 = random.choice(self.cities).title()
                sentence = template.format(city=city, city2=city2)
                entities = []
                for c in [city, city2]:
                    pos = sentence.find(c)
                    if pos >= 0:
                        entities.append((pos, pos + len(c), "LOCATION"))
            else:
                sentence = template.format(city=city)
                pos = sentence.find(city)
                entities = [(pos, pos + len(city), "LOCATION")] if pos >= 0 else []

            samples.append((sentence, entities))

        return samples

    def generate_organization_samples(self, n: int) -> List[Tuple[str, List[Tuple[int, int, str]]]]:
        """Generate sentences containing organization names."""
        samples = []
        org_suffixes = ["Inc", "Corp", "LLC", "Ltd", "Company", "Group", "Foundation"]

        templates = [
            "{org} announced their quarterly earnings.",
            "I work at {org}.",
            "The merger between {org} and {org2} was approved.",
            "{org} is hiring new employees.",
            "According to {org} spokesperson...",
            "The {org} headquarters is downtown.",
            "Shares of {org} rose 5% today.",
        ]

        for _ in range(n):
            template = random.choice(templates)
            org_base = self.fake.company().split()[0]
            suffix = random.choice(org_suffixes)
            org = f"{org_base} {suffix}"

            if "{org2}" in template:
                org2_base = self.fake.company().split()[0]
                org2 = f"{org2_base} {random.choice(org_suffixes)}"
                sentence = template.format(org=org, org2=org2)
                entities = []
                for o in [org, org2]:
                    pos = sentence.find(o)
                    if pos >= 0:
                        entities.append((pos, pos + len(o), "ORGANIZATION"))
            else:
                sentence = template.format(org=org)
                pos = sentence.find(org)
                entities = [(pos, pos + len(org), "ORGANIZATION")] if pos >= 0 else []

            samples.append((sentence, entities))

        return samples

    def generate_datetime_samples(self, n: int) -> List[Tuple[str, List[Tuple[int, int, str]]]]:
        """Generate sentences containing dates/times."""
        samples = []
        templates = [
            "The meeting is scheduled for {date}.",
            "Born on {date}.",
            "Date: {date}",
            "The deadline is {date}.",
            "As of {date}, the policy changed.",
            "Effective {date}",
            "Event date: {date}",
        ]

        date_formats = [
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%Y-%m-%d",
            "%B %d, %Y",
            "%b %d, %Y",
            "%d %B %Y",
        ]

        for _ in range(n):
            template = random.choice(templates)
            date_obj = self.fake.date_object()
            date_str = date_obj.strftime(random.choice(date_formats))

            sentence = template.format(date=date_str)
            pos = sentence.find(date_str)
            entities = [(pos, pos + len(date_str), "DATE_TIME")] if pos >= 0 else []

            samples.append((sentence, entities))

        return samples

    def generate_negative_samples(self, n: int) -> List[Tuple[str, List[Tuple[int, int, str]]]]:
        """Generate sentences without entities (negative samples)."""
        samples = []
        templates = [
            "The weather is nice today.",
            "Please submit your report by end of day.",
            "The system is currently under maintenance.",
            "Click here to continue.",
            "Thank you for your patience.",
            "This document is confidential.",
            "All rights reserved.",
            "Version 2.0 released.",
            "Loading data, please wait...",
            "Error: File not found.",
            "The quick brown fox jumps over the lazy dog.",
            "Important: Read the instructions carefully.",
        ]

        for _ in range(n):
            sentence = random.choice(templates)
            # Add some variation
            if random.random() > 0.5:
                sentence = sentence + " " + random.choice(templates)
            samples.append((sentence, []))

        return samples


def create_training_data(
    entity_type: str,
    n_positive: int,
    n_negative: int,
    generator: SyntheticDataGenerator
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create training data for a specific entity type.

    Uses hard negative mining: includes samples from OTHER entity types
    as negatives to help the model distinguish between entity types.

    Args:
        entity_type: The entity type to train for
        n_positive: Number of positive samples
        n_negative: Number of negative samples
        generator: SyntheticDataGenerator instance

    Returns:
        (X, y) tuple of features and labels
    """
    logger.info(f"Generating {n_positive} positive and {n_negative} negative samples for {entity_type}")

    # Generate positive samples for this entity type
    if entity_type == "PERSON":
        positive_samples = generator.generate_person_samples(n_positive)
    elif entity_type == "LOCATION":
        positive_samples = generator.generate_location_samples(n_positive)
    elif entity_type == "ORGANIZATION":
        positive_samples = generator.generate_organization_samples(n_positive)
    elif entity_type == "DATE_TIME":
        positive_samples = generator.generate_datetime_samples(n_positive)
    else:
        raise ValueError(f"Unsupported entity type: {entity_type}")

    # Generate negative samples: mix of plain text AND other entity types
    # This is "hard negative mining" - helps the model distinguish entity types
    negative_samples = generator.generate_negative_samples(n_negative // 2)

    # Add samples from OTHER entity types as hard negatives
    hard_negatives_per_type = n_negative // 6  # Divide among other types

    if entity_type != "PERSON":
        negative_samples.extend(generator.generate_person_samples(hard_negatives_per_type))
    if entity_type != "LOCATION":
        negative_samples.extend(generator.generate_location_samples(hard_negatives_per_type))
    if entity_type != "ORGANIZATION":
        negative_samples.extend(generator.generate_organization_samples(hard_negatives_per_type))

    # Convert to features
    all_features = []
    all_labels = []

    for sentence, entities in positive_samples + negative_samples:
        tokens = tokenize(sentence)
        features_list = extract_features_with_context(sentence)
        feature_dicts = features_to_matrix(features_list)

        # Create labels based on entity spans
        for i, (token, start, end) in enumerate(tokens):
            # Check if token overlaps with any entity of this type
            is_entity = False
            for ent_start, ent_end, ent_type in entities:
                if ent_type == entity_type and start < ent_end and end > ent_start:
                    is_entity = True
                    break

            all_features.append(feature_dicts[i])
            all_labels.append(1 if is_entity else 0)

    # Convert to numpy arrays
    X = np.zeros((len(all_features), len(FEATURE_NAMES)))
    for i, feat_dict in enumerate(all_features):
        for j, name in enumerate(FEATURE_NAMES):
            X[i, j] = feat_dict.get(name, 0)

    y = np.array(all_labels)

    logger.info(f"Created {len(y)} training samples ({sum(y)} positive, {len(y) - sum(y)} negative)")
    return X, y


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    entity_type: str,
    params: Dict = None
) -> lgb.Booster:
    """
    Train a LightGBM classifier.

    Args:
        X: Feature matrix
        y: Labels
        entity_type: Entity type (for logging)
        params: LightGBM parameters

    Returns:
        Trained Booster model
    """
    # Calculate class weights for imbalanced data
    n_pos = sum(y)
    n_neg = len(y) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    default_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 15,           # Reduced to prevent overfitting
        'learning_rate': 0.03,      # Slower learning
        'feature_fraction': 0.7,    # More regularization
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_data_in_leaf': 20,     # Prevent overfitting to small groups
        'lambda_l1': 0.1,           # L1 regularization
        'lambda_l2': 0.1,           # L2 regularization
        'scale_pos_weight': scale_pos_weight,  # Handle class imbalance
        'verbose': -1,
        'num_threads': 4,
    }

    if params:
        default_params.update(params)

    # Split for validation
    n_train = int(0.8 * len(y))
    indices = np.random.permutation(len(y))
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    train_data = lgb.Dataset(X[train_idx], y[train_idx], feature_name=FEATURE_NAMES)
    val_data = lgb.Dataset(X[val_idx], y[val_idx], feature_name=FEATURE_NAMES, reference=train_data)

    logger.info(f"Training LightGBM classifier for {entity_type} (pos_weight={scale_pos_weight:.2f})...")

    model = lgb.train(
        default_params,
        train_data,
        num_boost_round=200,        # More rounds with early stopping
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=20),
            lgb.log_evaluation(period=50),
        ]
    )

    # Evaluate with threshold tuning
    val_preds = model.predict(X[val_idx])

    # Find optimal threshold using F1 score
    best_f1 = 0
    best_threshold = 0.5
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        val_preds_binary = (val_preds >= threshold).astype(int)
        tp = sum((val_preds_binary == 1) & (y[val_idx] == 1))
        fp = sum((val_preds_binary == 1) & (y[val_idx] == 0))
        fn = sum((val_preds_binary == 0) & (y[val_idx] == 1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # Final evaluation at best threshold
    val_preds_binary = (val_preds >= best_threshold).astype(int)
    tp = sum((val_preds_binary == 1) & (y[val_idx] == 1))
    fp = sum((val_preds_binary == 1) & (y[val_idx] == 0))
    fn = sum((val_preds_binary == 0) & (y[val_idx] == 1))
    tn = sum((val_preds_binary == 0) & (y[val_idx] == 0))

    accuracy = (tp + tn) / len(y[val_idx])
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    logger.info(f"Validation @ threshold={best_threshold} - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {best_f1:.3f}")

    return model


def save_model(model: lgb.Booster, entity_type: str, output_dir: Path):
    """Save trained model to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{entity_type.lower()}_classifier.txt"
    model.save_model(str(model_path))
    logger.info(f"Saved model to {model_path}")

    # Save metadata
    metadata = {
        "entity_type": entity_type,
        "trained_at": datetime.now().isoformat(),
        "feature_names": FEATURE_NAMES,
        "num_trees": model.num_trees(),
        "privacy_note": "Trained on synthetic data only. No real PII used.",
    }
    metadata_path = output_dir / f"{entity_type.lower()}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM NER classifiers")
    parser.add_argument(
        "--entity-type",
        choices=ENTITY_TYPES,
        help="Entity type to train (or use --all)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train classifiers for all entity types"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5000,
        help="Number of positive samples per entity type (default: 5000)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODEL_DIR,
        help=f"Output directory for models (default: {MODEL_DIR})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    if not args.entity_type and not args.all:
        parser.error("Must specify --entity-type or --all")

    entity_types = ENTITY_TYPES if args.all else [args.entity_type]

    logger.info("=" * 60)
    logger.info("LightGBM NER Training")
    logger.info("PRIVACY: Using synthetic data only. No real PII.")
    logger.info("=" * 60)

    generator = SyntheticDataGenerator(seed=args.seed)

    for entity_type in entity_types:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training classifier for: {entity_type}")
        logger.info(f"{'='*60}")

        # Create training data
        X, y = create_training_data(
            entity_type=entity_type,
            n_positive=args.samples,
            n_negative=args.samples,  # Equal negative samples
            generator=generator
        )

        # Train model
        model = train_classifier(X, y, entity_type)

        # Save model
        save_model(model, entity_type, args.output_dir)

    logger.info("\nTraining complete!")
    logger.info(f"Models saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
