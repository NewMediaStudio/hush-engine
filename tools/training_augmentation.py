#!/usr/bin/env python3
"""
Training Data Augmentation Module

Implements noise injection techniques based on the PMC paper findings
that achieved 91.1% F1 using:
- 30% probability random punctuation removal
- Text normalization
- Character-level noise (simulating OCR artifacts)

These augmentations help models become robust to real-world document noise
from OCR processing, scanning artifacts, and varied input quality.

Usage:
    from tools.training_augmentation import augment_training_sample, NoiseInjector

    # Augment a single sample
    augmented_text, updated_entities = augment_training_sample(
        text="John Smith lives at 123 Main St.",
        entities=[{"start": 0, "end": 10, "label": "PERSON"}]
    )

    # Batch augmentation
    injector = NoiseInjector()
    augmented_samples = injector.augment_batch(samples, multiplier=3)
"""

import random
import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AugmentationConfig:
    """Configuration for noise injection."""
    punctuation_removal_prob: float = 0.30  # PMC paper: 30%
    case_variation_prob: float = 0.10
    spacing_error_prob: float = 0.10
    char_substitution_prob: float = 0.05
    random_seed: Optional[int] = None


# Common OCR character substitutions (confusion matrix)
OCR_SUBSTITUTIONS = {
    'O': ['0', 'Q', 'D'],
    '0': ['O', 'D'],
    '1': ['l', 'I', '|', '7'],
    'l': ['1', 'I', '|'],
    'I': ['1', 'l', '|'],
    '5': ['S', 's'],
    'S': ['5', '$'],
    's': ['5'],
    '8': ['B', '&'],
    'B': ['8', '3'],
    '6': ['G', 'b'],
    'G': ['6', 'C'],
    'g': ['9', 'q'],
    '9': ['g', 'q'],
    'q': ['9', 'g'],
    '2': ['Z', 'z'],
    'Z': ['2'],
    'z': ['2'],
    'a': ['@', 'o'],
    'e': ['c', '€'],
    'c': ['e', '('],
    'm': ['rn', 'nn'],
    'rn': ['m'],
    'nn': ['m'],
    'w': ['vv', 'uu'],
    'vv': ['w'],
    'cl': ['d'],
    'd': ['cl'],
}

# Punctuation that can be removed without breaking meaning
REMOVABLE_PUNCTUATION = set('.,;:!?\'"-()[]{}')


def random_punctuation_removal(text: str, probability: float = 0.30) -> Tuple[str, Dict[int, int]]:
    """
    Randomly remove punctuation from text.

    Based on PMC paper: "Data Preprocessing adds minor random noise like
    punctuation removal (~30% probability)"

    Args:
        text: Input text
        probability: Probability of removing each punctuation mark

    Returns:
        Tuple of (modified text, position mapping from old to new indices)
    """
    result = []
    position_map = {}  # Maps old position to new position
    new_pos = 0

    for old_pos, char in enumerate(text):
        if char in REMOVABLE_PUNCTUATION and random.random() < probability:
            # Remove this punctuation
            position_map[old_pos] = new_pos  # Map to current position (will be next char)
            continue

        position_map[old_pos] = new_pos
        result.append(char)
        new_pos += 1

    return ''.join(result), position_map


def random_case_variation(text: str, probability: float = 0.10) -> str:
    """
    Introduce random case errors to simulate OCR artifacts.

    Args:
        text: Input text
        probability: Probability of changing case for each character

    Returns:
        Text with random case variations
    """
    result = []
    for char in text:
        if char.isalpha() and random.random() < probability:
            result.append(char.swapcase())
        else:
            result.append(char)
    return ''.join(result)


def random_spacing_errors(text: str, probability: float = 0.10) -> Tuple[str, Dict[int, int]]:
    """
    Add or remove spaces to simulate OCR spacing errors.

    Args:
        text: Input text
        probability: Probability of modifying spacing at each position

    Returns:
        Tuple of (modified text, position mapping)
    """
    result = []
    position_map = {}
    new_pos = 0

    for old_pos, char in enumerate(text):
        position_map[old_pos] = new_pos

        if char == ' ' and random.random() < probability:
            # Maybe remove space
            if random.random() < 0.5:
                continue  # Remove space

        result.append(char)
        new_pos += 1

        # Maybe add extra space after character
        if char != ' ' and random.random() < probability * 0.3:
            result.append(' ')
            new_pos += 1

    return ''.join(result), position_map


def random_char_substitution(text: str, probability: float = 0.05) -> Tuple[str, Dict[int, int]]:
    """
    Substitute characters with common OCR confusion pairs.

    Args:
        text: Input text
        probability: Probability of substituting each character

    Returns:
        Tuple of (modified text, position mapping)
    """
    result = []
    position_map = {}
    new_pos = 0

    i = 0
    while i < len(text):
        position_map[i] = new_pos
        char = text[i]

        # Check for multi-character patterns (e.g., 'rn' -> 'm')
        if i < len(text) - 1:
            two_char = text[i:i+2]
            if two_char in OCR_SUBSTITUTIONS and random.random() < probability:
                replacement = random.choice(OCR_SUBSTITUTIONS[two_char])
                result.append(replacement)
                new_pos += len(replacement)
                position_map[i+1] = new_pos
                i += 2
                continue

        # Single character substitution
        if char in OCR_SUBSTITUTIONS and random.random() < probability:
            replacement = random.choice(OCR_SUBSTITUTIONS[char])
            result.append(replacement)
            new_pos += len(replacement)
        else:
            result.append(char)
            new_pos += 1

        i += 1

    return ''.join(result), position_map


def update_entity_positions(
    entities: List[Dict[str, Any]],
    position_map: Dict[int, int],
    new_text_length: int
) -> List[Dict[str, Any]]:
    """
    Update entity positions after text modification.

    Args:
        entities: List of entity dicts with 'start' and 'end' keys
        position_map: Mapping from old positions to new positions
        new_text_length: Length of the modified text

    Returns:
        List of entities with updated positions
    """
    updated = []

    for entity in entities:
        old_start = entity.get('start', 0)
        old_end = entity.get('end', 0)

        # Find new positions
        new_start = position_map.get(old_start, old_start)

        # For end position, we need to find the mapped position
        # If exact end isn't in map, find closest previous position
        new_end = old_end
        for pos in range(old_end, -1, -1):
            if pos in position_map:
                new_end = position_map[pos]
                if pos < old_end:
                    new_end += 1  # Adjust if we found a position before end
                break

        # Validate positions
        new_start = max(0, min(new_start, new_text_length))
        new_end = max(new_start, min(new_end, new_text_length))

        # Skip if entity becomes empty
        if new_start >= new_end:
            continue

        updated_entity = entity.copy()
        updated_entity['start'] = new_start
        updated_entity['end'] = new_end
        updated.append(updated_entity)

    return updated


def augment_training_sample(
    text: str,
    entities: List[Dict[str, Any]],
    config: Optional[AugmentationConfig] = None
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Augment a training sample with noise injection.

    Applies noise while preserving entity boundaries as much as possible.

    Args:
        text: Original text
        entities: List of entity annotations with 'start', 'end', 'label' keys
        config: Augmentation configuration

    Returns:
        Tuple of (augmented text, updated entities)
    """
    if config is None:
        config = AugmentationConfig()

    if config.random_seed is not None:
        random.seed(config.random_seed)

    current_text = text
    current_entities = entities.copy()

    # Apply punctuation removal (main augmentation from PMC paper)
    if random.random() < 0.8:  # Apply to 80% of samples
        new_text, position_map = random_punctuation_removal(
            current_text, config.punctuation_removal_prob
        )
        current_entities = update_entity_positions(
            current_entities, position_map, len(new_text)
        )
        current_text = new_text

    # Apply case variation (less aggressive, doesn't change positions)
    if random.random() < 0.5:  # Apply to 50% of samples
        current_text = random_case_variation(current_text, config.case_variation_prob)

    # Apply spacing errors (less frequent, can break entity alignment)
    if random.random() < 0.3:  # Apply to 30% of samples
        new_text, position_map = random_spacing_errors(
            current_text, config.spacing_error_prob
        )
        current_entities = update_entity_positions(
            current_entities, position_map, len(new_text)
        )
        current_text = new_text

    # Apply character substitution (OCR simulation, least frequent)
    if random.random() < 0.2:  # Apply to 20% of samples
        new_text, position_map = random_char_substitution(
            current_text, config.char_substitution_prob
        )
        current_entities = update_entity_positions(
            current_entities, position_map, len(new_text)
        )
        current_text = new_text

    return current_text, current_entities


class NoiseInjector:
    """
    Batch noise injection for training data augmentation.

    Based on PMC paper methodology for improving model robustness.
    """

    def __init__(self, config: Optional[AugmentationConfig] = None):
        """
        Initialize the noise injector.

        Args:
            config: Augmentation configuration
        """
        self.config = config or AugmentationConfig()

    def augment_sample(
        self,
        text: str,
        entities: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Augment a single sample."""
        return augment_training_sample(text, entities, self.config)

    def augment_batch(
        self,
        samples: List[Dict[str, Any]],
        multiplier: int = 3,
        include_original: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Augment a batch of training samples.

        Args:
            samples: List of samples with 'text' and 'entities' keys
            multiplier: Number of augmented versions to create per sample
            include_original: Whether to include original samples in output

        Returns:
            List of augmented samples
        """
        result = []

        for sample in samples:
            text = sample.get('text', '')
            entities = sample.get('entities', [])

            # Include original
            if include_original:
                result.append(sample.copy())

            # Generate augmented versions
            for i in range(multiplier):
                aug_text, aug_entities = self.augment_sample(text, entities)

                # Create augmented sample
                aug_sample = sample.copy()
                aug_sample['text'] = aug_text
                aug_sample['entities'] = aug_entities
                aug_sample['augmented'] = True
                aug_sample['augmentation_idx'] = i
                result.append(aug_sample)

        return result

    def augment_bio_data(
        self,
        tokens: List[str],
        labels: List[str],
        multiplier: int = 3
    ) -> List[Tuple[List[str], List[str]]]:
        """
        Augment BIO-format training data.

        Args:
            tokens: List of tokens
            labels: List of BIO labels
            multiplier: Number of augmented versions

        Returns:
            List of (tokens, labels) tuples
        """
        result = [(tokens.copy(), labels.copy())]  # Include original

        # Reconstruct text and entities
        text = ' '.join(tokens)
        entities = []
        current_entity = None
        char_pos = 0

        for i, (token, label) in enumerate(zip(tokens, labels)):
            token_start = char_pos
            token_end = char_pos + len(token)

            if label.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    'start': token_start,
                    'end': token_end,
                    'label': label[2:]
                }
            elif label.startswith('I-') and current_entity:
                current_entity['end'] = token_end
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

            char_pos = token_end + 1  # +1 for space

        if current_entity:
            entities.append(current_entity)

        # Generate augmented versions
        for _ in range(multiplier):
            aug_text, aug_entities = self.augment_sample(text, entities)

            # Tokenize augmented text and rebuild BIO labels
            aug_tokens = aug_text.split()
            aug_labels = ['O'] * len(aug_tokens)

            # Map entities back to tokens
            char_pos = 0
            for i, token in enumerate(aug_tokens):
                token_start = char_pos
                token_end = char_pos + len(token)

                for entity in aug_entities:
                    if entity['start'] <= token_start < entity['end']:
                        if entity['start'] == token_start:
                            aug_labels[i] = f"B-{entity['label']}"
                        else:
                            aug_labels[i] = f"I-{entity['label']}"
                        break

                char_pos = token_end + 1

            result.append((aug_tokens, aug_labels))

        return result


def demo():
    """Demonstrate augmentation capabilities."""
    print("=" * 70)
    print("TRAINING DATA AUGMENTATION DEMO")
    print("Based on PMC paper methodology (91.1% F1)")
    print("=" * 70)

    # Sample data
    samples = [
        {
            "text": "John Smith lives at 123 Main Street, New York, NY 10001.",
            "entities": [
                {"start": 0, "end": 10, "label": "PERSON"},
                {"start": 20, "end": 55, "label": "ADDRESS"}
            ]
        },
        {
            "text": "Contact: jane.doe@email.com or call (555) 123-4567",
            "entities": [
                {"start": 9, "end": 27, "label": "EMAIL"},
                {"start": 36, "end": 50, "label": "PHONE"}
            ]
        },
        {
            "text": "SSN: 123-45-6789, Credit Card: 4532-1234-5678-9010",
            "entities": [
                {"start": 5, "end": 16, "label": "SSN"},
                {"start": 31, "end": 50, "label": "CREDIT_CARD"}
            ]
        }
    ]

    injector = NoiseInjector()

    for sample in samples:
        print(f"\nOriginal: {sample['text']}")
        print(f"Entities: {sample['entities']}")

        for i in range(3):
            aug_text, aug_entities = injector.augment_sample(
                sample['text'], sample['entities']
            )
            print(f"  Aug {i+1}: {aug_text}")
            print(f"          {aug_entities}")

    print("\n" + "=" * 70)
    print("Batch augmentation example:")
    augmented = injector.augment_batch(samples[:1], multiplier=2)
    for i, aug in enumerate(augmented):
        print(f"  Sample {i}: {aug['text'][:50]}... (augmented: {aug.get('augmented', False)})")


if __name__ == "__main__":
    demo()
