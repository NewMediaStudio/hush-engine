#!/usr/bin/env python3
"""
AI4Privacy Dataset Converter for Hush Engine

Converts the ai4privacy/pii-masking-300k dataset to the format expected by
the LightGBM NER training pipeline (train_lgbm_ner.py).

Dataset Source: https://huggingface.co/datasets/ai4privacy/pii-masking-300k

The ai4privacy dataset provides:
- source_text: Original text with PII
- target_text: Text with [LABEL] placeholders
- privacy_mask: Array of {value, start, end, label} annotations
- mbert_bio_labels: BIO tags for mBERT tokens (optional)

This script:
1. Maps ai4privacy entity types to Hush Engine entity types
2. Extracts character-level entity spans
3. Outputs training data compatible with our feature extractor

PRIVACY NOTICE:
- This dataset contains SYNTHETIC PII data, not real personal information
- All data is generated/anonymized by AI4Privacy for training purposes
- No real user data is processed

Usage:
    # Install dependencies first
    pip install datasets

    # Basic conversion (PERSON entities only)
    python tools/convert_ai4privacy.py --entity-type PERSON --output tests/data/ai4privacy

    # Convert all supported entity types
    python tools/convert_ai4privacy.py --all --output tests/data/ai4privacy

    # Limit samples for quick testing
    python tools/convert_ai4privacy.py --all --max-samples 1000 --output tests/data/ai4privacy

    # Filter by language
    python tools/convert_ai4privacy.py --all --language English --output tests/data/ai4privacy
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import random

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# ENTITY TYPE MAPPING
# ============================================================================

# Map ai4privacy labels to Hush Engine entity types
# The ai4privacy dataset has 27+ fine-grained labels; we consolidate them
# into Hush Engine's broader categories

ENTITY_TYPE_MAPPING = {
    # PERSON - Name components
    "GIVENNAME1": "PERSON",
    "GIVENNAME2": "PERSON",      # Middle names
    "LASTNAME1": "PERSON",
    "LASTNAME2": "PERSON",
    "LASTNAME3": "PERSON",
    "TITLE": "PERSON",           # Mr., Mrs., Dr., etc.
    "USERNAME": "PERSON",        # Could also be excluded

    # LOCATION - Geographic entities
    "CITY": "LOCATION",
    "COUNTRY": "LOCATION",
    "STATE": "LOCATION",
    "POSTCODE": "LOCATION",      # ZIP codes as location

    # ADDRESS - Street-level addresses
    "STREET": "ADDRESS",
    "BUILDING": "ADDRESS",
    "SECADDRESS": "ADDRESS",     # Secondary address (Apt, Suite)

    # ORGANIZATION - Company/institution names
    # Note: ai4privacy doesn't have a direct ORGANIZATION label
    # but some contexts may be inferred

    # DATE_TIME - Temporal entities
    "DATE": "DATE_TIME",
    "TIME": "DATE_TIME",
    "BOD": "DATE_TIME",          # Birth of Date

    # FINANCIAL - Financial identifiers
    "IBAN": "FINANCIAL",
    "BIC": "FINANCIAL",
    "CREDITCARD": "FINANCIAL",
    "ACCOUNTNUMBER": "FINANCIAL",

    # CONTACT - Communication identifiers
    "EMAIL": "EMAIL_ADDRESS",
    "TEL": "PHONE_NUMBER",
    "IP": "IP_ADDRESS",

    # GOVERNMENT IDs - Official identifiers
    "SOCIALNUMBER": "SSN",
    "IDCARD": "SSN",             # Generic ID mapped to SSN
    "PASSPORT": "SSN",
    "DRIVERLICENSE": "SSN",

    # OTHER/EXCLUDED - Labels we don't train for
    "SEX": None,                 # Gender - too granular
    "PASS": None,                # Passwords - not relevant for OCR
    "GEOCOORD": "COORDINATES",   # Geographic coordinates

    # Additional labels found in the dataset
    "ACCOUNTNAME": None,         # Account names - context dependent
    "CURRENCYCODE": None,        # Currency codes (USD, EUR)
    "CURRENCYSYMBOL": None,      # Currency symbols
    "CURRENCYNAME": None,        # Currency names
    "AMOUNT": "FINANCIAL",       # Money amounts
    "VEHICLEVRM": None,          # Vehicle registration
    "AGE": None,                 # Age values
    "JOBDESCRIPTOR": None,       # Job titles
    "JOBTITLE": None,            # Job titles
    "JOBTYPE": None,             # Job types
    "JOBAREA": None,             # Job areas
    "PIN": None,                 # PINs
    "MASKEDNUMBER": None,        # Masked numbers
    "BITCOINADDRESS": "FINANCIAL",
    "ETHEREUMADDRESS": "FINANCIAL",
    "LITECOINADDRESS": "FINANCIAL",
    "ORDINALDIRECTION": None,    # North, South, etc.
    "BUILDINGNUMBER": "ADDRESS", # Building numbers
    "SECONDARYADDRESS": "ADDRESS",
    "STREETADDRESS": "ADDRESS",
    "COUNTY": "LOCATION",
    "NEARBYGPSCOORDINATE": "COORDINATES",
    "GENDER": None,
    "MIDDLENAME": "PERSON",
    "PREFIX": "PERSON",          # Name prefixes
    "SUFFIX": "PERSON",          # Name suffixes (Jr., III)
    "FULLNAME": "PERSON",
    "FIRSTNAME": "PERSON",
    "LASTNAME": "PERSON",
    "MAC": None,                 # MAC addresses
    "USERAGENT": None,           # User agents
    "URL": "URL",
    "IPV4": "IP_ADDRESS",
    "IPV6": "IP_ADDRESS",
    "PASSWORD": None,
    "CREDITCARDNUMBER": "FINANCIAL",
    "CREDITCARDISSUER": None,
    "CREDITCARDCVV": None,
    "CURRENCY": "FINANCIAL",
    "COMPANYNAME": "ORGANIZATION",
    "COMPANY": "ORGANIZATION",
    "PHONENUMBER": "PHONE_NUMBER",
    "PHONEIMEI": None,
    "SSN": "SSN",
    "DISPLAYNAME": "PERSON",
    "DOB": "DATE_TIME",
    "ZIPCODE": "LOCATION",
    "MOBILE": "PHONE_NUMBER",
    "LANDLINE": "PHONE_NUMBER",
    "FAX": "PHONE_NUMBER",
    "BIRTHDATE": "DATE_TIME",
    "DATE_TIME": "DATE_TIME",
}

# Hush Engine entity types we train for
HUSH_ENTITY_TYPES = {
    "PERSON",
    "LOCATION",
    "ADDRESS",
    "ORGANIZATION",
    "DATE_TIME",
    "FINANCIAL",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "IP_ADDRESS",
    "SSN",
    "COORDINATES",
    "URL",
}


@dataclass
class EntitySpan:
    """Represents an entity span in text."""
    start: int
    end: int
    label: str
    original_label: str
    text: str


@dataclass
class TrainingSample:
    """A training sample with text and entity annotations."""
    text: str
    entities: List[EntitySpan]
    source_id: str
    language: str


@dataclass
class ConversionStats:
    """Statistics about the conversion process."""
    total_records: int = 0
    records_with_entities: int = 0
    records_skipped: int = 0
    entities_by_type: Dict[str, int] = field(default_factory=Counter)
    original_labels: Dict[str, int] = field(default_factory=Counter)
    unmapped_labels: Dict[str, int] = field(default_factory=Counter)
    languages: Dict[str, int] = field(default_factory=Counter)
    sentences_extracted: int = 0
    sentences_with_entities: int = 0
    sentences_negative: int = 0


# ============================================================================
# SENTENCE EXTRACTION
# ============================================================================

# Sentence boundary pattern - handles common abbreviations
SENTENCE_SPLIT_PATTERN = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z])|'  # Standard sentence boundaries
    r'(?<=\n)\s*(?=\S)|'        # Newline boundaries
    r'(?<=[.!?])\s*$'           # End of text
)

# Common abbreviations that shouldn't split sentences
ABBREVIATIONS = {
    'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'vs', 'etc', 'inc', 'ltd',
    'corp', 'no', 'vol', 'dept', 'est', 'approx', 'st', 'ave', 'rd', 'blvd'
}


def split_into_sentences(text: str) -> List[Tuple[int, int, str]]:
    """
    Split text into sentences with their character offsets.

    This uses a simple regex-based approach that handles common cases.
    For production use, consider using spaCy or nltk for better accuracy.

    Args:
        text: The source text to split

    Returns:
        List of (start, end, sentence_text) tuples
    """
    sentences = []

    # Simple sentence boundary detection
    # Split on .!? followed by space and capital letter, or newlines
    pattern = re.compile(
        r'(?<=[.!?])\s+(?=[A-Z"\'(])|'  # Period/!/? + space + capital
        r'\n\s*\n|'                       # Double newlines (paragraph breaks)
        r'(?<=\n)(?=[A-Z])'               # Single newline + capital
    )

    last_end = 0
    for match in pattern.finditer(text):
        start = last_end
        end = match.start()

        # Skip if this would create an empty sentence
        sentence = text[start:end].strip()
        if sentence and len(sentence) >= 3:
            sentences.append((start, end, sentence))

        last_end = match.end()

    # Handle the last sentence
    if last_end < len(text):
        sentence = text[last_end:].strip()
        if sentence and len(sentence) >= 3:
            sentences.append((last_end, len(text), sentence))

    # If no sentences found, return the whole text as one sentence
    if not sentences and text.strip():
        sentences.append((0, len(text), text.strip()))

    return sentences


def extract_sentence_samples(
    sample: 'TrainingSample',
    negative_sample_rate: float = 0.1,
    min_tokens: int = 3,
    max_tokens: int = 50
) -> List['TrainingSample']:
    """
    Extract sentence-level samples from a document-level sample.

    This addresses the class imbalance problem by:
    1. Splitting documents into individual sentences
    2. Keeping all sentences with entities (positive samples)
    3. Sampling a fraction of empty sentences (negative samples)
    4. Renumbering entity positions relative to sentence start

    Args:
        sample: A document-level TrainingSample
        negative_sample_rate: Fraction of empty sentences to keep (0.0-1.0)
        min_tokens: Minimum token count to keep a sentence
        max_tokens: Maximum token count to keep a sentence

    Returns:
        List of sentence-level TrainingSample objects
    """
    sentence_samples = []
    text = sample.text

    # Split into sentences
    sentences = split_into_sentences(text)

    for sent_start, sent_end, sent_text in sentences:
        # Count tokens (rough approximation)
        token_count = len(sent_text.split())

        # Filter by token count
        if token_count < min_tokens or token_count > max_tokens:
            continue

        # Find entities within this sentence
        sentence_entities = []
        for entity in sample.entities:
            # Check if entity overlaps with this sentence
            if entity.start >= sent_start and entity.end <= sent_end:
                # Renumber position relative to sentence start
                new_entity = EntitySpan(
                    start=entity.start - sent_start,
                    end=entity.end - sent_start,
                    label=entity.label,
                    original_label=entity.original_label,
                    text=entity.text
                )
                sentence_entities.append(new_entity)

        # Decide whether to keep this sentence
        if sentence_entities:
            # Always keep sentences with entities (positive samples)
            sentence_samples.append(TrainingSample(
                text=sent_text,
                entities=sentence_entities,
                source_id=f"{sample.source_id}_s{len(sentence_samples)}",
                language=sample.language
            ))
        elif random.random() < negative_sample_rate:
            # Sample a fraction of empty sentences (negative samples)
            sentence_samples.append(TrainingSample(
                text=sent_text,
                entities=[],
                source_id=f"{sample.source_id}_neg{len(sentence_samples)}",
                language=sample.language
            ))

    return sentence_samples


def map_entity_type(ai4privacy_label: str) -> Optional[str]:
    """
    Map an ai4privacy label to a Hush Engine entity type.

    Args:
        ai4privacy_label: The label from the ai4privacy dataset

    Returns:
        Hush Engine entity type or None if unmapped
    """
    # Normalize label (uppercase, strip whitespace)
    normalized = ai4privacy_label.upper().strip()

    # Direct lookup
    if normalized in ENTITY_TYPE_MAPPING:
        return ENTITY_TYPE_MAPPING[normalized]

    # Try without numbers (GIVENNAME1 -> GIVENNAME)
    base_label = ''.join(c for c in normalized if not c.isdigit())
    if base_label in ENTITY_TYPE_MAPPING:
        return ENTITY_TYPE_MAPPING[base_label]

    return None


def parse_privacy_mask(
    source_text: str,
    privacy_mask: List[Dict[str, Any]],
    target_entity_type: Optional[str] = None
) -> List[EntitySpan]:
    """
    Parse the privacy_mask field into EntitySpan objects.

    Args:
        source_text: The original source text
        privacy_mask: List of {value, start, end, label} dictionaries
        target_entity_type: If specified, only return entities of this type

    Returns:
        List of EntitySpan objects
    """
    entities = []

    for mask in privacy_mask:
        original_label = mask.get("label", "")
        start = mask.get("start", 0)
        end = mask.get("end", 0)
        value = mask.get("value", "")

        # Map to Hush Engine type
        hush_type = map_entity_type(original_label)

        if hush_type is None:
            continue

        if target_entity_type and hush_type != target_entity_type:
            continue

        # Validate span
        if start < 0 or end > len(source_text) or start >= end:
            continue

        # Extract actual text from source
        actual_text = source_text[start:end]

        entities.append(EntitySpan(
            start=start,
            end=end,
            label=hush_type,
            original_label=original_label,
            text=actual_text
        ))

    # Sort by start position
    entities.sort(key=lambda e: e.start)

    return entities


def convert_to_bio_format(
    text: str,
    entities: List[EntitySpan]
) -> List[Tuple[str, int, int, str]]:
    """
    Convert text and entities to BIO-tagged tokens.

    Uses simple whitespace tokenization to match train_lgbm_ner.py.

    Args:
        text: The source text
        entities: List of EntitySpan objects

    Returns:
        List of (token, start, end, bio_tag) tuples
    """
    import re

    # Tokenize (matching feature_extractor.tokenize)
    tokens = []
    for match in re.finditer(r'\S+', text):
        token = match.group()
        start = match.start()
        end = match.end()
        tokens.append((token, start, end))

    # Assign BIO tags
    bio_tokens = []
    for token, t_start, t_end in tokens:
        # Find overlapping entity
        tag = "O"
        for entity in entities:
            # Check for overlap
            if t_start < entity.end and t_end > entity.start:
                # Determine if B (beginning) or I (inside)
                if t_start <= entity.start:
                    tag = f"B-{entity.label}"
                else:
                    tag = f"I-{entity.label}"
                break

        bio_tokens.append((token, t_start, t_end, tag))

    return bio_tokens


def convert_to_span_format(
    text: str,
    entities: List[EntitySpan]
) -> List[Tuple[int, int, str]]:
    """
    Convert entities to span format expected by train_lgbm_ner.py.

    Returns:
        List of (start, end, entity_type) tuples
    """
    return [(e.start, e.end, e.label) for e in entities]


class AI4PrivacyConverter:
    """
    Converts ai4privacy dataset to Hush Engine training format.
    """

    def __init__(
        self,
        target_entity_types: Optional[Set[str]] = None,
        language_filter: Optional[str] = None,
        max_samples: Optional[int] = None,
        split: str = "train"
    ):
        """
        Initialize the converter.

        Args:
            target_entity_types: Set of Hush entity types to extract (None = all)
            language_filter: Only include samples in this language (None = all)
            max_samples: Maximum number of samples to convert (None = all)
            split: Dataset split to use ("train" or "validation")
        """
        self.target_entity_types = target_entity_types or HUSH_ENTITY_TYPES
        self.language_filter = language_filter
        self.max_samples = max_samples
        self.split = split
        self.stats = ConversionStats()

    def load_dataset(self):
        """
        Load the ai4privacy dataset from Hugging Face.

        Returns:
            The dataset split
        """
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("Missing 'datasets' library. Install with: pip install datasets")
            sys.exit(1)

        logger.info("Loading ai4privacy/pii-masking-300k dataset...")
        dataset = load_dataset("ai4privacy/pii-masking-300k", split=self.split)
        logger.info(f"Loaded {len(dataset)} records from '{self.split}' split")

        return dataset

    def convert_record(self, record: Dict) -> Optional[TrainingSample]:
        """
        Convert a single dataset record to TrainingSample.

        Args:
            record: A record from the dataset

        Returns:
            TrainingSample or None if no relevant entities
        """
        self.stats.total_records += 1

        # Language filter
        language = record.get("language", "Unknown")
        self.stats.languages[language] += 1

        if self.language_filter and language != self.language_filter:
            self.stats.records_skipped += 1
            return None

        source_text = record.get("source_text", "")
        privacy_mask = record.get("privacy_mask", [])
        record_id = record.get("id", str(self.stats.total_records))

        if not source_text or not privacy_mask:
            self.stats.records_skipped += 1
            return None

        # Parse entities
        all_entities = []
        for mask in privacy_mask:
            original_label = mask.get("label", "")
            self.stats.original_labels[original_label] += 1

            # Map to Hush type
            hush_type = map_entity_type(original_label)

            if hush_type is None:
                self.stats.unmapped_labels[original_label] += 1
                continue

            if hush_type not in self.target_entity_types:
                continue

            start = mask.get("start", 0)
            end = mask.get("end", 0)

            # Validate span
            if start < 0 or end > len(source_text) or start >= end:
                continue

            actual_text = source_text[start:end]
            self.stats.entities_by_type[hush_type] += 1

            all_entities.append(EntitySpan(
                start=start,
                end=end,
                label=hush_type,
                original_label=original_label,
                text=actual_text
            ))

        if not all_entities:
            self.stats.records_skipped += 1
            return None

        self.stats.records_with_entities += 1

        return TrainingSample(
            text=source_text,
            entities=sorted(all_entities, key=lambda e: e.start),
            source_id=record_id,
            language=language
        )

    def convert_all(self) -> List[TrainingSample]:
        """
        Convert the entire dataset.

        Returns:
            List of TrainingSample objects
        """
        dataset = self.load_dataset()
        samples = []

        for i, record in enumerate(dataset):
            if self.max_samples and len(samples) >= self.max_samples:
                break

            sample = self.convert_record(record)
            if sample:
                samples.append(sample)

            if (i + 1) % 10000 == 0:
                logger.info(f"Processed {i + 1} records, {len(samples)} samples extracted")

        logger.info(f"Conversion complete: {len(samples)} samples from {self.stats.total_records} records")
        return samples

    def export_for_lgbm_training(
        self,
        samples: List[TrainingSample],
        output_dir: Path,
        entity_type: str
    ):
        """
        Export samples in format compatible with train_lgbm_ner.py.

        Creates a JSON file with (text, entities) pairs that can be loaded
        by a modified training script.

        Args:
            samples: List of TrainingSample objects
            output_dir: Output directory
            entity_type: The entity type being exported
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Filter samples for this entity type
        filtered_samples = []
        for sample in samples:
            relevant_entities = [
                (e.start, e.end, e.label)
                for e in sample.entities
                if e.label == entity_type
            ]
            if relevant_entities:
                filtered_samples.append({
                    "text": sample.text,
                    "entities": relevant_entities,
                    "source_id": sample.source_id,
                    "language": sample.language
                })

        # Export as JSON
        output_file = output_dir / f"{entity_type.lower()}_training_data.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "entity_type": entity_type,
                "source": "ai4privacy/pii-masking-300k",
                "sample_count": len(filtered_samples),
                "samples": filtered_samples
            }, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported {len(filtered_samples)} samples for {entity_type} to {output_file}")
        return output_file

    def export_bio_format(
        self,
        samples: List[TrainingSample],
        output_dir: Path,
        entity_type: Optional[str] = None
    ):
        """
        Export samples in CoNLL-style BIO format.

        Format:
            token\tstart\tend\tBIO_TAG

        Args:
            samples: List of TrainingSample objects
            output_dir: Output directory
            entity_type: If specified, only export this entity type
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        suffix = f"_{entity_type.lower()}" if entity_type else "_all"
        output_file = output_dir / f"bio_format{suffix}.tsv"

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("token\tstart\tend\ttag\tsample_id\n")

            for sample in samples:
                # Filter entities if entity_type specified
                if entity_type:
                    entities = [e for e in sample.entities if e.label == entity_type]
                else:
                    entities = sample.entities

                bio_tokens = convert_to_bio_format(sample.text, entities)

                for token, start, end, tag in bio_tokens:
                    # Escape tabs and newlines in token
                    clean_token = token.replace('\t', ' ').replace('\n', ' ')
                    f.write(f"{clean_token}\t{start}\t{end}\t{tag}\t{sample.source_id}\n")

                # Blank line between samples
                f.write("\n")

        logger.info(f"Exported BIO format to {output_file}")
        return output_file

    def print_stats(self):
        """Print conversion statistics."""
        print("\n" + "=" * 60)
        print("CONVERSION STATISTICS")
        print("=" * 60)

        print(f"\nRecords processed: {self.stats.total_records}")
        print(f"Records with entities: {self.stats.records_with_entities}")
        print(f"Records skipped: {self.stats.records_skipped}")

        print("\nEntities by Hush Engine type:")
        for entity_type, count in sorted(self.stats.entities_by_type.items(), key=lambda x: -x[1]):
            print(f"  {entity_type}: {count:,}")

        print("\nOriginal ai4privacy labels (top 20):")
        for label, count in self.stats.original_labels.most_common(20):
            mapped = map_entity_type(label) or "UNMAPPED"
            print(f"  {label} -> {mapped}: {count:,}")

        if self.stats.unmapped_labels:
            print("\nUnmapped labels (excluded):")
            for label, count in sorted(self.stats.unmapped_labels.items(), key=lambda x: -x[1])[:10]:
                print(f"  {label}: {count:,}")

        print("\nLanguages:")
        for lang, count in sorted(self.stats.languages.items(), key=lambda x: -x[1]):
            print(f"  {lang}: {count:,}")

        if self.stats.sentences_extracted > 0:
            print(f"\nSentence extraction:")
            print(f"  Total sentences: {self.stats.sentences_extracted:,}")
            print(f"  With entities: {self.stats.sentences_with_entities:,}")
            print(f"  Negative samples: {self.stats.sentences_negative:,}")
            if self.stats.sentences_extracted > 0:
                pos_ratio = self.stats.sentences_with_entities / self.stats.sentences_extracted * 100
                print(f"  Positive ratio: {pos_ratio:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ai4privacy dataset to Hush Engine training format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert PERSON entities only
    python tools/convert_ai4privacy.py --entity-type PERSON --output tests/data/ai4privacy

    # Convert all supported entity types
    python tools/convert_ai4privacy.py --all --output tests/data/ai4privacy

    # Quick test with 1000 samples
    python tools/convert_ai4privacy.py --all --max-samples 1000 --output tests/data/ai4privacy

    # English only
    python tools/convert_ai4privacy.py --all --language English --output tests/data/ai4privacy

    # Export in BIO format
    python tools/convert_ai4privacy.py --all --format bio --output tests/data/ai4privacy
        """
    )

    parser.add_argument(
        "--entity-type",
        choices=list(HUSH_ENTITY_TYPES),
        help="Specific entity type to convert"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Convert all supported entity types"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for converted data"
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Filter by language (e.g., 'English', 'French')"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to convert (for testing)"
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation"],
        default="train",
        help="Dataset split to use (default: train)"
    )
    parser.add_argument(
        "--format",
        choices=["json", "bio", "both"],
        default="json",
        help="Output format (default: json)"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only print statistics, don't export"
    )
    parser.add_argument(
        "--extract-sentences",
        action="store_true",
        help="Extract sentence-level samples to fix class imbalance (recommended for LightGBM training)"
    )
    parser.add_argument(
        "--negative-rate",
        type=float,
        default=0.1,
        help="Fraction of empty sentences to keep as negatives (default: 0.1)"
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=3,
        help="Minimum tokens per sentence (default: 3)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum tokens per sentence (default: 50)"
    )

    args = parser.parse_args()

    if not args.entity_type and not args.all:
        parser.error("Must specify --entity-type or --all")

    # Determine which entity types to process
    if args.all:
        target_types = HUSH_ENTITY_TYPES
    else:
        target_types = {args.entity_type}

    logger.info(f"Target entity types: {target_types}")

    # Create converter
    converter = AI4PrivacyConverter(
        target_entity_types=target_types,
        language_filter=args.language,
        max_samples=args.max_samples,
        split=args.split
    )

    # Convert dataset
    samples = converter.convert_all()

    # Extract sentence-level samples if requested
    if args.extract_sentences:
        logger.info("\nExtracting sentence-level samples...")
        sentence_samples = []
        for sample in samples:
            extracted = extract_sentence_samples(
                sample,
                negative_sample_rate=args.negative_rate,
                min_tokens=args.min_tokens,
                max_tokens=args.max_tokens
            )
            sentence_samples.extend(extracted)
            converter.stats.sentences_extracted += len(extracted)
            converter.stats.sentences_with_entities += sum(1 for s in extracted if s.entities)
            converter.stats.sentences_negative += sum(1 for s in extracted if not s.entities)

        logger.info(f"Extracted {len(sentence_samples)} sentences from {len(samples)} documents")
        logger.info(f"  - With entities: {converter.stats.sentences_with_entities}")
        logger.info(f"  - Negatives: {converter.stats.sentences_negative}")

        # Calculate class balance
        avg_doc_len = sum(len(s.text.split()) for s in samples) / max(len(samples), 1)
        avg_sent_len = sum(len(s.text.split()) for s in sentence_samples) / max(len(sentence_samples), 1)
        logger.info(f"  - Avg doc tokens: {avg_doc_len:.1f}")
        logger.info(f"  - Avg sentence tokens: {avg_sent_len:.1f}")

        samples = sentence_samples

    converter.print_stats()

    if args.stats_only:
        return

    # Export
    if args.format in ("json", "both"):
        if args.all:
            # Export each entity type separately
            for entity_type in target_types:
                converter.export_for_lgbm_training(samples, args.output, entity_type)
        else:
            converter.export_for_lgbm_training(samples, args.output, args.entity_type)

    if args.format in ("bio", "both"):
        if args.entity_type:
            converter.export_bio_format(samples, args.output, args.entity_type)
        else:
            converter.export_bio_format(samples, args.output)

    logger.info(f"\nOutput saved to: {args.output}")
    logger.info("\nNext steps:")
    logger.info("1. Review the exported data for quality")
    logger.info("2. Update train_lgbm_ner.py to load external data")
    logger.info("3. Run training with the new data")


if __name__ == "__main__":
    main()
