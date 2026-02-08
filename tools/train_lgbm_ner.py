#!/usr/bin/env python3
"""
Training Script for LightGBM/SVM NER Models

Generates synthetic training data using Faker and public datasets,
then trains LightGBM or SVM classifiers for each entity type.

Supports loading from ai4privacy/pii-masking-300k dataset with noise augmentation
based on PMC paper methodology (91.1% F1).

SVM classifier option based on Makhija 2020 showing 93%+ accuracy with simpler
model architecture and faster inference.

PRIVACY NOTICE:
- Training data is 100% synthetic (Faker) or from public datasets (ai4privacy)
- NO real user data is ever used
- NO user feedback files are read for training
- Models are trained locally, nothing is uploaded

Usage:
    # Train with synthetic data (default LightGBM)
    python tools/train_lgbm_ner.py --entity-type PERSON --samples 10000
    python tools/train_lgbm_ner.py --all --samples 5000

    # Train with SVM classifier instead of LightGBM
    python tools/train_lgbm_ner.py --entity-type PERSON --classifier svm --samples 10000
    python tools/train_lgbm_ner.py --all --classifier svm --samples 5000

    # Train with ai4privacy dataset + noise augmentation (PMC paper approach)
    python tools/train_lgbm_ner.py --entity-type PERSON --ai4privacy --augment
    python tools/train_lgbm_ner.py --all --ai4privacy --augment --samples 10000
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

# ai4privacy converted data directory (sentence-level for better class balance)
AI4PRIVACY_DIR = Path(__file__).parent.parent / "tests" / "data" / "ai4privacy" / "sentences"

# Supported entity types
ENTITY_TYPES = ["PERSON", "LOCATION", "ORGANIZATION", "DATE_TIME", "ADDRESS"]

# Entity type mapping from ai4privacy to our types
AI4PRIVACY_MAPPING = {
    "PERSON": "person_training_data.json",
    "ADDRESS": "address_training_data.json",
    "LOCATION": "location_training_data.json",
    "DATE_TIME": "date_time_training_data.json",
    "PHONE_NUMBER": "phone_number_training_data.json",
    "NATIONAL_ID": "ssn_training_data.json",
}


def load_ai4privacy_data(
    entity_type: str,
    max_samples: int = None,
    apply_augmentation: bool = True
) -> List[Tuple[str, List[Tuple[int, int, str]]]]:
    """
    Load training data from ai4privacy converted files.

    Args:
        entity_type: Entity type to load (PERSON, ADDRESS, etc.)
        max_samples: Maximum samples to load (None for all)
        apply_augmentation: Whether to apply noise augmentation

    Returns:
        List of (text, [(start, end, entity_type), ...]) tuples
    """
    if entity_type not in AI4PRIVACY_MAPPING:
        logger.warning(f"No ai4privacy data for {entity_type}, using synthetic data")
        return []

    data_file = AI4PRIVACY_DIR / AI4PRIVACY_MAPPING[entity_type]
    if not data_file.exists():
        logger.warning(f"ai4privacy data file not found: {data_file}")
        return []

    logger.info(f"Loading ai4privacy data from {data_file}")

    with open(data_file, 'r') as f:
        data = json.load(f)

    samples = data.get("samples", [])
    if max_samples:
        samples = samples[:max_samples]

    # Convert to expected format
    result = []
    for sample in samples:
        text = sample.get("text", "")
        entities = [
            (e[0], e[1], e[2]) for e in sample.get("entities", [])
        ]
        result.append((text, entities))

    logger.info(f"Loaded {len(result)} samples from ai4privacy")

    # Apply noise augmentation if requested
    if apply_augmentation and result:
        try:
            from tools.training_augmentation import NoiseInjector, AugmentationConfig

            config = AugmentationConfig(
                punctuation_removal_prob=0.30,  # PMC paper: 30%
                case_variation_prob=0.10,
                spacing_error_prob=0.10,
                char_substitution_prob=0.05,
            )
            injector = NoiseInjector(config)

            # Convert to expected format for NoiseInjector
            injector_samples = [
                {"text": text, "entities": [{"start": e[0], "end": e[1], "label": e[2]} for e in entities]}
                for text, entities in result
            ]

            # Augment with multiplier=2 (original + 2 augmented = 3x data)
            augmented = injector.augment_batch(injector_samples, multiplier=2, include_original=True)

            # Convert back
            result = []
            for sample in augmented:
                text = sample["text"]
                entities = [(e["start"], e["end"], e["label"]) for e in sample["entities"]]
                result.append((text, entities))

            logger.info(f"Augmented to {len(result)} samples (3x with noise injection)")

        except ImportError as e:
            logger.warning(f"Could not load augmentation module: {e}")
        except Exception as e:
            logger.warning(f"Augmentation failed: {e}")

    return result


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

        # Company suffixes for synthetic names
        org_suffixes = [
            "Inc", "Inc.", "Corp", "Corp.", "Corporation", "LLC", "L.L.C.",
            "Ltd", "Ltd.", "Limited", "PLC", "Company", "Co", "Co.",
            "Group", "Holdings", "Partners", "Associates", "Enterprises",
            "Industries", "International", "Global", "Solutions", "Services",
            "Technologies", "Systems", "Networks", "Capital", "Investments",
        ]

        # Diverse templates covering different contexts
        templates = [
            # Business/Finance
            "{org} announced their quarterly earnings.",
            "Shares of {org} rose 5% today.",
            "{org} stock fell after the announcement.",
            "The merger between {org} and {org2} was approved.",
            "{org} acquired {org2} for $2 billion.",
            "Investors are bullish on {org}.",
            "{org} reported record profits this quarter.",
            "The {org} IPO raised $500 million.",

            # Employment
            "I work at {org}.",
            "{org} is hiring new employees.",
            "She was promoted to VP at {org}.",
            "He left {org} to join {org2}.",
            "The {org} employees went on strike.",
            "According to {org} HR department...",

            # News/Media
            "According to {org} spokesperson...",
            "{org} denied the allegations.",
            "A {org} representative confirmed the news.",
            "{org} released a statement today.",
            "Sources at {org} say...",

            # Location/Facilities
            "The {org} headquarters is in New York.",
            "{org} opened a new office in London.",
            "The {org} factory produces 1000 units daily.",
            "Visit the {org} store on Main Street.",
            "{org} has offices worldwide.",

            # Products/Services
            "{org} launched a new product line.",
            "The {org} app has 10 million downloads.",
            "{org} services are now available online.",
            "Try {org} premium subscription.",

            # Legal/Regulatory
            "{org} was fined by regulators.",
            "The lawsuit against {org} was dismissed.",
            "{org} settled the case for $10 million.",
            "{org} must comply with new regulations.",

            # Partnerships
            "{org} partnered with {org2} on the project.",
            "{org} and {org2} signed a joint venture agreement.",
            "The {org} and {org2} collaboration produced results.",
        ]

        # Load real company names from database
        try:
            from hush_engine.data.companies_database import SP500_COMPANIES, MAJOR_INTERNATIONAL_COMPANIES
            real_companies = list(SP500_COMPANIES | MAJOR_INTERNATIONAL_COMPANIES)
            # Capitalize properly
            real_companies = [c.title() for c in real_companies]
        except ImportError:
            real_companies = []

        for i in range(n):
            template = random.choice(templates)

            # Mix of real companies (60%) and synthetic (40%)
            if real_companies and random.random() < 0.6:
                org = random.choice(real_companies)
                # Sometimes add a suffix to real company names
                if random.random() < 0.3:
                    org = f"{org} {random.choice(['Inc', 'Corp', 'LLC'])}"
            else:
                # Synthetic company name
                org_base = self.fake.company().split()[0]
                if random.random() < 0.7:  # 70% have suffix
                    suffix = random.choice(org_suffixes)
                    org = f"{org_base} {suffix}"
                else:
                    org = org_base.title()

            if "{org2}" in template:
                # Second organization
                if real_companies and random.random() < 0.6:
                    org2 = random.choice(real_companies)
                else:
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

    def generate_address_samples(self, n: int) -> List[Tuple[str, List[Tuple[int, int, str]]]]:
        """Generate sentences containing street addresses."""
        samples = []

        # Street types for address generation
        street_types = [
            "Street", "St", "Avenue", "Ave", "Road", "Rd", "Drive", "Dr",
            "Boulevard", "Blvd", "Lane", "Ln", "Way", "Court", "Ct", "Circle",
            "Place", "Pl", "Terrace", "Parkway", "Pkwy", "Highway", "Hwy",
        ]

        # Street names
        street_names = [
            "Main", "Oak", "Maple", "Cedar", "Park", "Lake", "Hill", "River",
            "Washington", "Lincoln", "Jefferson", "Franklin", "Madison",
            "First", "Second", "Third", "Fourth", "Fifth", "Elm", "Pine",
            "Walnut", "Cherry", "Sunset", "Highland", "Valley", "Spring",
        ]

        templates = [
            # Simple address
            "{address}",
            # With city
            "{address}, {city}",
            # With city and state
            "{address}, {city}, {state}",
            # With ZIP
            "{address}, {city}, {state} {zip}",
            # Contextual
            "Ship to: {address}",
            "Address: {address}",
            "Located at {address}",
            "Our office is at {address}",
            "Please send to {address}, {city}",
            "Meet me at {address}",
            "The property at {address} is for sale.",
            "Billing: {address}, {city}, {state} {zip}",
        ]

        states = ["CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI", "WA", "AZ", "MA", "CO"]

        for _ in range(n):
            template = random.choice(templates)

            # Generate address components
            street_num = str(random.randint(100, 9999))
            street_name = random.choice(street_names)
            street_type = random.choice(street_types)
            address = f"{street_num} {street_name} {street_type}"

            # Sometimes add apartment/suite
            if random.random() < 0.2:
                apt_num = random.randint(1, 999)
                apt_type = random.choice(["Apt", "Suite", "Unit", "#"])
                address = f"{address}, {apt_type} {apt_num}"

            city = random.choice(self.cities).title() if self.cities else self.fake.city()
            state = random.choice(states)
            zip_code = str(random.randint(10000, 99999))

            sentence = template.format(
                address=address,
                city=city,
                state=state,
                zip=zip_code
            )

            # Find the address span
            pos = sentence.find(address)
            if pos >= 0:
                # Determine end position based on template
                if "{city}" in template or "{state}" in template or "{zip}" in template:
                    # Full address with city/state/zip
                    end_markers = [zip_code, state, city]
                    end_pos = pos + len(address)
                    for marker in end_markers:
                        marker_pos = sentence.find(marker, pos)
                        if marker_pos >= 0:
                            end_pos = marker_pos + len(marker)
                    entities = [(pos, end_pos, "ADDRESS")]
                else:
                    entities = [(pos, pos + len(address), "ADDRESS")]
            else:
                entities = []

            samples.append((sentence, entities))

        return samples

    def generate_address_false_positives(self, n: int) -> List[Tuple[str, List[Tuple[int, int, str]]]]:
        """
        Generate sentences with common ADDRESS false positive patterns.

        These are phrases that pattern-based detectors often incorrectly
        classify as addresses, used as hard negatives for training.
        """
        samples = []

        # Form labels (common false positives)
        form_patterns = [
            "Street Address",
            "Billing Address",
            "Shipping Address",
            "Mailing Address",
            "Home Address",
            "Work Address",
            "Enter your address here",
            "Address Line 1",
            "Address Line 2",
            "City, State, ZIP",
            "Please provide your address",
            "Address Required",
            "No address on file",
            "Update Address",
        ]

        # Verb phrases with location words
        verb_patterns = [
            "The road ahead is clear.",
            "We went to the store.",
            "Based in the downtown area.",
            "Located in a prime area.",
            "Travel to different places.",
            "Move to a new location.",
            "Headed to the meeting.",
            "Send to the recipient.",
            "Ship to customer.",
            "Found in the document.",
        ]

        # Generic location descriptions
        generic_patterns = [
            "In the building next door.",
            "At the corner of the block.",
            "On the main floor.",
            "Near the entrance.",
            "By the parking lot.",
            "Across the street.",
            "Down the road.",
            "Around the corner.",
            "Behind the building.",
            "In the lobby.",
        ]

        # OCR artifact patterns
        ocr_patterns = [
            "User ID or Address",
            "Tax ID at location",
            "Post ID 12345",
            "Email at address below",
            "ID # Street",
            "The way forward",
            "On the path to success",
        ]

        all_patterns = form_patterns + verb_patterns + generic_patterns + ocr_patterns

        for _ in range(n):
            sentence = random.choice(all_patterns)
            # Add some variation
            if random.random() < 0.3:
                sentence = f"{sentence} {random.choice(all_patterns)}"
            samples.append((sentence, []))

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

    def generate_company_false_positives(self, n: int) -> List[Tuple[str, List[Tuple[int, int, str]]]]:
        """
        Generate sentences with common COMPANY false positive patterns.

        These are phrases that pattern-based detectors often incorrectly
        classify as company names, used as hard negatives for training.
        """
        samples = []

        # UI/Navigation patterns (common false positives)
        ui_patterns = [
            "Click on Expenses & Bills to view transactions.",
            "Navigate to Settings & Preferences for options.",
            "Go to Reports & Insights for analytics.",
            "Open Help & Support for assistance.",
            "Visit Sales & Get Paid to manage invoices.",
            "Check Billing & Payments for your subscription.",
            "Use Import & Export to transfer data.",
            "Access Accounts & Settings to configure.",
            "The Terms & Conditions apply to all users.",
            "Review Privacy & Security settings.",
        ]

        # Accounting/Business phrases (common false positives)
        accounting_patterns = [
            "Check the Profit & Loss statement.",
            "Review Assets & Liabilities in the balance sheet.",
            "The Research & Development budget was approved.",
            "The Mergers & Acquisitions team reviewed the deal.",
            "Submit to Human Resources for processing.",
            "Contact Customer Service for support.",
            "The Quality Assurance team approved the release.",
            "Supply Chain management optimized operations.",
            "Business Development identified new opportunities.",
            "Health & Safety protocols must be followed.",
        ]

        # Hyphenated adjective patterns (common false positives)
        hyphenated_patterns = [
            "This is a well-known procedure.",
            "Apply the cross-verified method.",
            "Use high-value transactions only.",
            "The self-employed individual filed taxes.",
            "Follow the long-term strategy.",
            "Check real-time updates regularly.",
            "Submit year-end reports by January.",
            "The tax-exempt status was confirmed.",
            "Review non-compliance issues carefully.",
            "This is a full-time position.",
        ]

        # Generic phrases that look like companies but aren't
        generic_patterns = [
            "The International committee met today.",
            "Our Global team reviewed the proposal.",
            "The Solutions department was restructured.",
            "Services were temporarily unavailable.",
            "The Group decided to proceed.",
            "Holdings were transferred yesterday.",
            "Management approved the budget.",
            "Consulting reviewed the project.",
            "Technologies division expanded.",
            "Partners agreed to the terms.",
        ]

        # Time/measurement patterns (numeric + word combos)
        time_patterns = [
            "Please allow 7 business days for processing.",
            "The project took 36 months to complete.",
            "We need 4 hours to finish.",
            "Delivery within 14 business days.",
            "The lease runs for 5 years.",
            "Complete within 30 days of receipt.",
            "Allow 2 weeks for shipping.",
            "Project duration: 6 months.",
            "Response time: 24 hours.",
            "Valid for 12 months from issue.",
        ]

        # Collect all patterns
        all_patterns = (
            ui_patterns + accounting_patterns + hyphenated_patterns +
            generic_patterns + time_patterns
        )

        # Generate samples - these are all NEGATIVE (no entities)
        for i in range(n):
            sentence = random.choice(all_patterns)
            samples.append((sentence, []))

        return samples


def create_training_data(
    entity_type: str,
    n_positive: int,
    n_negative: int,
    generator: SyntheticDataGenerator,
    use_ai4privacy: bool = False,
    apply_augmentation: bool = True
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
        use_ai4privacy: Whether to load from ai4privacy dataset
        apply_augmentation: Whether to apply noise augmentation (PMC paper)

    Returns:
        (X, y) tuple of features and labels
    """
    logger.info(f"Generating {n_positive} positive and {n_negative} negative samples for {entity_type}")

    # Generate positive samples for this entity type
    positive_samples = []

    # Try ai4privacy data first if requested
    if use_ai4privacy:
        ai4privacy_samples = load_ai4privacy_data(
            entity_type, max_samples=n_positive, apply_augmentation=apply_augmentation
        )
        if ai4privacy_samples:
            positive_samples = ai4privacy_samples
            logger.info(f"Using {len(positive_samples)} samples from ai4privacy")

    # Fall back to synthetic data if needed
    if not positive_samples:
        if entity_type == "PERSON":
            positive_samples = generator.generate_person_samples(n_positive)
        elif entity_type == "LOCATION":
            positive_samples = generator.generate_location_samples(n_positive)
        elif entity_type == "ORGANIZATION":
            positive_samples = generator.generate_organization_samples(n_positive)
        elif entity_type == "DATE_TIME":
            positive_samples = generator.generate_datetime_samples(n_positive)
        elif entity_type == "ADDRESS":
            positive_samples = generator.generate_address_samples(n_positive)
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

    if entity_type != "ADDRESS":
        negative_samples.extend(generator.generate_address_samples(hard_negatives_per_type))

    # For ORGANIZATION, add company-specific false positive examples
    # These help the model distinguish real company names from common phrases
    if entity_type == "ORGANIZATION":
        company_fp_samples = generator.generate_company_false_positives(n_negative // 4)
        negative_samples.extend(company_fp_samples)
        logger.info(f"Added {len(company_fp_samples)} company-specific hard negatives")

    # For ADDRESS, add address-specific false positive examples
    if entity_type == "ADDRESS":
        address_fp_samples = generator.generate_address_false_positives(n_negative // 4)
        negative_samples.extend(address_fp_samples)
        logger.info(f"Added {len(address_fp_samples)} address-specific hard negatives")

    # Convert to features with negative undersampling
    # To fix class imbalance, we keep all positive tokens but only sample
    # a fraction of negative tokens per sentence
    all_features = []
    all_labels = []
    neg_sample_rate = 0.15  # Keep 15% of negative tokens per sentence

    for sentence, entities in positive_samples + negative_samples:
        tokens = tokenize(sentence)
        features_list = extract_features_with_context(sentence)
        feature_dicts = features_to_matrix(features_list)

        # First pass: identify positive tokens
        positive_indices = set()
        for i, (token, start, end) in enumerate(tokens):
            for ent_start, ent_end, ent_type in entities:
                if ent_type == entity_type and start < ent_end and end > ent_start:
                    positive_indices.add(i)
                    break

        # Second pass: add positive tokens + sampled negatives + context tokens
        for i, (token, start, end) in enumerate(tokens):
            is_entity = i in positive_indices

            if is_entity:
                # Always keep positive tokens
                all_features.append(feature_dicts[i])
                all_labels.append(1)
            elif i > 0 and (i - 1) in positive_indices:
                # Keep tokens immediately after entities (right context)
                all_features.append(feature_dicts[i])
                all_labels.append(0)
            elif i < len(tokens) - 1 and (i + 1) in positive_indices:
                # Keep tokens immediately before entities (left context)
                all_features.append(feature_dicts[i])
                all_labels.append(0)
            elif random.random() < neg_sample_rate:
                # Undersample other negative tokens
                all_features.append(feature_dicts[i])
                all_labels.append(0)

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
        'num_leaves': 31,           # More capacity for learning
        'learning_rate': 0.05,      # Faster learning with better class balance
        'feature_fraction': 0.8,    # Less regularization
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 10,     # Allow smaller leaf nodes
        'lambda_l1': 0.01,          # Light L1 regularization
        'lambda_l2': 0.01,          # Light L2 regularization
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


def train_svm_classifier(
    X: np.ndarray,
    y: np.ndarray,
    entity_type: str,
) -> Tuple:
    """
    Train an SVM classifier with RBF kernel.

    Based on Makhija 2020 showing SVM achieves 93%+ accuracy for NER,
    with simpler model architecture and faster inference than LightGBM.

    Args:
        X: Feature matrix
        y: Labels
        entity_type: Entity type (for logging)

    Returns:
        Tuple of (trained SVC model, fitted StandardScaler)
    """
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler

    logger.info(f"Training SVM classifier for {entity_type}...")

    # Split for validation
    n_train = int(0.8 * len(y))
    indices = np.random.permutation(len(y))
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Scale features for SVM (important for RBF kernel)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Calculate class weights for imbalanced data
    n_pos = sum(y_train)
    n_neg = len(y_train) - n_pos
    class_weight = {0: 1.0, 1: n_neg / n_pos if n_pos > 0 else 1.0}

    logger.info(f"Class weight for positive class: {class_weight[1]:.2f}")

    # Train SVM with RBF kernel
    model = SVC(
        kernel='rbf',
        probability=True,  # Enable probability estimates
        class_weight=class_weight,
        random_state=42,
        C=1.0,  # Regularization parameter
        gamma='scale',  # 1 / (n_features * X.var())
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate with threshold tuning
    val_probs = model.predict_proba(X_val_scaled)[:, 1]

    # Find optimal threshold using F1 score
    best_f1 = 0
    best_threshold = 0.5
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        val_preds_binary = (val_probs >= threshold).astype(int)
        tp = sum((val_preds_binary == 1) & (y_val == 1))
        fp = sum((val_preds_binary == 1) & (y_val == 0))
        fn = sum((val_preds_binary == 0) & (y_val == 1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # Final evaluation at best threshold
    val_preds_binary = (val_probs >= best_threshold).astype(int)
    tp = sum((val_preds_binary == 1) & (y_val == 1))
    fp = sum((val_preds_binary == 1) & (y_val == 0))
    fn = sum((val_preds_binary == 0) & (y_val == 1))
    tn = sum((val_preds_binary == 0) & (y_val == 0))

    accuracy = (tp + tn) / len(y_val)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    logger.info(f"Validation @ threshold={best_threshold} - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {best_f1:.3f}")

    return model, scaler


def save_model(model, entity_type: str, output_dir: Path, classifier_type: str = "lgbm", scaler=None):
    """Save trained model to disk.

    Args:
        model: Trained model (LightGBM Booster or sklearn SVC)
        entity_type: Entity type name
        output_dir: Directory to save model files
        classifier_type: "lgbm" or "svm"
        scaler: StandardScaler for SVM (required when classifier_type="svm")
    """
    import joblib

    output_dir.mkdir(parents=True, exist_ok=True)

    if classifier_type == "svm":
        # Save SVM model and scaler using joblib
        model_path = output_dir / f"{entity_type.lower()}_svm.pkl"
        scaler_path = output_dir / f"{entity_type.lower()}_scaler.pkl"

        joblib.dump(model, str(model_path))
        logger.info(f"Saved SVM model to {model_path}")

        if scaler is not None:
            joblib.dump(scaler, str(scaler_path))
            logger.info(f"Saved scaler to {scaler_path}")

        # Save metadata
        metadata = {
            "entity_type": entity_type,
            "classifier_type": "svm",
            "trained_at": datetime.now().isoformat(),
            "feature_names": FEATURE_NAMES,
            "kernel": model.kernel,
            "n_support_vectors": int(sum(model.n_support_)),
            "privacy_note": "Trained on synthetic data only. No real PII used.",
        }
    else:
        # Save LightGBM model
        model_path = output_dir / f"{entity_type.lower()}_classifier.txt"
        model.save_model(str(model_path))
        logger.info(f"Saved LightGBM model to {model_path}")

        # Save metadata
        metadata = {
            "entity_type": entity_type,
            "classifier_type": "lgbm",
            "trained_at": datetime.now().isoformat(),
            "feature_names": FEATURE_NAMES,
            "num_trees": model.num_trees(),
            "privacy_note": "Trained on synthetic data only. No real PII used.",
        }

    metadata_path = output_dir / f"{entity_type.lower()}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM/SVM NER classifiers")
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
        "--classifier",
        choices=["lgbm", "svm"],
        default="lgbm",
        help="Classifier type: lgbm (LightGBM) or svm (SVM with RBF kernel). Default: lgbm"
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
    parser.add_argument(
        "--ai4privacy",
        action="store_true",
        help="Use ai4privacy/pii-masking-300k dataset for training"
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply noise augmentation (PMC paper: 30% punctuation removal, etc.)"
    )
    parser.add_argument(
        "--smote",
        action="store_true",
        help="Apply SMOTE oversampling to balance classes 1:1 (requires imbalanced-learn)"
    )

    args = parser.parse_args()

    if not args.entity_type and not args.all:
        parser.error("Must specify --entity-type or --all")

    entity_types = ENTITY_TYPES if args.all else [args.entity_type]

    logger.info("=" * 60)
    if args.classifier == "svm":
        logger.info("SVM NER Training (RBF kernel)")
    else:
        logger.info("LightGBM NER Training")
    logger.info(f"CLASSIFIER: {args.classifier.upper()}")
    if args.ai4privacy:
        logger.info("DATA SOURCE: ai4privacy/pii-masking-300k dataset")
    else:
        logger.info("DATA SOURCE: Synthetic (Faker)")
    if args.augment:
        logger.info("AUGMENTATION: Enabled (PMC paper: 30% noise injection)")
    if args.smote:
        logger.info("SMOTE: Enabled (class balancing via oversampling)")
    logger.info("PRIVACY: Using synthetic/public data only. No real PII.")
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
            generator=generator,
            use_ai4privacy=args.ai4privacy,
            apply_augmentation=args.augment
        )

        # Apply SMOTE if requested (after feature extraction, before training)
        if args.smote:
            try:
                from imblearn.over_sampling import SMOTE
                n_pos_before = sum(y)
                n_neg_before = len(y) - n_pos_before
                logger.info(f"Before SMOTE: {n_pos_before} positive, {n_neg_before} negative")

                smote = SMOTE(random_state=args.seed)
                X, y = smote.fit_resample(X, y)

                n_pos_after = sum(y)
                n_neg_after = len(y) - n_pos_after
                logger.info(f"After SMOTE: {n_pos_after} positive, {n_neg_after} negative (balanced 1:1)")
            except ImportError:
                logger.warning("imbalanced-learn not installed. Skipping SMOTE.")
                logger.warning("Install with: pip install imbalanced-learn")

        # Train model based on classifier type
        if args.classifier == "svm":
            model, scaler = train_svm_classifier(X, y, entity_type)
            save_model(model, entity_type, args.output_dir, classifier_type="svm", scaler=scaler)
        else:
            model = train_classifier(X, y, entity_type)
            save_model(model, entity_type, args.output_dir, classifier_type="lgbm")

    logger.info("\nTraining complete!")
    logger.info(f"Models saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
