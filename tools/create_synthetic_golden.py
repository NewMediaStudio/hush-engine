#!/usr/bin/env python3
"""
Create a synthetic golden test set using Faker.

This generates clean, well-formatted PII data for accurate benchmarking.
Unlike the arrow/parquet datasets which have OCR noise and non-standard formats,
this produces data that matches real-world PII patterns.

Usage:
    python tools/create_synthetic_golden.py --samples 500 --output tests/data/synthetic_golden.json
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

try:
    from faker import Faker
except ImportError:
    print("Error: faker not installed. Run: pip install faker")
    sys.exit(1)


# Entity templates - each creates text with embedded PII
TEMPLATES = [
    # Contact info
    {
        "template": "Please contact {name} at {email} or call {phone}.",
        "entities": ["PERSON", "EMAIL", "PHONE"]
    },
    {
        "template": "My name is {name} and I live at {address}.",
        "entities": ["PERSON", "ADDRESS"]
    },
    {
        "template": "Send the invoice to {name}, {email}, by {date}.",
        "entities": ["PERSON", "EMAIL", "DATE_TIME"]
    },
    # Financial
    {
        "template": "Customer {name} paid with card {credit_card} on {date}.",
        "entities": ["PERSON", "CREDIT_CARD", "DATE_TIME"]
    },
    {
        "template": "Account holder: {name}, SSN: {ssn}, DOB: {date}.",
        "entities": ["PERSON", "NATIONAL_ID", "DATE_TIME"]
    },
    # Business
    {
        "template": "Reach out to {name} at {company} via {email}.",
        "entities": ["PERSON", "COMPANY", "EMAIL"]
    },
    {
        "template": "{company} headquarters is located at {address}.",
        "entities": ["COMPANY", "ADDRESS"]
    },
    # Medical/HR
    {
        "template": "Patient {name}, age {age}, scheduled for {date}.",
        "entities": ["PERSON", "AGE", "DATE_TIME"]
    },
    {
        "template": "Employee {name} (SSN: {ssn}) started on {date}.",
        "entities": ["PERSON", "NATIONAL_ID", "DATE_TIME"]
    },
    # Multi-entity
    {
        "template": "From: {name} <{email}>\nPhone: {phone}\nAddress: {address}",
        "entities": ["PERSON", "EMAIL", "PHONE", "ADDRESS"]
    },
    {
        "template": "Dear {name},\n\nYour order #{order_id} will ship to {address} on {date}.",
        "entities": ["PERSON", "ADDRESS", "DATE_TIME"]
    },
    # Simple single-entity
    {
        "template": "Call me at {phone} after {date}.",
        "entities": ["PHONE", "DATE_TIME"]
    },
    {
        "template": "Email: {email}",
        "entities": ["EMAIL"]
    },
    {
        "template": "Ship to: {address}",
        "entities": ["ADDRESS"]
    },
    {
        "template": "Cardholder: {name}, Card: {credit_card}",
        "entities": ["PERSON", "CREDIT_CARD"]
    },
]


class SyntheticGoldenGenerator:
    """Generate synthetic PII samples using Faker."""

    def __init__(self, seed: int = 42):
        self.fake = Faker(['en_US'])
        Faker.seed(seed)
        random.seed(seed)

    def generate_entity(self, entity_type: str) -> str:
        """Generate a single entity value."""
        generators = {
            "PERSON": lambda: self.fake.name(),
            "EMAIL": lambda: self.fake.email(),
            "PHONE": lambda: self.fake.phone_number(),
            "ADDRESS": lambda: self.fake.address().replace('\n', ', '),
            "DATE_TIME": lambda: random.choice([
                self.fake.date(),
                self.fake.date_time().strftime("%Y-%m-%d %H:%M"),
                self.fake.date_this_year().strftime("%B %d, %Y"),
                self.fake.date_of_birth().strftime("%m/%d/%Y"),
            ]),
            "CREDIT_CARD": lambda: self.fake.credit_card_number(),
            "NATIONAL_ID": lambda: self.fake.ssn(),
            "COMPANY": lambda: self.fake.company(),
            "AGE": lambda: str(random.randint(18, 85)),
            "URL": lambda: self.fake.url(),
        }

        generator = generators.get(entity_type)
        if generator:
            return generator()
        return ""

    def generate_sample(self, template_info: Dict) -> Dict[str, Any]:
        """Generate a single sample from a template."""
        template = template_info["template"]
        entity_types = template_info["entities"]

        # Generate entity values
        ground_truth = {}
        replacements = {}

        for entity_type in entity_types:
            value = self.generate_entity(entity_type)

            # Map to replacement key
            key_map = {
                "PERSON": "name",
                "EMAIL": "email",
                "PHONE": "phone",
                "ADDRESS": "address",
                "DATE_TIME": "date",
                "CREDIT_CARD": "credit_card",
                "NATIONAL_ID": "ssn",
                "COMPANY": "company",
                "AGE": "age",
                "URL": "url",
            }

            key = key_map.get(entity_type, entity_type.lower())
            replacements[key] = value

            # Add to ground truth
            if entity_type not in ground_truth:
                ground_truth[entity_type] = []
            ground_truth[entity_type].append(value)

        # Handle order_id specially (not a PII entity)
        if "{order_id}" in template:
            replacements["order_id"] = str(random.randint(10000, 99999))

        # Generate text
        text = template.format(**replacements)

        return {
            "text": text,
            "ground_truth": ground_truth,
        }

    def generate_golden_set(self, num_samples: int) -> Dict[str, Any]:
        """Generate a complete golden test set."""
        samples = []
        entity_counts = {}

        for i in range(num_samples):
            # Pick a random template
            template_info = random.choice(TEMPLATES)
            sample = self.generate_sample(template_info)

            sample["id"] = i
            sample["source"] = "faker_synthetic"
            samples.append(sample)

            # Track entity counts
            for entity_type, values in sample["ground_truth"].items():
                entity_counts[entity_type] = entity_counts.get(entity_type, 0) + len(values)

        total_entities = sum(entity_counts.values())

        return {
            "version": "3.0",
            "description": "Synthetic golden set generated with Faker (clean, well-formatted PII)",
            "created": datetime.now().isoformat(),
            "seed": 42,
            "total_samples": len(samples),
            "total_entities": total_entities,
            "composition": entity_counts,
            "samples": samples,
        }


def main():
    parser = argparse.ArgumentParser(description="Create synthetic golden test set")
    parser.add_argument("--samples", type=int, default=500, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="tests/data/synthetic_golden.json",
                        help="Output file path")
    args = parser.parse_args()

    print(f"Generating {args.samples} synthetic samples with Faker...")

    generator = SyntheticGoldenGenerator(seed=args.seed)
    golden_set = generator.generate_golden_set(args.samples)

    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(golden_set, f, indent=2)

    print(f"\nGenerated {golden_set['total_samples']} samples with {golden_set['total_entities']} entities")
    print(f"\nEntity distribution:")
    for entity_type, count in sorted(golden_set['composition'].items(), key=lambda x: -x[1]):
        print(f"  {entity_type}: {count}")

    print(f"\nSaved to: {output_path}")

    # Show sample
    print("\nSample entry:")
    sample = golden_set['samples'][0]
    print(f"  Text: {sample['text'][:100]}...")
    print(f"  Ground truth: {sample['ground_truth']}")


if __name__ == "__main__":
    main()
