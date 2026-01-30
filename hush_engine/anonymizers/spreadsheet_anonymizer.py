"""
Spreadsheet anonymization - Replace PII with synthetic data
"""

import pandas as pd
from typing import Dict, List
from faker import Faker
import hashlib


class SpreadsheetAnonymizer:
    """
    Anonymizes spreadsheets by replacing PII with consistent synthetic data
    """

    def __init__(self):
        self.faker = Faker()
        self._cache = {}  # For consistent replacements

    def anonymize_dataframe(
        self,
        df: pd.DataFrame,
        entity_map: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """
        Anonymize a DataFrame based on detected entity types

        Args:
            df: Input DataFrame
            entity_map: Dict mapping column names to entity types
                       e.g., {"email_col": ["EMAIL_ADDRESS"], "name_col": ["PERSON"]}

        Returns:
            Anonymized DataFrame
        """
        df_copy = df.copy()

        for column, entity_types in entity_map.items():
            if column not in df_copy.columns:
                continue

            # Apply anonymization based on primary entity type
            primary_type = entity_types[0] if entity_types else "UNKNOWN"

            df_copy[column] = df_copy[column].apply(
                lambda x: self._anonymize_value(x, primary_type)
            )

        return df_copy

    def _anonymize_value(self, value, entity_type: str):
        """
        Replace a single value with synthetic data

        Uses deterministic hashing to ensure same input → same output
        """
        if pd.isna(value):
            return value

        # Create deterministic seed from value
        seed = int(hashlib.md5(str(value).encode("utf-8")).hexdigest()[:8], 16)
        self.faker.seed_instance(seed)

        # Generate replacement based on entity type
        replacements = {
            "PERSON": lambda: self.faker.name(),
            "EMAIL_ADDRESS": lambda: f"user{seed % 10000}@example.com",
            "PHONE_NUMBER": lambda: f"555-{seed % 10000:04d}",
            "LOCATION": lambda: self.faker.city(),
            "CREDIT_CARD": lambda: "****-****-****-****",
            "AWS_ACCESS_KEY": lambda: f"AKIAIOSFODNN7{seed % 10000000:07d}",
            "STRIPE_KEY": lambda: "sk_live_[REDACTED]",
        }

        generator = replacements.get(entity_type, lambda: f"[REDACTED-{seed % 1000}]")
        return generator()
