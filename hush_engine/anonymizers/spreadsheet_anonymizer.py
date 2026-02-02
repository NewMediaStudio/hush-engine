"""
Spreadsheet anonymization - Replace PII with [REDACTED] labels

Supports all entity types from the PII Reference (docs/PII_REFERENCE.md):
- Personal Identity: PERSON, FACE, GENDER, NRP
- Contact Information: EMAIL_ADDRESS, PHONE_NUMBER, URL
- Government IDs: US_SSN, SSN, PASSPORT, DRIVERS_LICENSE, DRIVER_LICENSE, ITIN, UK_NHS
- Financial: CREDIT_CARD, BANK_NUMBER, IBAN_CODE, FINANCIAL, CRYPTO
- Location: LOCATION, COORDINATES
- Medical: MEDICAL
- Technical/Device: IP_ADDRESS, DEVICE_ID, VEHICLE_ID
- Document Elements: DATE_TIME, QR_CODE, BARCODE
- Organization: COMPANY
- API Keys: AWS_ACCESS_KEY, STRIPE_KEY
"""

import pandas as pd
from typing import Dict, List


class SpreadsheetAnonymizer:
    """
    Anonymizes spreadsheets by replacing PII with [REDACTED] labels.
    """

    def __init__(self):
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
                lambda x, et=primary_type: self._anonymize_value(x, et)
            )

        return df_copy

    def _anonymize_value(self, value, entity_type: str):
        """
        Replace a single value with [REDACTED] label.
        """
        if pd.isna(value):
            return value

        # Human-readable labels for entity types
        labels = {
            # Personal Identity
            "PERSON": "[REDACTED NAME]",
            "FACE": "[REDACTED FACE]",
            "GENDER": "[REDACTED]",
            "NRP": "[REDACTED]",

            # Contact Information
            "EMAIL_ADDRESS": "[REDACTED EMAIL]",
            "PHONE_NUMBER": "[REDACTED PHONE]",
            "URL": "[REDACTED URL]",

            # Government IDs
            "US_SSN": "[REDACTED SSN]",
            "SSN": "[REDACTED SSN]",
            "PASSPORT": "[REDACTED PASSPORT]",
            "DRIVERS_LICENSE": "[REDACTED DL]",
            "DRIVER_LICENSE": "[REDACTED DL]",
            "ITIN": "[REDACTED ITIN]",
            "UK_NHS": "[REDACTED NHS]",

            # Financial
            "CREDIT_CARD": "[REDACTED CC]",
            "BANK_NUMBER": "[REDACTED BANK]",
            "IBAN_CODE": "[REDACTED IBAN]",
            "FINANCIAL": "[REDACTED]",
            "CRYPTO": "[REDACTED CRYPTO]",

            # Location
            "LOCATION": "[REDACTED LOCATION]",
            "COORDINATES": "[REDACTED COORDS]",

            # Medical
            "MEDICAL": "[REDACTED MEDICAL]",

            # Technical / Device
            "IP_ADDRESS": "[REDACTED IP]",
            "DEVICE_ID": "[REDACTED DEVICE]",
            "VEHICLE_ID": "[REDACTED VIN]",

            # Document Elements
            "DATE_TIME": "[REDACTED DATE]",
            "QR_CODE": "[REDACTED QR]",
            "BARCODE": "[REDACTED BARCODE]",

            # Organization
            "COMPANY": "[REDACTED COMPANY]",

            # API Keys / Secrets
            "AWS_ACCESS_KEY": "[REDACTED KEY]",
            "STRIPE_KEY": "[REDACTED KEY]",
        }

        return labels.get(entity_type, f"[REDACTED]")
