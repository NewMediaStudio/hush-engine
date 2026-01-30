"""
PII Detection using Microsoft Presidio
"""

from dataclasses import dataclass
from typing import List, Dict
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
import pandas as pd
import threading


@dataclass
class PIIEntity:
    """Represents a detected PII entity"""
    entity_type: str  # e.g., "EMAIL_ADDRESS", "PHONE_NUMBER", "AWS_KEY"
    text: str
    start: int
    end: int
    confidence: float


class PIIDetector:
    """
    Detects PII and technical secrets in text and structured data

    Uses Presidio with custom recognizers for:
    - Standard PII (names, emails, phones, SSNs)
    - Technical secrets (API keys, tokens, credentials)
    """

    def __init__(self):
        """Initialize Presidio analyzer with custom recognizers"""
        # Thread lock for analyzer access
        self._lock = threading.Lock()
        
        # Use Presidio WITHOUT NLP engine (regex-only mode)
        # This avoids the spaCy threading issue but won't detect PERSON entities
        self.analyzer = AnalyzerEngine(nlp_engine=None)

        # Add custom recognizers for technical secrets
        self._add_technical_recognizers()
        # Add credit card recognizers with common patterns
        self._add_credit_card_recognizers()
        # Add location recognizers (e.g. Canadian address format)
        self._add_location_recognizers()
        # Add currency recognizers
        self._add_currency_recognizers()
        # Add company recognizers
        self._add_company_recognizers()
        
        # Denylist of common words that should not be detected as PII
        # These are often document headers (e.g. "Email:", "Phone:")
        self.denylist = {
            "email", "phone", "name", "address", "date", "subject", "to", "from", "cc", "bcc",
            "first name", "last name", "middle name", "street", "city", "province", "state", "zip", "postal",
            "country", "mobile", "fax", "tel", "website", "url",
            "apartment", "unit", "suite", "floor", "level", "building", "po box"
        }

    def _add_location_recognizers(self):
        """Add pattern recognizers for address formats (e.g. Canadian place, province, postal code)."""
        # International street type designators
        # North American street types (US/Canada)
        na_street_types = r"(?:Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Boulevard|Blvd\.?|Drive|Dr\.?|Lane|Ln\.?|Court|Ct\.?|Circle|Cir\.?|Way|Place|Pl\.?|Terrace|Ter\.?|Parkway|Pkwy\.?|Highway|Hwy\.?|Crescent|Cres\.?|Trail)"
        
        # UK/Irish street types
        uk_street_types = r"(?:Road|Street|Lane|Avenue|Drive|Close|Gardens|Square|Crescent|Terrace|Grove|Place|Mews|Court|Row|Walk|Green|Park|Rise|Hill|Way|View)"
        
        # Australian/NZ street types (includes NA + UK plus specific types)
        au_street_types = r"(?:Parade|Esplanade|Promenade|Circuit)"
        
        # Combined English-speaking street types
        en_street_types = rf"(?:{na_street_types}|{uk_street_types}|{au_street_types})"
        
        # European street prefixes
        eu_street_prefixes = r"(?:Rue|Via|Calle|Avenida|Rua|Straße|Strasse|Platz|Allee|Plein)"
        
        # Canadian province abbreviations
        provinces = r"(AB|BC|MB|NB|NL|NS|NT|NU|ON|PE|QC|SK|YT)"
        # Canadian postal code: A1A 1A1 or A1A1A1
        postal = r"[A-Z]\d[A-Z] ?\d[A-Z]\d"

        # 1. Full format: "123 Main St, Desboro, ON N0H 1K0" or "Desboro, ON N0H 1K0"
        # We allow numbers and words at the start for street addresses and cities
        canadian_address_full = PatternRecognizer(
            supported_entity="LOCATION",
            patterns=[
                Pattern(
                    name="canadian_address_full",
                    # More permissive: matches anything that looks like an address leading up to PROV POSTAL
                    # Handles optional comma and multiple spaces
                    regex=rf"[\w\s\-.'']+,?\s+{provinces}\s+{postal}",
                    score=0.9,
                )
            ],
        )

        # 2. Province + postal only: "ON N0H 1K0"
        canadian_address_prov_postal = PatternRecognizer(
            supported_entity="LOCATION",
            patterns=[
                Pattern(
                    name="canadian_address_prov_postal",
                    regex=rf"\b{provinces}\s+{postal}\b",
                    score=0.85,
                )
            ],
        )

        # 3. City + Province: "Toronto, ON" or "Toronto ON"
        canadian_city_prov = PatternRecognizer(
            supported_entity="LOCATION",
            patterns=[
                Pattern(
                    name="canadian_city_prov",
                    regex=rf"\b[A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*,?\s+{provinces}\b",
                    score=0.6,
                )
            ],
        )

        # 4. Postal code only: "N0H 1K0"
        # High confidence for Canadian postal code format
        canadian_postal_only = PatternRecognizer(
            supported_entity="LOCATION",
            patterns=[
                Pattern(
                    name="canadian_postal_only",
                    regex=rf"\b{postal}\b",
                    score=0.7,
                )
            ],
        )

        # International street address recognizers
        
        # 5. Basic street address with number: "12 Crane Ave", "221B Baker Street"
        street_address_with_number = PatternRecognizer(
            supported_entity="LOCATION",
            patterns=[
                Pattern(
                    name="street_address_numbered",
                    # Matches: number (with optional letter) + street name (1-4 words) + street type
                    # Supports directional prefixes: N, S, E, W, North, South, etc.
                    regex=rf"\b\d{{1,6}}[A-Z]?\s+(?:(?:North|South|East|West|N\.?|S\.?|E\.?|W\.?)\s+)?[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){{0,4}}\s+{en_street_types}\b",
                    score=0.85,
                )
            ],
        )
        
        # 6. Street name with type (no number): "Crane Avenue", "Baker Street"
        street_address_no_number = PatternRecognizer(
            supported_entity="LOCATION",
            patterns=[
                Pattern(
                    name="street_name_only",
                    # Matches: street name (1-3 words) + street type
                    # Lower confidence to avoid false positives
                    regex=rf"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){{0,3}}\s+{en_street_types}\b",
                    score=0.70,
                )
            ],
        )
        
        # 7. European street formats: "Rue de la Paix", "Via Roma", "Calle Mayor"
        european_street_address = PatternRecognizer(
            supported_entity="LOCATION",
            patterns=[
                Pattern(
                    name="european_street",
                    # Matches: European prefix + optional "de/la/del/des" + street name
                    regex=rf"\b{eu_street_prefixes}\s+(?:de\s+(?:la\s+|l')?|del\s+|des\s+)?[A-Z][a-zA-Z\s'-]{{2,40}}\b",
                    score=0.80,
                )
            ],
        )
        
        # 8. PO Box addresses: "PO Box 123", "P.O. Box 456"
        po_box_address = PatternRecognizer(
            supported_entity="LOCATION",
            patterns=[
                Pattern(
                    name="po_box",
                    # Matches: P.O. Box or PO Box (with various punctuation) + number
                    regex=r"\b(?:P\.?\s*O\.?\s+)?Box\s+\d{1,6}\b",
                    score=0.90,
                )
            ],
        )
        
        # 9. Unit/Apartment addresses: "Unit 5, 12 Crane Ave", "Apt 3B, 100 Main Street"
        unit_street_address = PatternRecognizer(
            supported_entity="LOCATION",
            patterns=[
                Pattern(
                    name="unit_apartment",
                    # Matches: Unit/Apt/Suite + number/letter + comma + street address
                    regex=rf"\b(?:Unit|Apt\.?|Apartment|Suite|Ste\.?)\s+[0-9A-Z]+,\s+\d{{1,6}}[A-Z]?\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){{0,4}}\s+{en_street_types}\b",
                    score=0.85,
                )
            ],
        )
        
        # Register all recognizers
        self.analyzer.registry.add_recognizer(canadian_address_full)
        self.analyzer.registry.add_recognizer(canadian_address_prov_postal)
        self.analyzer.registry.add_recognizer(canadian_city_prov)
        self.analyzer.registry.add_recognizer(canadian_postal_only)
        self.analyzer.registry.add_recognizer(street_address_with_number)
        self.analyzer.registry.add_recognizer(street_address_no_number)
        self.analyzer.registry.add_recognizer(european_street_address)
        self.analyzer.registry.add_recognizer(po_box_address)
        self.analyzer.registry.add_recognizer(unit_street_address)

    def _add_currency_recognizers(self):
        """Add pattern recognizers for currency amounts."""
        # Generic currency regex: symbol followed by numbers with commas and decimals
        # Supports $, £, €, ¥, ₹, ₽, 元 and handles optional space
        currency_regex = r"(\$|£|€|¥|₹|₽|元)\s?\d{1,3}(,\d{3})*(\.\d{2})?\b"
        
        # Word-based currency regex: USD, CAD, EUR, etc. followed by numbers
        word_currency_regex = r"\b(USD|CAD|EUR|GBP|JPY|AUD|CNY)\s?\d{1,3}(,\d{3})*(\.\d{2})?\b"

        currency_recognizer = PatternRecognizer(
            supported_entity="CURRENCY",
            patterns=[
                Pattern(
                    name="currency_symbol",
                    regex=currency_regex,
                    score=0.8,
                ),
                Pattern(
                    name="currency_code",
                    regex=word_currency_regex,
                    score=0.8,
                )
            ],
        )

        self.analyzer.registry.add_recognizer(currency_recognizer)

    def _add_company_recognizers(self):
        """Add pattern recognizers for company names and legal designations."""
        # Legal designations for companies
        # Supports: Ltd, Inc, LLC, GmbH, S.A., PLC, AG, Corp, etc.
        designations = [
            r"Ltd\.?", r"Limited",
            r"Inc\.?", r"Incorporated",
            r"Co\.?", r"Company",
            r"Corp\.?", r"Corporation",
            r"LLC", r"GmbH", r"S\.A\.?", r"PLC", r"AG",
            r"N\.?V\.?", r"B\.?V\.?", r"S\.?R\.?L\.?", r"S\.?A\.?S\.?",
            r"KGaA", r"S\.?E\.?"
        ]
        designations_pattern = "|".join(designations)
        
        # Company name regex: Capitalized words followed by a legal designation
        # Matches "Alleles Company Ltd.", "Apple Inc.", "Siemens AG", etc.
        company_regex = rf"\b[A-Z][a-zA-Z0-9&',.-]+(?:\s+[A-Z][a-zA-Z0-9&',.-]+)*\s+(?:{designations_pattern})\b"

        company_recognizer = PatternRecognizer(
            supported_entity="COMPANY",
            patterns=[
                Pattern(
                    name="company_name",
                    regex=company_regex,
                    score=0.7,
                )
            ],
            context=["company", "inc", "ltd", "corp", "limited", "firm", "business"]
        )

        self.analyzer.registry.add_recognizer(company_recognizer)

    def _add_credit_card_recognizers(self):
        """Add custom pattern recognizers for common credit card formats."""
        # Common credit card patterns (with optional spaces or dashes)
        # Visa: starts with 4, 16 digits
        # Mastercard: starts with 51-55 or 2221-2720, 16 digits
        # American Express: starts with 34 or 37, 15 digits
        # Discover: starts with 6011, 622126-622925, 644-649, or 65, 16 digits
        
        credit_card_recognizer = PatternRecognizer(
            supported_entity="CREDIT_CARD",
            patterns=[
                # Visa (4xxx xxxx xxxx xxxx)
                Pattern(
                    name="visa",
                    regex=r"\b4\d{3}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
                    score=0.9
                ),
                # Mastercard (51xx-55xx or 2221-2720)
                Pattern(
                    name="mastercard",
                    regex=r"\b(?:5[1-5]\d{2}|2(?:22[1-9]|2[3-9]\d|[3-6]\d{2}|7[0-1]\d|720))[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
                    score=0.9
                ),
                # American Express (34xx or 37xx)
                Pattern(
                    name="amex",
                    regex=r"\b3[47]\d{2}[\s-]?\d{6}[\s-]?\d{5}\b",
                    score=0.9
                ),
                # Discover (6011, 622126-622925, 644-649, 65)
                Pattern(
                    name="discover",
                    regex=r"\b(?:6011|65\d{2}|64[4-9]\d|622(?:1(?:2[6-9]|[3-9]\d)|[2-8]\d{2}|9(?:[01]\d|2[0-5])))[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
                    score=0.9
                ),
                # Generic 16-digit pattern (fallback)
                Pattern(
                    name="generic_16",
                    regex=r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
                    score=0.6
                ),
                # Generic 15-digit pattern (fallback for Amex)
                Pattern(
                    name="generic_15",
                    regex=r"\b\d{4}[\s-]?\d{6}[\s-]?\d{5}\b",
                    score=0.6
                )
            ],
            context=["card", "credit", "payment", "visa", "mastercard", "amex", "discover"]
        )

        self.analyzer.registry.add_recognizer(credit_card_recognizer)

    def _add_technical_recognizers(self):
        """Add custom pattern recognizers for API keys, tokens, etc."""
        # AWS Access Key
        aws_recognizer = PatternRecognizer(
            supported_entity="AWS_ACCESS_KEY",
            patterns=[
                Pattern(
                    name="aws_access_key",
                    regex=r"(AKIA|ASIA)[A-Z0-9]{16}",
                    score=0.8
                )
            ],
            context=["aws", "amazon", "key", "access", "secret"]
        )

        # Stripe Secret Key
        stripe_recognizer = PatternRecognizer(
            supported_entity="STRIPE_KEY",
            patterns=[
                Pattern(
                    name="stripe_secret",
                    regex=r"sk_live_[0-9a-zA-Z]{24,}",
                    score=0.9
                )
            ],
            context=["stripe", "secret", "api", "key"]
        )

        # TODO: Add more recognizers (GitHub tokens, Google API keys, etc.)

        self.analyzer.registry.add_recognizer(aws_recognizer)
        self.analyzer.registry.add_recognizer(stripe_recognizer)

    def analyze_text(self, text: str, language: str = "en") -> List[PIIEntity]:
        """
        Analyze text for PII entities

        Args:
            text: Input text to analyze
            language: Language code (default: "en")

        Returns:
            List of detected PII entities
        """
        results = self.analyzer.analyze(
            text=text,
            language=language,
            entities=None,  # Detect all entity types
            return_decision_process=False
        )

        entities = []
        for r in results:
            entity_type = r.entity_type
            if entity_type.startswith("US_"):
                entity_type = entity_type[3:]
                
            entity_text = text[r.start:r.end]
            # Filter out common words in the denylist (case-insensitive)
            if entity_text.lower().strip(": ") in self.denylist:
                continue
                
            entities.append(PIIEntity(
                entity_type=entity_type,
                text=entity_text,
                start=r.start,
                end=r.end,
                confidence=r.score
            ))

        return entities

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        sample_size: int = 20
    ) -> Dict[str, List[PIIEntity]]:
        """
        Analyze a DataFrame for PII, using sampling for efficiency

        Args:
            df: Input DataFrame
            sample_size: Number of rows to sample per column

        Returns:
            Dictionary mapping column names to detected entities
        """
        results = {}

        for column in df.columns:
            # Sample non-null values
            sample = df[column].dropna().sample(
                n=min(sample_size, len(df[column].dropna())),
                random_state=42
            )

            # Analyze concatenated sample
            sample_text = " | ".join(str(val) for val in sample)
            entities = self.analyze_text(sample_text)

            if entities:
                results[column] = entities

        return results
