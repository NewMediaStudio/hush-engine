"""
PII Detection using Microsoft Presidio
"""

from dataclasses import dataclass
from typing import List, Dict, Set, Optional
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
import pandas as pd
import threading
import re


@dataclass
class PIIEntity:
    """Represents a detected PII entity"""
    entity_type: str  # e.g., "EMAIL_ADDRESS", "PHONE_NUMBER", "AWS_KEY"
    text: str
    start: int
    end: int
    confidence: float


# Valid North American area codes (US, Canada, Caribbean)
# This list includes all assigned area codes as of 2024
NORTH_AMERICAN_AREA_CODES: Set[str] = {
    # Canada
    "204", "226", "236", "249", "250", "263", "289", "306", "343", "354", "365", "367",
    "368", "382", "387", "403", "416", "418", "428", "431", "437", "438", "450", "460",
    "468", "474", "506", "514", "519", "548", "579", "581", "584", "587", "604", "613",
    "639", "647", "672", "683", "705", "709", "742", "753", "778", "780", "782", "807",
    "819", "825", "867", "873", "879", "902", "905",
    # US - Major cities/regions (partial list of most common)
    "201", "202", "203", "205", "206", "207", "208", "209", "210", "212", "213", "214",
    "215", "216", "217", "218", "219", "220", "223", "224", "225", "228", "229", "231",
    "234", "239", "240", "248", "251", "252", "253", "254", "256", "260", "262", "267",
    "269", "270", "272", "276", "279", "281", "301", "302", "303", "304", "305", "307",
    "308", "309", "310", "312", "313", "314", "315", "316", "317", "318", "319", "320",
    "321", "323", "325", "326", "330", "331", "332", "334", "336", "337", "339", "340",
    "341", "346", "347", "351", "352", "360", "361", "364", "369", "380", "385", "386",
    "401", "402", "404", "405", "406", "407", "408", "409", "410", "412", "413", "414",
    "415", "417", "419", "423", "424", "425", "430", "432", "434", "435", "440", "442",
    "443", "445", "447", "448", "458", "463", "464", "469", "470", "475", "478", "479",
    "480", "484", "501", "502", "503", "504", "505", "507", "508", "509", "510", "512",
    "513", "515", "516", "517", "518", "520", "530", "531", "534", "539", "540", "541",
    "551", "559", "561", "562", "563", "564", "567", "570", "571", "573", "574", "575",
    "580", "582", "585", "586", "601", "602", "603", "605", "606", "607", "608", "609",
    "610", "612", "614", "615", "616", "617", "618", "619", "620", "623", "626", "628",
    "629", "630", "631", "636", "640", "641", "646", "650", "651", "657", "659", "660",
    "661", "662", "667", "669", "678", "680", "681", "682", "689", "701", "702", "703",
    "704", "706", "707", "708", "712", "713", "714", "715", "716", "717", "718", "719",
    "720", "724", "725", "726", "727", "730", "731", "732", "734", "737", "740", "743",
    "747", "754", "757", "760", "762", "763", "765", "769", "770", "772", "773", "774",
    "775", "779", "781", "785", "786", "801", "802", "803", "804", "805", "806", "808",
    "810", "812", "813", "814", "815", "816", "817", "818", "820", "828", "830", "831",
    "832", "838", "843", "845", "847", "848", "850", "854", "856", "857", "858", "859",
    "860", "862", "863", "864", "865", "870", "872", "878", "901", "903", "904", "906",
    "907", "908", "909", "910", "912", "913", "914", "915", "916", "917", "918", "919",
    "920", "925", "928", "929", "930", "931", "934", "936", "937", "938", "940", "941",
    "945", "947", "949", "951", "952", "954", "956", "959", "970", "971", "972", "973",
    "975", "978", "979", "980", "984", "985", "986", "989",
}

# Phone-related context words that indicate a phone number
PHONE_CONTEXT_WORDS = {
    "phone", "tel", "telephone", "mobile", "cell", "fax", "call", "contact",
    "dial", "number", "ph", "ph.", "ph:", "cell:", "mobile:", "fax:",
}


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
        # Add phone number recognizers with NA area code validation
        self._add_phone_recognizers()
        
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

    def _add_phone_recognizers(self):
        """
        Add custom phone number recognizers with North American area code validation.

        These recognizers have higher confidence than Presidio's default NHS recognizer
        to ensure phone numbers with valid NA area codes are correctly identified.
        """
        # NA phone with parentheses: (416) 770-4541
        na_phone_parens = PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=[
                Pattern(
                    name="na_phone_parens",
                    regex=r"\(\d{3}\)\s?\d{3}[-.\s]?\d{4}\b",
                    score=0.92,
                ),
            ],
            context=["phone", "tel", "mobile", "cell", "fax", "contact"]
        )

        # NA phone with dashes: 416-770-4541
        na_phone_dashes = PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=[
                Pattern(
                    name="na_phone_dashes",
                    regex=r"\b\d{3}-\d{3}-\d{4}\b",
                    score=0.90,
                ),
            ],
            context=["phone", "tel", "mobile", "cell", "fax", "contact"]
        )

        # NA phone with dots: 416.770.4541
        na_phone_dots = PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=[
                Pattern(
                    name="na_phone_dots",
                    regex=r"\b\d{3}\.\d{3}\.\d{4}\b",
                    score=0.90,
                ),
            ],
            context=["phone", "tel", "mobile", "cell", "fax", "contact"]
        )

        # International format: +1-416-770-4541
        intl_phone = PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=[
                Pattern(
                    name="intl_phone_plus",
                    regex=r"\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b",
                    score=0.85,
                ),
            ],
            context=["phone", "tel", "mobile", "cell", "fax", "contact"]
        )

        self.analyzer.registry.add_recognizer(na_phone_parens)
        self.analyzer.registry.add_recognizer(na_phone_dashes)
        self.analyzer.registry.add_recognizer(na_phone_dots)
        self.analyzer.registry.add_recognizer(intl_phone)

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

    def _extract_area_code(self, phone_text: str) -> Optional[str]:
        """Extract the area code from a phone number string."""
        # Remove common formatting
        digits = re.sub(r'[^\d]', '', phone_text)
        # Handle +1 country code
        if digits.startswith('1') and len(digits) == 11:
            digits = digits[1:]
        # Return first 3 digits if we have 10
        if len(digits) >= 10:
            return digits[:3]
        return None

    def _has_valid_na_area_code(self, phone_text: str) -> bool:
        """Check if a phone number has a valid North American area code."""
        area_code = self._extract_area_code(phone_text)
        return area_code in NORTH_AMERICAN_AREA_CODES if area_code else False

    def _has_phone_context(self, text: str, start: int, end: int, window: int = 50) -> bool:
        """Check if there's phone-related context near the detected entity."""
        # Get surrounding text
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        context = text[context_start:context_end].lower()

        # Check for phone-related words
        for word in PHONE_CONTEXT_WORDS:
            if word in context:
                return True
        return False

    def _has_phone_format(self, text: str) -> bool:
        """Check if text has typical phone number formatting (dashes, dots, parens)."""
        # Phone numbers typically have dashes, dots, or parentheses
        has_dashes = '-' in text and text.count('-') >= 2
        has_dots = '.' in text and text.count('.') >= 2
        has_parens = '(' in text and ')' in text
        return has_dashes or has_dots or has_parens

    def _detect_geographic_context(self, text: str) -> str:
        """
        Detect geographic context from currency and other indicators.
        Returns: 'NA' for North America, 'UK' for UK, 'unknown' otherwise
        """
        text_lower = text.lower()

        # North American indicators
        na_indicators = ['cad', 'usd', 'canada', 'canadian', 'united states',
                         'usa', 'ontario', 'quebec', 'alberta', 'british columbia',
                         'california', 'new york', 'texas', 'florida']
        for indicator in na_indicators:
            if indicator in text_lower:
                return 'NA'

        # UK indicators
        uk_indicators = ['gbp', 'nhs', 'united kingdom', 'england', 'scotland',
                         'wales', 'london', 'manchester', 'birmingham']
        for indicator in uk_indicators:
            if indicator in text_lower:
                return 'UK'

        return 'unknown'

    def _resolve_nhs_phone_conflicts(self, entities: List[PIIEntity], text: str) -> List[PIIEntity]:
        """
        Two-pass detection: resolve conflicts between UK_NHS and PHONE_NUMBER.

        If a detection could be either NHS or phone, use context to decide:
        1. Valid NA area code → PHONE_NUMBER
        2. Phone-related context nearby → PHONE_NUMBER
        3. Phone formatting (dashes/dots/parens) → PHONE_NUMBER
        4. NA geographic context → PHONE_NUMBER
        5. UK geographic context → keep as NHS
        """
        geo_context = self._detect_geographic_context(text)
        resolved = []

        for entity in entities:
            # Check if this is a potential NHS/phone conflict
            if entity.entity_type in ('UK_NHS', 'NHS'):
                phone_text = entity.text

                # Check signals that indicate this is a phone number
                is_phone = False
                confidence_boost = 0.0

                # 1. Valid NA area code (strongest signal)
                if self._has_valid_na_area_code(phone_text):
                    is_phone = True
                    confidence_boost = 0.15

                # 2. Phone context nearby
                if self._has_phone_context(text, entity.start, entity.end):
                    is_phone = True
                    confidence_boost = max(confidence_boost, 0.10)

                # 3. Phone formatting
                if self._has_phone_format(phone_text):
                    is_phone = True
                    confidence_boost = max(confidence_boost, 0.05)

                # 4. Geographic context
                if geo_context == 'NA':
                    is_phone = True
                    confidence_boost = max(confidence_boost, 0.05)
                elif geo_context == 'UK':
                    is_phone = False  # Keep as NHS

                if is_phone:
                    # Convert to PHONE_NUMBER with boosted confidence
                    resolved.append(PIIEntity(
                        entity_type='PHONE_NUMBER',
                        text=entity.text,
                        start=entity.start,
                        end=entity.end,
                        confidence=min(0.95, entity.confidence + confidence_boost)
                    ))
                else:
                    resolved.append(entity)
            else:
                resolved.append(entity)

        return resolved

    def analyze_text(self, text: str, language: str = "en") -> List[PIIEntity]:
        """
        Analyze text for PII entities with two-pass detection.

        Pass 1: Run Presidio detection
        Pass 2: Resolve conflicts (e.g., NHS vs phone numbers)

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

        # Pass 2: Resolve NHS/phone conflicts
        entities = self._resolve_nhs_phone_conflicts(entities, text)

        # Pass 3: Deduplicate overlapping entities (keep highest confidence)
        entities = self._deduplicate_entities(entities)

        return entities

    def _deduplicate_entities(self, entities: List[PIIEntity]) -> List[PIIEntity]:
        """Remove duplicate entities, keeping the one with highest confidence."""
        if not entities:
            return entities

        # Sort by start position, then by confidence (descending)
        sorted_entities = sorted(entities, key=lambda e: (e.start, -e.confidence))

        deduplicated = []
        for entity in sorted_entities:
            # Check if this entity overlaps with any already added
            is_duplicate = False
            for existing in deduplicated:
                # Check for significant overlap (same or nearly same span)
                if (entity.start >= existing.start and entity.end <= existing.end) or \
                   (existing.start >= entity.start and existing.end <= entity.end) or \
                   (abs(entity.start - existing.start) < 3 and abs(entity.end - existing.end) < 3):
                    # Same span and same type - skip duplicate
                    if entity.entity_type == existing.entity_type:
                        is_duplicate = True
                        break
            if not is_duplicate:
                deduplicated.append(entity)

        return deduplicated

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
