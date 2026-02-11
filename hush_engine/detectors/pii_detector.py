"""
PII Detection using Microsoft Presidio

Supports locale-aware detection for international documents.
"""

# CRITICAL: Pre-load LightGBM models BEFORE spaCy/presidio imports
# This avoids OpenMP library conflicts (libomp vs libiomp5) on macOS
try:
    from hush_engine.detectors import lgbm_preloader  # noqa: F401
except ImportError:
    try:
        from . import lgbm_preloader  # noqa: F401
    except ImportError:
        pass  # LightGBM pre-loading optional

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.context_aware_enhancers import LemmaContextAwareEnhancer
from presidio_analyzer.predefined_recognizers import PhoneRecognizer
import pandas as pd
import threading
import re
from pathlib import Path

# Phone number validation using Google's libphonenumber
try:
    import phonenumbers
    from phonenumbers import NumberParseException
    PHONENUMBERS_AVAILABLE = True
except ImportError:
    PHONENUMBERS_AVAILABLE = False

# National ID validation using python-stdnum (35+ countries)
try:
    from stdnum import luhn
    from stdnum.us import ssn as us_ssn
    from stdnum.gb import nino as uk_nino
    from stdnum.de import idnr as de_steuerid
    from stdnum.br import cpf as br_cpf
    from stdnum.it import codicefiscale as it_cf
    from stdnum.es import dni as es_dni
    from stdnum.pl import pesel as pl_pesel
    from stdnum.se import personnummer as se_personnummer
    from stdnum.nl import bsn as nl_bsn
    from stdnum.be import nn as be_nn
    STDNUM_AVAILABLE = True
except ImportError:
    STDNUM_AVAILABLE = False

# Address parsing using libpostal
try:
    from postal.parser import parse_address as _parse_address_raw
    from functools import lru_cache

    @lru_cache(maxsize=2000)
    def parse_address(text: str):
        """Cached version of libpostal parse_address for better performance."""
        return _parse_address_raw(text)

    LIBPOSTAL_AVAILABLE = True
except ImportError:
    LIBPOSTAL_AVAILABLE = False

    def parse_address(text: str):
        """Stub when libpostal not available."""
        return []

# URL extraction using urlextract (better than regex for bare domains, subdomains)
try:
    from urlextract import URLExtract
    URLEXTRACT_AVAILABLE = True
    _url_extractor = None  # Lazy-loaded singleton
except ImportError:
    URLEXTRACT_AVAILABLE = False
    _url_extractor = None

# Date parsing using dateparser (better than regex for natural language dates)
try:
    import dateparser
    DATEPARSER_AVAILABLE = True
except ImportError:
    DATEPARSER_AVAILABLE = False

# Cities database for LOCATION detection
try:
    from hush_engine.data.cities_database import get_cities_db, CitiesDatabase
    CITIES_DB_AVAILABLE = True
except ImportError:
    try:
        from .data.cities_database import get_cities_db, CitiesDatabase
        CITIES_DB_AVAILABLE = True
    except ImportError:
        CITIES_DB_AVAILABLE = False
        get_cities_db = None
        CitiesDatabase = None

# Countries database for LOCATION detection
try:
    from hush_engine.data.countries_database import get_countries_database, CountriesDatabase
    COUNTRIES_DB_AVAILABLE = True
except ImportError:
    try:
        from .data.countries_database import get_countries_database, CountriesDatabase
        COUNTRIES_DB_AVAILABLE = True
    except ImportError:
        COUNTRIES_DB_AVAILABLE = False
        get_countries_database = None
        CountriesDatabase = None

# Package version - import from central config
try:
    from hush_engine.detection_config import VERSION, get_config
except ImportError:
    from ..detection_config import VERSION, get_config

# Negative gazetteer for false positive suppression
try:
    from hush_engine.data.negative_gazetteer import is_negative_match, is_single_common_word
    NEGATIVE_GAZETTEER_AVAILABLE = True
except ImportError:
    try:
        from ..data.negative_gazetteer import is_negative_match, is_single_common_word
        NEGATIVE_GAZETTEER_AVAILABLE = True
    except ImportError:
        NEGATIVE_GAZETTEER_AVAILABLE = False
        is_negative_match = lambda text, entity_type: False
        is_single_common_word = lambda text, entity_type: False

# Rule-based heuristic verification for high-FP entity types
try:
    from hush_engine.detectors.heuristic_verifier import filter_with_heuristics
    HEURISTIC_VERIFIER_AVAILABLE = True
except ImportError:
    try:
        from .heuristic_verifier import filter_with_heuristics
        HEURISTIC_VERIFIER_AVAILABLE = True
    except ImportError:
        HEURISTIC_VERIFIER_AVAILABLE = False
        filter_with_heuristics = lambda entities, text, min_confidence=0.40: entities

# OCR artifact filtering (LARVPC rules)
try:
    from hush_engine.detectors.ocr_filter import filter_ocr_artifacts, is_ocr_garbage
    OCR_FILTER_AVAILABLE = True
except ImportError:
    try:
        from .ocr_filter import filter_ocr_artifacts, is_ocr_garbage
        OCR_FILTER_AVAILABLE = True
    except ImportError:
        OCR_FILTER_AVAILABLE = False
        filter_ocr_artifacts = lambda text, entity_type: (True, "ok")
        is_ocr_garbage = lambda text: False

# LightGBM-based COMPANY verification for false positive filtering
try:
    from hush_engine.detectors.company_verifier import verify_company_detection, is_company_verifier_available
    COMPANY_VERIFIER_AVAILABLE = is_company_verifier_available()
except ImportError:
    try:
        from .company_verifier import verify_company_detection, is_company_verifier_available
        COMPANY_VERIFIER_AVAILABLE = is_company_verifier_available()
    except ImportError:
        COMPANY_VERIFIER_AVAILABLE = False
        verify_company_detection = lambda text, company_text, start, end, confidence: (True, confidence)

# LightGBM-based ADDRESS verification for false positive filtering
try:
    from hush_engine.detectors.address_verifier import verify_address_detection, is_address_verifier_available
    ADDRESS_VERIFIER_AVAILABLE = is_address_verifier_available()
except ImportError:
    try:
        from .address_verifier import verify_address_detection, is_address_verifier_available
        ADDRESS_VERIFIER_AVAILABLE = is_address_verifier_available()
    except ImportError:
        ADDRESS_VERIFIER_AVAILABLE = False
        verify_address_detection = lambda text, addr_text, start, end, confidence: (True, confidence)

# Shannon entropy-based CREDENTIAL verification for secret detection
try:
    from hush_engine.detectors.credential_entropy import (
        analyze_credential_entropy,
        filter_credential_by_entropy,
        ENTROPY_THRESHOLDS,
    )
    CREDENTIAL_ENTROPY_AVAILABLE = True
except ImportError:
    try:
        from .credential_entropy import (
            analyze_credential_entropy,
            filter_credential_by_entropy,
            ENTROPY_THRESHOLDS,
        )
        CREDENTIAL_ENTROPY_AVAILABLE = True
    except ImportError:
        CREDENTIAL_ENTROPY_AVAILABLE = False
        analyze_credential_entropy = None
        filter_credential_by_entropy = None
        ENTROPY_THRESHOLDS = None
    except ImportError:
        ADDRESS_VERIFIER_AVAILABLE = False
        verify_address_detection = lambda text, addr_text, start, end, confidence: (True, confidence)

__version__ = VERSION

# Import locale support
from .locale import (
    Locale, calculate_locale_boost, detect_document_locale,
    get_locale_from_string, PATTERN_LOCALE_MAP
)

# Import text normalization for evasion defense
try:
    from hush_engine.preprocessing.text_normalizer import normalize_text, decode_and_scan
    TEXT_NORMALIZER_AVAILABLE = True
except ImportError:
    try:
        from ..preprocessing.text_normalizer import normalize_text, decode_and_scan
        TEXT_NORMALIZER_AVAILABLE = True
    except ImportError:
        TEXT_NORMALIZER_AVAILABLE = False
        normalize_text = lambda x: x
        decode_and_scan = lambda x: []

# Company NER using dictionary-based recognition (Fast Data Science)
try:
    from company_named_entity_recognition import find_companies
    COMPANY_NER_AVAILABLE = True
except (ImportError, FileNotFoundError, Exception) as e:
    # Package may be missing or have missing data files
    COMPANY_NER_AVAILABLE = False
    find_companies = None


@dataclass
class PIIEntity:
    """
    Represents a detected PII entity.

    Attributes:
        entity_type: The type of PII (e.g., "EMAIL_ADDRESS", "PHONE_NUMBER")
        text: The detected text
        start: Start position in the source text
        end: End position in the source text
        confidence: Detection confidence score (0.0 to 1.0)
        pattern_name: Name of the pattern that matched (optional)
        locale: ISO locale code the pattern is associated with (optional)
        recognition_metadata: Metadata from recognizer (e.g., detection_source)
    """
    entity_type: str  # e.g., "EMAIL_ADDRESS", "PHONE_NUMBER", "AWS_KEY"
    text: str
    start: int
    end: int
    confidence: float
    pattern_name: Optional[str] = None
    locale: Optional[str] = None
    recognition_metadata: Optional[dict] = None


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


def shannon_entropy(text: str) -> float:
    """
    Calculate Shannon entropy of a string.

    Used for credential detection: high-entropy strings (4.5-6.0) are likely
    passwords, API keys, or tokens. Low-entropy strings are likely false positives.

    Args:
        text: String to calculate entropy for

    Returns:
        Entropy value (0.0 to ~6.0 for printable ASCII)
    """
    if not text:
        return 0.0
    import math
    from collections import Counter
    freq = Counter(text)
    length = len(text)
    return -sum((count / length) * math.log2(count / length)
                for count in freq.values())


# Entropy thresholds for credential detection
CREDENTIAL_ENTROPY_MIN = 3.0  # Minimum entropy for secrets
CREDENTIAL_ENTROPY_MAX = 6.5  # Maximum entropy (theoretical max ~6.0 for base64)


class PIIDetector:
    """
    Detects PII and technical secrets in text and structured data

    Uses Presidio with custom recognizers for:
    - Standard PII (names, emails, phones, SSNs)
    - Technical secrets (API keys, tokens, credentials)
    """

    def __init__(self, enable_libpostal: bool = True):
        """Initialize Presidio analyzer with custom recognizers

        Args:
            enable_libpostal: If True, enable libpostal-based address detection.
                             Set to False for faster benchmark runs (30-40% speedup).
        """
        # Thread lock for analyzer access
        self._lock = threading.Lock()

        # Store config for introspection
        self._enable_libpostal = enable_libpostal

        # Configure NLP engine for context-aware detection
        # Uses spaCy for lemmatization and context enhancement
        try:
            nlp_config = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]
            }
            provider = NlpEngineProvider(nlp_configuration=nlp_config)
            nlp_engine = provider.create_engine()

            # Context enhancer: boost score by 0.35 when context words found
            # within a 5-token window (e.g., "Patient John Smith" boosts PERSON)
            context_enhancer = LemmaContextAwareEnhancer(
                context_similarity_factor=0.35,
                min_score_with_context_similarity=0.4,
                context_prefix_count=10,  # Wider window for structured/HTML text
                context_suffix_count=5
            )

            self.analyzer = AnalyzerEngine(
                nlp_engine=nlp_engine,
                context_aware_enhancer=context_enhancer
            )
            self._nlp_enabled = True
        except Exception:
            # Fallback to regex-only mode if spaCy not available
            self.analyzer = AnalyzerEngine(nlp_engine=None)
            self._nlp_enabled = False

        # Add custom recognizers for technical secrets
        self._add_technical_recognizers()
        # Add credit card recognizers with common patterns
        self._add_credit_card_recognizers()
        # Add location recognizers (e.g. Canadian address format)
        self._add_location_recognizers()
        # Add date/time recognizers (international date formats)
        self._add_datetime_recognizers()
        # Add age recognizers (age in years, "X years old", "Age: X")
        self._add_age_recognizers()
        # Add financial recognizers (currency, SWIFT codes, etc.)
        self._add_financial_recognizers()
        # Add company recognizers
        self._add_company_recognizers()
        # Add gender identity recognizers
        self._add_gender_recognizers()
        # Add coordinate recognizers (GPS, lat/long)
        self._add_coordinate_recognizers()
        # Add medical recognizers (conditions, medications, blood types, etc.)
        self._add_medical_recognizers()
        # Add Fast Data Science medical NER (broader coverage)
        self._add_fast_medical_recognizer()
        # Add phone number recognizers with NA area code validation
        self._add_phone_recognizers()
        # Add ID document recognizers (passports, driver's licenses)
        self._add_id_document_recognizers()
        # Add vehicle recognizers (VIN)
        self._add_vehicle_recognizers()
        # Add device ID recognizers (IMEI, MAC address, UUID)
        self._add_device_recognizers()
        # Add biometric ID recognizers (BIO-, fingerprint IDs, etc.)
        self._add_biometric_recognizers()
        # Add credential recognizers (passwords, API keys, PINs)
        self._add_credential_recognizers()
        # Add username recognizers (alphanumeric usernames, dot-separated)
        self._add_username_recognizers()
        # Add generic ID recognizers (alphanumeric codes)
        self._add_id_recognizers()
        # Add network recognizers (MAC addresses, cookies, session IDs)
        self._add_network_recognizers()
        # Add IP address recognizers (IPv4, IPv6)
        self._add_ip_address_recognizers()
        # Add SSN recognizer (emits NATIONAL_ID for unified ID handling)
        self._add_ssn_recognizers()
        # Add international national ID recognizers
        self._add_international_id_recognizers()
        # Add person name recognizers (title + name patterns)
        self._add_person_recognizers()
        # Remove Presidio's default SpacyRecognizer for PERSON to avoid duplicate detection
        self._remove_default_person_recognizers()
        # Remove Presidio's default US_SSN recognizer (we emit NATIONAL_ID instead)
        self._remove_default_ssn_recognizers()
        # Add URL recognizers (http, https, www, subdomains)
        self._add_url_recognizers()
        # Add obfuscated email recognizers ([at], spaced emails)
        self._add_obfuscated_email_recognizers()
        # Add generic alphanumeric ID patterns (customer IDs, reference numbers, etc.)
        self._add_generic_id_recognizers()
        # Add libpostal-based address detection (99.45% accuracy on global addresses)
        # Note: libpostal is the slowest recognizer (~30-40% of detection time)
        if enable_libpostal:
            self._add_libpostal_recognizer()

        # Denylist of common words that should not be detected as PII
        # These are often document headers (e.g. "Email:", "Phone:")
        self.denylist = {
            # Form labels
            "email", "phone", "name", "address", "date", "subject", "to", "from", "cc", "bcc",
            "first name", "last name", "middle name", "street", "city", "province", "state", "zip", "postal",
            "country", "mobile", "fax", "tel", "website", "url",
            "apartment", "unit", "suite", "floor", "level", "building", "po box",
            # UI/business terms (prevent SWIFT false positives like CUSTOMER -> CUST+OM+ER)
            "customer", "customers", "customer hub", "customer agent", "customer request", "customer requests",
            "overview", "dashboard", "portal", "service", "services", "support", "agent", "agents",
            "business", "request", "requests", "menu", "navigation", "header", "footer", "new customer",
            "help your business", "financial",
            # Common disclaimer/explanatory phrases (prevent address false positives)
            "business network", "billing purposes", "connected to", "may be connected", "used for",
            # Software/brand names (not person names)
            "creo", "adobe", "figma", "sketch", "canva", "illustrator", "photoshop", "indesign",
        }

    @staticmethod
    def getVersion() -> str:
        """
        Get the current version of the PII detector engine.

        Returns:
            str: Version string (e.g., "1.1.1")
        """
        return __version__

    @staticmethod
    def get_version() -> str:
        """
        Get the current version of the PII detector engine.
        Alias for getVersion() using Python naming convention.

        Returns:
            str: Version string (e.g., "1.1.1")
        """
        return __version__

    def _add_location_recognizers(self):
        """Add pattern recognizers for address formats (e.g. Canadian place, province, postal code)."""
        # International street type designators
        # North American street types (US/Canada)
        # Expanded to include less common but valid street types
        na_street_types = r"(?:Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Boulevard|Blvd\.?|Drive|Dr\.?|Lane|Ln\.?|Court|Ct\.?|Circle|Cir\.?|Way|Place|Pl\.?|Terrace|Ter\.?|Parkway|Pkwy\.?|Highway|Hwy\.?|Crescent|Cres\.?|Trail|Cliff|Cliffs|Ramp|Haven|Village|Commons|Plaza|Bend|Loop|Cove|Ridge|Heights|Point|Center|Centre|Pass|Run|Crossing|Path|Alley|Aly\.?|Pike|Fork|Branch|Glen|Hollow|Knoll|Landing|Manor|Meadow|Mill|Orchard|Overlook|Pointe|Shores|Spring|Summit|Valley|Vista|Estates|Gardens|Grove|Oaks|Pines|Woods)"
        
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
        # Canadian postal code: A1A 1A1 or A1A1A1 (Letter-Digit-Letter Digit-Letter-Digit)
        # Allow O/0 confusion from OCR (e.g., NOH 1KO vs N0H 1K0)
        # Digit positions: 2nd (can be 0-9 or O), 4th (0-9 or O), 6th (0-9 or O)
        postal = r"[A-Z][0-9O][A-Z] ?[0-9O][A-Z][0-9O]"

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
                    score=0.80,  # Boosted for better recall
                )
            ],
        )

        # US state abbreviations (all 50 states + DC + territories + freely associated states)
        us_states = r"(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY|DC|PR|VI|GU|AS|MP|FM|PW|MH)"

        # US ZIP code: 5 digits, optionally followed by -4 digits
        us_zip = r"\d{5}(?:-\d{4})?"

        # 5. US City, State ZIP: "Portland, OR 97201" or "New York, NY 10001-1234"
        us_address_full = PatternRecognizer(
            supported_entity="LOCATION",
            patterns=[
                Pattern(
                    name="us_city_state_zip",
                    regex=rf"\b[A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*,?\s+{us_states}\s+{us_zip}\b",
                    score=0.9,
                )
            ],
        )

        # 6. US City, State: "Portland, OR" or "New York, NY"
        us_city_state = PatternRecognizer(
            supported_entity="LOCATION",
            patterns=[
                Pattern(
                    name="us_city_state",
                    regex=rf"\b[A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*,?\s+{us_states}\b",
                    score=0.6,
                )
            ],
        )

        # 7. US State + ZIP: "OR 97201" or "NY 10001-1234"
        us_state_zip = PatternRecognizer(
            supported_entity="LOCATION",
            patterns=[
                Pattern(
                    name="us_state_zip",
                    regex=rf"\b{us_states}\s+{us_zip}\b",
                    score=0.85,
                )
            ],
        )

        # 8. US ZIP code only: "97201" or "10001-1234"
        # Lower confidence since 5-digit numbers are common
        us_zip_only = PatternRecognizer(
            supported_entity="LOCATION",
            patterns=[
                Pattern(
                    name="us_zip_only",
                    regex=rf"\b{us_zip}\b",
                    score=0.65,  # Boosted for better recall
                ),
                # Labeled ZIP patterns: "ZIP: 12345", "Postal Code: 90210"
                Pattern(
                    name="labeled_zip",
                    regex=r"(?:ZIP|Zip|zip|Postal\s*Code|POSTAL\s*CODE|Post\s*Code)[:\s]+\d{5}(?:-\d{4})?\b",
                    score=0.95,
                ),
            ],
            context=["zip", "postal", "code", "address", "city", "state"]
        )

        # 9. US State abbreviation only (standalone)
        # Very low confidence - only use when in context of other location data
        us_state_only = PatternRecognizer(
            supported_entity="LOCATION",
            patterns=[
                Pattern(
                    name="us_state_abbrev",
                    regex=rf"\b{us_states}\b",
                    score=0.45,
                )
            ],
            context=["state", "city", "address", "location", "zip", "postal", "from", "to", "ship", "mail"]
        )

        # 10. Full US State names (standalone)
        # Detects full state names like "Maryland", "New Jersey", "California"
        us_state_names = PatternRecognizer(
            supported_entity="LOCATION",
            patterns=[
                Pattern(
                    name="us_state_full",
                    regex=r"\b(?:Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|New Hampshire|New Jersey|New Mexico|New York|North Carolina|North Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|Rhode Island|South Carolina|South Dakota|Tennessee|Texas|Utah|Vermont|Virginia|Washington|West Virginia|Wisconsin|Wyoming|District of Columbia)\b",
                    score=0.70,
                ),
            ],
            context=["state", "from", "lives in", "address", "located", "born in", "moved to", "resident"]
        )

        # 10b. US County names: "Cook County", "Santa Clara County", "Anne Arundel County"
        us_county = PatternRecognizer(
            supported_entity="LOCATION",
            patterns=[
                Pattern(
                    name="us_county",
                    # Matches: 1-3 capitalized words + "County"
                    regex=r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\s+County\b",
                    score=0.85,
                ),
            ],
            context=["county", "jurisdiction", "court", "recorder", "assessor", "clerk", "sheriff"]
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

        # 8b. US Military/Diplomatic addresses (APO/FPO/DPO)
        # Formats: "Unit 3963 Box 6057, DPO AP 50469"
        #          "PSC 5135, Box 3465, APO AA 34653"
        #          "USS Carter, FPO AP 71467"
        #          "USNS Jones, FPO AE 70161"
        military_post_codes = r"(?:APO|FPO|DPO)"
        military_regions = r"(?:AA|AE|AP)"  # Americas, Europe/Africa, Pacific
        military_address = PatternRecognizer(
            supported_entity="LOCATION",
            patterns=[
                Pattern(
                    name="military_unit_box",
                    # Unit XXXX Box XXXX, DPO/APO/FPO AA/AE/AP XXXXX
                    regex=rf"\bUnit\s+\d{{1,5}}\s+Box\s+\d{{1,5}},?\s+{military_post_codes}\s+{military_regions}\s+\d{{5}}\b",
                    score=0.95,
                ),
                Pattern(
                    name="military_psc",
                    # PSC XXXX, Box XXXX, APO/FPO AA/AE/AP XXXXX
                    regex=rf"\bPSC\s+\d{{1,5}},?\s+Box\s+\d{{1,5}},?\s+{military_post_codes}\s+{military_regions}\s+\d{{5}}\b",
                    score=0.95,
                ),
                Pattern(
                    name="military_vessel",
                    # USS/USNS/USNV/USCGC Name, FPO/APO AA/AE/AP XXXXX
                    regex=rf"\b(?:USS|USNS|USNV|USCGC)\s+[A-Z][a-zA-Z]+,?\s+{military_post_codes}\s+{military_regions}\s+\d{{5}}\b",
                    score=0.95,
                ),
                Pattern(
                    name="military_post_generic",
                    # Any text followed by DPO/APO/FPO + region + ZIP
                    regex=rf"\b{military_post_codes}\s+{military_regions}\s+\d{{5}}\b",
                    score=0.85,
                ),
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

        # 10. Standalone apartment/unit: "APT 1104", "SUITE 200", "UNIT 5A", "FL 3"
        standalone_unit = PatternRecognizer(
            supported_entity="LOCATION",
            patterns=[
                Pattern(
                    name="standalone_apt",
                    # Matches: Apt/Unit/Suite/Floor + number (with optional letter)
                    regex=r"\b(?:APT|Apt\.?|APARTMENT|Apartment|UNIT|Unit|SUITE|Suite|STE|Ste\.?|FL|Floor|FLOOR|RM|Room|ROOM|Studio|STUDIO|Cabin|CABIN|Mansion|MANSION|Lodge|LODGE|Dept|DEPT|Section|SECTION|Flat|FLAT|Maisonette|MAISONETTE|Farmhouse|FARMHOUSE|Duplex|DUPLEX|Loft|LOFT|Dorm|DORM|Barracks|BARRACKS|Townhouse|TOWNHOUSE|Penthouse|PENTHOUSE|Ranch|RANCH|Villa|VILLA|Cottage|COTTAGE|Chalet|CHALET|Bungalow|BUNGALOW|Palace|PALACE|Pod|POD|Block|BLOCK|Basement|BASEMENT|Office|OFFICE|Level|LEVEL|Wing|WING|Bay|BAY|Annex|ANNEX|Garage|GARAGE)\s+[0-9]+[A-Za-z]?\b",
                    score=0.80,
                )
            ],
            context=["address", "mail", "deliver", "ship", "building", "floor",
                      "studio", "cabin", "mansion", "lodge", "dept", "section", "flat"]
        )

        # 11. US street address with abbreviated types: "01 INDIANA AV", "123 MAIN ST"
        # Handles ALL CAPS street names and abbreviated street types
        na_abbrev_street_types = r"(?:ST|AVE?|AV|RD|BLVD|DR|LN|CT|CIR|WAY|PL|TER|PKWY|HWY|CRES)"
        street_address_uppercase = PatternRecognizer(
            supported_entity="LOCATION",
            patterns=[
                Pattern(
                    name="street_address_caps",
                    # Matches: number + optional direction + name + abbreviated type (ALL CAPS)
                    regex=rf"\b\d{{1,6}}\s+(?:[NSEW]\.?\s+)?[A-Z][A-Z]+(?:\s+[A-Z][A-Z]+){{0,3}}\s+{na_abbrev_street_types}\b",
                    score=0.85,
                )
            ],
        )

        # 12. Indian city/state patterns (major cities and states)
        indian_states = r"(?:ASSAM|BIHAR|GOA|GUJARAT|HARYANA|HIMACHAL|JHARKHAND|KARNATAKA|KERALA|MADHYA|MAHARASHTRA|MANIPUR|MEGHALAYA|MIZORAM|NAGALAND|ODISHA|PUNJAB|RAJASTHAN|SIKKIM|TAMIL|TELANGANA|TRIPURA|UTTAR|UTTARAKHAND|WEST\s+BENGAL|Assam|Bihar|Goa|Gujarat|Haryana|Karnataka|Kerala|Maharashtra|Punjab|Rajasthan|Tamil\s+Nadu|Telangana|West\s+Bengal)"
        indian_cities = r"(?:GUWAHATI|MUMBAI|DELHI|BANGALORE|BENGALURU|CHENNAI|KOLKATA|HYDERABAD|PUNE|AHMEDABAD|JAIPUR|LUCKNOW|KANPUR|NAGPUR|INDORE|THANE|BHOPAL|VISAKHAPATNAM|PATNA|VADODARA|Guwahati|Mumbai|Delhi|Bangalore|Bengaluru|Chennai|Kolkata|Hyderabad|Pune|Ahmedabad|Jaipur|Lucknow|Kanpur)"

        indian_locality = PatternRecognizer(
            supported_entity="LOCATION",
            patterns=[
                Pattern(
                    name="indian_city",
                    regex=rf"\b{indian_cities}\b",
                    score=0.70,
                ),
                Pattern(
                    name="indian_state",
                    regex=rf"\b{indian_states}\b",
                    score=0.70,
                ),
                Pattern(
                    name="indian_locality_state",
                    # Matches: locality name + comma + state (e.g., "SUNDARPARA PT IV, ASSAM")
                    regex=rf"\b[A-Z][A-Za-z\s]+(?:PT|PART|PHASE)?\s*[IVX0-9]*,?\s+{indian_states}\b",
                    score=0.80,
                ),
            ],
            context=["india", "address", "pin", "pincode", "city", "state", "district"]
        )

        # =========================================
        # INTERNATIONAL POSTAL CODE PATTERNS
        # =========================================
        # Based on official sources and ISO standards
        # https://www.geoapify.com/postcode-formats-around-the-world/

        # Japan: NNN-NNNN (7 digits with hyphen) - 〒 prefix optional
        # Source: Wikipedia Postal_codes_in_Japan
        japan_postal = PatternRecognizer(
            supported_entity="LOCATION",
            patterns=[
                Pattern(
                    name="japan_postal",
                    regex=r"\b\d{3}-\d{4}\b",
                    score=0.85,
                ),
                Pattern(
                    name="japan_postal_symbol",
                    regex=r"〒\d{3}-\d{4}",
                    score=0.95,
                ),
            ],
            context=["郵便番号", "postal", "zip", "japan", "tokyo", "osaka", "jp"]
        )

        # UK: Complex alphanumeric (outcode + incode)
        # Format: A(A)N(N) NAA or A(A)N(A) NAA
        # Source: https://www.geoapify.com/postcode-formats-around-the-world/
        uk_postcode = PatternRecognizer(
            supported_entity="LOCATION",
            patterns=[
                Pattern(
                    name="uk_postcode",
                    regex=r"(?i)\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b",
                    score=0.85,
                ),
                # Higher confidence for full UK postcode with space (very distinctive format)
                Pattern(
                    name="uk_postcode_spaced",
                    regex=r"(?i)\b[A-Z]{1,2}\d[A-Z\d]?\s\d[A-Z]{2}\b",
                    score=0.92,
                ),
            ],
            context=["postcode", "post code", "uk", "england", "scotland", "wales", "london",
                      "address", "city", "town", "county"]
        )

        # Germany: 5 digits (NNNNN)
        # Source: Official format
        germany_plz = PatternRecognizer(
            supported_entity="LOCATION",
            patterns=[
                Pattern(
                    name="germany_plz",
                    regex=r"\b[0-9]{5}\b",
                    score=0.55,  # Boosted from 0.50 - requires context for disambiguation
                ),
            ],
            context=["plz", "postleitzahl", "germany", "deutschland", "berlin", "münchen", "hamburg"]
        )

        # India: 6 digits PIN code
        # Source: Official Indian postal system
        india_pin = PatternRecognizer(
            supported_entity="LOCATION",
            patterns=[
                Pattern(
                    name="india_pin",
                    regex=r"\b[1-9]\d{5}\b",  # First digit 1-9
                    score=0.60,  # Boosted from 0.55 for better recall
                ),
                Pattern(
                    name="india_pin_labeled",
                    # PIN: 783330, Pin-123456, PIN CODE: 110001
                    regex=r"\b(?:PIN|Pin)[:\s-]+[1-9]\d{5}\b",
                    score=0.95,
                ),
                Pattern(
                    name="india_pin_with_state",
                    # 783330, ASSAM or PIN: 783330, INDIA
                    regex=r"\b[1-9]\d{5},?\s*(?:INDIA|India|ASSAM|BIHAR|DELHI|MAHARASHTRA|KARNATAKA|TAMIL\s+NADU)\b",
                    score=0.90,
                ),
            ],
            context=["pin", "pincode", "pin code", "india", "delhi", "mumbai", "bangalore"]
        )

        # Bangladesh: Country and major cities
        bangladesh_location = PatternRecognizer(
            supported_entity="LOCATION",
            patterns=[
                Pattern(
                    name="bangladesh_country",
                    regex=r"\b(?i)BANGLADESH\b",
                    score=0.80,
                ),
                Pattern(
                    name="bangladesh_cities",
                    regex=r"\b(?i)(?:DHAKA|CHITTAGONG|KHULNA|RAJSHAHI|SYLHET|COMILLA)\b",
                    score=0.70,
                ),
            ],
            context=["address", "bangladesh", "bd", "country"]
        )

        # Brazil: NNNNN-NNN (CEP format)
        # Source: Official Brazilian postal system
        brazil_cep = PatternRecognizer(
            supported_entity="LOCATION",
            patterns=[
                Pattern(
                    name="brazil_cep",
                    regex=r"\b\d{5}-\d{3}\b",
                    score=0.90,
                ),
            ],
            context=["cep", "código postal", "brazil", "brasil", "são paulo", "rio"]
        )

        # South Korea: 5 digits (current format since 2015)
        # Source: Korea Post
        korea_postal = PatternRecognizer(
            supported_entity="LOCATION",
            patterns=[
                Pattern(
                    name="korea_postal",
                    regex=r"\b\d{5}\b",
                    score=0.55,  # Boosted from 0.50 - requires context
                ),
            ],
            context=["우편번호", "postal", "korea", "seoul", "busan", "kr"]
        )

        # China: 6 digits
        # Source: China Post
        china_postal = PatternRecognizer(
            supported_entity="LOCATION",
            patterns=[
                Pattern(
                    name="china_postal",
                    regex=r"\b[1-8]\d{5}\b",  # First digit 1-8
                    score=0.55,  # Boosted from 0.50 - requires context
                ),
            ],
            context=["邮政编码", "邮编", "postal", "china", "beijing", "shanghai", "cn"]
        )

        # Russia: 6 digits
        # Source: Russian Post
        russia_postal = PatternRecognizer(
            supported_entity="LOCATION",
            patterns=[
                Pattern(
                    name="russia_postal",
                    regex=r"\b[1-9]\d{5}\b",
                    score=0.55,  # Boosted from 0.50 - requires context
                ),
            ],
            context=["почтовый индекс", "индекс", "postal", "russia", "moscow", "ru"]
        )

        # Netherlands: NNNN AA (4 digits + 2 letters)
        # Source: PostNL
        netherlands_postal = PatternRecognizer(
            supported_entity="LOCATION",
            patterns=[
                Pattern(
                    name="netherlands_postal",
                    regex=r"\b\d{4}\s?[A-Z]{2}\b",
                    score=0.85,
                ),
            ],
            context=["postcode", "netherlands", "nederland", "amsterdam", "nl"]
        )

        # Australia: 4 digits
        # Source: Australia Post
        australia_postal = PatternRecognizer(
            supported_entity="LOCATION",
            patterns=[
                Pattern(
                    name="australia_postal",
                    regex=r"\b[0-9]{4}\b",
                    score=0.45,  # Very generic
                ),
            ],
            context=["postcode", "australia", "sydney", "melbourne", "au"]
        )

        # 19. Labeled address patterns: "Address: 123 Main St", "Shipping Address: ..."
        labeled_address = PatternRecognizer(
            supported_entity="LOCATION",
            patterns=[
                Pattern(
                    name="labeled_address_simple",
                    # Handles: "Address: 123 Main St" or "Mailing Address: 100 Oak Lane"
                    regex=r"(?:Address|Addr\.?|Shipping|Mailing|Billing|Home|Work|Residential|Business)\s*(?:Address)?[:\s]+\d{1,6}\s+[A-Za-z][A-Za-z\s',.-]{3,50}",
                    score=0.90,
                ),
                Pattern(
                    name="labeled_address_general",
                    # Handles: "Address: City, State ZIP" or multi-word addresses after label
                    regex=r"(?:Address|Addr\.?)[:\s]+[A-Z][A-Za-z\s',.-]{5,60}",
                    score=0.75,
                ),
                # Address Line 1/2 patterns (common in forms)
                Pattern(
                    name="address_line_labeled",
                    regex=r"(?:Address\s*Line\s*[12]|Street\s*Address|Street\s*Line)[:\s]+[A-Za-z0-9][A-Za-z0-9\s',.-]{5,50}",
                    score=0.88,
                ),
                # ALL CAPS labeled addresses
                Pattern(
                    name="labeled_address_caps",
                    regex=r"(?:ADDRESS|STREET|MAILING|SHIPPING|BILLING)[:\s]+[A-Z0-9][A-Z0-9\s',.-]{5,50}",
                    score=0.85,
                ),
            ],
        )

        # 20. International address formats
        international_address = PatternRecognizer(
            supported_entity="LOCATION",
            patterns=[
                # UK format: "123 High Street, London SW1A 1AA"
                Pattern(
                    name="uk_full_address",
                    regex=r"\b\d{1,4}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\s+(?:Street|Road|Lane|Avenue|Drive|Close|Gardens|Square|Crescent|Terrace|Grove|Place|Mews|Court|Row|Walk),?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?,?\s*[A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2}\b",
                    score=0.95,
                ),
                # Australian format: "123 Smith Street, Melbourne VIC 3000"
                Pattern(
                    name="au_full_address",
                    regex=r"\b\d{1,4}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\s+(?:Street|Road|Lane|Avenue|Drive|Parade|Circuit),?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?,?\s*(?:NSW|VIC|QLD|SA|WA|TAS|NT|ACT)\s*\d{4}\b",
                    score=0.95,
                ),
                # Indian format: "123 MG Road, Bangalore, Karnataka 560001"
                Pattern(
                    name="india_full_address",
                    regex=r"\b\d{1,4}\s+[A-Za-z\s.-]+,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?,\s*(?:Karnataka|Maharashtra|Tamil Nadu|Delhi|Gujarat|Rajasthan|West Bengal|Uttar Pradesh|Andhra Pradesh|Telangana)\s*\d{6}\b",
                    score=0.95,
                ),
            ],
        )

        # Register all recognizers
        self.analyzer.registry.add_recognizer(labeled_address)
        self.analyzer.registry.add_recognizer(international_address)
        self.analyzer.registry.add_recognizer(canadian_address_full)
        self.analyzer.registry.add_recognizer(canadian_address_prov_postal)
        self.analyzer.registry.add_recognizer(canadian_city_prov)
        self.analyzer.registry.add_recognizer(canadian_postal_only)
        self.analyzer.registry.add_recognizer(us_address_full)
        self.analyzer.registry.add_recognizer(us_city_state)
        self.analyzer.registry.add_recognizer(us_state_zip)
        self.analyzer.registry.add_recognizer(us_zip_only)
        self.analyzer.registry.add_recognizer(us_state_only)
        self.analyzer.registry.add_recognizer(us_state_names)
        self.analyzer.registry.add_recognizer(us_county)
        self.analyzer.registry.add_recognizer(street_address_with_number)
        self.analyzer.registry.add_recognizer(street_address_no_number)
        self.analyzer.registry.add_recognizer(european_street_address)
        self.analyzer.registry.add_recognizer(po_box_address)
        self.analyzer.registry.add_recognizer(military_address)
        self.analyzer.registry.add_recognizer(unit_street_address)
        self.analyzer.registry.add_recognizer(standalone_unit)
        self.analyzer.registry.add_recognizer(street_address_uppercase)
        self.analyzer.registry.add_recognizer(indian_locality)
        # International postal codes
        self.analyzer.registry.add_recognizer(japan_postal)
        self.analyzer.registry.add_recognizer(uk_postcode)
        self.analyzer.registry.add_recognizer(germany_plz)
        self.analyzer.registry.add_recognizer(india_pin)
        self.analyzer.registry.add_recognizer(bangladesh_location)
        self.analyzer.registry.add_recognizer(brazil_cep)
        self.analyzer.registry.add_recognizer(korea_postal)
        self.analyzer.registry.add_recognizer(china_postal)
        self.analyzer.registry.add_recognizer(russia_postal)
        self.analyzer.registry.add_recognizer(netherlands_postal)
        self.analyzer.registry.add_recognizer(australia_postal)

        # =========================================
        # ENHANCED ADDRESS COMPONENT PATTERNS
        # =========================================

        # Unit designator with hash: "#101", "# 5A", "Apt #3B"
        hash_unit = PatternRecognizer(
            supported_entity="LOCATION",
            patterns=[
                Pattern(
                    name="hash_unit_number",
                    # Matches: # followed by number and optional letter
                    regex=r"(?:Apt\.?|Unit|Suite|Ste\.?|Rm\.?|Room)?\s*#\s*[0-9]+[A-Za-z]?\b",
                    score=0.75,
                ),
            ],
            context=["address", "mail", "deliver", "apartment", "unit", "suite"]
        )

        # Care of patterns: "c/o John Smith", "C/O ABC Company"
        care_of_address = PatternRecognizer(
            supported_entity="LOCATION",
            patterns=[
                Pattern(
                    name="care_of",
                    regex=r"\b[Cc]/?[Oo]\.?\s+[A-Z][A-Za-z\s]+\b",
                    score=0.80,
                ),
                Pattern(
                    name="attention",
                    regex=r"\b(?:ATTN|Attn|Attention)[:\s]+[A-Z][A-Za-z\s]+\b",
                    score=0.80,
                ),
            ],
            context=["address", "mail", "deliver", "ship", "send"]
        )

        # Rural route and highway addresses: "RR 2 Box 45", "HC 73 Box 123"
        rural_route = PatternRecognizer(
            supported_entity="LOCATION",
            patterns=[
                Pattern(
                    name="rural_route",
                    # Rural Route: RR N Box NNN
                    regex=r"\b(?:RR|R\.R\.|Rural\s+Route)\s*\d+\s*(?:Box\s*\d+)?\b",
                    score=0.85,
                ),
                Pattern(
                    name="highway_contract",
                    # Highway Contract: HC N Box NNN
                    regex=r"\b(?:HC|H\.C\.|Highway\s+Contract)\s*\d+\s*(?:Box\s*\d+)?\b",
                    score=0.85,
                ),
                Pattern(
                    name="state_route",
                    # State/County Route: State Route 45, County Road 12
                    regex=r"\b(?:State\s+(?:Route|Rd|Road|Highway|Hwy)|County\s+(?:Road|Rd)|SR|CR)\s*\d+\b",
                    score=0.75,
                ),
                Pattern(
                    name="canadian_concession",
                    # Canadian concession road: "175265 Concession 6", "Lot 5 Concession 3"
                    regex=r"\b\d{1,6}\s+Concession\s+\d{1,3}\b",
                    score=0.90,
                ),
                Pattern(
                    name="canadian_lot_concession",
                    # Lot and concession: "Lot 5, Concession 3"
                    regex=r"\bLot\s+\d{1,4},?\s+Concession\s+\d{1,3}\b",
                    score=0.90,
                ),
                Pattern(
                    name="canadian_sideroad",
                    # Sideroad/Line patterns: "123 Sideroad 5", "456 Line 10"
                    regex=r"\b\d{1,6}\s+(?:Sideroad|Side\s+Road|Line)\s+\d{1,3}\b",
                    score=0.85,
                ),
                Pattern(
                    name="canadian_range_road",
                    # Range Road: "Range Road 123", "Township Road 456"
                    regex=r"\b(?:Range|Township)\s+Road\s+\d{1,4}\b",
                    score=0.85,
                ),
            ],
        )

        # Building/floor/room patterns: "Bldg A", "Tower 2", "Wing B"
        building_patterns = PatternRecognizer(
            supported_entity="LOCATION",
            patterns=[
                Pattern(
                    name="building_wing",
                    regex=r"\b(?:Bldg\.?|Building|Tower|Wing|Block|Annex|Barracks|Attic|Ranch|Townhouse|Penthouse|Cottage|Lodge|Villa|Chalet|Cabin|Quarters|Pavilion)\s+[A-Z0-9]+\b",
                    score=0.70,
                ),
                Pattern(
                    name="floor_level",
                    regex=r"\b(?:Floor|Fl\.?|Level|Lvl\.?)\s+[0-9]+[A-Za-z]?\b",
                    score=0.70,
                ),
            ],
            context=["address", "office", "building", "located"]
        )

        # Multi-line address continuation patterns (common in PDFs)
        # Captures when city/state is on separate line from street
        address_continuation = PatternRecognizer(
            supported_entity="LOCATION",
            patterns=[
                Pattern(
                    name="city_state_country",
                    # City, State, Country or City, State Zip Country
                    regex=r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]{2,3},?\s*(?:USA|Canada|UK|Australia|United\s+States|United\s+Kingdom)\b",
                    score=0.90,
                ),
            ],
        )

        # UK compound place names: Newcastle-under-Lyme, Stow cum Quy, Burton upon Trent
        uk_compound_places = PatternRecognizer(
            supported_entity="LOCATION",
            patterns=[
                Pattern(
                    name="uk_compound_hyphenated",
                    # X-under-Y, X-upon-Y, X-on-Y, X-de-la-Z (hyphenated)
                    regex=r"\b[A-Z][a-z]+(?:-(?:under|upon|on|in|by|le|la|de|en|cum|next|super)-[A-Za-z]+)+\b",
                    score=0.85,
                ),
                Pattern(
                    name="uk_compound_spaced",
                    # X under Y, X upon Y, X cum Y (spaced, with optional "the")
                    regex=r"\b[A-Z][a-z]+\s+(?:under|upon|on|in|by|le|la|de|en|cum|next|super)\s+(?:the\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b",
                    score=0.82,
                ),
            ],
        )

        # European street prefix patterns: Via, Rue, Route, Ruta, Camino, Calle, Strasse
        european_streets = PatternRecognizer(
            supported_entity="LOCATION",
            patterns=[
                Pattern(
                    name="italian_street",
                    # Via + name: "Via Piancraiolo", "Via della Ferrovia", "Via dell'Industria"
                    regex=r"\bVia\s+(?:del(?:la|le|l['']|lo)?\s+)?[A-Z][A-Za-z]+(?:\s+[A-Za-z]+){0,3}\b",
                    score=0.85,
                ),
                Pattern(
                    name="french_street",
                    # Rue/Route/Allée/Chemin + name: "Rue des Ecoles", "Route du Lac"
                    regex=r"\b(?:Rue|Route|All[ée]e|Chemin|Boulevard|Passage|Impasse|Place)\s+(?:du|de|des|la|le|l[''])?\s*[A-Z][A-Za-z]+(?:\s+[A-Za-z]+){0,3}\b",
                    score=0.85,
                ),
                Pattern(
                    name="spanish_street",
                    # Calle/Camino/Ruta/Avenida: "Camino de los Algarbes", "Calle Mayor"
                    regex=r"\b(?:Calle|Camino|Ruta|Avenida|Paseo|Carrera|Carretera)\s+(?:de\s+(?:los?\s+|las?\s+)?)?[A-Z][A-Za-z]+(?:\s+[A-Za-z]+){0,3}\b",
                    score=0.85,
                ),
                Pattern(
                    name="german_street",
                    # Strasse/Straße/Weg/Gasse: "Hauptstrasse", "Bergweg"
                    regex=r"\b[A-Z][a-z]+(?:stra[sß]e|weg|gasse|platz|allee)\b",
                    score=0.82,
                ),
            ],
        )

        # Register enhanced address patterns
        self.analyzer.registry.add_recognizer(hash_unit)
        self.analyzer.registry.add_recognizer(care_of_address)
        self.analyzer.registry.add_recognizer(rural_route)
        self.analyzer.registry.add_recognizer(building_patterns)
        self.analyzer.registry.add_recognizer(address_continuation)
        self.analyzer.registry.add_recognizer(uk_compound_places)
        self.analyzer.registry.add_recognizer(european_streets)

        # =========================================
        # CITIES DATABASE RECOGNIZER
        # =========================================
        # Uses the cities database for major world cities detection
        if CITIES_DB_AVAILABLE:
            from presidio_analyzer import EntityRecognizer, RecognizerResult

            class CityRecognizer(EntityRecognizer):
                """
                Recognizer that uses the cities database to detect major world cities.

                This improves LOCATION detection by recognizing city names from a
                curated database of ~500 major cities worldwide.
                """

                ENTITIES = ["LOCATION"]

                def __init__(self):
                    super().__init__(
                        supported_entities=self.ENTITIES,
                        supported_language="en",
                        name="CityRecognizer",
                    )
                    self._cities_db = get_cities_db()

                def load(self) -> None:
                    """Load the cities database."""
                    pass  # Already loaded in __init__

                def analyze(self, text: str, entities, nlp_artifacts=None):
                    """Detect city names in text."""
                    results = []

                    # Look for capitalized words that could be city names
                    # Pattern: Capitalized words (possibly multi-word like "New York")
                    city_pattern = re.compile(
                        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b"
                    )

                    for match in city_pattern.finditer(text):
                        potential_city = match.group(1)

                        # Check if this is a known city
                        if self._cities_db.is_city(potential_city):
                            # Get confidence boost based on city importance
                            boost = self._cities_db.get_confidence_boost(potential_city)
                            base_score = 0.70  # Base confidence for city match

                            results.append(
                                RecognizerResult(
                                    entity_type="LOCATION",
                                    start=match.start(),
                                    end=match.end(),
                                    score=min(0.95, base_score + boost),
                                    analysis_explanation=None,
                                    recognition_metadata={
                                        "recognizer_name": "CityRecognizer",
                                        "city_name": potential_city,
                                    }
                                )
                            )

                    return results

            # Register the cities recognizer
            self.analyzer.registry.add_recognizer(CityRecognizer())

        # =========================================
        # COUNTRIES DATABASE RECOGNIZER
        # =========================================
        # Uses the countries database for country name detection
        if COUNTRIES_DB_AVAILABLE:
            from presidio_analyzer import EntityRecognizer, RecognizerResult

            class CountryRecognizer(EntityRecognizer):
                """
                Recognizer that uses the countries database to detect country names.

                This improves LOCATION detection by recognizing country names from a
                curated database of ~200 countries and their variations.
                """

                ENTITIES = ["LOCATION"]

                def __init__(self):
                    super().__init__(
                        supported_entities=self.ENTITIES,
                        supported_language="en",
                        name="CountryRecognizer",
                    )
                    self._countries_db = get_countries_database()

                def load(self) -> None:
                    """Load the countries database."""
                    pass  # Already loaded in __init__

                def analyze(self, text: str, entities, nlp_artifacts=None):
                    """Detect country names in text."""
                    results = []

                    # Use the database's find_countries_in_text method
                    countries = self._countries_db.find_countries_in_text(text)

                    for country_name, start, end, confidence in countries:
                        results.append(
                            RecognizerResult(
                                entity_type="LOCATION",
                                start=start,
                                end=end,
                                score=confidence,
                                analysis_explanation=None,
                                recognition_metadata={
                                    "recognizer_name": "CountryRecognizer",
                                    "country_name": country_name,
                                }
                            )
                        )

                    return results

            # Register the countries recognizer
            self.analyzer.registry.add_recognizer(CountryRecognizer())

    def _add_datetime_recognizers(self):
        """Add pattern recognizers for international date formats."""
        # DD/MM/YYYY format (European, UK, Australia, most of the world)
        # Examples: 04/01/1945, 12/06/2033, 31/12/2024, 1/2/1965
        date_dmy_slash = PatternRecognizer(
            supported_entity="DATE_TIME",
            patterns=[
                Pattern(
                    name="date_dmy_slash",
                    # DD/MM/YYYY - day 1-31, month 1-12, year 1900-2099
                    # Supports optional leading zeros: 01/02/1965 or 1/2/1965
                    regex=r"\b(?:0?[1-9]|[12][0-9]|3[01])/(?:0?[1-9]|1[0-2])/(?:19|20)\d{2}\b",
                    score=0.90,  # Boosted for better recall
                ),
                Pattern(
                    name="date_dmy_dash",
                    # DD-MM-YYYY with optional leading zeros
                    regex=r"\b(?:0?[1-9]|[12][0-9]|3[01])-(?:0?[1-9]|1[0-2])-(?:19|20)\d{2}\b",
                    score=0.90,  # Boosted
                ),
                Pattern(
                    name="date_dmy_dot",
                    # DD.MM.YYYY (German format) with optional leading zeros
                    regex=r"\b(?:0?[1-9]|[12][0-9]|3[01])\.(?:0?[1-9]|1[0-2])\.(?:19|20)\d{2}\b",
                    score=0.90,  # Boosted
                ),
            ],
        )

        # MM/DD/YYYY format (US style)
        date_mdy_slash = PatternRecognizer(
            supported_entity="DATE_TIME",
            patterns=[
                Pattern(
                    name="date_mdy_slash",
                    # MM/DD/YYYY - month 1-12, day 1-31, year
                    regex=r"\b(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12][0-9]|3[01])/(?:19|20)\d{2}\b",
                    score=0.85,
                ),
            ],
        )

        # Labeled date patterns: "Date: 04/01/1945", "DATE OF BIRTH: 12/06/1990"
        labeled_date = PatternRecognizer(
            supported_entity="DATE_TIME",
            patterns=[
                Pattern(
                    name="labeled_date",
                    regex=r"(?:Date|DATE|Dated?)[:\s]+(?:0?[1-9]|[12][0-9]|3[01])[/\-.](?:0?[1-9]|1[0-2])[/\-.](?:19|20)\d{2}\b",
                    score=0.95,
                ),
            ],
        )

        # YYYY/MM/DD and YYYY-MM-DD formats (ISO 8601, Asian countries)
        date_ymd = PatternRecognizer(
            supported_entity="DATE_TIME",
            patterns=[
                Pattern(
                    name="date_ymd_slash",
                    # Supports optional leading zeros
                    regex=r"\b(?:19|20)\d{2}/(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12][0-9]|3[01])\b",
                    score=0.85,
                ),
                Pattern(
                    name="date_ymd_dash",
                    # ISO 8601 date: YYYY-MM-DD (supports optional leading zeros)
                    regex=r"\b(?:19|20)\d{2}-(?:0?[1-9]|1[0-2])-(?:0?[1-9]|[12][0-9]|3[01])\b",
                    score=0.90,  # ISO 8601 format - higher confidence
                ),
                Pattern(
                    name="date_iso8601_datetime",
                    # ISO 8601 datetime: 2024-01-15T10:30:00, 2024-01-15T10:30:00Z, 2024-01-15T10:30:00+05:30
                    regex=r"\b(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])T(?:[01][0-9]|2[0-3]):[0-5][0-9](?::[0-5][0-9])?(?:\.\d+)?(?:Z|[+\-](?:[01][0-9]|2[0-3]):?[0-5][0-9])?\b",
                    score=0.95,  # ISO 8601 datetime - very high confidence
                ),
                Pattern(
                    name="date_ymd_dot",
                    # YYYY.MM.DD format (some Asian countries)
                    regex=r"\b(?:19|20)\d{2}\.(?:0?[1-9]|1[0-2])\.(?:0?[1-9]|[12][0-9]|3[01])\b",
                    score=0.85,
                ),
            ],
        )

        # Date with arrow/text prefix (OCR artifacts): "→ 13/06/2023", "LEusuR Dei → 13/06/2023"
        date_with_prefix = PatternRecognizer(
            supported_entity="DATE_TIME",
            patterns=[
                Pattern(
                    name="date_arrow_prefix",
                    # Matches dates that follow arrow or other characters
                    regex=r"→\s*(?:0[1-9]|[12][0-9]|3[01])/(?:0[1-9]|1[0-2])/(?:19|20)\d{2}\b",
                    score=0.80,
                ),
            ],
        )

        # Birth date patterns: "DOB: 04/01/1945", "Date of Birth: 12/06/1990"
        date_birth = PatternRecognizer(
            supported_entity="DATE_TIME",
            patterns=[
                Pattern(
                    name="date_birth_dob",
                    regex=r"\b(?:DOB|D\.O\.B\.?|Date\s+of\s+Birth)[:\s]+(?:0?[1-9]|[12][0-9]|3[01])[/\-.](?:0?[1-9]|1[0-2])[/\-.](?:19|20)\d{2}\b",
                    score=0.95,
                ),
            ],
            context=["birth", "born", "dob", "date of birth", "birthday"]
        )

        # Partial date patterns: "31, 2023", "January 15, 2024", "Dec 31, 2023"
        date_partial = PatternRecognizer(
            supported_entity="DATE_TIME",
            patterns=[
                # Day, Year: "31, 2023" (common in documents after month name)
                Pattern(
                    name="date_day_comma_year",
                    regex=r"\b(?:0?[1-9]|[12][0-9]|3[01]),\s*(?:19|20)\d{2}\b",
                    score=0.75,
                ),
                # Month Day, Year: "January 15, 2024", "Dec 31, 2023"
                Pattern(
                    name="date_month_day_year",
                    regex=r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+(?:0?[1-9]|[12][0-9]|3[01]),?\s*(?:19|20)\d{2}\b",
                    score=0.90,
                ),
                # Month+Day ordinal (no space): "July30th 2015", "Jan1st 2024"
                Pattern(
                    name="date_month_ordinal_attached",
                    regex=r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?(?:0?[1-9]|[12][0-9]|3[01])(?:st|nd|rd|th)\s+(?:19|20)\d{2}\b",
                    score=0.90,
                ),
            ],
        )

        # Two-digit year formats: "12/22/75", "8/21/65", "11-26-83"
        date_2digit_year = PatternRecognizer(
            supported_entity="DATE_TIME",
            patterns=[
                # MM/DD/YY with slashes (US format with 2-digit year)
                Pattern(
                    name="date_mdy_2digit_slash",
                    regex=r"\b(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12][0-9]|3[01])/\d{2}\b",
                    score=0.85,
                ),
                # DD/MM/YY with slashes (European format with 2-digit year)
                Pattern(
                    name="date_dmy_2digit_slash",
                    regex=r"\b(?:0?[1-9]|[12][0-9]|3[01])/(?:0?[1-9]|1[0-2])/\d{2}\b",
                    score=0.80,  # Lower score - ambiguous with US format
                ),
                # MM-DD-YY with dashes
                Pattern(
                    name="date_mdy_2digit_dash",
                    regex=r"\b(?:0?[1-9]|1[0-2])-(?:0?[1-9]|[12][0-9]|3[01])-\d{2}\b",
                    score=0.85,
                ),
            ],
            context=["date", "dob", "born", "birth", "birthday"]
        )

        # Birth year and "born in" patterns
        date_birth_year = PatternRecognizer(
            supported_entity="DATE_TIME",
            patterns=[
                # "born in 1992", "Born: 1985", "born 1978"
                Pattern(
                    name="born_in_year",
                    regex=r"\b[Bb]orn\s*(?:in|:)?\s*(19|20)\d{2}\b",
                    score=0.90,
                ),
                # "birthday: 8/21", "Birthday: 12/25"
                Pattern(
                    name="birthday_date",
                    regex=r"\b[Bb]irthday\s*[:=]?\s*(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12][0-9]|3[01])\b",
                    score=0.90,
                ),
                # "b. 1965", "b.1978" (abbreviation for born)
                Pattern(
                    name="born_abbrev",
                    regex=r"\bb\.\s?(19|20)\d{2}\b",
                    score=0.75,
                ),
            ],
            context=["birth", "born", "birthday", "dob"]
        )

        # Ordinal date formats: "June 25th, 2022", "March 3rd, 2015", "25th June 2022"
        date_ordinal = PatternRecognizer(
            supported_entity="DATE_TIME",
            patterns=[
                # "June 25th, 2022", "March 3rd, 2015", "December 1st, 2020"
                Pattern(
                    name="month_ordinal_year",
                    regex=r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th),?\s+(?:19|20)\d{2}\b",
                    score=0.92,
                ),
                # "25th June 2022", "3rd March 2015"
                Pattern(
                    name="ordinal_month_year",
                    regex=r"\b\d{1,2}(?:st|nd|rd|th)\s+(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+(?:19|20)\d{2}\b",
                    score=0.92,
                ),
                # "Jun 25th, 2022", "Mar 3rd, 2015" (abbreviated months)
                Pattern(
                    name="month_abbrev_ordinal_year",
                    regex=r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2}(?:st|nd|rd|th),?\s+(?:19|20)\d{2}\b",
                    score=0.90,
                ),
            ],
            context=["date", "on", "dated", "as of", "born", "dob"]
        )

        self.analyzer.registry.add_recognizer(date_dmy_slash)
        self.analyzer.registry.add_recognizer(date_mdy_slash)
        self.analyzer.registry.add_recognizer(labeled_date)
        self.analyzer.registry.add_recognizer(date_ymd)
        self.analyzer.registry.add_recognizer(date_with_prefix)
        self.analyzer.registry.add_recognizer(date_birth)
        self.analyzer.registry.add_recognizer(date_partial)
        self.analyzer.registry.add_recognizer(date_2digit_year)
        self.analyzer.registry.add_recognizer(date_birth_year)
        self.analyzer.registry.add_recognizer(date_ordinal)

        # Standalone time patterns (work regardless of dateparser availability)
        time_patterns = PatternRecognizer(
            supported_entity="DATE_TIME",
            patterns=[
                Pattern(
                    name="time_european_h",
                    # European format: 10h30, 14h00, 8h45
                    regex=r"\b([01]?\d|2[0-3])h[0-5]\d\b",
                    score=0.80,
                ),
                Pattern(
                    name="time_24h_standalone",
                    # 24-hour time: 14:30, 08:45, 23:59 (not already in dateparser)
                    regex=r"\b([01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?\b",
                    score=0.75,
                ),
                Pattern(
                    name="time_12h_ampm",
                    # 12-hour time with AM/PM: 2:45 PM, 11:30 am
                    regex=r"\b(1[0-2]|0?[1-9]):[0-5]\d\s*[APap]\.?[Mm]\.?\b",
                    score=0.85,
                ),
                Pattern(
                    name="time_hour_ampm",
                    # Standalone hour with AM/PM: 5 AM, 10 PM, 2 pm
                    regex=r"\b(1[0-2]|0?[1-9])\s+[APap]\.?[Mm]\.?\b",
                    score=0.75,
                ),
                Pattern(
                    name="time_oclock",
                    # N o'clock: 3 o'clock, 15 o'clock
                    regex=r"\b\d{1,2}\s+o'clock\b",
                    score=0.80,
                ),
                Pattern(
                    name="date_month_slash_yy",
                    # Month/YY format: February/44, August/42, May/64
                    regex=r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)/\d{1,2}\b",
                    score=0.80,
                ),
            ],
            context=["time", "at", "from", "to", "until", "schedule", "appointment",
                     "meeting", "call", "pm", "am", "hour", "clock", "date", "birth"]
        )
        self.analyzer.registry.add_recognizer(time_patterns)

        # Add dateparser-based recognizer for natural language dates
        # This catches formats that regex misses: "Jul 13, 2009", "13th July 2009", "13/07/2009" (European)
        if DATEPARSER_AVAILABLE:
            from presidio_analyzer import EntityRecognizer, RecognizerResult, AnalysisExplanation

            class DateparserRecognizer(EntityRecognizer):
                """Custom recognizer using dateparser for natural language date detection."""

                ENTITIES = ["DATE_TIME"]
                # Patterns to find date-like text for dateparser validation
                DATE_CANDIDATE_PATTERNS = [
                    # Time patterns: 14:30, 2:45 PM, 18:35:00
                    re.compile(r'\b([01]?\d|2[0-3]):[0-5]\d(:[0-5]\d)?(\s*[APap][Mm])?\b'),
                    # 12-hour time: 3:45 PM, 11:30 AM
                    re.compile(r'\b(1[0-2]|0?[1-9]):[0-5]\d\s*[APap]\.?[Mm]\.?\b'),
                    # Month name patterns: "Jul 13, 2009", "13th July 2009", "July 13"
                    re.compile(r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[.,]?\s+\d{1,2}(?:st|nd|rd|th)?[.,]?\s*\d{2,4}?\b', re.IGNORECASE),
                    re.compile(r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[.,]?\s*\d{2,4}?\b', re.IGNORECASE),
                ]

                def __init__(self):
                    super().__init__(
                        supported_entities=self.ENTITIES,
                        supported_language="en",
                        name="DateparserRecognizer",
                    )

                def load(self):
                    pass

                def analyze(self, text, entities, nlp_artifacts=None):
                    results = []
                    seen_spans = set()

                    for pattern in self.DATE_CANDIDATE_PATTERNS:
                        for match in pattern.finditer(text):
                            candidate = match.group(0)
                            start, end = match.start(), match.end()

                            # Skip if we've already processed this span
                            if (start, end) in seen_spans:
                                continue

                            # Validate with dateparser
                            try:
                                parsed = dateparser.parse(candidate, settings={
                                    'STRICT_PARSING': True,
                                    'PREFER_DATES_FROM': 'past',
                                })
                                if parsed:
                                    seen_spans.add((start, end))
                                    # Determine confidence based on pattern type
                                    if ':' in candidate:  # Time pattern
                                        score = 0.80
                                    else:  # Date pattern
                                        score = 0.85

                                    explanation = AnalysisExplanation(
                                        recognizer=self.name,
                                        original_score=score,
                                        pattern_name="dateparser_natural",
                                        pattern=None,
                                        validation_result=None,
                                    )
                                    results.append(
                                        RecognizerResult(
                                            entity_type="DATE_TIME",
                                            start=start,
                                            end=end,
                                            score=score,
                                            analysis_explanation=explanation,
                                        )
                                    )
                            except Exception:
                                pass  # Skip invalid dates

                    return results

            self.analyzer.registry.add_recognizer(DateparserRecognizer())

    def _add_age_recognizers(self):
        """Add pattern recognizers for age detection.

        Detects age in various formats:
        - "15 years old", "75 yrs old", "25 y/o"
        - "Age: 45", "AGE 30", "age=25"
        - "aged 30", "Aged: 55"
        - "Years: 75", "years old: 42"
        """
        age_recognizer = PatternRecognizer(
            supported_entity="AGE",
            patterns=[
                # "X years old", "X yrs old", "X y/o"
                Pattern(
                    name="age_years_old",
                    regex=r"\b(\d{1,3})\s*(?:years?\s*old|yrs?\s*old|y/?o)\b",
                    score=0.95,
                ),
                # "Age: X", "AGE X", "age=X", "age - X"
                Pattern(
                    name="age_labeled",
                    regex=r"\b[Aa][Gg][Ee]\s*[:=\-]?\s*(\d{1,3})\b",
                    score=0.90,
                ),
                # "aged X", "Aged: X"
                Pattern(
                    name="age_aged",
                    regex=r"\b[Aa]ged\s*[:=]?\s*(\d{1,3})\b",
                    score=0.90,
                ),
                # "Years: X" (in forms/tables)
                Pattern(
                    name="age_years_label",
                    regex=r"\b[Yy]ears?\s*[:=]\s*(\d{1,3})\b",
                    score=0.75,
                ),
                # "X-year-old", "X year old" (as adjective)
                Pattern(
                    name="age_hyphenated",
                    regex=r"\b(\d{1,3})[\-\s]year[\-\s]old\b",
                    score=0.90,
                ),
            ],
            context=["age", "aged", "years", "old", "born", "birthday", "dob"]
        )

        # Gender-specific age patterns: "M28", "43F", "28m", "78M/32F"
        age_gender = PatternRecognizer(
            supported_entity="AGE",
            patterns=[
                # "28M", "43F", "28m", "43f" (age followed by gender)
                Pattern(
                    name="age_gender_suffix",
                    regex=r"\b\d{1,2}[MFmf]\b",
                    score=0.85,
                ),
                # "M28", "F43" (gender followed by age)
                Pattern(
                    name="gender_age_prefix",
                    regex=r"\b[MFmf]\d{1,2}\b",
                    score=0.85,
                ),
                # "78M/32F", "25M/24F" (couples/pairs)
                Pattern(
                    name="age_couple",
                    regex=r"\b\d{1,2}[MFmf]/\d{1,2}[MFmf]\b",
                    score=0.90,
                ),
                # "i'm 55", "I'm 28", "I am 35"
                Pattern(
                    name="i_am_age",
                    regex=r"\b[Ii](?:'m|'m| am)\s+\d{1,2}\b",
                    score=0.85,
                ),
                # "(M, 28)", "(F, 43)", "(28, M)", "(43, F)"
                Pattern(
                    name="age_gender_parens",
                    regex=r"\([MFmf],?\s*\d{1,2}\)|\(\d{1,2},?\s*[MFmf]\)",
                    score=0.85,
                ),
            ],
            context=["age", "male", "female", "gender", "dating", "m/f", "yo"]
        )

        self.analyzer.registry.add_recognizer(age_recognizer)
        self.analyzer.registry.add_recognizer(age_gender)

    def _add_financial_recognizers(self):
        """Add pattern recognizers for financial data (currency, SWIFT codes, etc.)."""
        # Generic currency regex: symbol followed by numbers (with or without commas)
        # Supports $, £, €, ¥, ₹, ₽ and handles optional space and sign
        # Handles: $1234.56, $1,234.56, €3750.25, ¥450000, $ 202.02, -$500.00, + $1,234.00
        # Currency with symbol: $100, $1,234.56, $ 202.02, -$500, ($500.00)
        # Improved to handle: space after symbol, any digit count, optional decimals (1-2 places)
        currency_regex = r"[+\-]?\s?[\$£€¥₹₽]\s?[+\-]?\s?\d+(?:,\d{3})*(?:\.\d{1,2})?\b"

        # Parenthetical negative amounts: ($500.00), (€1,234.56)
        paren_negative_regex = r"\([\$£€¥₹₽]\s?\d+(?:,\d{3})*(?:\.\d{1,2})?\)"

        # DISABLED: Signed amounts pattern causes too many FPs with table formatting
        # Handles: - 1,027.00, + 1,749.00, -500.00, -500, +1000 (decimals now optional)
        # signed_amount_regex = r"\b[+\-]\s?\d+(?:,\d{3})*(?:\.\d{1,2})?\b"

        # Bare currency amounts: $0, $100, $452, $ 5, €50
        # Improved to match any amount including single digit and space after symbol
        bare_currency_regex = r"[\$£€¥₹₽]\s?\d+(?:,\d{3})*(?:\.\d{1,2})?\b"

        # US Routing number (9 digits with label)
        routing_number_regex = r"\b(?:Routing\s*(?:Number|No\.?|#)?)[:\s]+\d{9}\b"

        # Word-based currency regex: USD, CAD, EUR, INR, etc. followed by numbers
        # Handles signed amounts: USD -500.00, EUR +1,234.00, USD 0, CAD 50
        word_currency_regex = r"\b(?:USD|CAD|EUR|GBP|JPY|AUD|CNY|INR|CHF|SGD|HKD|NZD|SEK|NOK|DKK|MXN|BRL|ZAR|KRW|THB|MYR|PHP|IDR|VND)\s?[+\-]?\s?\d+(?:,\d{3})*(?:\.\d{1,2})?\b"

        # Indian currency notation: "INR 500 Crores", "Rs. 10 Lakhs", "₹50 Crore"
        # Lakhs = 100,000, Crores = 10,000,000
        indian_currency_regex = r"\b(?:INR|Rs\.?|₹)\s?\d{1,3}(?:,?\d{2,3})*(?:\.\d{1,2})?\s*(?:Crores?|Lakhs?|Cr\.?|L)\b"

        # SWIFT/BIC code regex: 8 or 11 characters
        # Format: AAAA BB CC DDD (bank code + country + location + optional branch)
        # Comprehensive ISO 3166-1 alpha-2 country codes
        country_codes = (
            "AD|AE|AF|AG|AI|AL|AM|AO|AQ|AR|AS|AT|AU|AW|AX|AZ|"
            "BA|BB|BD|BE|BF|BG|BH|BI|BJ|BL|BM|BN|BO|BQ|BR|BS|BT|BV|BW|BY|BZ|"
            "CA|CC|CD|CF|CG|CH|CI|CK|CL|CM|CN|CO|CR|CU|CV|CW|CX|CY|CZ|"
            "DE|DJ|DK|DM|DO|DZ|EC|EE|EG|EH|ER|ES|ET|FI|FJ|FK|FM|FO|FR|"
            "GA|GB|GD|GE|GF|GG|GH|GI|GL|GM|GN|GP|GQ|GR|GS|GT|GU|GW|GY|"
            "HK|HM|HN|HR|HT|HU|ID|IE|IL|IM|IN|IO|IQ|IR|IS|IT|JE|JM|JO|JP|"
            "KE|KG|KH|KI|KM|KN|KP|KR|KW|KY|KZ|LA|LB|LC|LI|LK|LR|LS|LT|LU|LV|LY|"
            "MA|MC|MD|ME|MF|MG|MH|MK|ML|MM|MN|MO|MP|MQ|MR|MS|MT|MU|MV|MW|MX|MY|MZ|"
            "NA|NC|NE|NF|NG|NI|NL|NO|NP|NR|NU|NZ|OM|PA|PE|PF|PG|PH|PK|PL|PM|PN|PR|PS|PT|PW|PY|"
            "QA|RE|RO|RS|RU|RW|SA|SB|SC|SD|SE|SG|SH|SI|SJ|SK|SL|SM|SN|SO|SR|SS|ST|SV|SX|SY|SZ|"
            "TC|TD|TF|TG|TH|TJ|TK|TL|TM|TN|TO|TR|TT|TV|TW|TZ|UA|UG|UM|US|UY|UZ|"
            "VA|VC|VE|VG|VI|VN|VU|WF|WS|YE|YT|ZA|ZM|ZW"
        )
        swift_regex = rf"\b[A-Z]{{4}}(?:{country_codes})[A-Z0-9]{{2}}(?:[A-Z0-9]{{3}})?\b"

        # Bank code regex: "Bank code: 7214" or "Bank Code: 1234567"
        bank_code_regex = r"\b[Bb]ank\s+[Cc]ode[:\s]+\d{3,10}\b"

        # Branch code regex: "Branch code: 001" or "Branch Code: 12345"
        branch_code_regex = r"\b[Bb]ranch\s+[Cc]ode[:\s]+\d{3,10}\b"

        # Labeled SWIFT/BIC: "SWIFT: HASEHKHHXXX", "BIC: DEUTDEFF"
        labeled_swift_regex = rf"\b(?:SWIFT|BIC)\s*[:=]\s*[A-Z]{{4}}(?:{country_codes})[A-Z0-9]{{2}}(?:[A-Z0-9]{{3}})?\b"

        currency_recognizer = PatternRecognizer(
            supported_entity="FINANCIAL",
            patterns=[
                Pattern(
                    name="currency_symbol",
                    regex=currency_regex,
                    score=0.65,  # Reduced from 0.8 for better precision
                ),
                # REMOVED: signed_amount pattern causes too many FPs with table formatting
                # like "- 11", "+ 50", bullet points, and list items
                # Pattern(
                #     name="signed_amount",
                #     regex=signed_amount_regex,
                #     score=0.70,
                # ),
                Pattern(
                    name="bare_currency",
                    regex=bare_currency_regex,
                    score=0.75,  # Simple currency amounts like $0, $100, $452
                ),
                Pattern(
                    name="paren_negative",
                    regex=paren_negative_regex,
                    score=0.80,  # Parenthetical negative: ($500.00), (€1,234)
                ),
                Pattern(
                    name="routing_number",
                    regex=routing_number_regex,
                    score=0.85,  # US bank routing numbers
                ),
                Pattern(
                    name="currency_code",
                    regex=word_currency_regex,
                    score=0.8,
                ),
                Pattern(
                    name="swift_bic_code",
                    regex=swift_regex,
                    score=0.70,  # Reduced from 0.85 for better precision
                ),
                Pattern(
                    name="bank_code",
                    regex=bank_code_regex,
                    score=0.85,
                ),
                Pattern(
                    name="branch_code",
                    regex=branch_code_regex,
                    score=0.85,
                ),
                Pattern(
                    name="indian_currency",
                    regex=indian_currency_regex,
                    score=0.85,
                ),
                Pattern(
                    name="labeled_swift_bic",
                    regex=labeled_swift_regex,
                    score=0.95,  # Higher score for labeled codes
                )
            ],
            context=["swift", "bic", "bank", "transfer", "wire", "iban", "routing", "branch", "inr", "rupee", "crore", "lakh", "amount", "balance", "total", "credit", "debit"]
        )

        # IBAN (International Bank Account Number)
        # Format: 2-letter country code + 2 check digits + up to 30 alphanumeric BBAN
        # Country-specific lengths vary from 15 (Norway) to 34 (Malta) characters
        # Source: ISO 13616 / ECBS standard

        # Common IBAN country codes and their lengths
        # DE: 22, FR: 27, GB: 22, ES: 24, IT: 27, NL: 18, BE: 16, AT: 20, CH: 21
        # Each country has specific BBAN structure

        iban_recognizer = PatternRecognizer(
            supported_entity="IBAN_CODE",  # Use IBAN_CODE to match Presidio's built-in entity type
            patterns=[
                # Standard IBAN with spaces (DE89 3704 0044 0532 0130 00)
                Pattern(
                    name="iban_spaced",
                    regex=r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){2,8}\b",
                    score=0.90,
                ),
                # Compact IBAN without spaces (DE89370400440532013000)
                Pattern(
                    name="iban_compact",
                    regex=r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b",
                    score=0.85,
                ),
                # Common European IBANs with known country prefixes
                # Germany: DE + 20 chars
                Pattern(
                    name="iban_de",
                    regex=r"\bDE\d{2}\s?(?:\d{4}\s?){4}\d{2}\b",
                    score=0.95,
                ),
                # UK: GB + 20 chars (2 check + 4 bank + 6 sort + 8 account)
                Pattern(
                    name="iban_gb",
                    regex=r"\bGB\d{2}\s?[A-Z]{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{2}\b",
                    score=0.95,
                ),
                # France: FR + 25 chars
                Pattern(
                    name="iban_fr",
                    regex=r"\bFR\d{2}\s?(?:\d{4}\s?){5}\d{3}[A-Z0-9]\b",
                    score=0.95,
                ),
                # Netherlands: NL + 16 chars
                Pattern(
                    name="iban_nl",
                    regex=r"\bNL\d{2}\s?[A-Z]{4}\s?\d{4}\s?\d{4}\s?\d{2}\b",
                    score=0.95,
                ),
                # Spain: ES + 22 chars
                Pattern(
                    name="iban_es",
                    regex=r"\bES\d{2}\s?(?:\d{4}\s?){5}\b",
                    score=0.95,
                ),
                # Italy: IT + 25 chars
                Pattern(
                    name="iban_it",
                    regex=r"\bIT\d{2}\s?[A-Z]\d{3}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{3}\b",
                    score=0.95,
                ),
            ],
            context=["iban", "bank account", "account number", "wire transfer",
                     "sepa", "swift", "bic", "bank", "international"]
        )

        # Labeled bank account numbers: "Beneficiary account number: 1030 1784 7086 1"
        # Captures account numbers that follow descriptive labels
        labeled_bank_account = PatternRecognizer(
            supported_entity="BANK_NUMBER",
            patterns=[
                Pattern(
                    name="labeled_account_number",
                    # Matches: Label + colon + number (with spaces/dashes)
                    regex=r"(?:account\s*(?:number|no\.?|#)?|beneficiary|routing|sort\s*code)[:\s]+[\d\s\-]{8,20}",
                    score=0.85,
                ),
                Pattern(
                    name="account_number_grouped",
                    # Matches: groups of digits like "1030 1784 7086 1"
                    regex=r"\b\d{4}\s\d{4}\s\d{4}(?:\s\d{1,4})?\b",
                    score=0.70,
                ),
            ],
            context=["account", "beneficiary", "bank", "transfer", "routing", "sort code", "wire"]
        )

        # Canadian Business Number (BN)
        # Format: 9-digit BN + program identifier (RT, RP, RC, RR, etc.) + 4-digit reference
        # Example: "808921738 RT0001", "123456789RT0001", "808 921 738 RT 0001"
        # RT = GST/HST, RP = Payroll, RC = Corporate Income Tax, RR = Registered Charity
        # RZ = Information Returns, RM = Import/Export
        canadian_bn_recognizer = PatternRecognizer(
            supported_entity="FINANCIAL",
            patterns=[
                Pattern(
                    name="canadian_bn_spaced",
                    # 9 digits (with optional spaces) + program ID + 4 digits
                    regex=r"\b\d{3}\s?\d{3}\s?\d{3}\s?(?:RT|RP|RC|RR|RZ|RM)\s?\d{4}\b",
                    score=0.90,
                ),
                Pattern(
                    name="canadian_bn_compact",
                    # 9 digits + program ID + 4 digits (no spaces)
                    regex=r"\b\d{9}(?:RT|RP|RC|RR|RZ|RM)\d{4}\b",
                    score=0.90,
                ),
                Pattern(
                    name="canadian_bn_labeled",
                    # With label: "BN: 808921738 RT0001", "Business Number: 123456789RT0001"
                    regex=r"(?:BN|Business\s+Number|GST/HST|GST|HST)\s*[:#]?\s*\d{9}\s?(?:RT|RP|RC|RR|RZ|RM)?\s?\d{0,4}",
                    score=0.95,
                ),
            ],
            context=["business number", "bn", "gst", "hst", "cra", "canada revenue", "tax", "payroll", "employer"]
        )

        self.analyzer.registry.add_recognizer(currency_recognizer)
        self.analyzer.registry.add_recognizer(iban_recognizer)
        self.analyzer.registry.add_recognizer(labeled_bank_account)
        self.analyzer.registry.add_recognizer(canadian_bn_recognizer)

    def _add_person_recognizers(self):
        """Add recognizers for person names using smart cascade NER.

        PERSON detection strategy (via PersonRecognizer):
        1. Pattern matching: Title + Name patterns (Dr. Smith), labeled names (Name: John)
        2. Dictionary lookup: name-dataset for contextless names (spreadsheets)
        3. Standard NER: spaCy for fast, reliable detection
        4. Advanced NER: GLiNER/Flair for ambiguous names (when available)

        Falls back to pattern-only mode if NER engines unavailable.
        Respects DEFAULT_INTEGRATIONS config for heavy model loading.
        """
        try:
            from .person_recognizer import get_person_recognizer, is_person_ner_available
            try:
                from hush_engine.detection_config import DEFAULT_INTEGRATIONS
            except ImportError:
                from ..detection_config import DEFAULT_INTEGRATIONS

            # Check config to determine mode
            # Use "accurate" only if heavy models are explicitly enabled
            use_heavy = (
                DEFAULT_INTEGRATIONS.get("gliner", False) or
                DEFAULT_INTEGRATIONS.get("flair", False) or
                DEFAULT_INTEGRATIONS.get("transformers", False)
            )
            mode = "accurate" if use_heavy else "balanced"

            person_recognizer = get_person_recognizer(mode=mode)
            self.analyzer.registry.add_recognizer(person_recognizer)

            # Log which mode was used (don't call is_person_ner_available() as it loads heavy models)
            import sys
            sys.stderr.write(f"[PIIDetector] PersonRecognizer loaded (mode={mode})\n")

        except ImportError:
            # Fallback to basic pattern recognizers if person_recognizer module unavailable
            import sys
            sys.stderr.write("[PIIDetector] PersonRecognizer unavailable, using basic patterns\n")
            self._add_person_pattern_recognizers()

    def _add_person_pattern_recognizers(self):
        """Fallback: Basic pattern recognizers for person names.

        Used when PersonRecognizer module is unavailable.
        """
        # Titles that indicate a person name follows
        titles = r"(?:Dr|DR|dr|Mr|MR|mr|Mrs|MRS|mrs|Ms|MS|ms|Miss|MISS|Prof|PROF|prof|Professor|Rev|REV|rev|Reverend|Sr|JR|Jr|Esq|Hon|Capt|Col|Gen|Lt|Maj|Sgt)"

        # HIGH CONFIDENCE: Title + Name patterns
        person_recognizer = PatternRecognizer(
            supported_entity="PERSON",
            patterns=[
                Pattern(
                    name="title_name",
                    regex=rf"\b{titles}\.?\s+[A-Za-z][A-Za-z]+(?:\s+[A-Za-z][A-Za-z]+)?\b",
                    score=0.90,
                ),
                Pattern(
                    name="title_full_name",
                    regex=rf"\b{titles}\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b",
                    score=0.95,
                ),
            ],
        )
        self.analyzer.registry.add_recognizer(person_recognizer)

        # CONTEXT-REQUIRED: Labeled name patterns
        labeled_name_recognizer = PatternRecognizer(
            supported_entity="PERSON",
            patterns=[
                Pattern(
                    name="labeled_name",
                    regex=r"(?:(?:First|Last|Full|Middle|Given|Family|Sur|Patient|Client|Customer|Applicant|Cardholder|Account\s*Holder|Beneficiary|Employee|Contact)\s*)?Name\s*[:]\s*[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2}",
                    score=0.90,
                ),
                Pattern(
                    name="labeled_person",
                    regex=r"(?:Patient|Client|Applicant|Cardholder|Beneficiary|Employee|Recipient|Sender|Owner|Holder)\s*[:]\s*[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2}",
                    score=0.85,
                ),
            ],
        )
        self.analyzer.registry.add_recognizer(labeled_name_recognizer)

    def _remove_default_person_recognizers(self):
        """Remove Presidio's default SpacyRecognizer for PERSON to avoid duplicate detection.

        We use our custom PersonRecognizer which has better precision through
        multi-engine cascade and confidence scoring. The default SpacyRecognizer
        adds many false positives.
        """
        try:
            # Get all recognizers and find default spaCy-based ones that detect PERSON
            recognizers_to_remove = []
            for recognizer in self.analyzer.registry.recognizers:
                # Skip our custom PersonRecognizer
                if recognizer.name == "PersonRecognizer":
                    continue
                # Check if this recognizer detects PERSON
                if hasattr(recognizer, 'supported_entities'):
                    supported = recognizer.supported_entities
                    if "PERSON" in supported:
                        # Only remove SpacyRecognizer, not our pattern recognizers
                        if "Spacy" in recognizer.name or recognizer.name == "NlpRecognizer":
                            recognizers_to_remove.append(recognizer)

            # Remove the duplicate recognizers
            for recognizer in recognizers_to_remove:
                try:
                    self.analyzer.registry.recognizers.remove(recognizer)
                    import sys
                    sys.stderr.write(f"[PIIDetector] Removed default {recognizer.name} for PERSON\n")
                except ValueError:
                    pass  # Already removed

        except Exception as e:
            import sys
            sys.stderr.write(f"[PIIDetector] Warning: Could not remove default PERSON recognizers: {e}\n")

    def _remove_default_ssn_recognizers(self):
        """Remove Presidio's built-in US_SSN recognizer.

        We use our custom SSN recognizer which emits NATIONAL_ID for unified ID handling.
        The default US_SSN recognizer causes duplicate detections and inconsistent entity types.
        """
        try:
            recognizers_to_remove = []
            for recognizer in self.analyzer.registry.recognizers:
                if hasattr(recognizer, 'supported_entities'):
                    if "US_SSN" in recognizer.supported_entities:
                        recognizers_to_remove.append(recognizer)

            for recognizer in recognizers_to_remove:
                try:
                    self.analyzer.registry.recognizers.remove(recognizer)
                except ValueError:
                    pass  # Already removed
        except Exception:
            pass

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

        # Business suffix words (common company name endings)
        business_suffixes = [
            r"Partners", r"Partnership",
            r"Associates", r"Association",
            r"Group", r"Holdings",
            r"Agency", r"Agencies",
            r"Studio", r"Studios", r"Design",
            r"Foundation", r"Trust", r"Alliance", r"Institute", r"Society",
            r"Solutions", r"Services",
            r"Management", r"Consulting",
            r"International", r"Worldwide", r"Global",
            r"Brands", r"Products",
            r"Systems", r"Technologies", r"Tech",
            r"Research", r"Pharmaceuticals", r"Biotechnologies", r"Biotech",
            r"Entertainment", r"Media",
            r"Realty", r"Properties", r"Investments",
            r"Clinic", r"Hospital", r"Medical",
            r"Law\s+Firm", r"Legal",
            r"House", r"Press", r"Publishing",
            # Additional business indicators from ground truth
            r"Logistics", r"Analytics", r"Labs", r"Laboratory",
            r"Ventures", r"Capital", r"Finance", r"Financial",
            r"Insurance", r"Energy", r"Power", r"Electric",
            r"Automotive", r"Motors", r"Auto",
            r"Airlines", r"Airways", r"Transport", r"Transportation",
            r"Network", r"Networks", r"Communications", r"Telecom",
        ]
        business_suffix_pattern = "|".join(business_suffixes)

        # Company name regex: Capitalized words followed by a legal designation
        # Matches "Alleles Company Ltd.", "Apple Inc.", "Siemens AG", etc.
        company_legal_regex = rf"\b[A-Z][a-zA-Z0-9&',.-]+(?:\s+[A-Z][a-zA-Z0-9&',.-]+)*\s+(?:{designations_pattern})\b"

        # Company name with business suffix: "Johnson & Partners", "Davis Creative Agency"
        company_suffix_regex = rf"\b[A-Z][a-zA-Z0-9&',.-]+(?:\s+[A-Z&][a-zA-Z0-9&',.-]+)*\s+(?:{business_suffix_pattern})\b"

        # Hyphenated company names (e.g., "Jackson-Guzman", "Hewlett-Packard")
        # Two capitalized words connected by hyphen - reduced score to avoid false positives
        hyphenated_company_regex = r"\b[A-Z][a-z]+(?:-[A-Z][a-z]+)+\b"

        # Company names with ampersand (e.g., "Johnson & Johnson", "Ernst & Young")
        ampersand_company_regex = r"\b[A-Z][a-z]+\s*&\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"

        # Multi-name company format: "Name, Name and Name" (e.g., "Nguyen, Turner and Mcgee")
        # This is a very common pattern in the training data (41% of companies)
        multi_name_company_regex = r"\b[A-Z][a-z]+,\s+[A-Z][a-z]+\s+and\s+[A-Z][a-z]+\b"

        # "Name and Sons/Brothers/Daughters" pattern (e.g., "Mccann and Sons", "Martin and Sons")
        # Common Faker company format for family businesses
        and_family_company_regex = r"\b[A-Z][a-z]+\s+and\s+(?:Sons|Brothers|Daughters|Sisters|Family)\b"

        # Hyphenated company with context: "at Name-Name", "Name-Name headquarters"
        # Higher score than bare hyphenated pattern because context confirms company usage
        # Uses lookbehind to capture only the company name, not the preposition
        hyphenated_context_company_regex = r"(?:(?<=\bat\s)|(?<=\bfor\s)|(?<=\bby\s)|(?<=\bfrom\s)|(?<=\bwith\s))[A-Z][a-z]+-[A-Z][a-z]+\b|[A-Z][a-z]+-[A-Z][a-z]+(?=\s+headquarters\b)"

        # Company names after context keywords (e.g., "at Jackson-Guzman", "for Andrade LLC")
        # Match capitalized phrase following company context
        context_company_regex = r"(?:at|for|by|from|to|of|with)\s+([A-Z][a-zA-Z]+(?:[-\s][A-Z][a-zA-Z]+)*)"

        # Canadian numbered corporations: "2378238 Ontario Inc.", "1234567 B.C. Ltd."
        # Format: 6-7 digit number + Canadian province/territory + legal designation
        canadian_provinces = r"(?:Ontario|Quebec|Québec|Alberta|British\s+Columbia|B\.?C\.?|Manitoba|Saskatchewan|Nova\s+Scotia|New\s+Brunswick|Newfoundland|Prince\s+Edward\s+Island|P\.?E\.?I\.?|Northwest\s+Territories|N\.?W\.?T\.?|Yukon|Nunavut|Canada)"
        canadian_numbered_corp_regex = rf"\b\d{{6,7}}\s+{canadian_provinces}\s+(?:{designations_pattern})\b"

        # Numeric prefix companies (3M, 7-Eleven, 6 & 10 Euro Market)
        numeric_prefix_company_regex = r"\b\d+(?:\s*[-&]\s*\d+)?(?:\s+[A-Z][a-zA-Z]+)+(?:\s+(?:Company|Inc|LLC|Ltd|Corp|Group|Market|Store|Shop|Center|Centre))?\b"

        # "The X Company" pattern - "The ABC Company", "The Acme Corporation"
        the_company_regex = rf"\b[Tt]he\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\s+(?:{designations_pattern}|Company|Corporation|Group|Foundation)\b"

        # ALL CAPS company names with suffixes: "ALLELES DESIGN", "ACME CORP"
        all_caps_company_regex = rf"\b[A-Z]{{2,}}(?:\s+[A-Z]{{2,}})*\s+(?:{designations_pattern.upper()}|COMPANY|CORPORATION|GROUP|SOLUTIONS|SERVICES|INTERNATIONAL|ENTERTAINMENT|MEDIA|TECHNOLOGIES|INDUSTRIES|SYSTEMS|PARTNERS|ASSOCIATES|CONSULTING|LOGISTICS|VENTURES|CAPITAL|HOLDINGS)\b"

        # CamelCase company names (common in tech): "VeriTrust", "SwiftFlow", "TechCorp"
        # Must have at least 2 capital letters to avoid matching regular words
        camelcase_company_regex = r"\b[A-Z][a-z]+[A-Z][a-z]+(?:[A-Z][a-z]+)*\b"

        # Labeled company: "Company: ABC Corp", "Employer: XYZ Ltd"
        labeled_company_regex = rf"\b(?:Company|Employer|Organization|Business|Firm|Client|Vendor)\s*[:\-]\s*([A-Z][a-zA-Z0-9&',.\s-]+(?:{designations_pattern}))"

        company_recognizer = PatternRecognizer(
            supported_entity="COMPANY",
            patterns=[
                Pattern(
                    name="company_legal",
                    regex=company_legal_regex,
                    score=0.85,
                ),
                Pattern(
                    name="company_suffix",
                    regex=company_suffix_regex,
                    score=0.75,
                ),
                Pattern(
                    name="company_multi_name",
                    regex=multi_name_company_regex,
                    score=0.85,  # High score - very reliable pattern
                ),
                Pattern(
                    name="company_and_family",
                    regex=and_family_company_regex,
                    score=0.85,  # High score - "X and Sons" is very reliable
                ),
                Pattern(
                    name="company_hyphenated_context",
                    regex=hyphenated_context_company_regex,
                    score=0.80,  # High score - context confirms company usage
                ),
                Pattern(
                    name="company_hyphenated",
                    regex=hyphenated_company_regex,
                    score=0.40,  # Reduced from 0.55 - too many false positives
                ),
                Pattern(
                    name="company_ampersand",
                    regex=ampersand_company_regex,
                    score=0.70,
                ),
                Pattern(
                    name="canadian_numbered_corp",
                    regex=canadian_numbered_corp_regex,
                    score=0.90,  # High score - very specific pattern
                ),
                Pattern(
                    name="numeric_prefix_company",
                    regex=numeric_prefix_company_regex,
                    score=0.70,  # Companies like 3M, 7-Eleven
                ),
                Pattern(
                    name="the_company",
                    regex=the_company_regex,
                    score=0.80,  # "The ABC Company" pattern
                ),
                Pattern(
                    name="all_caps_company",
                    regex=all_caps_company_regex,
                    score=0.75,  # ALL CAPS company names
                ),
                Pattern(
                    name="labeled_company",
                    regex=labeled_company_regex,
                    score=0.90,  # Labeled company names - high confidence
                ),
                # CamelCase pattern disabled - too many false positives
            ],
            context=["company", "inc", "ltd", "corp", "limited", "firm", "business", "employer", "organization", "corporation", "enterprise", "client", "vendor", "headquarters", "hq"]
        )

        self.analyzer.registry.add_recognizer(company_recognizer)

        # Add dictionary-based company NER (company-named-entity-recognition library)
        if COMPANY_NER_AVAILABLE:
            self._add_company_ner_recognizer()

    def _add_company_ner_recognizer(self):
        """
        Add dictionary-based company name recognition using company-named-entity-recognition.

        This library provides high-confidence detection of known company names
        by linking them to a dictionary of real company entities.
        """
        from presidio_analyzer import EntityRecognizer, RecognizerResult

        class CompanyNERRecognizer(EntityRecognizer):
            """Custom recognizer using company-named-entity-recognition library."""

            ENTITIES = ["COMPANY"]

            def __init__(self):
                super().__init__(
                    supported_entities=self.ENTITIES,
                    supported_language="en",
                    name="CompanyNERRecognizer"
                )

            def load(self):
                """No model to load - dictionary-based."""
                pass

            def analyze(self, text, entities, nlp_artifacts=None):
                """Find company names using dictionary lookup."""
                results = []
                if not text:
                    return results

                # Tokenize text (simple whitespace tokenization)
                tokens = text.split()

                # Find companies using the library
                try:
                    found = find_companies(tokens)
                    for company_info in found:
                        # company_info is typically (start_idx, end_idx, company_name, entity_id)
                        if isinstance(company_info, tuple) and len(company_info) >= 3:
                            start_idx, end_idx, company_name = company_info[:3]

                            # Calculate character positions
                            char_start = sum(len(t) + 1 for t in tokens[:start_idx])
                            char_end = sum(len(t) + 1 for t in tokens[:end_idx]) - 1

                            results.append(RecognizerResult(
                                entity_type="COMPANY",
                                start=char_start,
                                end=char_end,
                                score=0.90,  # High confidence - dictionary match
                            ))
                except Exception:
                    pass  # Gracefully handle library errors

                return results

        self.analyzer.registry.add_recognizer(CompanyNERRecognizer())

    def _add_gender_recognizers(self):
        """
        Add pattern recognizers for gender identities and biological sex designations.

        This recognizer detects various gender identity terms including:
        - Binary identities (male, female, man, woman)
        - Transgender terms (trans, MTF, FTM, transgender)
        - Non-binary identities (non-binary, genderqueer, genderfluid, agender, etc.)
        - Medical sex designations (AFAB, AMAB, intersex)
        - Cultural identities (Two-Spirit, Third Gender)
        """
        # Core binary gender terms with word boundaries
        # Note: These are common words, so we require context to detect them
        # Include uppercase variants (MALE, FEMALE) for form fields
        binary_terms = [
            r"[Mm][Aa][Ll][Ee]", r"[Ff][Ee][Mm][Aa][Ll][Ee]",
            r"[Mm][Aa][Nn]", r"[Ww][Oo][Mm][Aa][Nn]",
            r"[Bb][Oo][Yy]", r"[Gg][Ii][Rr][Ll]",
            r"[Mm]asculine", r"[Ff]eminine",
        ]

        # Transgender terms
        trans_terms = [
            r"[Tt]ransgender", r"[Tt]rans[- ]?[Ww]oman", r"[Tt]rans[- ]?[Mm]an",
            r"[Tt]rans[- ]?[Mm]ale", r"[Tt]rans[- ]?[Ff]emale",
            r"[Tt]rans[- ]?[Mm]asc(?:uline)?", r"[Tt]rans[- ]?[Ff]em(?:inine)?",
            r"MTF", r"FTM", r"M2F", r"F2M",
        ]

        # Non-binary and other gender identity terms
        nonbinary_terms = [
            r"[Nn]on[- ]?[Bb]inary", r"[Ee]nby", r"NB",
            r"[Gg]enderqueer", r"[Gg]ender[- ]?[Ff]luid",
            r"[Aa]gender", r"[Bb]igender", r"[Tt]rigender",
            r"[Pp]angender", r"[Pp]olygender", r"[Mm]ultigender",
            r"[Dd]emigender", r"[Dd]emigirl", r"[Dd]emiboy",
            r"[Aa]ndrogyn(?:e|ous)", r"[Nn]eutrois",
            r"[Gg]ender[- ]?[Nn]eutral", r"[Gg]ender[- ]?[Ee]xpansive",
            r"[Gg]ender[- ]?[Nn]on[- ]?[Cc]onforming", r"GNC",
            r"[Ii]ntergender", r"[Xx]enogender",
            r"[Oo]mnigender", r"[Aa]pogender",
        ]

        # Cultural and medical terms
        cultural_medical_terms = [
            r"[Tt]wo[- ]?[Ss]pirit", r"[Tt]hird[- ]?[Gg]ender",
            r"[Hh]ijra", r"[Mm]uxe", r"[Ff]a'afafine",
            r"[Cc]isgender", r"[Cc]is[- ]?[Mm]an", r"[Cc]is[- ]?[Ww]oman",
            r"AFAB", r"AMAB", r"DFAB", r"DMAB",
            r"[Aa]ssigned\s+(?:[Mm]ale|[Ff]emale)\s+[Aa]t\s+[Bb]irth",
            r"[Ii]ntersex", r"[Hh]ermaphrodite",
            r"[Ss]ex\s+[Aa]t\s+[Bb]irth",
        ]

        # Combine all terms
        all_terms = binary_terms + trans_terms + nonbinary_terms + cultural_medical_terms
        gender_pattern = r"\b(?:" + "|".join(all_terms) + r")\b"

        gender_recognizer = PatternRecognizer(
            supported_entity="GENDER",
            patterns=[
                Pattern(
                    name="gender_identity",
                    regex=gender_pattern,
                    score=0.75,
                )
            ],
            context=["gender", "sex", "identity", "identifies", "pronoun", "assigned", "birth"]
        )

        self.analyzer.registry.add_recognizer(gender_recognizer)

        # Single-letter gender codes (M/F) - very low base score, rely on context boost
        gender_letter_recognizer = PatternRecognizer(
            supported_entity="GENDER",
            patterns=[
                Pattern(
                    name="gender_single_letter",
                    regex=r"\b[MF]\b",
                    score=0.35,
                )
            ],
            context=["gender", "sex", "male", "female", "m/f", "f/m"]
        )

        self.analyzer.registry.add_recognizer(gender_letter_recognizer)

        # Phrase-based gender responses (common in forms and surveys)
        gender_phrase_recognizer = PatternRecognizer(
            supported_entity="GENDER",
            patterns=[
                Pattern(
                    name="gender_prefer_not_to_disclose",
                    regex=r"\b[Pp]refer\s+not\s+to\s+(?:disclose|say|answer|specify)\b",
                    score=0.80,
                ),
                Pattern(
                    name="gender_other",
                    regex=r"\b[Oo]ther\b",
                    score=0.30,
                ),
            ],
            context=["gender", "sex", "identity", "male", "female", "m/f", "f/m"]
        )

        self.analyzer.registry.add_recognizer(gender_phrase_recognizer)

    def _add_coordinate_recognizers(self):
        """
        Add pattern recognizers for geographic coordinates (latitude/longitude).

        Detects various coordinate formats:
        - Decimal degrees: 40.7128, -74.0060 or 40.7128° N, 74.0060° W
        - Degrees minutes seconds: 40°42'46"N 74°0'22"W
        - GPS coordinates: N 40° 42.767', W 074° 00.367'
        """
        # Decimal degrees with optional direction
        # Matches: 40.7128, -74.0060 or 40.7128°N, 74.0060°W
        # Now accepts 1-8 decimal places (was 3-8, too restrictive)
        decimal_coords = r"\b-?\d{1,3}\.\d{1,8}°?\s*[NSEW]?\b"

        # Decimal degree pairs (latitude, longitude)
        # Accepts 1-8 decimal places
        decimal_pair = r"\b-?\d{1,3}\.\d{1,8}\s*,\s*-?\d{1,3}\.\d{1,8}\b"

        # Degrees, minutes, seconds: 40°42'46"N
        dms_coord = r"\b\d{1,3}°\s*\d{1,2}['\u2032]\s*\d{1,2}(?:\.\d+)?[\"″\u2033]?\s*[NSEW]\b"

        # Degrees decimal minutes: N 40° 42.767'
        ddm_coord = r"\b[NSEW]\s*\d{1,3}°\s*\d{1,2}\.\d+['\u2032]\b"

        # Full coordinate pair with directions
        coord_pair_directions = r"\b\d{1,3}°\s*\d{1,2}['\u2032]\s*\d{1,2}(?:\.\d+)?[\"″\u2033]?\s*[NS]\s*,?\s*\d{1,3}°\s*\d{1,2}['\u2032]\s*\d{1,2}(?:\.\d+)?[\"″\u2033]?\s*[EW]\b"

        # GPS format: GPS: 40.7128, -74.0060
        gps_labeled = r"\bGPS[:\s]+[-\d.,\s°NSEW]+\b"

        # Lat/Long labeled: Lat: 40.7128, Long: -74.0060
        # Accepts 1-8 decimal places
        latlong_labeled = r"\b(?:Lat(?:itude)?|Long(?:itude)?)[:\s]+-?\d{1,3}\.\d{1,8}\b"

        coordinate_recognizer = PatternRecognizer(
            supported_entity="COORDINATES",
            patterns=[
                Pattern(name="decimal_pair", regex=decimal_pair, score=0.90),
                Pattern(name="coord_pair_dms", regex=coord_pair_directions, score=0.95),
                Pattern(name="dms_coord", regex=dms_coord, score=0.85),
                Pattern(name="ddm_coord", regex=ddm_coord, score=0.85),
                Pattern(name="decimal_coords", regex=decimal_coords, score=0.70),
                Pattern(name="gps_labeled", regex=gps_labeled, score=0.95),
                Pattern(name="latlong_labeled", regex=latlong_labeled, score=0.90),
            ],
            context=["gps", "coordinates", "location", "latitude", "longitude", "lat", "long", "position", "geo"]
        )

        self.analyzer.registry.add_recognizer(coordinate_recognizer)

    def _add_medical_recognizers(self):
        """
        Add pattern recognizers for medical information including:
        - Blood types (A+, B-, O+, AB-, etc.)
        - Common medications (drug name patterns and specific drugs)
        - Medical conditions and diagnoses
        - Medical codes (ICD-10 patterns)
        - Vital signs and lab values
        """
        # Blood types: A, B, AB, O with +/- or positive/negative
        blood_type_patterns = [
            r"\b(?:blood\s+)?(?:type|group)[:\s]+(?:AB|A|B|O)[+-]?\b",
            r"\b(?:AB|A|B|O)[+-]\s*(?:positive|negative|pos|neg)?\b",
            r"\b(?:AB|A|B|O)\s+(?:positive|negative|pos|neg)\b",
            r"\bRh[+-]\b",
            r"\bRh\s*(?:positive|negative|pos|neg)\b",
        ]

        # Common medication name patterns (drug suffixes)
        # These suffixes are commonly used in pharmaceutical naming
        medication_suffixes = [
            r"\b\w+(?:cillin|mycin|cycline|floxacin|azole|prazole|sartan|pril|olol|dipine|statin|mab|nib|tinib|zumab|ximab)\b",
            r"\b\w+(?:phen|profen|oxin|azepam|zolam|barbital|codone|morphone|fentanyl|methadone)\b",
            r"\b\w+(?:vir|navir|tegravir|buvir)\b",  # Antivirals
            r"\b\w+(?:sone|solone|nisone|olone)\b",  # Corticosteroids
        ]

        # Common specific drug names (partial list of frequently prescribed)
        common_drugs = [
            "Aspirin", "Ibuprofen", "Acetaminophen", "Tylenol", "Advil", "Motrin",
            "Metformin", "Lisinopril", "Atorvastatin", "Lipitor", "Amlodipine",
            "Omeprazole", "Prilosec", "Losartan", "Gabapentin", "Hydrocodone",
            "Amoxicillin", "Azithromycin", "Zithromax", "Ciprofloxacin", "Metoprolol",
            "Levothyroxine", "Synthroid", "Prednisone", "Albuterol", "Ventolin",
            "Insulin", "Warfarin", "Coumadin", "Xarelto", "Eliquis", "Plavix",
            "Prozac", "Zoloft", "Lexapro", "Xanax", "Ativan", "Valium", "Ambien",
            "Adderall", "Ritalin", "Viagra", "Cialis", "Oxycodone", "Percocet",
            "Morphine", "Fentanyl", "Tramadol", "Methadone", "Suboxone", "Naloxone",
            "Humira", "Enbrel", "Remicade", "Keytruda", "Opdivo", "Herceptin",
            # Cannabis/marijuana related
            "Cannabis", "Marijuana", "THC", "CBD", "Cannabidiol", "Dronabinol", "Marinol",
            "Epidiolex", "Sativex", "Nabilone", "Cesamet",
        ]
        # Case-insensitive pattern for drug names
        common_drugs_pattern = r"\b(?i)(?:" + "|".join(common_drugs) + r")\b"

        # Medical conditions and diagnoses patterns
        condition_patterns = [
            # Diabetes variations
            r"\b(?:Type\s*[12]\s*)?[Dd]iabetes(?:\s+[Mm]ellitus)?\b",
            r"\bT[12]DM\b",  # Type 1/2 Diabetes Mellitus abbreviation
            # Heart conditions
            r"\b(?:coronary\s+)?(?:artery|heart)\s+disease\b",
            r"\b(?:atrial\s+)?fibrillation\b",
            r"\bCHF\b",  # Congestive Heart Failure
            r"\bCAD\b",  # Coronary Artery Disease
            r"\b(?:myo|peri)?carditis\b",
            # Cancer types
            r"\b\w+(?:carcinoma|sarcoma|lymphoma|leukemia|melanoma|blastoma)\b",
            r"\b(?:breast|lung|colon|prostate|pancreatic|ovarian|liver|brain)\s+cancer\b",
            r"\b(?i)cancer\b",
            r"\b(?i)tumo(?:u)?r\b",
            # Mental health
            r"\b(?:major\s+)?depress(?:ion|ive\s+disorder)\b",
            r"\b(?:generalized\s+)?anxiety(?:\s+disorder)?\b",
            r"\b(?:bipolar|schizophren|PTSD|OCD|ADHD|ADD)\b",
            # Common conditions
            r"\b(?:hyper|hypo)tension\b",
            r"\b(?:hyper|hypo)thyroid(?:ism)?\b",
            r"\basthma\b",
            r"\bCOPD\b",
            r"\b(?:chronic\s+)?(?:kidney|renal)\s+disease\b",
            r"\bCKD\b",
            r"\b(?:rheumatoid\s+)?arthritis\b",
            r"\bosteoporosis\b",
            r"\bepilepsy\b",
            r"\b(?:Parkinson|Alzheimer|Huntington)(?:'?s)?(?:\s+disease)?\b",
            r"\bmultiple\s+sclerosis\b",
            r"\bMS\b(?=\s+(?:patient|diagnosis|treatment))",
            r"\bHIV(?:/AIDS)?\b",
            r"\b(?:Hepatitis|Hep)\s*[ABC]\b",
            r"\bCOVID-?19\b",
            r"\bpneumonia\b",
            r"\bstroke\b",
            r"\baneurysm\b",
            # Infectious diseases
            r"\b(?i)malaria\b",
            r"\b(?i)leprosy\b",
            r"\b(?i)leishmaniasis\b",
            r"\b(?i)syphilis\b",
            r"\b(?i)tuberculosis\b",
            r"\b(?i)cholera\b",
            r"\b(?i)typhoid\b",
            r"\b(?i)dengue\b",
            r"\b(?i)ebola\b",
            r"\b(?i)zika\b",
            r"\b(?i)rabies\b",
            r"\b(?i)tetanus\b",
            r"\b(?i)measles\b",
            r"\b(?i)mumps\b",
            r"\b(?i)rubella\b",
            r"\b(?i)polio(?:myelitis)?\b",
            r"\b(?i)smallpox\b",
            r"\b(?i)anthrax\b",
            r"\b(?i)meningitis\b",
            r"\b(?i)encephalitis\b",
            # Mental health expanded
            r"\b(?i)mental\s+(?:illness|disorder|health\s+(?:condition|disorder))\b",
            r"\b(?i)psychosis\b",
            r"\b(?i)dementia\b",
            r"\b(?i)autism(?:\s+spectrum)?\b",
            r"\b(?i)eating\s+disorder\b",
            r"\b(?i)anorexia(?:\s+nervosa)?\b",
            r"\b(?i)bulimia(?:\s+nervosa)?\b",
            r"\b(?i)substance\s+(?:abuse|use\s+disorder)\b",
            r"\b(?i)addiction\b",
            r"\b(?i)alcoholism\b",
            # Additional conditions
            r"\b(?i)fibromyalgia\b",
            r"\b(?i)GERD\b",
            r"\b(?i)gastroesophageal\s+reflux\b",
            r"\b(?i)celiac(?:\s+disease)?\b",
            r"\b(?i)gout\b",
        ]

        # Body parts and organs (can be sensitive in medical contexts)
        # Note: Using case variations instead of (?i) flag to avoid deprecation warning
        body_part_patterns = [
            # Major organs (case-insensitive via alternation)
            r"(?i)\b(?:heart|liver|kidney|lung|brain|spleen|pancreas|gallbladder|bladder|stomach|intestine|colon)\b",
            r"(?i)\b(?:eyes?|ears?|nose|throat|tongue|teeth|gums|tonsils)\b",
            r"(?i)\b(?:skin|bone|muscle|tendon|ligament|cartilage|joint)\b",
            r"(?i)\b(?:artery|arteries|vein|veins|blood\s+vessel)\b",
            r"(?i)\b(?:thyroid|adrenal|pituitary|prostate|ovary|ovaries|uterus|testes|testicle)\b",
            # Body regions
            r"(?i)\b(?:chest|abdomen|pelvis|groin|spine|vertebra|vertebrae)\b",
            r"(?i)\b(?:skull|rib|ribs|femur|tibia|fibula|humerus|radius|ulna)\b",
            # Systems
            r"(?i)\b(?:nervous\s+system|cardiovascular|respiratory|digestive|urinary|reproductive)\b",
        ]

        # ICD-10 code patterns (letter followed by digits, optionally with decimal)
        icd_patterns = [
            r"\b[A-TV-Z]\d{2}(?:\.\d{1,4})?\b",  # ICD-10-CM format
        ]

        # Lab values and vital signs
        lab_vital_patterns = [
            r"\bBP[:\s]+\d{2,3}/\d{2,3}\b",  # Blood pressure short form
            r"\b[Bb]lood\s+[Pp]ressure[:\s]+\d{2,3}/\d{2,3}\b",  # Blood Pressure: 120/80
            r"\b(?:blood\s+)?(?:glucose|sugar)[:\s]+\d{2,3}\s*(?:mg/dL|mmol/L)?\b",
            r"\bA1[Cc][:\s]+\d{1,2}(?:\.\d)?%?\b",  # HbA1c
            r"\bHbA1[Cc][:\s]+\d{1,2}(?:\.\d)?%?\b",  # HbA1c: 6.5%
            r"\b[Ll]ab\s+[Rr]esults?[:\s]+HbA1[Cc]\s+\d{1,2}(?:\.\d)?%?\b",  # Lab Results: HbA1c 6.5%
            r"\b[Ll]ab\s+[Rr]esults?[:\s]+\w+\s+[\d.]+\b",  # Lab Results: TSH 2.1
            r"\bBMI[:\s]+\d{1,2}(?:\.\d)?\b",
            r"\bGFR[:\s]+\d{1,3}\b",
            r"\bcreatinine[:\s]+\d{1,2}(?:\.\d{1,2})?\b",
            r"\bcholesterol[:\s]+\d{2,3}\b",
            r"\bLDL[:\s]+\d{2,3}\b",
            r"\bHDL[:\s]+\d{2,3}\b",
            r"\btriglycerides[:\s]+\d{2,4}\b",
            r"\bTSH[:\s]+[\d.]+\b",  # TSH: 2.1
        ]

        # Health plan beneficiary numbers (from ground truth)
        # Format: state prefix + numbers, or alphanumeric codes
        health_plan_patterns = [
            r"\b[A-Z]{2}[-]?\d{7,10}\b",  # PA-0004382965, MN001526348
            r"\b[A-Z]{2,4}-\d{4}-\d{4}-\d{2}\b",  # AET-9345-2178-65
            r"\b[A-Z0-9]{4}-[A-Z0-9]{3}-[A-Z0-9]{4}\b",  # 4F92-KL7-PT39
            r"\b[A-Z]\d{7}-\d{2}\b",  # H1284657-03
        ]

        # Medical record numbers (MRN) patterns
        mrn_patterns = [
            r"\b00\d{8}\b",  # 0009271658
            r"\b[A-Z]{2,3}-\d{8}\b",  # LAC-00018325
            r"\b[A-Z]-\d{2}-\d{6}\b",  # M-25-000349
            r"\b[A-Z]\d{6}\b",  # A345982
            r"\b(?:MRN|MR#|Medical\s+Record)[:\s]*\d{6,10}\b",  # Labeled MRN
        ]

        # Combine all patterns
        all_patterns = (
            blood_type_patterns +
            medication_suffixes +
            [common_drugs_pattern] +
            condition_patterns +
            body_part_patterns +
            icd_patterns +
            lab_vital_patterns
        )

        # Create recognizers for different medical subcategories
        medical_recognizer = PatternRecognizer(
            supported_entity="MEDICAL",
            patterns=[
                Pattern(name="blood_type", regex=blood_type_patterns[0], score=0.85),
                Pattern(name="blood_type_simple", regex=blood_type_patterns[1], score=0.75),
                Pattern(name="blood_type_word", regex=blood_type_patterns[2], score=0.80),
                Pattern(name="rh_factor", regex=blood_type_patterns[3], score=0.80),
                Pattern(name="rh_factor_word", regex=blood_type_patterns[4], score=0.80),
                Pattern(name="medication_suffix_1", regex=medication_suffixes[0], score=0.80),
                Pattern(name="medication_suffix_2", regex=medication_suffixes[1], score=0.80),
                Pattern(name="medication_suffix_3", regex=medication_suffixes[2], score=0.80),
                Pattern(name="medication_suffix_4", regex=medication_suffixes[3], score=0.80),
                Pattern(name="common_drugs", regex=common_drugs_pattern, score=0.85),
                Pattern(name="icd_code", regex=icd_patterns[0], score=0.70),
            ],
            context=["patient", "diagnosis", "treatment", "prescription", "medication", "drug",
                     "condition", "disease", "blood", "type", "medical", "health", "doctor",
                     "hospital", "clinic", "pharmacy", "dose", "mg", "ml"]
        )

        # Separate recognizer for conditions (case-insensitive matching needed)
        condition_recognizer = PatternRecognizer(
            supported_entity="MEDICAL",
            patterns=[Pattern(name=f"condition_{i}", regex=p, score=0.80)
                      for i, p in enumerate(condition_patterns)],
            context=["patient", "diagnosis", "diagnosed", "treatment", "condition",
                     "disease", "history", "medical", "chronic", "acute"]
        )

        # Lab values recognizer
        lab_recognizer = PatternRecognizer(
            supported_entity="MEDICAL",
            patterns=[Pattern(name=f"lab_{i}", regex=p, score=0.75)
                      for i, p in enumerate(lab_vital_patterns)],
            context=["lab", "test", "result", "value", "level", "reading", "vital", "signs"]
        )

        # Body parts recognizer (organs, body regions)
        body_part_recognizer = PatternRecognizer(
            supported_entity="MEDICAL",
            patterns=[Pattern(name=f"body_part_{i}", regex=p, score=0.70)
                      for i, p in enumerate(body_part_patterns)],
            context=["patient", "examination", "pain", "surgery", "transplant", "organ",
                     "injury", "damaged", "removed", "biopsy", "scan", "x-ray", "mri", "ct"]
        )

        # Health plan beneficiary number recognizer
        health_plan_recognizer = PatternRecognizer(
            supported_entity="MEDICAL",
            patterns=[Pattern(name=f"health_plan_{i}", regex=p, score=0.80)
                      for i, p in enumerate(health_plan_patterns)],
            context=["health", "plan", "beneficiary", "insurance", "member", "subscriber",
                     "policy", "coverage", "claim", "copay", "deductible"]
        )

        # Medical record number recognizer
        mrn_recognizer = PatternRecognizer(
            supported_entity="MEDICAL",
            patterns=[Pattern(name=f"mrn_{i}", regex=p, score=0.80)
                      for i, p in enumerate(mrn_patterns)],
            context=["medical", "record", "patient", "chart", "file", "mrn", "hospital",
                     "clinic", "admission", "visit", "encounter"]
        )

        self.analyzer.registry.add_recognizer(medical_recognizer)
        self.analyzer.registry.add_recognizer(condition_recognizer)
        self.analyzer.registry.add_recognizer(lab_recognizer)
        self.analyzer.registry.add_recognizer(body_part_recognizer)
        self.analyzer.registry.add_recognizer(health_plan_recognizer)
        self.analyzer.registry.add_recognizer(mrn_recognizer)

    def _add_fast_medical_recognizer(self):
        """
        Add Fast Data Science medical NER recognizer for broader coverage.

        Uses lightweight, zero-dependency libraries to detect diseases,
        conditions, drugs, and medications with MeSH code normalization.

        Falls back gracefully if libraries are not installed.
        """
        try:
            from .medical_recognizer import get_medical_recognizer, is_medical_ner_available

            if is_medical_ner_available():
                recognizer = get_medical_recognizer()
                if recognizer:
                    self.analyzer.registry.add_recognizer(recognizer)
        except ImportError:
            pass  # Libraries not installed, pattern-based detection still active

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
                    score=0.95,  # Boosted for better recall
                ),
            ],
            context=["phone", "tel", "mobile", "cell", "fax", "contact"]
        )

        # NA phone with dashes: 416-770-4541 or 1-416-770-4541 or 001-416-770-4541
        na_phone_dashes = PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=[
                Pattern(
                    name="na_phone_dashes",
                    regex=r"\b\d{3}-\d{3}-\d{4}\b",
                    score=0.95,  # Boosted for better recall
                ),
                Pattern(
                    name="na_phone_dashes_with_1",
                    # Matches: 1-XXX-XXX-XXXX (toll-free and long-distance format)
                    regex=r"\b1-\d{3}-\d{3}-\d{4}\b",
                    score=0.95,  # Same score as standard format
                ),
                Pattern(
                    name="na_phone_dashes_with_001",
                    # Matches: 001-XXX-XXX-XXXX (international dialing prefix for US/Canada)
                    # Faker generates this format; 001 is the IDD prefix for +1
                    regex=r"\b001-\d{3}-\d{3}-\d{4}\b",
                    score=0.95,  # Same score as standard format
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
                    score=0.95,  # Boosted for better recall
                ),
            ],
            context=["phone", "tel", "mobile", "cell", "fax", "contact"]
        )

        # Labeled phone patterns: "Phone: 555-123-4567", "Tel: (555) 123-4567"
        labeled_phone = PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=[
                Pattern(
                    name="labeled_phone",
                    regex=r"(?:Phone|Tel|Mobile|Cell|Fax|Contact)[:\s]+\(?[\d]{3}\)?[-.\s]?[\d]{3}[-.\s]?[\d]{4}\b",
                    score=0.98,
                ),
            ],
        )

        # International format: +1-416-770-4541
        intl_phone = PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=[
                Pattern(
                    name="intl_phone_plus",
                    regex=r"\+\d{1,3}[-.\s]?\(?\d{1,5}\)?[-.\s]?\d{1,5}[-.\s]?\d{1,9}\b",
                    score=0.85,
                ),
            ],
            context=["phone", "tel", "mobile", "cell", "fax", "contact"]
        )

        # Vanity phone numbers: 800-453-BANK, 1-800-FLOWERS
        # Uses letters that map to phone digits (ABC=2, DEF=3, etc.)
        vanity_phone = PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=[
                Pattern(
                    name="vanity_phone_dash",
                    # Matches: NNN-NNN-WORD or 1-NNN-WORD where WORD has letters
                    regex=r"\b1?-?\d{3}-\d{0,3}[A-Z]{2,}\b",
                    score=0.85,
                ),
                Pattern(
                    name="vanity_phone_mixed",
                    # Matches: 800-453-BANK (digits and letters in last part)
                    regex=r"\b\d{3}-\d{3}-[A-Z0-9]{4,}\b",
                    score=0.85,
                ),
            ],
            context=["call", "phone", "dial", "contact", "toll-free", "hotline"]
        )

        # European-style phone: 0XXX XXXXXXX (4 digits starting with 0, space, rest)
        # Common in UK, Germany, Netherlands, Australia, etc.
        # Examples: 0475 4429797, 0304 2215930, 0932 173 536, 0182 603 292
        european_phone = PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=[
                Pattern(
                    name="european_4digit_space",
                    # Matches: 0XXX XXXXXXX or 0XXX XXX XXXX (with optional spaces)
                    regex=r"\b0\d{3}\s+\d{3,4}\s?\d{3,4}\b",
                    score=0.85,
                ),
                Pattern(
                    name="european_4digit_compact",
                    # Matches: 0XXXXXXXXXX (10-11 digits starting with 0)
                    regex=r"\b0\d{9,10}\b",
                    score=0.80,
                ),
                Pattern(
                    name="european_4_3_3",
                    # Matches: 0XXX XXX XXX (4+3+3 format, common in many countries)
                    # Examples: 0182 603 292, 0714 975 0536
                    regex=r"\b0\d{3}\s\d{3}\s\d{3}\b",
                    score=0.85,
                ),
                Pattern(
                    name="european_4_7",
                    # Matches: 0XXX XXXXXXX (4+7 format)
                    # Examples: 0673 9869964, 0714 9750536
                    regex=r"\b0\d{3}\s\d{7}\b",
                    score=0.85,
                ),
            ],
            context=["phone", "tel", "mobile", "cell", "fax", "contact", "ring", "call"]
        )

        # South African phone: +27 XX XXX XXXX
        # Examples: +27 37 486 7849
        south_african_phone = PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=[
                Pattern(
                    name="za_intl_spaced",
                    # +27 XX XXX XXXX
                    regex=r"\+27\s?\d{2}\s?\d{3}\s?\d{4}\b",
                    score=0.92,
                ),
                Pattern(
                    name="za_local",
                    # 0XX XXX XXXX (local format)
                    regex=r"\b0\d{2}\s?\d{3}\s?\d{4}\b",
                    score=0.85,
                ),
            ],
            context=["phone", "tel", "mobile", "cell", "contact", "south africa", "za"]
        )

        # Indian phone: +91 XXXXX XXXXX or 0XXXX-XXXXXX
        # Examples: +91 98765 43210, 011-2334-5678
        indian_phone = PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=[
                Pattern(
                    name="in_mobile",
                    # +91 XXXXX XXXXX or +91 XXXXXXXXXX
                    regex=r"\+91[\s-]?\d{5}[\s-]?\d{5}\b",
                    score=0.92,
                ),
                Pattern(
                    name="in_landline",
                    # 0XX-XXXX-XXXX or 0XXX-XXX-XXXX
                    regex=r"\b0\d{2,4}[-\s]?\d{3,4}[-\s]?\d{4}\b",
                    score=0.85,
                ),
            ],
            context=["phone", "tel", "mobile", "cell", "contact", "india"]
        )

        # NA phone with spaces: 416 770 4541
        na_phone_spaces = PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=[
                Pattern(
                    name="na_phone_spaces",
                    regex=r"\b\d{3}\s\d{3}\s\d{4}\b",
                    score=0.85,
                ),
            ],
            context=["phone", "tel", "mobile", "cell", "fax", "contact"]
        )

        # Compact NA phone: 4167704541 (10 digits, validated with area code)
        compact_na_phone = PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=[
                Pattern(
                    name="na_phone_compact",
                    # 10 digits starting with valid NA area code (2-9 for first digit)
                    regex=r"\b[2-9]\d{2}[2-9]\d{6}\b",
                    score=0.75,  # Lower score - more prone to false positives
                ),
            ],
            context=["phone", "tel", "mobile", "cell", "fax", "contact", "call"]
        )

        # International with varied spacing: +XX XXX XXX XXXX, +XX-XXX-XXX-XXXX
        intl_phone_varied = PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=[
                Pattern(
                    name="intl_phone_spaces",
                    # +CC (area) rest or +CC area-rest
                    regex=r"\+\d{1,3}\s+\d{2,4}\s+\d{3,4}\s+\d{3,4}\b",
                    score=0.90,
                ),
                Pattern(
                    name="intl_phone_parens_space",
                    # +CC (area) rest
                    regex=r"\+\d{1,3}\s*\(\d{2,4}\)\s*\d{3,4}[-.\s]?\d{3,4}\b",
                    score=0.90,
                ),
            ],
            context=["phone", "tel", "mobile", "cell", "fax", "contact", "international"]
        )

        # Australian phone: +61 X XXXX XXXX, 04XX XXX XXX
        australian_phone = PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=[
                Pattern(
                    name="au_mobile",
                    # Australian mobile: 04XX XXX XXX
                    regex=r"\b04\d{2}\s?\d{3}\s?\d{3}\b",
                    score=0.90,
                ),
                Pattern(
                    name="au_landline",
                    # Australian landline: (0X) XXXX XXXX
                    regex=r"\(0[2-9]\)\s?\d{4}\s?\d{4}\b",
                    score=0.90,
                ),
                Pattern(
                    name="au_intl",
                    # +61 X XXXX XXXX
                    regex=r"\+61\s?\d\s?\d{4}\s?\d{4}\b",
                    score=0.92,
                ),
            ],
            context=["phone", "tel", "mobile", "cell", "contact", "australia"]
        )

        # UK phone: +44 XXXX XXXXXX, 07XXX XXXXXX, 01XXX XXXXXX
        uk_phone = PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=[
                Pattern(
                    name="uk_mobile",
                    # UK mobile: 07XXX XXXXXX or 07XXX-XXXXXX
                    regex=r"\b07\d{3}[\s\-]?\d{6}\b",
                    score=0.90,
                ),
                Pattern(
                    name="uk_intl",
                    # +44 XXXX XXXXXX
                    regex=r"\+44[\s\-]?\d{4}[\s\-]?\d{6}\b",
                    score=0.92,
                ),
                Pattern(
                    name="uk_landline",
                    # UK landline: 01XXX XXXXXX, 02X XXXX XXXX
                    regex=r"\b0[12]\d{2,3}[\s\-]\d{5,6}\b",
                    score=0.85,
                ),
                Pattern(
                    name="uk_landline_dash",
                    # UK landline with dash: 01926-97469, 01221-370813
                    regex=r"\b0\d{4}[\-]\d{5,6}\b",
                    score=0.85,
                ),
            ],
            context=["phone", "tel", "mobile", "cell", "contact", "uk", "call"]
        )

        # Phone numbers with extensions: (636)734-8519x84099, 555-123-4567 ext 123
        phone_with_extension = PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=[
                # (XXX)XXX-XXXXxNNNNN - parentheses with extension
                Pattern(
                    name="phone_ext_parens",
                    regex=r"\(\d{3}\)\s?\d{3}[-.\s]?\d{4}\s?[xX#]\s?\d{1,6}\b",
                    score=0.95,
                ),
                # XXX-XXX-XXXX x NNNNN - dashes/dots with extension
                Pattern(
                    name="phone_ext_dashes",
                    regex=r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\s?(?:x|ext\.?|extension)\s?\d{1,6}\b",
                    score=0.95,
                ),
                # 001-XXX-XXX-XXXXxNNNNN - IDD prefix with extension
                Pattern(
                    name="phone_ext_001",
                    regex=r"\b001-\d{3}-\d{3}-\d{4}\s?[xX#]\s?\d{1,6}\b",
                    score=0.95,
                ),
                # +1-XXX-XXX-XXXXxNNNNN - international prefix with extension
                Pattern(
                    name="phone_ext_plus1",
                    regex=r"\+1-\d{3}-\d{3}-\d{4}\s?[xX#]\s?\d{1,6}\b",
                    score=0.95,
                ),
                # Redacted phone: ***-***-1906, XXX-XXX-1234
                Pattern(
                    name="redacted_phone_stars",
                    regex=r"\*{3}[-.\s]?\*{3}[-.\s]?\d{4}\b",
                    score=0.85,
                ),
                Pattern(
                    name="redacted_phone_x",
                    regex=r"[Xx]{3}[-.\s]?[Xx]{3}[-.\s]?\d{4}\b",
                    score=0.85,
                ),
            ],
            context=["phone", "tel", "mobile", "cell", "fax", "contact", "ext", "extension"]
        )

        # Heavily spaced phone numbers: "9 6 4 0 7 2 4 3 9 3 6", "1 800 555 1234"
        # These are common in OCR output or obfuscated formats
        # Pattern matches 8-15 digits with various spacing patterns
        # NOTE: Scores must be >= 0.70 to pass PHONE_NUMBER threshold in Pass 8
        heavily_spaced_phone = PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=[
                Pattern(
                    name="spaced_digits_phone",
                    # Matches: d d d d d d d d (8-15 single digits with spaces)
                    # e.g., "9 6 4 0 1 2 3 4 5 6"
                    regex=r"\b(\d\s+){7,14}\d\b",
                    score=0.75,  # High score - OCR spacing is strong phone signal
                ),
                Pattern(
                    name="spaced_digits_with_separators",
                    # Matches mixed separators: 9-6-4 0 7 2-4 3-9 3 6
                    regex=r"\b(\d[-.\s]+){7,14}\d\b",
                    score=0.72,  # Good score - mixed separators still indicate phone
                ),
                Pattern(
                    name="ocr_spaced_with_plus",
                    # Matches international OCR: +1 8 0 0 5 5 5 1 2 3 4
                    regex=r"\+\s*(\d\s*){8,15}\b",
                    score=0.88,  # Higher than intl_phone (0.85) to win deduplication
                ),
                Pattern(
                    name="ocr_grouped_spaces",
                    # Matches grouped OCR: 1 800 555 1234, 416 555 1234
                    # Groups of 1-4 digits separated by single spaces
                    regex=r"\b\d{1,4}(?:\s+\d{1,4}){2,5}\b",
                    score=0.70,  # Meets threshold - common OCR phone format
                ),
                Pattern(
                    name="ocr_minimal_spacing",
                    # Matches minimal OCR spacing: digits with optional single spaces
                    # e.g., "9640123456" with occasional spaces
                    regex=r"\b(?:\d\s*){8,15}\b",
                    score=0.55,  # Low score - too broad, needs validation boost
                ),
            ],
            context=["phone", "tel", "mobile", "cell", "fax", "contact", "number", "call"]
        )

        # International 00-prefix phone with mixed separators (dots, dashes, spaces)
        # Matches: "0064.744259477", "0004-33 859.5757", "0067-24684.7522"
        intl_00_prefix = PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=[
                Pattern(
                    name="intl_00_dots",
                    # 00CC.NNNNNNNNN (country code + dot + rest)
                    regex=r"\b00\d{2}[-.]\d{5,10}\b",
                    score=0.88,
                ),
                Pattern(
                    name="intl_00_mixed",
                    # 00CC-NN NNN.NNNN or 00CC.NN.NNN.NNNN
                    regex=r"\b00\d{2}[-.\s]\d{1,5}[-.\s]\d{3,5}[-.\s]?\d{0,5}\b",
                    score=0.88,
                ),
                Pattern(
                    name="intl_00_compact",
                    # 00CCNNNNNNNNN (13-15 digits starting with 00)
                    regex=r"\b00\d{11,13}\b",
                    score=0.82,
                ),
            ],
            context=["phone", "tel", "mobile", "cell", "fax", "contact", "international"]
        )

        # European phone with dots as separators
        # Matches: "013417878.8587", "01-72.06-94.36", "0629.15429898"
        european_phone_dots = PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=[
                Pattern(
                    name="euro_phone_dots",
                    # 0NNNNNNN.NNNN (European with dot separator)
                    regex=r"\b0\d{4,9}\.\d{4,8}\b",
                    score=0.82,
                ),
                Pattern(
                    name="euro_phone_mixed_dots",
                    # NN-NN.NN-NN.NN or NN.NN.NN.NN.NN (European dot/dash mix)
                    regex=r"\b\d{2}[-.]?\d{2}\.\d{2}[-.]?\d{2}\.\d{2}\b",
                    score=0.85,
                ),
                Pattern(
                    name="euro_phone_dot_continuous",
                    # 0XXX.XXXXXXXXX (European with dot and continuous digits)
                    regex=r"\b0\d{3,4}\.\d{6,10}\b",
                    score=0.80,
                ),
                Pattern(
                    name="euro_phone_dot_groups",
                    # 010.155.741.8175 (dot-separated groups)
                    regex=r"\b0\d{2}\.\d{3}\.\d{3}\.\d{4}\b",
                    score=0.85,
                ),
            ],
            context=["phone", "tel", "mobile", "cell", "fax", "contact"]
        )

        # Japanese phone: 070-1920-3719, 03-1234-5678
        japanese_phone = PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=[
                Pattern(
                    name="jp_mobile",
                    # Japanese mobile: 070/080/090-XXXX-XXXX
                    regex=r"\b0[789]0[-.\s]?\d{4}[-.\s]?\d{4}\b",
                    score=0.90,
                ),
                Pattern(
                    name="jp_landline",
                    # Japanese landline: 0X-XXXX-XXXX or 0XX-XXX-XXXX
                    regex=r"\b0\d{1,2}[-.\s]?\d{3,4}[-.\s]?\d{4}\b",
                    score=0.85,
                ),
                Pattern(
                    name="jp_intl",
                    # +81-X-XXXX-XXXX
                    regex=r"\+81[-.\s]?\d{1,2}[-.\s]?\d{3,4}[-.\s]?\d{4}\b",
                    score=0.92,
                ),
            ],
            context=["phone", "tel", "mobile", "cell", "contact", "japan", "jp"]
        )

        self.analyzer.registry.add_recognizer(na_phone_parens)
        self.analyzer.registry.add_recognizer(na_phone_dashes)
        self.analyzer.registry.add_recognizer(na_phone_dots)
        self.analyzer.registry.add_recognizer(labeled_phone)
        self.analyzer.registry.add_recognizer(intl_phone)
        self.analyzer.registry.add_recognizer(vanity_phone)
        self.analyzer.registry.add_recognizer(european_phone)
        self.analyzer.registry.add_recognizer(na_phone_spaces)
        self.analyzer.registry.add_recognizer(compact_na_phone)
        self.analyzer.registry.add_recognizer(intl_phone_varied)
        self.analyzer.registry.add_recognizer(australian_phone)
        self.analyzer.registry.add_recognizer(uk_phone)
        self.analyzer.registry.add_recognizer(south_african_phone)
        self.analyzer.registry.add_recognizer(indian_phone)
        self.analyzer.registry.add_recognizer(phone_with_extension)
        self.analyzer.registry.add_recognizer(heavily_spaced_phone)
        self.analyzer.registry.add_recognizer(japanese_phone)
        self.analyzer.registry.add_recognizer(intl_00_prefix)
        self.analyzer.registry.add_recognizer(european_phone_dots)

        # Flexible mixed-separator phone pattern (dots + dashes + spaces combined)
        # Catches phones like: 0138-96 042.3987, +284-383.385-7028, 00461.203 543-4671
        mixed_separator_phone = PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=[
                Pattern(
                    name="mixed_sep_intl_plus",
                    # +CC then 8-12 digits with mixed separators
                    regex=r"\+\d{1,4}[.\-\s]\d{2,5}[.\-\s]\d{2,5}[.\-\s]?\d{2,5}",
                    score=0.88,
                ),
                Pattern(
                    name="mixed_sep_intl_00",
                    # 00CC then 8-12 digits with mixed separators
                    regex=r"00\d{2,3}[.\-\s]\d{2,5}[.\-\s]\d{2,5}[.\-\s]?\d{2,5}",
                    score=0.85,
                ),
                Pattern(
                    name="mixed_sep_intl_00_dot",
                    # 00.CC-XX XX XX (dot after 00 international prefix)
                    regex=r"00[.\-]\d{2}[.\-\s]\d{2}[\s]\d{2}[\s]\d{2}",
                    score=0.85,
                ),
                Pattern(
                    name="mixed_sep_intl_00_space",
                    # 00147 832542, 002-275 5228, 002672332 3506 (00 + digits with space/dash)
                    regex=r"00\d{1,3}[.\-\s]\d{3,6}[\s]?\d{3,6}",
                    score=0.82,
                ),
                Pattern(
                    name="mixed_sep_local",
                    # Local format: 0XXX-XX XXX.XXXX or similar with 3+ groups
                    regex=r"\b0\d{2,4}[.\-\s]\d{2,5}[.\-\s]\d{3,5}[.\-\s]?\d{0,5}\b",
                    score=0.80,
                ),
                Pattern(
                    name="mixed_sep_local_2group",
                    # Two-group local: 09545 55592, 0103-663399916, 0509.19132656
                    regex=r"\b0\d{3,5}[.\-\s]\d{4,10}\b",
                    score=0.78,
                ),
                Pattern(
                    name="mixed_sep_generic",
                    # Any digit group with mixed dots/dashes/spaces (10+ digits total)
                    regex=r"\b\d{2,5}[.\-]\d{2,5}[\s]\d{2,5}[.\-]?\d{2,5}\b",
                    score=0.78,
                ),
                Pattern(
                    name="mixed_sep_4group",
                    # 4+ groups with any separator: 05-93 19.08-21, 02 33.97-89.54
                    regex=r"\b0\d{1,2}[.\-\s]\d{2,4}[.\-\s]\d{2,4}[.\-\s]\d{2,4}[.\-\s]?\d{0,4}\b",
                    score=0.80,
                ),
            ],
            context=["phone", "tel", "mobile", "cell", "fax", "contact", "call", "number"]
        )
        self.analyzer.registry.add_recognizer(mixed_separator_phone)

        # Short local phone formats: 0XXXX.XXXXXXX, 0XX-XXXXXXX (no spaces, dot/dash only)
        short_local_phone = PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=[
                Pattern(
                    name="local_dot_nosep",
                    # 06196.175469, 0333.48455417, 0509.19132656
                    regex=r"\b0\d{2,4}\.\d{6,10}\b",
                    score=0.78,
                ),
                Pattern(
                    name="local_dash_short",
                    # 055-7436469, 063-8615170
                    regex=r"\b0\d{2,3}-\d{6,8}\b",
                    score=0.78,
                ),
                Pattern(
                    name="local_space_long",
                    # 097 8611392, 07433 381400
                    regex=r"\b0\d{2,4}\s\d{6,8}\b",
                    score=0.78,
                ),
            ],
            context=["phone", "tel", "mobile", "cell", "fax", "contact", "call", "number"]
        )
        self.analyzer.registry.add_recognizer(short_local_phone)

        # 4-3-4 digit format: "5140-790-6744", "9922.726.0323"
        phone_4_3_4 = PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=[
                Pattern(
                    name="phone_4_3_4_dash",
                    regex=r"\b\d{4}-\d{3}-\d{4}\b",
                    score=0.82,
                ),
                Pattern(
                    name="phone_4_3_4_dot",
                    regex=r"\b\d{4}\.\d{3}\.\d{4}\b",
                    score=0.82,
                ),
            ],
            context=["phone", "tel", "mobile", "cell", "fax", "contact", "call"]
        )
        self.analyzer.registry.add_recognizer(phone_4_3_4)

        # Mixed dot-dash without spaces: "3945.181-0029", "9922-726.0323"
        phone_mixed_dot_dash = PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=[
                Pattern(
                    name="phone_dot_then_dash",
                    regex=r"\b\d{3,4}\.\d{3}-\d{4}\b",
                    score=0.80,
                ),
                Pattern(
                    name="phone_dash_then_dot",
                    regex=r"\b\d{3,4}-\d{3}\.\d{4}\b",
                    score=0.80,
                ),
            ],
            context=["phone", "tel", "mobile", "cell", "fax", "contact", "call"]
        )
        self.analyzer.registry.add_recognizer(phone_mixed_dot_dash)

        # Regional phone recognizer using Google's phonenumbers library
        # Provides comprehensive validation for international phone formats
        if PHONENUMBERS_AVAILABLE:
            regional_phone = PhoneRecognizer(
                supported_regions=["US", "CA", "GB", "AU", "DE", "FR", "IN", "JP", "BR", "MX",
                                   "IT", "ES", "NL", "BE", "CH", "AT", "PL", "SE", "NO", "DK",
                                   "ZA", "NZ", "SG", "HK", "KR", "CN", "RU", "AE", "SA", "IL"],
                context=["phone", "tel", "mobile", "cell", "fax", "contact", "number", "call"],
                supported_language="en"
            )
            self.analyzer.registry.add_recognizer(regional_phone)

        # UK NHS Number: XXX XXX XXXX (10 digits with spaces)
        # High confidence when context words are present
        uk_nhs_recognizer = PatternRecognizer(
            supported_entity="UK_NHS",
            patterns=[
                # Standard format with spaces: 123 456 7890
                Pattern(
                    name="uk_nhs_spaces",
                    regex=r"\b\d{3}\s\d{3}\s\d{4}\b",
                    score=0.85,
                ),
                # Format with dashes: 123-456-7890
                Pattern(
                    name="uk_nhs_dashes",
                    regex=r"\b\d{3}-\d{3}-\d{4}\b",
                    score=0.70,  # Lower score - conflicts with phone format
                ),
            ],
            context=["nhs", "national health", "health service", "nhs number",
                     "patient", "hospital", "clinic", "gp", "surgery", "uk"]
        )
        self.analyzer.registry.add_recognizer(uk_nhs_recognizer)

    def _add_credit_card_recognizers(self):
        """Add custom pattern recognizers for common credit card formats."""
        # Common credit card patterns (with optional spaces or dashes)
        # Visa: starts with 4, 13/16/19 digits
        # Mastercard: starts with 51-55 or 2221-2720, 16 digits
        # American Express: starts with 34 or 37, 15 digits
        # Discover: starts with 6011, 622126-622925, 644-649, or 65, 16 digits
        # JCB: starts with 3528-3589, 15/16 digits; old JCB: 1800, 2131, 15 digits
        # Diners Club: starts with 300-305, 36, 38, 14-16 digits
        # UnionPay: starts with 62, 16-19 digits

        credit_card_recognizer = PatternRecognizer(
            supported_entity="CREDIT_CARD",
            patterns=[
                # === HIGH-CONFIDENCE BRAND-SPECIFIC PATTERNS ===

                # Visa 16-digit (4xxx xxxx xxxx xxxx) - with optional separators
                Pattern(
                    name="visa_16",
                    regex=r"\b4\d{3}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
                    score=0.9
                ),
                # Visa 13-digit (old format, less common)
                Pattern(
                    name="visa_13",
                    regex=r"\b4\d{12}\b",
                    score=0.85
                ),
                # Visa 19-digit (extended format for some regions)
                Pattern(
                    name="visa_19",
                    regex=r"\b4\d{18}\b",
                    score=0.85
                ),
                # Mastercard (51xx-55xx or 2221-2720) - 16 digits
                Pattern(
                    name="mastercard",
                    regex=r"\b(?:5[1-5]\d{2}|2(?:22[1-9]|2[3-9]\d|[3-6]\d{2}|7[0-1]\d|720))[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
                    score=0.9
                ),
                # Mastercard continuous 16-digit (no separators)
                Pattern(
                    name="mastercard_continuous",
                    regex=r"\b(?:5[1-5]|2[2-7])\d{14}\b",
                    score=0.85
                ),
                # American Express (34xx or 37xx) - 15 digits with Amex separator pattern
                Pattern(
                    name="amex",
                    regex=r"\b3[47]\d{2}[\s-]?\d{6}[\s-]?\d{5}\b",
                    score=0.9
                ),
                # American Express continuous 15-digit
                Pattern(
                    name="amex_continuous",
                    regex=r"\b3[47]\d{13}\b",
                    score=0.9
                ),
                # JCB modern (3528-3589) - 16 digits
                Pattern(
                    name="jcb_16",
                    regex=r"\b35(?:2[89]|[3-8]\d)[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
                    score=0.9
                ),
                # JCB modern continuous 16-digit
                Pattern(
                    name="jcb_16_continuous",
                    regex=r"\b35(?:2[89]|[3-8]\d)\d{12}\b",
                    score=0.9
                ),
                # JCB modern (3528-3589) - 15 digits
                Pattern(
                    name="jcb_15",
                    regex=r"\b35(?:2[89]|[3-8]\d)[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{3}\b",
                    score=0.9
                ),
                # JCB old format (1800, 2131) - 15 digits
                Pattern(
                    name="jcb_old_15",
                    regex=r"\b(?:1800|2131)\d{11}\b",
                    score=0.9
                ),
                # Diners Club (300-305, 36, 38) - 14 digits with separators
                Pattern(
                    name="diners_14",
                    regex=r"\b(?:30[0-5]\d|36\d{2}|38\d{2})[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{2}\b",
                    score=0.9
                ),
                # Diners Club continuous 14-digit
                Pattern(
                    name="diners_14_continuous",
                    regex=r"\b(?:30[0-5]|36|38)\d{11,12}\b",
                    score=0.85
                ),
                # UnionPay (62) - 16 digits
                Pattern(
                    name="unionpay_16",
                    regex=r"\b62\d{2}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
                    score=0.85
                ),
                # UnionPay continuous 16-digit
                Pattern(
                    name="unionpay_16_continuous",
                    regex=r"\b62\d{14}\b",
                    score=0.85
                ),
                # UnionPay (62) - 19 digits
                Pattern(
                    name="unionpay_19",
                    regex=r"\b62\d{2}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{3}\b",
                    score=0.85
                ),
                # UnionPay continuous 19-digit
                Pattern(
                    name="unionpay_19_continuous",
                    regex=r"\b62\d{17}\b",
                    score=0.85
                ),
                # Discover (6011, 622126-622925, 644-649, 65) - 16 digits
                Pattern(
                    name="discover",
                    regex=r"\b(?:6011|65\d{2}|64[4-9]\d|622(?:1(?:2[6-9]|[3-9]\d)|[2-8]\d{2}|9(?:[01]\d|2[0-5])))[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
                    score=0.9
                ),
                # Discover continuous 16-digit
                Pattern(
                    name="discover_continuous",
                    regex=r"\b(?:6011|65\d{2}|64[4-9]\d)\d{12}\b",
                    score=0.85
                ),

                # === MAESTRO PATTERNS (variable length 12-19) ===

                # Maestro specific prefixes (5018, 5020, 5038, 6304, 6759, 6761-6763)
                Pattern(
                    name="maestro_specific",
                    regex=r"\b(?:5018|5020|5038|6304|6759|676[1-3])\d{8,15}\b",
                    score=0.8
                ),
                # Maestro general (50, 56-69) - 12-13 digits
                Pattern(
                    name="maestro_12_13",
                    regex=r"\b(?:50|5[6-9]|6[0-9])\d{10,11}\b",
                    score=0.75
                ),

                # === GENERIC FALLBACK PATTERNS (lower confidence) ===

                # Generic 16-digit with separators (fallback)
                Pattern(
                    name="generic_16_sep",
                    regex=r"\b\d{4}[\s-]\d{4}[\s-]\d{4}[\s-]\d{4}\b",
                    score=0.75
                ),
                # Generic 16-digit continuous (fallback) - requires Luhn validation
                Pattern(
                    name="generic_16_continuous",
                    regex=r"\b\d{16}\b",
                    score=0.6
                ),
                # Generic 15-digit with Amex-style separator pattern
                Pattern(
                    name="generic_15_sep",
                    regex=r"\b\d{4}[\s-]?\d{6}[\s-]?\d{5}\b",
                    score=0.6
                ),
                # Generic 15-digit continuous
                Pattern(
                    name="generic_15_continuous",
                    regex=r"\b\d{15}\b",
                    score=0.55
                ),
                # Generic 14-digit continuous (Diners)
                Pattern(
                    name="generic_14_continuous",
                    regex=r"\b\d{14}\b",
                    score=0.5
                ),
                # Generic 13-digit continuous (old Visa)
                Pattern(
                    name="generic_13_continuous",
                    regex=r"\b\d{13}\b",
                    score=0.5
                ),
                # Generic 12-digit pattern with card context (fallback for Maestro/other cards)
                Pattern(
                    name="generic_12",
                    regex=r"\b\d{12}\b",
                    score=0.5
                )
            ],
            context=["card", "credit", "payment", "visa", "mastercard", "amex", "discover", "maestro", "debit", "cvv", "cvc", "expir", "jcb", "diners", "unionpay"]
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

        # GitHub Personal Access Tokens (classic and fine-grained)
        github_recognizer = PatternRecognizer(
            supported_entity="CREDENTIAL",
            patterns=[
                Pattern(
                    name="github_pat_classic",
                    # Classic PAT: ghp_xxxx (40 chars)
                    regex=r"ghp_[A-Za-z0-9]{36,}",
                    score=0.95,
                ),
                Pattern(
                    name="github_pat_fine_grained",
                    # Fine-grained PAT: github_pat_xxxx
                    regex=r"github_pat_[A-Za-z0-9]{22,}",
                    score=0.95,
                ),
                Pattern(
                    name="github_oauth",
                    # OAuth tokens: gho_xxxx
                    regex=r"gho_[A-Za-z0-9]{36,}",
                    score=0.95,
                ),
                Pattern(
                    name="github_user_server",
                    # User-to-server tokens: ghu_xxxx
                    regex=r"ghu_[A-Za-z0-9]{36,}",
                    score=0.95,
                ),
                Pattern(
                    name="github_server_server",
                    # Server-to-server tokens: ghs_xxxx
                    regex=r"ghs_[A-Za-z0-9]{36,}",
                    score=0.95,
                ),
                Pattern(
                    name="github_refresh",
                    # Refresh tokens: ghr_xxxx
                    regex=r"ghr_[A-Za-z0-9]{36,}",
                    score=0.95,
                ),
            ],
            context=["github", "token", "pat", "api", "key", "secret"]
        )

        # Google/GCP API Keys
        google_recognizer = PatternRecognizer(
            supported_entity="CREDENTIAL",
            patterns=[
                Pattern(
                    name="google_api_key",
                    # Google API Key: AIza followed by 35 chars
                    regex=r"AIza[0-9A-Za-z\-_]{35}",
                    score=0.90,
                ),
                Pattern(
                    name="google_oauth",
                    # Google OAuth: 24-char client ID/secret
                    regex=r"\d{12}-[a-z0-9]{32}\.apps\.googleusercontent\.com",
                    score=0.90,
                ),
            ],
            context=["google", "gcp", "api", "key", "cloud", "firebase"]
        )

        # Generic API Keys and Tokens
        # Scores reduced: max 0.60 + context boost 0.35 = 0.95 < 0.99 threshold
        generic_credential_recognizer = PatternRecognizer(
            supported_entity="CREDENTIAL",
            patterns=[
                Pattern(
                    name="bearer_token",
                    # Bearer tokens in Authorization headers
                    regex=r"Bearer\s+[A-Za-z0-9_\-\.]{20,}",
                    score=0.60,
                ),
                Pattern(
                    name="api_key_labeled",
                    # Labeled API keys: api_key=xxx, apiKey: xxx
                    regex=r"(?:api[_\-]?key|apikey)\s*[=:]\s*['\"]?[A-Za-z0-9_\-]{20,}['\"]?",
                    score=0.60,
                ),
                Pattern(
                    name="secret_labeled",
                    # Labeled secrets: secret=xxx, SECRET_KEY: xxx
                    regex=r"(?:secret|SECRET)[_\-]?(?:key|KEY)?\s*[=:]\s*['\"]?[A-Za-z0-9_\-]{16,}['\"]?",
                    score=0.50,
                ),
                Pattern(
                    name="password_labeled",
                    # Labeled passwords - require quoted value to reduce false positives
                    regex=r"(?:password|passwd|pwd)\s*[=:]\s*['\"][^\s'\"\n]{8,}['\"]",
                    score=0.50,
                ),
                Pattern(
                    name="token_labeled",
                    # Labeled tokens: token=xxx, access_token: xxx
                    regex=r"(?:access[_\-]?token|auth[_\-]?token|TOKEN)\s*[=:]\s*['\"]?[A-Za-z0-9_\-\.]{20,}['\"]?",
                    score=0.50,
                ),
            ],
            context=["api", "key", "token", "secret", "password", "credential", "auth", "bearer"]
        )

        # Slack Tokens - specific patterns, lower scores to stay under threshold
        slack_recognizer = PatternRecognizer(
            supported_entity="CREDENTIAL",
            patterns=[
                Pattern(
                    name="slack_bot_token",
                    regex=r"xoxb-[0-9]{10,13}-[0-9]{10,13}-[a-zA-Z0-9]{24}",
                    score=0.60,
                ),
                Pattern(
                    name="slack_user_token",
                    regex=r"xoxp-[0-9]{10,13}-[0-9]{10,13}-[0-9]{10,13}-[a-f0-9]{32}",
                    score=0.60,
                ),
                Pattern(
                    name="slack_workspace_token",
                    regex=r"xoxa-[0-9]{10,13}-[0-9]{10,13}-[a-zA-Z0-9]{24}",
                    score=0.60,
                ),
            ],
            context=["slack", "token", "bot", "webhook"]
        )

        # NPM Tokens
        npm_recognizer = PatternRecognizer(
            supported_entity="CREDENTIAL",
            patterns=[
                Pattern(
                    name="npm_token",
                    regex=r"npm_[A-Za-z0-9]{36}",
                    score=0.60,
                ),
            ],
            context=["npm", "token", "registry", "publish"]
        )

        # Private Keys (PEM format headers)
        private_key_recognizer = PatternRecognizer(
            supported_entity="CREDENTIAL",
            patterns=[
                Pattern(
                    name="private_key_header",
                    regex=r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----",
                    score=0.95,
                ),
                Pattern(
                    name="private_key_openssh",
                    regex=r"-----BEGIN\s+OPENSSH\s+PRIVATE\s+KEY-----",
                    score=0.95,
                ),
            ],
            context=["key", "private", "ssh", "rsa", "pem", "certificate"]
        )

        # Password patterns (high-entropy strings with special characters)
        # Patterns like: z9P@m3Kt#7nY5, PurpleElephant!SkyDolphin
        password_recognizer = PatternRecognizer(
            supported_entity="CREDENTIAL",
            patterns=[
                # Mixed case + digits + special chars (8-20 chars): z9P@m3Kt#7nY5
                Pattern(
                    name="complex_password",
                    regex=r"\b(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@#$%^&*!])[A-Za-z\d@#$%^&*!]{8,20}\b",
                    score=0.75,
                ),
                # CamelCase with special chars: PurpleElephant!SkyDolphin
                Pattern(
                    name="passphrase_special",
                    regex=r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+[!@#$%^&*][A-Z][a-z]+(?:[A-Z][a-z]+)*\b",
                    score=0.80,
                ),
                # REMOVED: word_year_password and pin_6digit patterns
                # These were too broad and caused 14k+ false positives
                # word_year_password matched "Michael1995", "California2020", "January2023"
                # pin_6digit matched any 6-digit number (dates, ZIPs, phone fragments)
                # Only labeled credentials are now detected via generic_credential_recognizer
            ],
            context=["password", "pwd", "pass", "secret", "pin", "passcode",
                     "passphrase", "credentials", "auth", "login", "unlock"]
        )

        self.analyzer.registry.add_recognizer(aws_recognizer)
        self.analyzer.registry.add_recognizer(stripe_recognizer)
        self.analyzer.registry.add_recognizer(github_recognizer)
        self.analyzer.registry.add_recognizer(google_recognizer)
        self.analyzer.registry.add_recognizer(generic_credential_recognizer)
        self.analyzer.registry.add_recognizer(slack_recognizer)
        self.analyzer.registry.add_recognizer(npm_recognizer)
        self.analyzer.registry.add_recognizer(private_key_recognizer)
        self.analyzer.registry.add_recognizer(password_recognizer)

    def _add_id_document_recognizers(self):
        """
        Add pattern recognizers for government-issued ID documents.

        Detects:
        - Passport numbers (US, Canada, Germany, France with context)
        - Driver's license numbers (US states, Canada provinces, UK)

        Note: Patterns are strict to avoid false positives. Context words
        significantly boost confidence for ambiguous patterns.
        """
        # ===================
        # PASSPORT PATTERNS
        # ===================
        # Canadian Passport: 2 letters + 6 digits
        ca_passport = r"\b[A-Z]{2}\d{6}\b"

        # Extended passport: 2 letters + 7-8 digits (various countries)
        extended_passport = r"\b[A-Z]{2}\d{7,8}\b"

        # German Passport: C + F/G/H/J/K + 7 alphanumeric (biometric)
        de_passport = r"\bC[FGHJK][A-Z0-9]{7}\b"

        # French Passport: 2 digits + 2 letters + 5 digits
        fr_passport = r"\b\d{2}[A-Z]{2}\d{5}\b"

        # US Passport: 1 letter + 8 digits (new format only - more distinctive)
        us_passport_new = r"\b[A-Z]\d{8}\b"

        passport_recognizer = PatternRecognizer(
            supported_entity="PASSPORT",
            patterns=[
                Pattern(name="ca_passport", regex=ca_passport, score=0.85),
                Pattern(name="extended_passport", regex=extended_passport, score=0.85),
                Pattern(name="de_passport", regex=de_passport, score=0.90),
                Pattern(name="fr_passport", regex=fr_passport, score=0.85),
                Pattern(name="us_passport_new", regex=us_passport_new, score=0.80),
            ],
            context=["passport", "travel document", "passport number", "passport no",
                     "passeport", "reisepass", "pasaporte", "passaporto",
                     "visa", "immigration", "border", "customs", "travel"]
        )

        # ===========================
        # DRIVER'S LICENSE PATTERNS
        # ===========================
        # Based on official government sources and Microsoft Purview definitions
        # https://learn.microsoft.com/en-us/purview/sit-defn-eu-drivers-license-number

        # --- US STATE FORMATS ---
        # Florida: 1 letter + 12 digits (highly distinctive)
        fl_dl = r"\b[A-Z]\d{12}\b"

        # Illinois: 1 letter + 11 digits (highly distinctive)
        il_dl = r"\b[A-Z]\d{11}\b"

        # California-style: 1 letter + 7 digits (requires context)
        ca_dl = r"\b[A-Z]\d{7}\b"

        # --- CANADA ---
        # Ontario (Canada): 1 letter + 14 digits with dashes (highly distinctive)
        on_dl = r"\b[A-Z]\d{4}[-\s]\d{5}[-\s]\d{5}\b"

        # --- UK ---
        # UK: 16 characters - SSSSS (5 letters or 9) + Y (decade) + MMDDY (5 digits)
        # + CC (2 initials or 9) + DDDDD (5 digits)
        # Source: https://ukdriving.org.uk/driving-licence-number/
        uk_dl = r"\b[A-Z9]{5}\d[0-6]\d{4}[A-Z9]{2}\d{5}\b"

        # UK format alternate: with spaces (SSSSS YMMDD CC DDD DD)
        uk_dl_spaced = r"\b[A-Z9]{5}\s?\d[0-6]\d{4}\s?[A-Z9]{2}\s?\d{5}\b"

        # Northern Ireland: 8 digits only
        ni_dl = r"\b\d{8}\b"

        # --- GERMANY ---
        # German: 11 alphanumeric - [A-Za-z0-9] + 2 digits + 6 alphanumeric + 1 digit + 1 alphanumeric
        # Source: Microsoft Purview sit-defn-germany-drivers-license-number
        de_dl = r"\b[A-Za-z0-9]\d{2}[A-Za-z0-9]{6}\d[A-Za-z0-9]\b"

        # --- FRANCE ---
        # French: 12 digits
        # Source: Microsoft Purview sit-defn-france-drivers-license-number
        fr_dl = r"\b\d{12}\b"

        # --- JAPAN ---
        # Japanese: 12 digits
        # Source: Microsoft Purview sit-defn-japan-drivers-license-number
        jp_dl = r"\b\d{12}\b"

        # --- AUSTRALIA ---
        # Victoria: 8-10 digits
        # Source: https://lookuptax.com/docs/how-to-verify/australia-drivers-License-format
        au_vic_dl = r"\b\d{8,10}\b"

        # NSW: 8 digits
        au_nsw_dl = r"\b\d{8}\b"

        # Queensland: 8 digits starting with 0
        au_qld_dl = r"\b0\d{7}\b"

        # General Australian: letter + 5-6 digits (some states)
        au_letter_dl = r"\b[A-Z]\d{5,6}\b"

        drivers_license_recognizer = PatternRecognizer(
            supported_entity="DRIVERS_LICENSE",
            patterns=[
                # High confidence - distinctive formats
                Pattern(name="fl_dl", regex=fl_dl, score=0.90),
                Pattern(name="il_dl", regex=il_dl, score=0.90),
                Pattern(name="on_dl", regex=on_dl, score=0.95),
                Pattern(name="uk_dl", regex=uk_dl, score=0.95),
                Pattern(name="uk_dl_spaced", regex=uk_dl_spaced, score=0.90),
                Pattern(name="de_dl", regex=de_dl, score=0.85),
                # Medium confidence - need context
                Pattern(name="ca_dl", regex=ca_dl, score=0.70),
                Pattern(name="au_qld_dl", regex=au_qld_dl, score=0.70),
                Pattern(name="au_letter_dl", regex=au_letter_dl, score=0.65),
                # Lower confidence - generic digit formats need strong context
                # Note: fr_dl and jp_dl (12 digits) need context to avoid false positives
            ],
            context=["driver", "license", "licence", "DL", "driver's license",
                     "drivers license", "driving license", "driving licence",
                     "license number", "licence number", "DL#", "DLN",
                     "operator", "motor vehicle", "DMV", "MVA", "RMV",
                     "permis de conduire", "führerschein", "licencia",
                     "運転免許", "运驾照", "führerscheinnummer"]
        )

        # Separate recognizer for generic digit-only patterns (need strong context)
        generic_dl_recognizer = PatternRecognizer(
            supported_entity="DRIVERS_LICENSE",
            patterns=[
                # 12-digit formats (France, Japan) - very generic, need context
                Pattern(name="dl_12_digit", regex=r"\b\d{12}\b", score=0.50),
                # 8-digit formats (Australia, NI) - generic, need context
                Pattern(name="dl_8_digit", regex=r"\b\d{8}\b", score=0.45),
            ],
            context=["driver", "license", "licence", "DL", "driving",
                     "permis", "führerschein", "licencia", "運転免許",
                     "license number", "licence number"]
        )

        self.analyzer.registry.add_recognizer(passport_recognizer)
        self.analyzer.registry.add_recognizer(drivers_license_recognizer)
        self.analyzer.registry.add_recognizer(generic_dl_recognizer)

    def _add_generic_id_recognizers(self):
        """
        Add pattern recognizers for generic alphanumeric IDs.

        Detects:
        - Customer/client IDs: CL1293746, 22USR46123
        - Reference numbers: SM-78321, MK7831
        - Mixed alphanumeric codes: 21MKT456Z
        """
        # Pattern 1: 2-3 uppercase letters + dash + 4-6 digits
        # Examples: SM-78321, CL-12345, REF-123456
        id_letter_dash_digits = r"\b[A-Z]{2,3}-\d{4,6}\b"

        # Pattern 2: 2 uppercase letters + 4-7 digits (no dash)
        # Examples: CL1293746, MK7831, ID12345
        id_letters_digits = r"\b[A-Z]{2}\d{4,7}\b"

        # Pattern 3: Digits + letters + digits pattern (mixed)
        # Examples: 21MKT456Z, 22USR46123
        id_mixed_pattern = r"\b\d{2}[A-Z]{2,4}\d{3,5}[A-Z]?\b"

        # Pattern 4: Airport-style 3-letter + dash + 6-8 digits
        # Examples: LAX-12345678, JFK-123456
        id_airport_code = r"\b[A-Z]{3}-\d{6,8}\b"

        # Pattern 5: Pure digits (6-10) with context
        # Examples: 123456, 1234567890 (customer ID, account number)
        id_digits_only = r"\b\d{6,10}\b"

        # Pattern 6: Short hex IDs (8 chars)
        # Examples: a1b2c3d4, DEADBEEF
        id_hex_short = r"\b[a-fA-F0-9]{8}\b"

        generic_id_recognizer = PatternRecognizer(
            supported_entity="ID",
            patterns=[
                Pattern(name="id_letter_dash_digits", regex=id_letter_dash_digits, score=0.85),
                Pattern(name="id_letters_digits", regex=id_letters_digits, score=0.55),  # Lowered: generic pattern creates massive FPs
                Pattern(name="id_mixed_pattern", regex=id_mixed_pattern, score=0.55),  # Lowered: needs context boost
                Pattern(name="id_airport_code", regex=id_airport_code, score=0.70),  # Lowered: needs some context
                Pattern(name="id_digits_only", regex=id_digits_only, score=0.40),  # Reduced - needs strong context
                Pattern(name="id_hex_short", regex=id_hex_short, score=0.55),  # Reduced - needs context
            ],
            context=["id", "reference", "number", "account", "customer", "client",
                     "case", "ticket", "order", "tracking", "confirmation", "ref",
                     "member", "subscriber", "policy", "claim", "identifier", "session"]
        )

        self.analyzer.registry.add_recognizer(generic_id_recognizer)

    def _add_vehicle_recognizers(self):
        """
        Add pattern recognizers for vehicle identification.

        Detects:
        - VIN (Vehicle Identification Number) - ISO 3779 standard
        - License plates (international formats)
        """
        # VIN: 17 characters, alphanumeric excluding I, O, Q
        # Position 9 is check digit (0-9 or X)
        # Format: WMI (3) + VDS (6) + VIS (8)
        vin_regex = r"\b[A-HJ-NPR-Z0-9]{17}\b"

        # More specific VIN pattern with known manufacturer prefixes
        # World Manufacturer Identifiers for common manufacturers
        # 1, 4, 5 = USA, 2 = Canada, 3 = Mexico, J = Japan, K = Korea,
        # S = UK, W = Germany, Z = Italy, Y = Sweden/Finland
        vin_with_wmi = r"\b[1-5JKSWZY][A-HJ-NPR-Z0-9]{16}\b"

        vin_recognizer = PatternRecognizer(
            supported_entity="VEHICLE_ID",
            patterns=[
                Pattern(name="vin_specific", regex=vin_with_wmi, score=0.90),
                Pattern(name="vin_generic", regex=vin_regex, score=0.80),
            ],
            context=["VIN", "vehicle identification", "vehicle identification number",
                     "chassis", "chassis number", "serial", "car", "automobile",
                     "truck", "motorcycle", "vehicle", "registration", "title",
                     "odometer", "carfax", "autocheck"]
        )

        self.analyzer.registry.add_recognizer(vin_recognizer)

        # License Plate patterns (international formats)
        # These are highly variable by jurisdiction but common patterns include:
        license_plate_recognizer = PatternRecognizer(
            supported_entity="VEHICLE_ID",
            patterns=[
                # European style: XX-999-XX, XX 999 XX (France, Belgium, etc.)
                Pattern(
                    name="eu_license_2_3_2",
                    regex=r"\b[A-Z]{2}[-\s]?\d{3}[-\s]?[A-Z]{2}\b",
                    score=0.80,
                ),
                # European style: XXX-9999, XXX 9999 (Netherlands, etc.)
                Pattern(
                    name="eu_license_3_4",
                    regex=r"\b[A-Z]{3}[-\s]?\d{3,4}\b",
                    score=0.30,  # Reduced - matches acronyms+numbers, needs context
                ),
                # UK style: XX99 XXX (new format) or X999 XXX (old format)
                Pattern(
                    name="uk_license_new",
                    regex=r"\b[A-Z]{2}\d{2}\s?[A-Z]{3}\b",
                    score=0.80,
                ),
                # German style: X-XX 9999, XX-XX 9999 (city code + letters + numbers)
                Pattern(
                    name="de_license",
                    regex=r"\b[A-Z]{1,3}[-\s][A-Z]{1,2}\s?\d{1,4}\b",
                    score=0.75,
                ),
                # US/Canada style: ABC-1234, ABC 1234, 1ABC234
                Pattern(
                    name="us_license_3_4",
                    regex=r"\b[A-Z]{3}[-\s]?\d{4}\b",
                    score=0.35,  # Reduced - generic format, needs context
                ),
                Pattern(
                    name="us_license_mixed",
                    regex=r"\b\d[A-Z]{3}\d{3}\b",
                    score=0.75,
                ),
                # Generic: 2-3 letters + space + 4 digits + space + 2 letters (e.g., "BB 6512 ZY")
                Pattern(
                    name="generic_license_space",
                    regex=r"\b[A-Z]{2,3}\s\d{4}\s[A-Z]{2}\b",
                    score=0.70,  # Reduced from 0.85 for better precision
                ),
                # Japanese style: 99-99 or hiragana + numbers
                Pattern(
                    name="jp_license",
                    regex=r"\b\d{2}[-\s]?\d{2,4}\b",
                    score=0.30,  # Very generic, context required
                ),
            ],
            context=["license", "plate", "registration", "vehicle", "car", "truck",
                     "automobile", "license plate", "number plate", "reg", "tag",
                     "license_plate", "licence", "matriculation"]
        )

        self.analyzer.registry.add_recognizer(license_plate_recognizer)

    def _add_device_recognizers(self):
        """
        Add pattern recognizers for device identifiers.

        Detects:
        - IMEI (International Mobile Equipment Identity) - 15 digits with specific format
        - MAC Address (Media Access Control) - 6 hex pairs with separators
        - UUID (Universally Unique Identifier) - 8-4-4-4-12 format with dashes

        Note: Patterns are strict to avoid false positives. Only well-formatted
        identifiers with clear separators are detected with high confidence.
        """
        # MAC Address: 6 pairs of hex digits WITH SEPARATORS (high specificity)
        # Formats: XX:XX:XX:XX:XX:XX or XX-XX-XX-XX-XX-XX
        mac_colon = r"\b([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b"
        mac_dash = r"\b([0-9A-Fa-f]{2}-){5}[0-9A-Fa-f]{2}\b"

        # UUID: Standard format 8-4-4-4-12 WITH DASHES (highly distinctive)
        uuid_standard = r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"

        # IMEI: 15 digits
        # Formatted: XX-XXXXXX-XXXXXX-X or similar
        imei_formatted = r"\b\d{2}[-\s]\d{6}[-\s]\d{6}[-\s]\d\b"
        # Plain 15 digits - common TAC prefixes: 35, 86, 49, 01, 45, 36, 87, 46, etc.
        # These are assigned to device manufacturers
        imei_plain = r"\b(?:35|86|49|01|45|36|87|46|91|92|99|01|02|03|04)\d{13}\b"

        mac_recognizer = PatternRecognizer(
            supported_entity="DEVICE_ID",
            patterns=[
                Pattern(name="mac_colon", regex=mac_colon, score=0.95),
                Pattern(name="mac_dash", regex=mac_dash, score=0.95),
            ],
            context=["MAC", "MAC address", "hardware", "network", "ethernet",
                     "wifi", "wireless", "adapter", "interface", "NIC",
                     "physical address", "hardware address"]
        )

        uuid_recognizer = PatternRecognizer(
            supported_entity="DEVICE_ID",
            patterns=[
                Pattern(name="uuid_standard", regex=uuid_standard, score=0.95),
            ],
            context=["UUID", "GUID", "unique identifier", "device ID",
                     "identifier", "device identifier", "tracking"]
        )

        # IMEI recognizer - formatted and plain versions
        imei_recognizer = PatternRecognizer(
            supported_entity="DEVICE_ID",
            patterns=[
                Pattern(name="imei_formatted", regex=imei_formatted, score=0.95),
                Pattern(name="imei_plain", regex=imei_plain, score=0.85),
            ],
            context=["IMEI", "mobile", "phone", "device", "handset", "cellular",
                     "equipment", "identity", "serial"]
        )

        self.analyzer.registry.add_recognizer(mac_recognizer)
        self.analyzer.registry.add_recognizer(uuid_recognizer)
        self.analyzer.registry.add_recognizer(imei_recognizer)

    def _add_biometric_recognizers(self):
        """
        Add pattern recognizers for biometric identifiers.

        Detects:
        - Biometric IDs with prefixes (BIO-, BIOMETRIC-, FP-, SCAN-)
        """
        biometric_recognizer = PatternRecognizer(
            supported_entity="BIOMETRIC",
            patterns=[
                # BIO- prefix followed by 8-12 digits: BIO-7459126830
                Pattern(
                    name="bio_prefix_digits",
                    regex=r"\bBIO[-_]?\d{8,12}\b",
                    score=0.95,
                ),
                # BIOMETRIC- prefix: BIOMETRIC-123456789
                Pattern(
                    name="biometric_prefix",
                    regex=r"\bBIOMETRIC[-_]?\d{6,12}\b",
                    score=0.95,
                ),
                # FP- or SCAN- prefix
                Pattern(
                    name="fp_scan_prefix",
                    regex=r"\b(?:FP|SCAN)[-_]?[A-Z0-9]{6,12}\b",
                    score=0.90,
                ),
                # Letter + 8-12 digits (expanded for better recall): D64837291598, M72983456187
                Pattern(
                    name="biometric_letter_digits",
                    regex=r"\b[A-Z]\d{8,12}\b",
                    score=0.85,  # Boosted from 0.75 for better recall
                ),
            ],
            context=["biometric", "fingerprint", "iris", "scan", "retina", "face",
                     "facial", "voice", "palm", "vein", "biometric ID", "bio",
                     "identification", "authentication", "scanner"]
        )

        self.analyzer.registry.add_recognizer(biometric_recognizer)

    def _add_credential_recognizers(self):
        """
        Add pattern recognizers for credentials.

        Detects:
        - Passwords with context labels (password=, pwd:, etc.)
        - PINs with context (pin:, passcode=)
        - API keys with common prefixes (sk_live_, pk_live_, api_key_, etc.)
        """
        # Password patterns - labeled passwords in configs/logs
        # Matches: password=mysecret123, pwd: hunter2, pass = "secret"
        password_labeled = r'(?:password|passwd|pwd|pass|secret)\s*[:=]\s*["\']?[^\s"\',]{4,}["\']?'

        # API key patterns with common prefixes
        # Stripe: sk_live_xxx, pk_live_xxx, sk_test_xxx
        # Generic: api_key_xxx, apikey_xxx, api-key-xxx
        api_key_prefixed = r'\b(?:sk_live_|pk_live_|sk_test_|pk_test_|api_key_|apikey_|api-key-)[a-zA-Z0-9]{16,}\b'

        # Bearer tokens (JWT-like)
        bearer_token = r'\bBearer\s+[a-zA-Z0-9\-_]+\.[a-zA-Z0-9\-_]+\.[a-zA-Z0-9\-_]+\b'

        # PIN patterns with context
        # Matches: pin: 1234, PIN = 123456, passcode: 0000
        pin_labeled = r'(?:pin|passcode|security code)\s*[:=]\s*\d{4,8}\b'

        # Short password with special characters
        # Matches: "]8Szx", "}ICd2", "{AR93;FE{l", "X1p83<A", "rz@8Bim"
        credential_special_chars_short = r'[a-zA-Z0-9]{2,}[!@#$%^&*()_+=\[\]{};:\'",<.>/?\\|-][a-zA-Z0-9]{1,}'

        # Password with mixed symbols
        # Matches: "{R12L-w#VIg", "cuHUga^8dE", "Zn_3@8B"
        credential_mixed_symbols = r'[{}\[\]<>+=\\|]{1}[a-zA-Z0-9]{2,}[!@#$%^&*]{1}'

        credential_recognizer = PatternRecognizer(
            supported_entity="CREDENTIAL",
            patterns=[
                # Scores reduced: max 0.60 + context boost 0.35 = 0.95 < 0.99 threshold
                Pattern(name="password_labeled", regex=password_labeled, score=0.55),
                Pattern(name="api_key_prefixed", regex=api_key_prefixed, score=0.60),
                Pattern(name="bearer_token", regex=bearer_token, score=0.55),
                Pattern(name="pin_labeled", regex=pin_labeled, score=0.50),
                # NEW: Short password patterns with special characters
                Pattern(name="credential_special_chars_short", regex=credential_special_chars_short, score=0.60),
                Pattern(name="credential_mixed_symbols", regex=credential_mixed_symbols, score=0.65),
            ],
            context=["password", "secret", "key", "token", "pin", "passcode",
                     "credential", "auth", "authentication", "api", "access"]
        )

        self.analyzer.registry.add_recognizer(credential_recognizer)

    def _add_username_recognizers(self):
        """
        Add pattern recognizers for usernames.

        Detects:
        - Alphanumeric usernames with digits (e.g., "1995LA", "91msalji", "50taham")
        - Usernames with dot separators (e.g., "setsuko.arnaz", "1958célina.grintschacher")
        - Long alphanumeric usernames (e.g., "tyefcfowvlrtvxxj719154")
        - Mixed case alphanumeric (e.g., "hviotd04")
        - Hyphenated usernames (e.g., "claude-michel")
        - IDCARD patterns (e.g., "IDCARD_D(JR34446NF")
        - Simple lowercase names (e.g., "narumi", "arseta")
        """
        # Alphanumeric username with digits: letters followed by digits OR digits followed by letters
        # Matches: "1995LA", "91msalji", "50taham", "supansa1948"
        username_alphanum_digits = r'\b[a-z]{2,}[0-9]{2,}\b|\b[0-9]{2,}[a-z]{2,}\b'

        # Username with dot separator: lowercase words separated by dot (require lowercase)
        # Matches: "setsuko.arnaz", "john.doe"
        username_dot_separator = r'\b[a-z]+\.[a-z]+\b'

        # Long alphanumeric username: many lowercase letters followed by digits
        # Matches: "tyefcfowvlrtvxxj719154", "qfttcrfhnurgqy8562"
        username_long_alphanum = r'\b[a-z]{8,}[0-9]+\b'

        # Mixed case alphanumeric: lowercase letters followed by 2+ digits
        # Matches: "hviotd04", "seranoludot"
        username_mixed = r'\b[a-z]+[0-9]{2,}\b'

        # Hyphenated username: lowercase words separated by hyphen
        # Matches: "claude-michel", "jean-pierre"
        username_hyphenated = r'\b[a-z]+-[a-z]+\b'

        # IDCARD pattern: IDCARD followed by underscore, letter, parenthesis, alphanumeric code
        # Matches: "IDCARD_D(JR34446NF", "IDCARD_E(RD50242AA", "IDCARD_F(4280327245"
        username_idcard = r'\bIDCARD_[A-Z]\([A-Z0-9]+\b'

        username_recognizer = PatternRecognizer(
            supported_entity="USERNAME",
            patterns=[
                Pattern(name="username_alphanum_digits", regex=username_alphanum_digits, score=0.65),
                Pattern(name="username_dot_separator", regex=username_dot_separator, score=0.60),
                Pattern(name="username_long_alphanum", regex=username_long_alphanum, score=0.70),
                Pattern(name="username_mixed", regex=username_mixed, score=0.55),
                Pattern(name="username_hyphenated", regex=username_hyphenated, score=0.58),
                Pattern(name="username_idcard", regex=username_idcard, score=0.75),
            ],
            context=["username", "user", "login", "account", "profile", "handle", "userid"]
        )

        self.analyzer.registry.add_recognizer(username_recognizer)

    def _add_id_recognizers(self):
        """
        Add pattern recognizers for generic ID formats.

        Detects:
        - Alphanumeric ID codes (e.g., "MX35698XH", "HG67819UU", "SX13700MN")
        - Format: 2-3 letters + 4-6 digits + 2-3 letters
        """
        from presidio_analyzer import Pattern, PatternRecognizer

        # Standard pattern: 2-3 uppercase letters + 4-6 digits + 2-3 uppercase letters
        # Matches: "MX35698XH", "HG67819UU", "SX13700MN", "QZ42138BZ"
        id_standard = r'\b[A-Z]{2,3}\d{4,6}[A-Z]{2,3}\b'

        # Alternative: 1-2 letters + 5-8 digits (no trailing letters)
        # Matches: "PM74058", "JA77673"
        id_simple = r'\b[A-Z]{1,2}\d{5,8}\b'

        id_recognizer = PatternRecognizer(
            supported_entity="ID",
            patterns=[
                Pattern(name="id_standard", regex=id_standard, score=0.75),
                Pattern(name="id_simple", regex=id_simple, score=0.60),
            ],
            context=["id", "identifier", "code", "reference", "number"]
        )

        self.analyzer.registry.add_recognizer(id_recognizer)

    def _add_network_recognizers(self):
        """
        Add pattern recognizers for network identifiers.

        Detects:
        - MAC addresses (already in device recognizer, but also emitted as NETWORK)
        - HTTP cookies with session identifiers
        - Session IDs
        """
        # MAC Address: 6 pairs of hex digits
        # Formats: XX:XX:XX:XX:XX:XX or XX-XX-XX-XX-XX-XX
        mac_colon = r"\b([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b"
        mac_dash = r"\b([0-9A-Fa-f]{2}-){5}[0-9A-Fa-f]{2}\b"

        # HTTP cookie patterns (common session cookie names)
        # Matches: session_id=abc123, PHPSESSID=xxx, JSESSIONID=xxx
        http_cookie = r'(?:session_id|sessionid|PHPSESSID|JSESSIONID|csrftoken|_ga|_gid|cf_clearance)\s*[:=]\s*[a-zA-Z0-9_\-]{10,}'

        # Full cookie strings with attributes (from ground truth)
        # Matches: consent_agreement=zxv82b7n9l; Path=/; Expires=...
        full_cookie = r'[a-z_]+=[a-zA-Z0-9_\-]+;\s*Path=/'

        # Generic session ID patterns
        # Matches: sid=xxx, session=xxx
        session_id = r'\b(?:sid|session|sess_id)\s*[:=]\s*[a-zA-Z0-9_\-]{16,}\b'

        # Device identifiers (hex strings from ground truth)
        device_hex = r'\b(?=[a-f0-9]*[a-f])[a-f0-9]{16}\b'
        device_uuid = r'\b[A-F0-9]{8}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{12}\b'

        network_recognizer = PatternRecognizer(
            supported_entity="NETWORK",
            patterns=[
                Pattern(name="mac_colon", regex=mac_colon, score=0.90),
                Pattern(name="mac_dash", regex=mac_dash, score=0.90),
                Pattern(name="http_cookie", regex=http_cookie, score=0.85),
                Pattern(name="full_cookie", regex=full_cookie, score=0.85),
                Pattern(name="session_id", regex=session_id, score=0.80),
                # Re-enabled device_hex with context-dependent scoring
                Pattern(name="device_hex_16char", regex=device_hex, score=0.75),
                Pattern(name="device_uuid", regex=device_uuid, score=0.90),
            ],
            context=["mac", "address", "cookie", "session", "device", "network",
                     "ethernet", "wifi", "hardware", "identifier", "tracking",
                     "imei", "mobile", "phone"]
        )

        self.analyzer.registry.add_recognizer(network_recognizer)

    def _add_ip_address_recognizers(self):
        """
        Add pattern recognizers for IP addresses (IPv4 and IPv6).

        Detects:
        - IPv4: Standard dotted decimal (192.168.1.1)
        - IPv4 with port: 192.168.1.1:8080
        - IPv6: Full and compressed formats
        """
        # IPv4: Four octets (0-255) separated by dots
        # Each octet: 0-255 (handles leading zeros)
        ipv4_octet = r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"
        ipv4_pattern = rf"\b{ipv4_octet}\.{ipv4_octet}\.{ipv4_octet}\.{ipv4_octet}\b"

        # IPv4 with optional port
        ipv4_with_port = rf"\b{ipv4_octet}\.{ipv4_octet}\.{ipv4_octet}\.{ipv4_octet}(?::\d{{1,5}})?\b"

        # IPv6: Full format (8 groups of 4 hex digits)
        ipv6_full = r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"
        # Compressed IPv6 with :: in middle (e.g., 2001:db8::8a2e:370:7334, fe80::1)
        ipv6_compressed_mid = r"\b(?:[0-9a-fA-F]{1,4}:){1,6}:(?:[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4}){0,5})\b"
        # Compressed IPv6 with leading :: (e.g., ::1, ::ffff:192.168.1.1)
        ipv6_compressed_start = r"(?<![:\w])::(?:[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4}){0,6})(?![:\w])"
        # IPv6-mapped IPv4 (e.g., ::ffff:192.168.1.1)
        ipv6_mapped_ipv4 = rf"(?<![:\w])::ffff:{ipv4_octet}\.{ipv4_octet}\.{ipv4_octet}\.{ipv4_octet}(?![:\w])"

        ip_recognizer = PatternRecognizer(
            supported_entity="IP_ADDRESS",
            patterns=[
                Pattern(name="ipv4_standard", regex=ipv4_pattern, score=0.90),  # Boosted from 0.85
                Pattern(name="ipv4_with_port", regex=ipv4_with_port, score=0.95),  # Boosted from 0.90
                Pattern(name="ipv6_full", regex=ipv6_full, score=0.95),
                Pattern(name="ipv6_compressed_mid", regex=ipv6_compressed_mid, score=0.90),  # Boosted from 0.85
                Pattern(name="ipv6_compressed_start", regex=ipv6_compressed_start, score=0.85),  # Boosted from 0.80
                Pattern(name="ipv6_mapped_ipv4", regex=ipv6_mapped_ipv4, score=0.90),
            ],
            context=["ip", "ip address", "ipv4", "ipv6", "server", "host", "network",
                     "address", "connection", "client", "remote", "local", "localhost"]
        )

        self.analyzer.registry.add_recognizer(ip_recognizer)

    def _add_url_recognizers(self):
        """
        Add pattern recognizers for URL detection.

        Detects:
        - Full URLs with http/https protocol
        - WWW URLs without protocol
        - Subdomain URLs (mail.domain.com, blog.domain.org, etc.)
        """
        url_recognizer = PatternRecognizer(
            supported_entity="URL",
            patterns=[
                # Full URLs with protocol (http/https)
                Pattern(
                    name="url_http_https",
                    regex=r"\bhttps?://[^\s<>\"{}|\\^`\[\]]+",
                    score=0.95,
                ),
                # WWW prefix (no protocol)
                Pattern(
                    name="url_www",
                    regex=r"\bwww\.[a-z0-9][a-z0-9\-]*(?:\.[a-z0-9][a-z0-9\-]*)*\.[a-z]{2,}(?:/[^\s<>\"{}|\\^`\[\]]*)?",
                    score=0.90,
                ),
                # Subdomain URLs (mail.domain.com, blog.domain.org, etc.)
                Pattern(
                    name="url_subdomain",
                    regex=r"\b(?:mail|blog|news|login|app|api|cdn|ftp|admin|portal|dashboard)\.[a-z][a-z0-9_\-]*(?:\.[a-z]{2,})+(?:/[^\s<>\"{}|\\^`\[\]]*)?",
                    score=0.85,
                ),
                # Bare domain with path (domain.com/path)
                Pattern(
                    name="url_bare_with_path",
                    regex=r"\b[a-z0-9][a-z0-9\-]*\.(?:com|org|net|edu|gov|io|co|app|dev|ai|me|info|biz|us|uk|ca|de|fr|jp|cn|ru|br|in|au)/[^\s<>\"{}|\\^`\[\]]+",
                    score=0.85,
                ),
                # Common TLDs without path (example.com, test.io, app.dev)
                # Must have alphanumeric before TLD and context to avoid false positives
                Pattern(
                    name="url_bare_domain",
                    regex=r"\b[a-z][a-z0-9\-]{1,62}\.(?:com|org|net|io|co|app|dev|ai)\b",
                    score=0.70,  # Lower score - needs context
                ),
                # Bank/company domains commonly found in PDFs (usbank.com, chase.com)
                Pattern(
                    name="url_bank_domain",
                    regex=r"\b(?:us)?bank\.com|chase\.com|wellsfargo\.com|citi\.com|bofa\.com|capitalone\.com|discover\.com|paypal\.com|venmo\.com|zelle\.com\b",
                    score=0.90,
                ),
                # OCR-friendly URL pattern (allows minor spacing issues after domain)
                Pattern(
                    name="url_ocr_friendly",
                    regex=r"\b[a-z][a-z0-9]{2,30}\s*\.\s*(?:com|org|net|gov|edu|io|co)\b",
                    score=0.75,
                ),
                # Common financial/bank websites in documents
                Pattern(
                    name="url_financial_sites",
                    regex=r"\b(?:bankofamerica|wellsfargo|citibank|hsbc|barclays|natwest|lloyds|santander|"
                          r"fidelity|vanguard|schwab|etrade|robinhood|ameritrade|merrill|"
                          r"axa|prudential|metlife|allstate|geico|statefarm|progressive|"
                          r"stripe|square|shopify|quickbooks|xero|freshbooks|wave)\.(?:com|co\.uk|ca|net)\b",
                    score=0.90,
                ),
                # Extended TLD support for international URLs
                Pattern(
                    name="url_extended_tld",
                    regex=r"\b[a-z][a-z0-9\-]{1,40}\.(?:co\.uk|com\.au|com\.br|co\.nz|com\.sg|co\.jp|co\.kr|co\.in|de|fr|it|es|nl|be|at|ch|se|no|dk|fi|pl|cz|ru|cn|jp|kr|tw|hk|sg|my|th|id|ph|vn|za|ng|eg|ae|sa|il)\b",
                    score=0.80,
                ),
                # URL preceded by common context words (visit, go to, at)
                Pattern(
                    name="url_with_context",
                    regex=r"(?:visit|go\s+to|at|see|check)\s+([a-z][a-z0-9\-]*\.(?:com|org|net|gov|edu|io|co|uk))",
                    score=0.88,
                ),
            ],
            context=["url", "website", "webpage", "link", "site", "http", "www", "visit", "browse", "go to", "check out"]
        )
        self.analyzer.registry.add_recognizer(url_recognizer)

        # Add urlextract-based recognizer for better detection of bare domains and edge cases
        if URLEXTRACT_AVAILABLE:
            self._add_urlextract_recognizer()

    def _add_urlextract_recognizer(self):
        """
        Add urlextract-based URL recognizer for comprehensive URL detection.

        urlextract is better than regex for:
        - Bare domains (example.com, test.io)
        - Subdomains without common prefixes (api.stripe.com)
        - IP addresses with ports (192.168.1.1:8080)
        - International TLDs
        - URLs with complex paths
        """
        from presidio_analyzer import EntityRecognizer, RecognizerResult, AnalysisExplanation

        class URLExtractRecognizer(EntityRecognizer):
            """Custom recognizer using urlextract library."""

            ENTITIES = ["URL"]

            def __init__(self):
                super().__init__(
                    supported_entities=self.ENTITIES,
                    supported_language="en",
                    name="URLExtractRecognizer",
                )
                self._extractor = None

            def load(self) -> None:
                """Lazy-load the URL extractor."""
                global _url_extractor
                if _url_extractor is None:
                    _url_extractor = URLExtract()
                self._extractor = _url_extractor

            def analyze(self, text: str, entities, nlp_artifacts=None):
                """Extract URLs from text using urlextract."""
                if "URL" not in entities:
                    return []

                if self._extractor is None:
                    self.load()

                results = []
                try:
                    # Get URLs with their positions
                    for url, (start, end) in self._extractor.gen_urls(text, with_schema_only=False):
                        # Skip very short matches (likely false positives)
                        if len(url) < 4:
                            continue

                        # Skip email-like patterns (handled by EMAIL recognizer)
                        if '@' in url and not url.startswith(('http', 'ftp', 'mailto')):
                            continue

                        # Skip localhost without port (too generic)
                        if url.lower() == 'localhost':
                            continue

                        # Determine confidence based on URL characteristics
                        score = 0.80  # Base score (boosted)

                        # Boost for protocol
                        if url.startswith(('http://', 'https://', 'ftp://')):
                            score = 0.95
                        # Boost for www
                        elif url.lower().startswith('www.'):
                            score = 0.92
                        # Boost for paths
                        elif '/' in url:
                            score = 0.88
                        # Boost for common TLDs
                        elif any(url.lower().endswith(tld) for tld in ['.com', '.org', '.net', '.io', '.co', '.edu', '.gov', '.uk', '.de', '.fr', '.ca', '.au']):
                            score = 0.85

                        explanation = AnalysisExplanation(
                            recognizer=self.name,
                            original_score=score,
                            pattern_name="urlextract",
                            pattern=None,
                            validation_result=None,
                        )

                        results.append(
                            RecognizerResult(
                                entity_type="URL",
                                start=start,
                                end=end,
                                score=score,
                                analysis_explanation=explanation,
                                recognition_metadata={
                                    "recognizer_name": self.name,
                                    "detected_url": url,
                                }
                            )
                        )
                except Exception:
                    # If urlextract fails, return empty results
                    pass

                return results

        self.analyzer.registry.add_recognizer(URLExtractRecognizer())

    def _add_obfuscated_email_recognizers(self):
        """
        Add recognizers for obfuscated email addresses.

        Detects emails that have been intentionally obfuscated to avoid spam:
        - [at] format: user[at]domain.com
        - Spaced format: user @ domain . com
        - (at) format: user(at)domain.com
        """
        # [at] format: user[at]domain.com, user [at] domain.com
        obfuscated_email = PatternRecognizer(
            supported_entity="EMAIL_ADDRESS",
            patterns=[
                # [at] or (at) format
                Pattern(
                    name="email_at_brackets",
                    regex=r"\b[\w\.\+\-]+\s*[\[\(]at[\]\)]\s*[\w\-]+(?:\.[\w\-]+)+\b",
                    score=0.85,
                ),
                # Spaced @ format: user @ domain . com
                Pattern(
                    name="email_spaced",
                    regex=r"\b[\w\.\+\-]+\s+@\s+[\w\-]+(?:\s*\.\s*[\w\-]+)+\b",
                    score=0.80,
                ),
                # "at" word format: user at domain dot com
                Pattern(
                    name="email_at_word",
                    regex=r"\b[\w\.\+\-]+\s+at\s+[\w\-]+\s+dot\s+[\w\-]+\b",
                    score=0.75,
                ),
            ],
            context=["email", "contact", "mail", "address", "reach", "send"]
        )
        self.analyzer.registry.add_recognizer(obfuscated_email)

    def _add_libpostal_recognizer(self):
        """
        Add libpostal-based address recognizer for high-accuracy global address detection.

        libpostal achieves 99.45% accuracy on global addresses using a CRF model trained
        on OpenStreetMap data. It handles international address formats that regex patterns
        often miss.

        Strategy:
        1. Find candidate address-like strings using broad patterns
        2. Validate candidates using libpostal's parser
        3. Accept as LOCATION if parser returns meaningful address components
        """
        if not LIBPOSTAL_AVAILABLE:
            return

        # Create a custom recognizer class for libpostal
        class LibpostalAddressRecognizer(PatternRecognizer):
            """
            Hybrid recognizer: regex for candidate detection + libpostal for validation.
            """

            def __init__(self):
                # Broad patterns to find address candidates
                patterns = [
                    # Street number + street name pattern (e.g., "123 Main St")
                    Pattern(
                        name="street_address",
                        regex=r"\b\d+\s+[A-Za-z][A-Za-z\s\-\.'']{2,40}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct|Circle|Cir|Place|Pl|Terrace|Ter|Highway|Hwy|Parkway|Pkwy)\b",
                        score=0.50,  # Low initial score, will be boosted by libpostal
                    ),
                    # PO Box pattern
                    Pattern(
                        name="po_box",
                        regex=r"\b(?:P\.?O\.?\s*Box|Post\s+Office\s+Box)\s+\d+\b",
                        score=0.85,
                    ),
                    # Number + words with comma + more words (address-like structure)
                    Pattern(
                        name="address_with_comma",
                        regex=r"\b\d+\s+[A-Za-z][A-Za-z\s\-\.'']{2,30},\s*[A-Za-z][A-Za-z\s\-]{2,30}\b",
                        score=0.50,
                    ),
                ]

                super().__init__(
                    supported_entity="LOCATION",
                    patterns=patterns,
                    context=["address", "street", "location", "mail", "ship", "deliver", "reside"],
                    name="LibpostalAddressRecognizer",
                )

            def analyze(self, text: str, entities, nlp_artifacts=None):
                """Override analyze to add libpostal validation."""
                # Get initial pattern matches
                pattern_results = super().analyze(text, entities, nlp_artifacts)

                validated_results = []
                for result in pattern_results:
                    # Extract the matched text
                    matched_text = text[result.start:result.end]

                    # Validate with libpostal
                    try:
                        parsed = parse_address(matched_text)

                        # Check if libpostal found meaningful address components
                        component_types = {comp[1] for comp in parsed}

                        # Required: must have at least house_number OR road
                        # Boost if we have multiple address components
                        address_components = {'house_number', 'road', 'house', 'po_box', 'unit'}
                        location_components = {'city', 'suburb', 'state', 'postcode', 'country', 'state_district'}

                        has_street = bool(component_types & address_components)
                        has_location = bool(component_types & location_components)
                        num_components = len(component_types & (address_components | location_components))

                        # Stricter validation: require 3+ components OR (street AND location)
                        # This reduces FPs from single-token locations like "Denver" or "Main Street"
                        if num_components >= 3 or (has_street and has_location):
                            # Boost confidence based on component count
                            boost = min(0.45, num_components * 0.10)
                            new_score = min(0.99, result.score + boost)

                            # Create new result with boosted score
                            from presidio_analyzer import RecognizerResult
                            validated_results.append(
                                RecognizerResult(
                                    entity_type=result.entity_type,
                                    start=result.start,
                                    end=result.end,
                                    score=new_score,
                                    analysis_explanation=result.analysis_explanation,
                                    recognition_metadata={
                                        "recognizer_name": "LibpostalAddressRecognizer",
                                        "libpostal_components": list(component_types),
                                    }
                                )
                            )
                    except Exception:
                        # If libpostal fails, keep original result but with lower confidence
                        validated_results.append(result)

                return validated_results

        # Also add a recognizer for international address formats that libpostal excels at
        # but that might not match the regex patterns above
        class LibpostalInternationalRecognizer(PatternRecognizer):
            """
            Recognizer for international address formats using libpostal.
            """

            def __init__(self):
                patterns = [
                    # European style: street name + number (e.g., "Rue de Rivoli 15")
                    Pattern(
                        name="eu_street_number",
                        regex=r"\b(?:Rue|Via|Calle|Avenida|Strasse|Straße|Rua|Piazza|Platz|Plein)\s+[A-Za-z\s\-'']+\s*\d{1,5}\b",
                        score=0.60,
                    ),
                    # Japanese style: prefecture/city + ward + block (simplified)
                    Pattern(
                        name="jp_address",
                        regex=r"\b[A-Za-z]+(?:ken|shi|ku|cho|machi)\b",
                        score=0.50,
                    ),
                    # UK postcode followed by country
                    Pattern(
                        name="uk_postcode_area",
                        regex=r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b",
                        score=0.70,
                    ),
                    # Apartment/Unit/Suite patterns often indicate addresses
                    Pattern(
                        name="unit_indicator",
                        regex=r"\b(?:Apt|Apartment|Unit|Suite|Ste|Floor|Fl)\.?\s*#?\d+\b",
                        score=0.55,
                    ),
                ]

                super().__init__(
                    supported_entity="LOCATION",
                    patterns=patterns,
                    context=["address", "location", "office", "residence", "shipping"],
                    name="LibpostalInternationalRecognizer",
                )

            def analyze(self, text: str, entities, nlp_artifacts=None):
                """Override to validate with libpostal."""
                pattern_results = super().analyze(text, entities, nlp_artifacts)

                validated_results = []
                for result in pattern_results:
                    matched_text = text[result.start:result.end]

                    try:
                        parsed = parse_address(matched_text)
                        component_types = {comp[1] for comp in parsed}

                        # Stricter validation: require 3+ components OR (street AND location)
                        address_components = {'house_number', 'road', 'house', 'po_box', 'unit'}
                        location_components = {'city', 'suburb', 'state', 'postcode', 'country'}
                        meaningful = address_components | location_components

                        has_street = bool(component_types & address_components)
                        has_location = bool(component_types & location_components)
                        num_components = len(component_types & meaningful)

                        if num_components >= 3 or (has_street and has_location):
                            boost = min(0.35, num_components * 0.08)
                            new_score = min(0.95, result.score + boost)

                            from presidio_analyzer import RecognizerResult
                            validated_results.append(
                                RecognizerResult(
                                    entity_type=result.entity_type,
                                    start=result.start,
                                    end=result.end,
                                    score=new_score,
                                    analysis_explanation=result.analysis_explanation,
                                    recognition_metadata={
                                        "recognizer_name": "LibpostalInternationalRecognizer",
                                        "libpostal_components": list(component_types),
                                    }
                                )
                            )
                    except Exception:
                        validated_results.append(result)

                return validated_results

        # Register both recognizers
        self.analyzer.registry.add_recognizer(LibpostalAddressRecognizer())
        self.analyzer.registry.add_recognizer(LibpostalInternationalRecognizer())

    def _add_ssn_recognizers(self):
        """
        Add custom SSN recognizer for test/fake SSNs.

        Presidio's built-in US_SSN recognizer validates area numbers and
        rejects those with reserved/invalid prefixes (000, 666, 900-999).
        This recognizer catches those patterns for comprehensive detection.
        """
        # SSN pattern with any area number (including reserved ones)
        # Format: XXX-XX-XXXX or XXX XX XXXX or XXX.XX.XXXX
        ssn_all_areas = r"\b\d{3}[-\s.]?\d{2}[-\s.]?\d{4}\b"

        ssn_recognizer = PatternRecognizer(
            supported_entity="NATIONAL_ID",  # SSN consolidated under NATIONAL_ID
            patterns=[
                # High-confidence hyphenated SSN: 123-45-6789 (canonical format)
                Pattern(
                    name="ssn_hyphenated",
                    regex=r"\b\d{3}-\d{2}-\d{4}\b",
                    score=0.95,  # Boosted - canonical SSN format
                ),
                # SSN with spaces: 123 45 6789
                Pattern(
                    name="ssn_spaced",
                    regex=r"\b\d{3}\s\d{2}\s\d{4}\b",
                    score=0.90,  # High confidence - proper 3-2-4 format
                ),
                Pattern(name="ssn_all_areas", regex=ssn_all_areas, score=0.85),
                # SSN with dots: 886.134.4633 or 661.98.2231
                Pattern(
                    name="ssn_dot_separator",
                    regex=r"\b\d{3}\.\d{2,3}\.\d{4}\b",
                    score=0.90,  # Boosted for better recall
                ),
                # EIN (Employer Identification Number): XX-XXXXXXX
                Pattern(
                    name="ein_standard",
                    regex=r"\b\d{2}-\d{7}\b",
                    score=0.85,  # High confidence for standard EIN format
                ),
                # Partially masked SSN: XXX-XX-1234 or ***-**-1234
                Pattern(
                    name="ssn_masked",
                    regex=r"\b[Xx*]{3}[-\s][Xx*]{2}[-\s]\d{4}\b",
                    score=0.90,  # High confidence - clearly intentionally masked
                ),
            ],
            context=["ssn", "social security", "social security number", "ss#", "ss #",
                     "tax id", "tin", "taxpayer", "ein", "employer id", "fein", "federal id"]
        )

        # SSN with spaces or no separators (needs stronger context)
        # Format: XXX XXX XXXX (3-3-4 with spaces) or XXXXXXXXX (9 digits no sep)
        # Low base scores - context boost required to reach threshold
        ssn_alternative = PatternRecognizer(
            supported_entity="NATIONAL_ID",  # SSN consolidated under NATIONAL_ID
            patterns=[
                # "755 979 3272" - 3-3-4 with spaces (unusual but found in some docs)
                Pattern(
                    name="ssn_3_3_4_spaces",
                    regex=r"\b\d{3}\s+\d{3}\s+\d{4}\b",
                    score=0.48,  # Above 0.45 threshold; phone-format filter handles overlap
                ),
                # "838-703-8103" - 3-3-4 with dashes (conflicts with phone, needs context)
                Pattern(
                    name="ssn_3_3_4_dashes",
                    regex=r"\b\d{3}-\d{3}-\d{4}\b",
                    score=0.48,  # Above 0.45 threshold; phone-format filter handles overlap
                ),
                # "092.255.5602" - 3-3-4 with dots
                Pattern(
                    name="ssn_3_3_4_dots",
                    regex=r"\b\d{3}\.\d{3}\.\d{4}\b",
                    score=0.48,  # Above 0.45 threshold; phone-format filter handles overlap
                ),
                # "044034803" - 9 digits no separators (very generic, needs context)
                # Score at 0.40 to pass min_score_with_context_similarity (0.40)
                # Without context: 0.40 < 0.55 threshold → filtered
                # With context: 0.40 + 0.35 = 0.75 → passes threshold
                Pattern(
                    name="ssn_no_sep",
                    regex=r"\b\d{9}\b",
                    score=0.40,  # Must be >= 0.40 for context boost to apply
                ),
                # 10 digits no separators: 9186107809, 2925802399
                Pattern(
                    name="ssn_10_digit",
                    regex=r"\b\d{10}\b",
                    score=0.35,  # Low - very generic, needs strong context
                ),
                # 4-4-4 with dashes (some international formats): 2918-7097-9313
                Pattern(
                    name="ssn_4_4_4_dashes",
                    regex=r"\b\d{4}-\d{4}-\d{4}\b",
                    score=0.35,  # Low - matches credit card-like, requires context
                ),
            ],
            context=["ssn", "social security", "social security number", "ss#", "ss #",
                     "social", "tax id", "tin", "taxpayer", "employee id", "national id",
                     "personal id", "citizen id", "identification", "id number",
                     "passport", "driver", "license", "licence", "participant",
                     "id", "number", "card", "student"]
        )

        self.analyzer.registry.add_recognizer(ssn_recognizer)
        self.analyzer.registry.add_recognizer(ssn_alternative)

        # Dotted ID patterns (ground truth uses broader definitions)
        # Format: X.XXX.XXX.XXX or XXX.XXX.XXX (dotted number sequences with context)
        # These could be mistaken for IP addresses without context
        dotted_id = PatternRecognizer(
            supported_entity="NATIONAL_ID",
            patterns=[
                # 10-digit dotted format: 1.234.567.890 or 123.456.7890 or 619.349.8188
                Pattern(
                    name="dotted_10_digit",
                    regex=r"\b\d{3}\.\d{3}\.\d{4}\b",
                    score=0.72,  # High - common national ID format
                ),
                # 12-digit dotted format: 123.456.789.012
                Pattern(
                    name="dotted_12_digit",
                    regex=r"\b\d{3}\.\d{3}\.\d{3}\.\d{3}\b",
                    score=0.65,
                ),
                # 9-digit dotted format: 123.456.789
                Pattern(
                    name="dotted_9_digit",
                    regex=r"\b\d{3}\.\d{3}\.\d{3}\b",
                    score=0.60,  # Common format
                ),
                # Swiss AHV format: 756.xxxx.xxxx.xx
                Pattern(
                    name="swiss_ahv",
                    regex=r"\b756\.\d{4}\.\d{4}\.\d{2}\b",
                    score=0.90,  # Very distinctive format (always starts with 756)
                ),
                # Generic 13-digit dotted: NNN.NNNN.NNNN.NN
                Pattern(
                    name="dotted_13_digit",
                    regex=r"\b\d{3}\.\d{4}\.\d{4}\.\d{2}\b",
                    score=0.70,
                ),
            ],
            context=["id", "national id", "id number", "identification", "identity",
                     "citizen", "personal id", "id card", "id-nummer", "kennung",
                     "número de identificación", "numéro d'identification", "ssn", "social"]
        )
        self.analyzer.registry.add_recognizer(dotted_id)

        # Alphanumeric national ID patterns (letter prefix + digits)
        # Examples: Z47895814, K3985823, E43171991
        alphanumeric_id = PatternRecognizer(
            supported_entity="NATIONAL_ID",
            patterns=[
                # Letter + 7-9 digits: Z47895814, K3985823
                Pattern(
                    name="alpha_prefix_id",
                    regex=r"\b[A-Z]\d{7,9}\b",
                    score=0.75,  # High - distinctive format
                ),
                # 2 letters + 6-8 digits: AB12345678
                Pattern(
                    name="alpha2_prefix_id",
                    regex=r"\b[A-Z]{2}\d{6,8}\b",
                    score=0.75,
                ),
                # 2 letters + 5 digits + 2 letters (UK NINO-like): QD95456MT, RG95837PQ
                Pattern(
                    name="alpha_mixed_id",
                    regex=r"\b[A-Z]{2}\d{4,6}[A-Z]{1,2}\b",
                    score=0.70,
                ),
                # CURP/RFC-like: DANCO-411081-9-013, MALAN-308037-9-763, DUSCA-004192-DA-658
                Pattern(
                    name="curp_like_id",
                    regex=r"\b[A-Z]{4,6}[-.]?\d{6}[-.]?[A-Z\d]{1,2}[-.]?\d{3}\b",
                    score=0.80,
                ),
                # Italian fiscal code-like: DESAN0622749530, NANNE6030669080 (5+ letters + 10 digits)
                Pattern(
                    name="fiscal_code_long",
                    regex=r"\b[A-Z]{4,6}\d{10}\b",
                    score=0.75,
                ),
                # Alphanumeric with dots: L3.82.I5Z3P4E.1, R56JT8873119
                Pattern(
                    name="alpha_dot_mixed",
                    regex=r"\b[A-Z]\d+\.[A-Z\d]+\.[A-Z\d]+\.\d+\b",
                    score=0.65,
                ),
                # Spaced CURP: MULUA 910056 9 416, GERAL 952255 GW 330, OKBA9 511279 9 950
                Pattern(
                    name="curp_spaced",
                    regex=r"\b[A-Z]{3,6}\d?\s+\d{6}\s+\d\s+\d{3}\b",
                    score=0.80,
                ),
                Pattern(
                    name="curp_spaced_letters",
                    regex=r"\b[A-Z]{3,6}\d?\s+\d{6}\s+[A-Z]{1,2}\s+\d{3}\b",
                    score=0.80,
                ),
                # Generic alphanumeric ID: 8-16 chars with mixed letters/numbers
                # Examples: 30ZU7O8Z (8), R56JT8873119 (12), PAMUK355316PL457 (16), 4K4G3Q51DHF (11)
                # Very generic - requires context to avoid FPs
                Pattern(
                    name="generic_alphanum_id",
                    regex=r"\b(?=.*[A-Z])(?=.*\d)[A-Z\d]{8,16}\b",
                    score=0.40,  # Low - requires context boost
                ),
                # Accented name + digits: GÜLLE5612269725, YÂSIN-307067-9-595
                # Supports Unicode letters with diacritics
                Pattern(
                    name="accented_name_digits_id",
                    regex=r"\b[A-ZÀÂÇÉÈÊËÏÎÔÙÛÜŸÆŒ]{3,}[\-\s]?\d{5,}\b",
                    score=0.50,
                ),
                # Dotted format: 324.202.4929 (3-3-4 digits)
                Pattern(
                    name="dotted_numeric_id",
                    regex=r"\b\d{3}\.\d{3}\.\d{4}\b",
                    score=0.65,
                ),
                # French passport/ID: G4FRA15JX19798500410FRANTZ, N0.FRA.97IC6340.5.331104.DARIYAN
                Pattern(
                    name="french_passport_compact",
                    regex=r"\b[A-Z\d]{2}FRA\d{2}[A-Z]{2}\d{4,}[A-Z]+\b",
                    score=0.85,
                ),
                Pattern(
                    name="french_passport_dotted",
                    regex=r"\b[A-Z\d]{2}[.\-\s]?FRA[.\-\s]?\d{2}[A-Z]{2}\d{4}[.\-\s]?\d[.\-\s]?\d{6}[.\-\s]?[A-Z]+\b",
                    score=0.85,
                ),
                # Belgian eID: Jos.Kle.23.G.61.4.NXO, Amr-Duc-19-M-55-7-BXL
                Pattern(
                    name="belgian_eid",
                    regex=r"\b[A-Z][a-z]+[.\-][A-Z][a-z]+[.\-]\d{2}[.\-][A-Z][.\-]\d{2}[.\-]\d[.\-][A-Z]{3}\b",
                    score=0.85,
                ),
                # Multi-part dash format: 1-67-03-31164-226-92, 2-74-01-06585-693-02
                Pattern(
                    name="multi_dash_id",
                    regex=r"\b\d{1,2}[-]\d{2}[-]\d{2}[-]\d{4,5}[-]\d{3}[-]\d{2}\b",
                    score=0.85,
                ),
            ],
            context=["id", "national id", "id number", "identification", "identity",
                     "passport", "license", "permit", "registration", "number"]
        )
        self.analyzer.registry.add_recognizer(alphanumeric_id)

        # French INSEE/Social Security format: XX.XX.XX.XX.XXX.XXX.XX or similar
        # Examples: 59.16.11.53.R58.9, 55.22.39.23.G80.3
        french_insee = PatternRecognizer(
            supported_entity="NATIONAL_ID",
            patterns=[
                # French INSEE format with letters
                Pattern(
                    name="french_insee_alpha",
                    regex=r"\b\d{2}\.\d{2}\.\d{2}\.\d{2}\.[A-Z]\d{2}\.\d\b",
                    score=0.85,  # Very distinctive format
                ),
                # Standard French NIR (13 digits + 2 key)
                Pattern(
                    name="french_nir_spaced",
                    regex=r"\b[12]\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{3}\s?\d{3}\s?\d{2}\b",
                    score=0.80,
                ),
            ],
            context=["insee", "nir", "sécurité sociale", "social security", "numéro"]
        )
        self.analyzer.registry.add_recognizer(french_insee)

    def _add_international_id_recognizers(self):
        """
        Add pattern recognizers for international national ID numbers.

        Detects national identification numbers from various countries:
        - UK National Insurance Number: AB 12 34 56 C
        - Canadian SIN: 123 456 789
        - German Steuer-ID: 12 345 678 901
        - French INSEE (NIR): 1 85 12 75 108 123 45
        - Japanese My Number: 1234 5678 9012
        - Indian Aadhaar: 1234 5678 9012
        - Australian TFN: 123 456 789
        - Brazilian CPF: 123.456.789-00
        - South Korean RRN: 850315-1234567
        - Italian Codice Fiscale: RSSSFO90L62H501Z
        - Spanish DNI/NIE: 12345678A or X1234567A
        - Polish PESEL: 78110812345 (11 digits)
        - Swedish Personnummer: 197811081234 or YYMMDD-XXXX
        - Danish CPR: 081178-1234
        - Norwegian Fødselsnummer: 07038712345 (11 digits)
        - Singapore NRIC: S8282828A
        - Irish PPS: 1234567WA
        - Chinese ID: 18 digits
        - Taiwan ID: A123456789
        """
        # UK National Insurance Number: 2 letters, 6 digits, 1 letter
        # Format: AB 12 34 56 C or AB123456C
        uk_nino = r"\b[A-CEGHJ-PR-TW-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b"

        # Canadian SIN: 9 digits in groups of 3
        # Format: 123 456 789 or 123-456-789
        canadian_sin = r"\b\d{3}[\s-]\d{3}[\s-]\d{3}\b"

        # German Steuer-ID (Tax ID): 11 digits
        # Format: 12 345 678 901 or 12345678901
        german_steuer = r"\b\d{2}\s?\d{3}\s?\d{3}\s?\d{3}\b"

        # French INSEE (NIR): 15 digits (13 + 2 digit key)
        # Format: 1 85 03 75 108 123 45 (sex, year, month, dept, commune, order, key)
        french_insee = r"\b[12]\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{3}\s?\d{3}\s?\d{2}\b"

        # Japanese My Number: 12 digits
        # Format: 1234 5678 9012 or 123456789012
        japanese_mynumber = r"\b\d{4}\s?\d{4}\s?\d{4}\b"

        # Indian Aadhaar: 12 digits
        # Format: 1234 5678 9012 or 123456789012
        indian_aadhaar = r"\b\d{4}\s?\d{4}\s?\d{4}\b"

        # Australian TFN: 9 digits
        # Format: 123 456 789 or 123-456-789
        australian_tfn = r"\b\d{3}\s?\d{3}\s?\d{3}\b"

        # Brazilian CPF: 11 digits
        # Format: 123.456.789-00
        brazilian_cpf = r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b"

        # South Korean RRN: 13 digits (6 + 7 with dash)
        # Format: 850315-1234567 (birthdate + gender/birth year indicator + serial)
        korean_rrn = r"\b\d{6}-[1-8]\d{6}\b"

        # Italian Codice Fiscale: 16 alphanumeric chars
        # Format: RSSSFO90L62H501Z
        italian_cf = r"\b[A-Z]{6}\d{2}[A-EHLMPR-T]\d{2}[A-Z]\d{3}[A-Z]\b"

        # Spanish DNI: 8 digits + letter, NIE: X/Y/Z + 7 digits + letter
        spanish_dni = r"\b\d{8}[A-Z]\b"
        spanish_nie = r"\b[XYZ]\d{7}[A-Z]\b"

        # Polish PESEL: 11 digits
        polish_pesel = r"\b\d{11}\b"

        # Swedish Personnummer: YYYYMMDD-XXXX or YYMMDD-XXXX
        swedish_pn = r"\b(?:19|20)?\d{6}[-+]\d{4}\b"

        # Danish CPR: DDMMYY-XXXX
        danish_cpr = r"\b\d{6}-\d{4}\b"

        # Norwegian Fødselsnummer: 11 digits (DDMMYYXXXXX)
        norwegian_fn = r"\b\d{11}\b"

        # Singapore NRIC/FIN: S/T/F/G + 7 digits + letter
        singapore_nric = r"\b[STFG]\d{7}[A-Z]\b"

        # Irish PPS: 7 digits + 1-2 letters
        irish_pps = r"\b\d{7}[A-W][A-IW]?\b"

        # Chinese ID: 18 digits (last can be X)
        chinese_id = r"\b\d{17}[\dX]\b"

        # Taiwan ID: 1 letter + 9 digits
        taiwan_id = r"\b[A-Z]\d{9}\b"

        # UAE Emirates ID: 784-YYYY-NNNNNNN-C
        uae_eid = r"\b784-\d{4}-\d{7}-\d\b"

        # Create recognizers for different ID types
        uk_nino_recognizer = PatternRecognizer(
            supported_entity="NATIONAL_ID",
            patterns=[
                Pattern(name="uk_nino", regex=uk_nino, score=0.85),
            ],
            context=["ni", "national insurance", "nino", "ni number", "insurance number"]
        )

        canadian_sin_recognizer = PatternRecognizer(
            supported_entity="NATIONAL_ID",
            patterns=[
                # Reduced from 0.75 - overlaps with phone patterns (XXX-XXX-XXX)
                Pattern(name="canadian_sin", regex=canadian_sin, score=0.50),
            ],
            context=["sin", "social insurance", "social insurance number", "sn", "sin number"]
        )

        german_id_recognizer = PatternRecognizer(
            supported_entity="NATIONAL_ID",
            patterns=[
                # Reduced from 0.70 - overlaps with phone patterns
                Pattern(name="german_steuer", regex=german_steuer, score=0.45),
            ],
            context=["steuer", "tax", "steuer-id", "steuernummer", "identifikationsnummer", "steuerliche"]
        )

        french_insee_recognizer = PatternRecognizer(
            supported_entity="NATIONAL_ID",
            patterns=[
                Pattern(name="french_insee", regex=french_insee, score=0.85),
            ],
            context=["insee", "nir", "securité sociale", "carte vitale", "numéro de sécurité"]
        )

        japanese_id_recognizer = PatternRecognizer(
            supported_entity="NATIONAL_ID",
            patterns=[
                # Reduced from 0.70 - identical pattern to Indian Aadhaar
                Pattern(name="japanese_mynumber", regex=japanese_mynumber, score=0.50),
            ],
            context=["my number", "マイナンバー", "individual number", "kojin bango", "mynumber", "japan"]
        )

        indian_aadhaar_recognizer = PatternRecognizer(
            supported_entity="NATIONAL_ID",
            patterns=[
                # Reduced from 0.70 - identical pattern to Japanese My Number
                Pattern(name="indian_aadhaar", regex=indian_aadhaar, score=0.50),
            ],
            context=["aadhaar", "aadhar", "uid", "unique identification", "india", "indian"]
        )

        brazilian_cpf_recognizer = PatternRecognizer(
            supported_entity="NATIONAL_ID",
            patterns=[
                Pattern(name="brazilian_cpf", regex=brazilian_cpf, score=0.95),
            ],
            context=["cpf", "cadastro", "pessoa física"]
        )

        korean_rrn_recognizer = PatternRecognizer(
            supported_entity="NATIONAL_ID",
            patterns=[
                Pattern(name="korean_rrn", regex=korean_rrn, score=0.90),
            ],
            context=["rrn", "resident registration", "주민등록번호"]
        )

        italian_cf_recognizer = PatternRecognizer(
            supported_entity="NATIONAL_ID",
            patterns=[
                Pattern(name="italian_cf", regex=italian_cf, score=0.90),
            ],
            context=["codice fiscale", "cf", "fiscal code"]
        )

        spanish_id_recognizer = PatternRecognizer(
            supported_entity="NATIONAL_ID",
            patterns=[
                Pattern(name="spanish_dni", regex=spanish_dni, score=0.80),
                Pattern(name="spanish_nie", regex=spanish_nie, score=0.85),
            ],
            context=["dni", "nie", "documento nacional", "número de identidad"]
        )

        nordic_id_recognizer = PatternRecognizer(
            supported_entity="NATIONAL_ID",
            patterns=[
                Pattern(name="swedish_pn", regex=swedish_pn, score=0.80),
                Pattern(name="danish_cpr", regex=danish_cpr, score=0.80),
            ],
            context=["personnummer", "cpr", "cpr-nummer", "personal number", "fødselsnummer"]
        )

        singapore_nric_recognizer = PatternRecognizer(
            supported_entity="NATIONAL_ID",
            patterns=[
                Pattern(name="singapore_nric", regex=singapore_nric, score=0.90),
            ],
            context=["nric", "fin", "national registration"]
        )

        irish_pps_recognizer = PatternRecognizer(
            supported_entity="NATIONAL_ID",
            patterns=[
                Pattern(name="irish_pps", regex=irish_pps, score=0.80),
            ],
            context=["pps", "ppsn", "personal public service"]
        )

        chinese_id_recognizer = PatternRecognizer(
            supported_entity="NATIONAL_ID",
            patterns=[
                Pattern(name="chinese_id", regex=chinese_id, score=0.80),
            ],
            context=["身份证", "id card", "identity card", "shenfenzheng"]
        )

        taiwan_id_recognizer = PatternRecognizer(
            supported_entity="NATIONAL_ID",
            patterns=[
                Pattern(name="taiwan_id", regex=taiwan_id, score=0.80),
            ],
            context=["身分證", "national id", "arc", "resident certificate"]
        )

        uae_eid_recognizer = PatternRecognizer(
            supported_entity="NATIONAL_ID",
            patterns=[
                Pattern(name="uae_eid", regex=uae_eid, score=0.95),
            ],
            context=["emirates id", "eid", "emirates", "uae id"]
        )

        # South African ID: YYMMDD SSSS CAZ (13 digits)
        # Format: 6 digit DOB + 4 digit sequence + 1 citizenship + 1 random + 1 checksum
        # Uses Luhn algorithm for checksum validation
        za_id = r"\b\d{6}\s?\d{4}\s?\d{3}\b"

        za_id_recognizer = PatternRecognizer(
            supported_entity="NATIONAL_ID",
            patterns=[
                Pattern(name="za_id", regex=za_id, score=0.85),
            ],
            context=["south african", "sa id", "id number", "identity", "home affairs", "rsa"]
        )

        # Register all international ID recognizers
        self.analyzer.registry.add_recognizer(uk_nino_recognizer)
        self.analyzer.registry.add_recognizer(canadian_sin_recognizer)
        self.analyzer.registry.add_recognizer(german_id_recognizer)
        self.analyzer.registry.add_recognizer(french_insee_recognizer)
        self.analyzer.registry.add_recognizer(japanese_id_recognizer)
        self.analyzer.registry.add_recognizer(indian_aadhaar_recognizer)
        self.analyzer.registry.add_recognizer(brazilian_cpf_recognizer)
        self.analyzer.registry.add_recognizer(korean_rrn_recognizer)
        self.analyzer.registry.add_recognizer(italian_cf_recognizer)
        self.analyzer.registry.add_recognizer(spanish_id_recognizer)
        self.analyzer.registry.add_recognizer(nordic_id_recognizer)
        self.analyzer.registry.add_recognizer(singapore_nric_recognizer)
        self.analyzer.registry.add_recognizer(irish_pps_recognizer)
        self.analyzer.registry.add_recognizer(chinese_id_recognizer)
        self.analyzer.registry.add_recognizer(taiwan_id_recognizer)
        self.analyzer.registry.add_recognizer(uae_eid_recognizer)
        self.analyzer.registry.add_recognizer(za_id_recognizer)

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

    def _normalize_phone_for_validation(self, phone_text: str) -> str:
        """
        Normalize phone number text for libphonenumber validation.

        This is critical for OCR-recovered phone numbers that have spaces between
        digits like "9 6 4 0 1 2 3 4 5 6" or "1 800 555 1234".

        Two-pass normalization:
        1. First pass: Strip all whitespace and non-numeric chars (except leading +)
        2. Second pass: Reconstruct a parseable format for libphonenumber

        Args:
            phone_text: Raw phone number text (may have OCR spacing)

        Returns:
            Normalized phone number string suitable for libphonenumber parsing
        """
        # Check if this looks like heavily spaced OCR output
        # Typical pattern: single digits separated by spaces
        spaced_pattern = re.match(r'^(\d\s+){7,14}\d$', phone_text.strip())
        mixed_spaced = re.match(r'^(\d[-.\s]+){7,14}\d$', phone_text.strip())

        if spaced_pattern or mixed_spaced:
            # First pass: Extract just the digits (preserve leading +)
            has_plus = phone_text.strip().startswith('+')
            digits = re.sub(r'[^\d]', '', phone_text)

            if has_plus:
                return '+' + digits

            # For 10-11 digit numbers, assume North American format
            if len(digits) == 10:
                # Format as XXX-XXX-XXXX for better parsing
                return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
            elif len(digits) == 11 and digits.startswith('1'):
                # Format as 1-XXX-XXX-XXXX
                return f"1-{digits[1:4]}-{digits[4:7]}-{digits[7:]}"
            else:
                # Return as-is with + prefix for international parsing
                return '+' + digits if len(digits) > 10 else digits

        # Normalize international dialing prefix 001- to +1 for libphonenumber
        # 001 is the IDD (International Direct Dialing) prefix used in many countries
        # to dial the US/Canada (+1), but phonenumbers expects +1 format
        if re.match(r'^001[-.\s]', phone_text):
            return '+1' + phone_text[3:]

        # Check for European 00-prefix international dialing (must be before partial_spaced)
        stripped = phone_text.strip()
        if stripped.startswith('00') and not stripped.startswith('001'):
            # Convert 00XX to +XX format for phonenumbers
            digits = re.sub(r'[^\d]', '', stripped)
            return '+' + digits[2:]  # Remove leading 00

        # Check for partially spaced numbers like "416 555 1234" or "1 800 555 1234"
        # These have groups of digits separated by spaces
        partial_spaced = re.match(r'^[\d\s]+$', phone_text.strip())
        if partial_spaced:
            digits = re.sub(r'\s+', '', phone_text)
            if len(digits) >= 10:
                return digits

        # Skip standard parenthesized formats - phonenumbers handles them natively
        if re.match(r'^\+?\(?\d', stripped) and '(' in phone_text and ')' in phone_text:
            return phone_text

        # Handle mixed separators (dash+dot+space): strip all non-digit chars
        # This catches formats like '+33 70-242-8659', '0106 41.195-8207', etc.
        has_plus = phone_text.strip().startswith('+')
        has_mixed_seps = bool(re.search(r'[-.]', phone_text) and re.search(r'[\s]', phone_text)) or \
                         bool(re.search(r'[-]', phone_text) and re.search(r'[.]', phone_text))
        if has_mixed_seps or re.search(r'[^\d\s()+\-.]', phone_text):
            digits = re.sub(r'[^\d]', '', phone_text)
            if has_plus:
                return '+' + digits
            return digits  # Don't add '+' - let phonenumbers use default_region

        # For standard formats, just return as-is
        return phone_text

    def _validate_phone_with_phonenumbers(self, phone_text: str, default_region: str = "US") -> str:
        """
        Validate a phone number using Google's libphonenumber library.

        Two-pass normalization is applied BEFORE validation:
        1. First pass: Normalize OCR-spaced numbers (e.g., "9 6 4 0 1 2 3 4 5 6")
        2. Second pass: Validate with libphonenumber

        Args:
            phone_text: The phone number text to validate
            default_region: Default region code for parsing (e.g., "US", "GB", "CA")

        Returns:
            "valid" if the phone number is definitely valid (boost confidence)
            "possible" if the number is possible but not confirmed (small boost for OCR)
            "invalid" if the phone number is definitely invalid (reduce confidence)
            "unknown" if validation is unavailable or inconclusive (keep as is)
        """
        if not PHONENUMBERS_AVAILABLE:
            return "unknown"  # Fall back to pattern-only validation if library unavailable

        # First pass: Normalize spaced/OCR phone numbers before validation
        normalized_phone = self._normalize_phone_for_validation(phone_text)

        # Detect if this is an OCR-spaced pattern (strong signal for phone numbers)
        is_ocr_spaced = bool(re.match(r'^(\d\s+){7,14}\d$', phone_text.strip()) or
                            re.match(r'^(\d[-.\s]+){7,14}\d$', phone_text.strip()) or
                            re.match(r'^\+\s*(\d\s*){8,15}$', phone_text.strip()))

        try:
            # Try to parse the normalized phone number
            parsed = phonenumbers.parse(normalized_phone, default_region)

            # Check if it's a valid number
            if phonenumbers.is_valid_number(parsed):
                return "valid"  # Definitely valid - boost confidence

            # Check if it's a possible number (less strict check)
            if phonenumbers.is_possible_number(parsed):
                # For OCR-spaced numbers, "possible" is good enough - the spacing
                # pattern itself is strong evidence this is a phone number
                if is_ocr_spaced:
                    return "possible"  # OCR pattern + possible = small boost
                return "unknown"  # Standard numbers need stronger validation

            return "invalid"  # Definitely invalid - reduce confidence
        except NumberParseException:
            # If parsing fails, it's likely not a valid phone number format
            return "invalid"

    def _validate_national_id_with_stdnum(self, id_text: str, country_hint: str = None) -> str:
        """
        Validate a national ID using python-stdnum library.

        Supports checksum validation for 35+ countries:
        - US SSN (Social Security Number)
        - UK NINO (National Insurance Number)
        - German Steuer-ID (Tax ID)
        - Brazilian CPF
        - Italian Codice Fiscale
        - Spanish DNI/NIE
        - Polish PESEL
        - Swedish Personnummer
        - Dutch BSN
        - Belgian National Number

        Args:
            id_text: The national ID text to validate
            country_hint: Optional country code hint (e.g., "US", "GB")

        Returns:
            "valid" if the ID passes checksum validation
            "invalid" if the ID fails validation
            "unknown" if validation is unavailable or inconclusive
        """
        if not STDNUM_AVAILABLE:
            return "unknown"

        # Normalize: remove spaces, dots, dashes for validation
        normalized = re.sub(r'[\s.-]', '', id_text.upper())

        # Try US SSN (9 digits, XXX-XX-XXXX format)
        if len(normalized) == 9 and normalized.isdigit():
            try:
                if us_ssn.is_valid(id_text):
                    return "valid"
            except Exception:
                pass

        # Try UK NINO (2 letters + 6 digits + 1 letter)
        if len(normalized) == 9 and normalized[:2].isalpha() and normalized[-1].isalpha():
            try:
                if uk_nino.is_valid(id_text):
                    return "valid"
            except Exception:
                pass

        # Try German Steuer-ID (11 digits)
        if len(normalized) == 11 and normalized.isdigit():
            try:
                if de_steuerid.is_valid(normalized):
                    return "valid"
            except Exception:
                pass

        # Try Brazilian CPF (11 digits, XXX.XXX.XXX-XX format)
        if len(normalized) == 11 and normalized.isdigit():
            try:
                if br_cpf.is_valid(id_text):
                    return "valid"
            except Exception:
                pass

        # Try Italian Codice Fiscale (16 alphanumeric)
        if len(normalized) == 16:
            try:
                if it_cf.is_valid(normalized):
                    return "valid"
            except Exception:
                pass

        # Try Spanish DNI (8 digits + 1 letter)
        if len(normalized) == 9 and normalized[:8].isdigit() and normalized[8].isalpha():
            try:
                if es_dni.is_valid(normalized):
                    return "valid"
            except Exception:
                pass

        # Try Polish PESEL (11 digits)
        if len(normalized) == 11 and normalized.isdigit():
            try:
                if pl_pesel.is_valid(normalized):
                    return "valid"
            except Exception:
                pass

        # Try Swedish Personnummer (10 or 12 digits)
        if (len(normalized) == 10 or len(normalized) == 12) and normalized.replace('-', '').isdigit():
            try:
                if se_personnummer.is_valid(id_text):
                    return "valid"
            except Exception:
                pass

        # Try Dutch BSN (8-9 digits)
        if (len(normalized) == 8 or len(normalized) == 9) and normalized.isdigit():
            try:
                if nl_bsn.is_valid(normalized):
                    return "valid"
            except Exception:
                pass

        # Try Belgian National Number (11 digits)
        if len(normalized) == 11 and normalized.isdigit():
            try:
                if be_nn.is_valid(normalized):
                    return "valid"
            except Exception:
                pass

        # Try Luhn checksum (generic, used by many IDs)
        if normalized.isdigit() and len(normalized) >= 8:
            try:
                if luhn.is_valid(normalized):
                    return "possible"  # Luhn valid but not country-specific
            except Exception:
                pass

        return "unknown"

    def _get_phone_region_from_context(self, full_text: str, start: int, end: int) -> str:
        """
        Determine the likely region of a phone number based on surrounding context.

        Args:
            full_text: The full text containing the phone number
            start: Start position of the phone number
            end: End position of the phone number

        Returns:
            ISO country code (e.g., "US", "GB", "CA")
        """
        # Get surrounding context
        window = 200
        context_start = max(0, start - window)
        context_end = min(len(full_text), end + window)
        context = full_text[context_start:context_end].lower()

        # Check for country indicators
        region_indicators = {
            "US": ["usa", "united states", "america", "usd", "$", "california", "new york",
                   "texas", "florida", "illinois", "ohio", "georgia", "michigan"],
            "CA": ["canada", "canadian", "cad", "ontario", "quebec", "alberta",
                   "british columbia", "toronto", "vancouver", "montreal"],
            "GB": ["uk", "united kingdom", "england", "scotland", "wales", "gbp", "£",
                   "london", "manchester", "birmingham", "nhs"],
            "AU": ["australia", "australian", "aud", "sydney", "melbourne", "brisbane"],
            "DE": ["germany", "german", "deutschland", "eur", "€", "berlin", "munich"],
            "FR": ["france", "french", "paris", "lyon", "marseille"],
            "IN": ["india", "indian", "inr", "₹", "mumbai", "delhi", "bangalore"],
            "BR": ["brazil", "brazilian", "brl", "r$", "são paulo", "rio de janeiro"],
            "MX": ["mexico", "mexican", "mxn", "mexico city"],
            "JP": ["japan", "japanese", "jpy", "¥", "tokyo", "osaka"],
        }

        for region, indicators in region_indicators.items():
            for indicator in indicators:
                if indicator in context:
                    return region

        # Default to US if no specific indicators found
        return "US"

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

    def _detect_labeled_pii(self, text: str) -> List[PIIEntity]:
        """Detect PII values inside labeled fields (XML tags, JSON keys, template variables).

        Real-world structured data often has explicit labels for PII fields.
        This pass detects values inside name-related tags/keys that pattern-based
        and NER-based recognizers miss (especially for unusual international names).
        """
        entities = []
        # XML name tags: <name>X</name>, <FirstName>X</FirstName>, etc.
        xml_name_tags = re.finditer(
            r'<(first_?name|last_?name|given_?name|family_?name|middle_?name|'
            r'sur_?name|name)\d*>([^<]{2,40})</',
            text, re.IGNORECASE
        )
        for m in xml_name_tags:
            value = m.group(2).strip()
            if value and len(value) >= 2 and not value.isdigit():
                start = m.start(2)
                end = m.start(2) + len(m.group(2))
                entities.append(PIIEntity(
                    entity_type='PERSON',
                    text=value,
                    start=start,
                    end=end,
                    confidence=0.85,
                    pattern_name='labeled_xml_name',
                    locale=None,
                    recognition_metadata={'recognizer_name': 'LabeledPIIDetector'}
                ))

        # JSON name keys: "first_name": "X", "First Name": "X", "GivenName1": "X", etc.
        json_name_keys = re.finditer(
            r'"(first[_\s]?name|last[_\s]?name|given[_\s]?name|family[_\s]?name|'
            r'middle[_\s]?name|sur[_\s]?name|student[_\s]?name|'
            r'name|GivenName|LastName|FullName|givename|Illustrator)\d*"\s*:\s*"([^"]{2,40})"',
            text, re.IGNORECASE
        )
        for m in json_name_keys:
            value = m.group(2).strip()
            if value and len(value) >= 2 and not value.isdigit():
                start = m.start(2)
                end = m.start(2) + len(m.group(2))
                entities.append(PIIEntity(
                    entity_type='PERSON',
                    text=value,
                    start=start,
                    end=end,
                    confidence=0.80,
                    pattern_name='labeled_json_name',
                    locale=None,
                    recognition_metadata={'recognizer_name': 'LabeledPIIDetector'}
                ))

        # Template variables with name context: {{X}}, <<X>>
        # Template delimiters are strong PII indicators in structured data
        template_patterns = [
            (r'\{\{([A-Z][A-Za-zÀ-ÿ]{1,30})\}\}', 'labeled_template_curly'),
            (r'<<([A-Z][A-Za-zÀ-ÿ]{1,30})>>', 'labeled_template_angle'),
        ]
        for pattern, pname in template_patterns:
            for m in re.finditer(pattern, text):
                value = m.group(1).strip()
                if value and len(value) >= 2:
                    # Check for name/data context in surrounding text
                    ctx_start = max(0, m.start() - 150)
                    preceding = text[ctx_start:m.start()].lower()
                    name_context = any(kw in preceding for kw in [
                        'name', 'dear', 'student', 'person', 'given', 'first',
                        'last', 'recipient', 'participant', 'contact',
                        'default', 'value', 'data', 'field', 'template',
                    ])
                    # Also match if template starts a greeting pattern
                    if name_context or re.search(r'(?:dear|hi|hello)\s*$', preceding, re.IGNORECASE):
                        start = m.start(1)
                        end = m.end(1)
                        entities.append(PIIEntity(
                            entity_type='PERSON',
                            text=value,
                            start=start,
                            end=end,
                            confidence=0.75,
                            pattern_name=pname,
                            locale=None,
                            recognition_metadata={'recognizer_name': 'LabeledPIIDetector'}
                        ))

        # Free-text name labels: "Name: X", "Student Name: X", "Cardholder: X", etc.
        # Handles plain text, Markdown bold (**Name:** X), and bracket formats ([X])
        label_patterns = [
            # "Name: X" or "Given_Name: X" or "Student Name: X" with optional Markdown bold
            # Allow optional whitespace, brackets, or ** between colon and value
            (r'(?:\*\*)?(?:(?:Student|Given|First|Last|Middle|Full|Family|Sur|'
             r'Applicant|Contact)\s*)?(?:_)?Name(?:\s*\d)?(?:\*\*)?'
             r"\s*[:]\s*(?:[\[\*\s])*([A-ZÀ-ÿ][A-Za-zÀ-ÿ\-''']+(?:\s+[A-ZÀ-ÿ][A-Za-zÀ-ÿ\-''']+){0,2})",
             'labeled_text_name', 0.80),
            # "Cardholder: X, Card:" or "Account holder: X, SSN:" pattern
            (r"(?:Card|Account)\s*holder:\s*([A-Z][A-Za-zÀ-ÿ\-''']+(?:\s+[A-Z][A-Za-zÀ-ÿ\-''']+){0,2})\s*,",
             'labeled_cardholder', 0.85),
            # Greeting patterns: "Dear X," or "Dear X,\n"
            (r"Dear\s+([A-ZÀ-ÿ][A-Za-zÀ-ÿ\-''']+(?:\s+[A-ZÀ-ÿ][A-Za-zÀ-ÿ\-''']+){0,2})\s*[,\n]",
             'labeled_greeting', 0.90),
            # Role labels without "Name" suffix: "Team Member: X", "Participant: X"
            (r'(?:\*\*)?(?:Team\s+Member|Participant|Panelist|Illustrator)(?:\*\*)?'
             r"\s*[:]\s*(?:[\[\*\s])*([A-ZÀ-ÿ][A-Za-zÀ-ÿ\-''']+(?:\s+[A-ZÀ-ÿ][A-Za-zÀ-ÿ\-''']+){0,2})",
             'labeled_text_role', 0.75),
            # Bold/markdown names: **FirstName LastName**
            (r"\*\*([A-ZÀ-ÿ][A-Za-zÀ-ÿ\-''']+(?:\s+[A-ZÀ-ÿ][A-Za-zÀ-ÿ\-''']+){0,2})\*\*",
             'labeled_bold_name', 0.70),
        ]
        for pattern, pname, score in label_patterns:
            for m in re.finditer(pattern, text):
                value = m.group(1).strip(' *[]')
                if value and len(value) >= 2 and not value.isdigit():
                    # Skip if value looks like a label/heading (all caps or common words)
                    if value.isupper() and len(value) > 3:
                        continue
                    _val_lower = value.lower()
                    if _val_lower in {'name', 'none', 'n/a', 'unknown', 'other',
                                      'esquire', 'countess', 'selectman', 'major',
                                      'captain', 'sergeant', 'infant', 'deacon',
                                      'reverend', 'checking', 'absence', 'through'}:
                        continue
                    # Skip values ending in common non-name suffixes
                    if len(value.split()) == 1 and len(value) > 4:
                        if _val_lower.endswith(('ing', 'tion', 'ment', 'ness', 'ence', 'ance', 'ity')):
                            continue
                    start = m.start(1)
                    end = m.end(1)
                    entities.append(PIIEntity(
                        entity_type='PERSON',
                        text=value,
                        start=start,
                        end=end,
                        confidence=score,
                        pattern_name=pname,
                        locale=None,
                        recognition_metadata={'recognizer_name': 'LabeledPIIDetector'}
                    ))

        # ========================================================================
        # NATIONAL_ID from labeled fields (XML tags, text labels)
        # ========================================================================
        # XML tags: <social_number>X</social_number>, <national_id>X</national_id>, etc.
        nid_xml_tags = re.finditer(
            r'<(social[_\s]?(?:security[_\s]?)?number|national[_\s]?id|'
            r'id[_\s]?number|ssn|tin|tax[_\s]?id|'
            r'identification[_\s]?number|personal[_\s]?id|citizen[_\s]?id)>'
            r'([^<]{3,30})</',
            text, re.IGNORECASE
        )
        for m in nid_xml_tags:
            value = m.group(2).strip()
            if value and len(value) >= 3:
                start = m.start(2)
                end = m.start(2) + len(m.group(2))
                entities.append(PIIEntity(
                    entity_type='NATIONAL_ID',
                    text=value,
                    start=start,
                    end=end,
                    confidence=0.85,
                    pattern_name='labeled_xml_national_id',
                    locale=None,
                    recognition_metadata={'recognizer_name': 'LabeledPIIDetector'}
                ))

        # XML tags for ID cards and passports
        id_xml_tags = re.finditer(
            r'<(id[_\s]?card|passport[_\s]?(?:number)?|driver[_\s]?license|'
            r'license[_\s]?number)>'
            r'([^<]{3,30})</',
            text, re.IGNORECASE
        )
        for m in id_xml_tags:
            value = m.group(2).strip()
            if value and len(value) >= 3:
                start = m.start(2)
                end = m.start(2) + len(m.group(2))
                entities.append(PIIEntity(
                    entity_type='NATIONAL_ID',
                    text=value,
                    start=start,
                    end=end,
                    confidence=0.80,
                    pattern_name='labeled_xml_id_card',
                    locale=None,
                    recognition_metadata={'recognizer_name': 'LabeledPIIDetector'}
                ))

        # JSON keys: "Social_Number": "X", "SSN": "X", etc.
        nid_json_keys = re.finditer(
            r'"(Social[_\s]?(?:Security[_\s]?)?Number|National[_\s]?ID|'
            r'ID[_\s]?Number|SSN|TIN|Tax[_\s]?ID|'
            r'Identification[_\s]?Number|Personal[_\s]?ID|Citizen[_\s]?ID|'
            r'Driver[_\s]?License[_\s]?(?:Number)?|Passport[_\s]?(?:Number)?|'
            r'ID[_\s]?Card)"\s*:\s*"([^"]{3,30})"',
            text, re.IGNORECASE
        )
        for m in nid_json_keys:
            value = m.group(2).strip()
            if value and len(value) >= 3 and value.lower() not in {'none', 'n/a', 'unknown'}:
                start = m.start(2)
                end = m.start(2) + len(m.group(2))
                entities.append(PIIEntity(
                    entity_type='NATIONAL_ID',
                    text=value,
                    start=start,
                    end=end,
                    confidence=0.85,
                    pattern_name='labeled_json_national_id',
                    locale=None,
                    recognition_metadata={'recognizer_name': 'LabeledPIIDetector'}
                ))

        # ========================================================================
        # ADDRESS (LOCATION) from labeled fields
        # ========================================================================
        # XML tags: <city>X</city>, <postcode>X</postcode>, <country>X</country>, etc.
        addr_xml_tags = re.finditer(
            r'<(city|town|postcode|post_?code|zip_?code|country|state|'
            r'province|region|county|district|municipality|suburb|'
            r'address|street|street_?address)>'
            r'([^<]{2,60})</',
            text, re.IGNORECASE
        )
        for m in addr_xml_tags:
            value = m.group(2).strip()
            if value and len(value) >= 2:
                # Skip noise values
                if value.lower() in {'none', 'n/a', 'unknown', 'other', 'null', ''}:
                    continue
                start = m.start(2)
                end = m.start(2) + len(m.group(2))
                entities.append(PIIEntity(
                    entity_type='LOCATION',
                    text=value,
                    start=start,
                    end=end,
                    confidence=0.85,
                    pattern_name='labeled_xml_address',
                    locale=None,
                    recognition_metadata={'recognizer_name': 'LabeledPIIDetector'}
                ))

        # JSON keys: "City": "X", "Postcode": "X", "Country": "X", etc.
        addr_json_keys = re.finditer(
            r'"(City|Town|Postcode|Post_?code|Zip_?code|ZIP|Country|State|'
            r'Province|Region|County|District|Municipality|Suburb|'
            r'Address|Street|Street_?Address|Location|Residence)"\s*:\s*"([^"]{2,60})"',
            text, re.IGNORECASE
        )
        for m in addr_json_keys:
            value = m.group(2).strip()
            if value and len(value) >= 2:
                if value.lower() in {'none', 'n/a', 'unknown', 'other', 'null', ''}:
                    continue
                start = m.start(2)
                end = m.start(2) + len(m.group(2))
                entities.append(PIIEntity(
                    entity_type='LOCATION',
                    text=value,
                    start=start,
                    end=end,
                    confidence=0.80,
                    pattern_name='labeled_json_address',
                    locale=None,
                    recognition_metadata={'recognizer_name': 'LabeledPIIDetector'}
                ))

        # YAML/text labels: "City: X", "Country: X", "Postcode: X"
        addr_text_labels = re.finditer(
            r'(?:City|Town|Postcode|Post[_\s]?code|Country|State|Province|County)\s*:\s*'
            r'([A-ZÀ-ÿ][A-Za-zÀ-ÿ\s\-]{1,40}?)(?:\s*[,\n"}]|$)',
            text
        )
        for m in addr_text_labels:
            value = m.group(1).strip()
            if value and len(value) >= 2:
                if value.lower() in {'none', 'n/a', 'unknown', 'other', 'null'}:
                    continue
                start = m.start(1)
                end = m.start(1) + len(value)
                entities.append(PIIEntity(
                    entity_type='LOCATION',
                    text=value,
                    start=start,
                    end=end,
                    confidence=0.80,
                    pattern_name='labeled_text_address',
                    locale=None,
                    recognition_metadata={'recognizer_name': 'LabeledPIIDetector'}
                ))

        # ========================================================================
        # PHONE from labeled fields (XML tags)
        # ========================================================================
        phone_xml_tags = re.finditer(
            r'<(tel|phone|telephone|mobile|cell|fax|contact_?number)>'
            r'([^<]{5,25})</',
            text, re.IGNORECASE
        )
        for m in phone_xml_tags:
            value = m.group(2).strip()
            if value and len(value) >= 5:
                digits = re.sub(r'\D', '', value)
                if len(digits) >= 7:  # Must have at least 7 digits
                    start = m.start(2)
                    end = m.start(2) + len(m.group(2))
                    entities.append(PIIEntity(
                        entity_type='PHONE_NUMBER',
                        text=value,
                        start=start,
                        end=end,
                        confidence=0.90,
                        pattern_name='labeled_xml_phone',
                        locale=None,
                        recognition_metadata={'recognizer_name': 'LabeledPIIDetector'}
                    ))

        # JSON phone keys: "Tel": "X", "Phone": "X"
        phone_json_keys = re.finditer(
            r'"(Tel|Phone|Telephone|Mobile|Cell|Fax|Contact_?Number)"\s*:\s*"([^"]{5,25})"',
            text, re.IGNORECASE
        )
        for m in phone_json_keys:
            value = m.group(2).strip()
            if value and len(value) >= 5:
                digits = re.sub(r'\D', '', value)
                if len(digits) >= 7:
                    start = m.start(2)
                    end = m.start(2) + len(m.group(2))
                    entities.append(PIIEntity(
                        entity_type='PHONE_NUMBER',
                        text=value,
                        start=start,
                        end=end,
                        confidence=0.88,
                        pattern_name='labeled_json_phone',
                        locale=None,
                        recognition_metadata={'recognizer_name': 'LabeledPIIDetector'}
                    ))

        return entities

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
                        confidence=min(0.95, entity.confidence + confidence_boost),
                        recognition_metadata=entity.recognition_metadata
                    ))
                else:
                    resolved.append(entity)
            else:
                resolved.append(entity)

        return resolved

    def _is_ocr_artifact(self, text: str) -> bool:
        """
        Detect likely OCR garbage that should be filtered from PII detection.

        Uses LARVPC (Low Alphanumeric Ratio, Vowel-Poor Characters) rules:
        - Random mixed case within words (not acronyms): "biRshday", "mea bYut"
        - Unusual punctuation or underscores within text
        - Incomplete words ending with dash
        - Very short fragments
        - Low alphanumeric density (<50% for tokens >5 chars)
        - Extreme consonant/vowel ratio (<10% vowels in alphabetic strings)
        - Multiple distinct punctuation marks (OCR noise)
        - Repeated identical characters (4+ consecutive)

        Args:
            text: The detected text to check

        Returns:
            True if the text appears to be an OCR artifact
        """
        if not text or len(text.strip()) < 3:
            return True

        # Random mixed case within words (not acronyms): "biRshday", "mea bYut"
        # Pattern: lowercase followed by uppercase followed by lowercase
        if re.search(r'[a-z][A-Z][a-z]', text):
            return True

        # Contains underscores with letters (OCR artifacts like "repre_eIt")
        if re.search(r'[a-zA-Z]_[a-zA-Z]', text):
            return True

        # Ends with dash (incomplete word)
        if text.rstrip().endswith('-'):
            return True

        # Contains newlines or excessive whitespace (OCR line breaks)
        if '\n' in text or '\r' in text or '  ' in text:
            return True

        # LARVPC Rule 1: Alphanumeric density < 50% for tokens > 5 chars
        if len(text) > 5:
            alnum_count = sum(1 for c in text if c.isalnum())
            if alnum_count / len(text) < 0.50:
                return True

        # LARVPC Rule 2: 4+ identical consecutive characters (e.g., "XXXX", "----")
        if re.search(r'(.)\1{3,}', text):
            return True

        # LARVPC Rule 3: Consonant/vowel ratio for alphabetic strings
        # If < 10% vowels in a purely alphabetic string, likely garbage
        alpha_only = ''.join(c for c in text.lower() if c.isalpha())
        if len(alpha_only) > 5:
            vowels = set('aeiou')
            vowel_count = sum(1 for c in alpha_only if c in vowels)
            vowel_ratio = vowel_count / len(alpha_only)
            if vowel_ratio < 0.10:
                return True

        # LARVPC Rule 4: Multiple distinct punctuation marks (>2 types)
        # Common in OCR misinterpretation of graphical elements
        punct_chars = set(c for c in text if c in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
        if len(punct_chars) > 2:
            return True

        # Check for words with random capitalization in the middle
        words = text.split()
        for word in words:
            if len(word) > 4:
                # Skip acronyms (all caps) or proper nouns (first letter cap only)
                if word.isupper() or (word[0].isupper() and word[1:].islower()):
                    continue
                # Skip hyphenated proper nouns (e.g., "Burgess-Patterson", "Hewlett-Packard")
                # Each hyphen-separated part should be checked individually
                if '-' in word:
                    parts = word.split('-')
                    if all(p and p[0].isupper() and (len(p) <= 1 or p[1:].islower()) for p in parts):
                        continue
                # Skip apostrophe proper nouns (e.g., "O'Brien", "D'Angelo", "O'Malley")
                # Each apostrophe-separated part should be checked individually
                if "'" in word or "\u2019" in word:
                    parts = re.split(r"['\u2019]", word)
                    if all(p and p[0].isupper() and (len(p) <= 1 or p[1:].islower()) for p in parts):
                        continue
                # Check for random caps in middle of word
                mid = word[1:-1]
                if mid and any(c.isupper() for c in mid) and any(c.islower() for c in mid):
                    return True

        return False

    def _is_garbage_string(self, text: str) -> bool:
        """
        Detect OCR garbage strings using LARVPC-style rules.

        Returns True if the string is likely OCR noise, not real text.
        Used to filter false positives for COMPANY, PERSON, etc.
        """
        if not text or len(text) < 2:
            return False

        # Rule 1: Alphanumeric density < 50%
        alnum_count = sum(1 for c in text if c.isalnum())
        if len(text) > 0 and alnum_count / len(text) < 0.5:
            return True

        # Rule 2: Consecutive identical characters (4+)
        for i in range(len(text) - 3):
            if text[i] == text[i+1] == text[i+2] == text[i+3]:
                return True

        # Rule 3: Very low vowel ratio for alphabetic text
        alpha_chars = [c for c in text.lower() if c.isalpha()]
        if len(alpha_chars) >= 4:
            vowels = sum(1 for c in alpha_chars if c in 'aeiou')
            if vowels / len(alpha_chars) < 0.08:  # Less than 8% vowels
                return True

        # Rule 4: Excessive special characters mixed with letters (OCR artifact pattern)
        if len(text) >= 4:
            special_in_middle = sum(1 for c in text[1:-1] if not c.isalnum() and c != ' ')
            if special_in_middle >= len(text) * 0.3:  # 30%+ special chars
                return True

        return False

    def _filter_false_positives(self, entities: List[PIIEntity], text: str) -> List[PIIEntity]:
        """
        Filter out common false positive patterns.

        Removes:
        - DRIVER_LICENSE with low confidence or short length
        - DATE_TIME that look like phone numbers or addresses
        - BANK_NUMBER that are clearly other entity types
        - ITIN/MEDICAL_LICENSE false positives from passport patterns
        - NATIONAL_ID false positives from credit card, phone, IP patterns
        - PHONE_NUMBER false positives from IMEI, IP addresses, GPS coordinates
        - LOCATION false positives from URLs
        - URL false positives from email addresses
        - MEDICAL false positives from technical identifiers
        - FINANCIAL false positives from names and non-financial data
        """
        filtered = []

        # Pre-compute ADDRESS, COMPANY, and LOCATION spans for cross-entity PERSON filtering
        address_spans = [(e.start, e.end) for e in entities if e.entity_type == "ADDRESS"]
        company_spans = [(e.start, e.end) for e in entities if e.entity_type == "COMPANY"]
        location_spans = [(e.start, e.end, e.confidence) for e in entities if e.entity_type == "LOCATION"]

        for entity in entities:
            entity_text = entity.text.strip()

            # --- PERSON span cleanup (before general filters) ---
            # PersonRecognizer sometimes merges overlapping detections into overly-wide spans
            # that include trailing context (", Card:", newlines) or leading labels ("Cardholder:").
            # Clean these up BEFORE OCR artifact filtering, which rejects excess punctuation.
            if entity.entity_type == "PERSON":
                _ptrim = entity_text
                # Truncate at newline (names don't span lines)
                _pnl = _ptrim.find('\n')
                if _pnl > 0:
                    _ptrim = _ptrim[:_pnl].rstrip()
                # Truncate at ", Word:" pattern (", Card:", ", SSN:")
                _pcl = re.search(r',\s*\w+\s*:', _ptrim)
                if _pcl:
                    _ptrim = _ptrim[:_pcl.start()].rstrip()
                _ptrim = _ptrim.rstrip(' ,')
                # Strip leading person-context labels ("Cardholder: ", "Employee: ")
                _plbl = re.match(r'^(\w+):\s+', _ptrim)
                if _plbl:
                    _plw = _plbl.group(1).lower()
                    if _plw in {'cardholder', 'contact', 'patient', 'client', 'customer',
                                'applicant', 'employee', 'beneficiary', 'holder', 'owner',
                                'recipient', 'sender', 'guardian', 'witness', 'spouse',
                                'parent', 'agent', 'representative', 'name', 'person'}:
                        _ptrim = _ptrim[_plbl.end():]
                if _ptrim != entity_text and len(_ptrim) >= 2:
                    _poff = entity.text.find(_ptrim)
                    if _poff >= 0:
                        entity = PIIEntity(
                            entity_type=entity.entity_type,
                            text=_ptrim,
                            start=entity.start + _poff,
                            end=entity.start + _poff + len(_ptrim),
                            confidence=entity.confidence,
                            pattern_name=entity.pattern_name,
                            locale=entity.locale,
                            recognition_metadata=entity.recognition_metadata,
                        )
                        entity_text = _ptrim

            # CREDENTIAL filtering - Context-anchored entropy verification
            # Uses Shannon entropy + context anchor (trigger words within 10 tokens)
            # See credential_entropy.py for implementation details
            if entity.entity_type == "CREDENTIAL":
                if CREDENTIAL_ENTROPY_AVAILABLE:
                    # Use advanced context-anchored entropy verification
                    # Pass full text and position for context anchor checking
                    result = analyze_credential_entropy(
                        entity_text,
                        full_text=text,
                        position_start=entity.start,
                        position_end=entity.end
                    )
                    if not result.is_credential:
                        # Filter out non-credentials based on entropy + context analysis
                        continue
                    # Optionally adjust confidence based on entropy quality
                    # (currently disabled - entity confidence unchanged)
                else:
                    # Fallback: basic entropy check for unlabeled patterns
                    # Keep labeled credentials (password=, Bearer, api_key_) but filter noise
                    has_label = bool(re.search(
                        r'(?:password|passwd|pwd|pass|secret|api[_-]?key|bearer|token)\s*[:=]',
                        entity_text, re.IGNORECASE
                    ))
                    # If no label, must be high entropy to be valid
                    if not has_label and entity.confidence < 0.95:
                        entropy = shannon_entropy(entity_text)
                        # Low entropy without label = likely false positive
                        if entropy < 4.0 or len(entity_text) < 12:
                            continue

            # Filter USERNAME false positives
            if entity.entity_type == "USERNAME":
                text_lower = entity_text.lower()
                # Skip if it's a common English word (use existing negative gazetteer)
                if is_negative_match(entity_text, entity.entity_type):
                    continue
                # Skip if it looks like a sentence fragment (contains spaces)
                if ' ' in entity_text:
                    continue
                # Skip very short usernames (likely noise)
                if len(entity_text) < 3:
                    continue
                # Filter common English hyphenated words (not usernames)
                _common_hyphenated = {
                    'well-being', 'well-known', 'well-established', 'well-defined',
                    'up-to-date', 'state-of-the-art', 'self-esteem', 'self-care',
                    'self-awareness', 'self-confidence', 'self-improvement',
                    'anti-bullying', 'anti-fraud', 'anti-virus', 'anti-spam',
                    'late-night', 'long-term', 'short-term', 'full-time', 'part-time',
                    'real-time', 'on-site', 'off-site', 'in-person', 'on-line',
                    'day-to-day', 'face-to-face', 'one-on-one', 'step-by-step',
                    'high-quality', 'high-level', 'low-cost', 'low-risk',
                    'non-profit', 'non-verbal', 'non-fiction',
                    'problem-solving', 'decision-making', 'team-building',
                }
                if text_lower in _common_hyphenated:
                    continue
                # Filter HTML attributes (http-equiv, X-UA, etc.)
                if text_lower.startswith(('http-', 'x-', 'utf-', 'content-')):
                    continue

            # LARVPC OCR artifact filtering (early rejection of garbage)
            # Filters: low alphanumeric density, consonant-heavy strings, excess punctuation
            if OCR_FILTER_AVAILABLE and entity.entity_type in ("COMPANY", "PERSON", "ADDRESS", "LOCATION"):
                # Skip OCR filter for UK postcode patterns (consonant-heavy but legitimate)
                _is_postcode_fmt = (entity.entity_type in ("ADDRESS", "LOCATION") and
                                    bool(re.match(r'^[A-Za-z]{1,2}\d[A-Za-z\d]?\s?\d[A-Za-z]{2}$', entity_text.strip())))
                if not _is_postcode_fmt:
                    should_keep, reason = filter_ocr_artifacts(entity_text, entity.entity_type)
                    if not should_keep:
                        continue

            # Filter using negative gazetteer (common words, brand names, etc.)
            # This catches high-frequency false positives for PERSON, COMPANY, LOCATION
            if entity.entity_type in ("PERSON", "COMPANY", "LOCATION", "ADDRESS"):
                if is_negative_match(entity_text, entity.entity_type):
                    continue
                # For COMPANY: filter single common words without corporate suffix
                if is_single_common_word(entity_text, entity.entity_type):
                    continue

            # Filter OCR garbage strings for COMPANY and PERSON (LARVPC rules)
            if entity.entity_type in ("COMPANY", "PERSON"):
                if self._is_garbage_string(entity_text):
                    continue

            # Filter COMPANY-specific false positives (OCR artifacts, garbage patterns)
            if entity.entity_type == "COMPANY":
                # Filter phone-fragment + generic word: "860-0892 Address", "766-0392 Address"
                if re.match(r'^\d{3}[-.\s]\d{4}\s+\w+$', entity_text):
                    continue
                # Filter patterns like "6 Name", "IPV4_...", mixed case OCR artifacts
                # But NOT numbered corporations (e.g., "2378238 Ontario Inc.")
                _corp_suffixes = ('inc', 'ltd', 'llc', 'corp', 'limited', 'gmbh', 'plc', 'co')
                _has_corp_suffix = any(entity_text.lower().rstrip('.').endswith(s) for s in _corp_suffixes)
                if not _has_corp_suffix and re.match(r'^(\d+\s+\w+|IPV4_.*|.*\.\s+\w+.*)', entity_text):
                    continue
                # Filter mixed case anomalies like "addXEss", "dEtense" (OCR artifacts)
                if re.search(r'[a-z][A-Z]{2,}|[a-z][A-Z][a-z]', entity_text):
                    continue
                # Filter very short company names without corporate suffix
                if len(entity_text) <= 3 and entity.confidence < 0.85:
                    continue

            # Filter ADDRESS/LOCATION false positives
            if entity.entity_type in ("ADDRESS", "LOCATION"):
                # --- Span cleanup: strip "Address:" label prefix from detection ---
                # PatternRecognizer sometimes includes the field label in the span
                _addr_lbl = re.match(r'^(?:Address|Location|Addr)\s*:\s*', entity_text, re.IGNORECASE)
                if _addr_lbl:
                    _addr_rest = entity_text[_addr_lbl.end():]
                    if len(_addr_rest.strip()) >= 3:
                        _addr_off = _addr_lbl.end()
                        entity = PIIEntity(
                            entity_type=entity.entity_type,
                            text=_addr_rest,
                            start=entity.start + _addr_off,
                            end=entity.end,
                            confidence=entity.confidence,
                            pattern_name=entity.pattern_name,
                            locale=entity.locale,
                            recognition_metadata=entity.recognition_metadata,
                        )
                        entity_text = _addr_rest
                # Filter "Employee Dr" / "Customer Dr" - job title + Dr abbreviation
                # These are not addresses, just role + title abbreviation
                if re.match(r'^(?:Employee|Customer|Patient|Client|Cardholder|Manager|Director)\s+Dr\.?$', entity_text, re.IGNORECASE):
                    continue
                # Skip HTML tags and fragments
                if '<' in entity_text or '>' in entity_text:
                    continue
                text_lower = entity_text.lower().strip()

                # Filter "via username" patterns (e.g., "via john_doe", "via @handle")
                if re.match(r'^via\s+\S+$', text_lower) and ' ' not in text_lower.split('via ', 1)[-1].strip():
                    continue
                # Filter "Dear Name" patterns (salutations, not addresses)
                if re.match(r'^dear\s+\w+', text_lower):
                    continue
                # Filter phone-number fragments (NNN-NNNN) misdetected as address/location
                if re.match(r'^\d{3}[-.]\d{4}$', entity_text.strip()):
                    continue

                # Filter OCR garbage and malformed patterns
                if re.match(r'#\d+[a-z]', entity_text, re.IGNORECASE):
                    continue
                # Skip garbage check for UK postcode format (consonant-heavy but legitimate)
                _is_postcode_here = bool(re.match(
                    r'^[A-Za-z]{1,2}\d[A-Za-z\d]?\s?\d[A-Za-z]{2}$', entity_text.strip()))
                if not _is_postcode_here and self._is_garbage_string(entity_text):
                    continue
                if self._is_ocr_artifact(entity_text):
                    continue

                # Filter verb phrases and sentence patterns mistaken as addresses
                verb_patterns = [
                    r'^\d+\s+(is|are|was|were|has|have|can|will|would|could|should|may|might|must|do|does|did)\b',
                    r'^(also|just|only|still|even|now|then|here|there|been|being)\s+\w+',
                    r'^(run|build|send|ship|go|come|move|travel|went|came|been|get|got|make|made|take|took)\s+',
                    r'\b(we|you|they|he|she|it|i)\s+(are|is|was|were|have|has|had|will|would|can|could)\b',
                ]
                if any(re.search(p, entity_text, re.IGNORECASE) for p in verb_patterns):
                    continue

                # Filter numeric-only patterns (not addresses)
                # Exception: 5-digit US ZIP codes (e.g., "28074", "97201")
                if re.match(r'^\s*\d+\s*$', entity_text):
                    stripped = entity_text.strip()
                    if not re.match(r'^\d{5}(?:-\d{4})?$', stripped):
                        continue

                # Filter patterns like "123 to", "456 or", "789 and" (number + preposition/verb)
                if re.match(r'^\d+\s+(to|or|and|at|in|on|of|by|for|is|are|was|were|has|have|can|will|would|could|should)\b', text_lower):
                    continue

                # Filter common non-address label phrases
                non_address_phrases = {
                    'street address', 'mailing address', 'shipping address', 'billing address',
                    'home address', 'work address', 'email address', 'ip address', 'web address',
                    'address line', 'address field', 'address book', 'address bar',
                    'physical address', 'business address', 'residential address',
                    'contact address', 'return address', 'forwarding address',
                }
                if text_lower in non_address_phrases:
                    continue

                # Filter standalone direction/position words
                direction_words = {'north', 'south', 'east', 'west', 'northeast', 'northwest',
                                   'southeast', 'southwest', 'downtown', 'uptown', 'midtown',
                                   'central', 'upper', 'lower', 'inner', 'outer'}
                if text_lower in direction_words:
                    continue

                # Filter common standalone words that are not addresses
                standalone_non_addresses = {
                    'abroad', 'overseas', 'domestic', 'local', 'remote', 'virtual',
                    'online', 'offline', 'mobile', 'portable', 'temporary', 'permanent',
                    'primary', 'secondary', 'alternate', 'backup', 'default',
                    'pending', 'processing', 'completed', 'confirmed', 'verified',
                }
                if text_lower in standalone_non_addresses:
                    continue

                # Filter if it ends with common prepositions (incomplete pattern)
                if re.search(r'\s(in|on|at|to|of|by|for|from|with|as|or|and|but|so|if|than)$', text_lower):
                    continue

                # Filter if it starts with prepositions (sentence fragment)
                if re.match(r'^(in|on|at|to|of|by|for|from|with|as|the|a|an)\s+\w+$', text_lower) and len(text_lower.split()) <= 3:
                    if entity.confidence < 0.85:
                        continue

                # Check city/geo status upfront (used for short-text filter AND verifier bypass)
                words = entity_text.split()
                _is_known_city = False
                if CITIES_DB_AVAILABLE:
                    _cdb = get_cities_db()
                    _is_known_city = _cdb.is_city(entity_text.strip())
                _is_city_recognizer = (entity.recognition_metadata or {}).get('recognizer_name') == 'CityRecognizer'

                # Filter very short text (<5 chars) without strong indicators
                if len(entity_text) < 5:
                    # Keep US state abbreviations, ZIP codes, and known cities
                    is_state_abbrev = re.match(r'^[A-Z]{2}$', entity_text.strip())
                    is_zip = re.match(r'^\d{5}$', entity_text.strip())
                    _is_uk_postcode_short = bool(re.match(
                        r'^[A-Za-z]{1,2}\d[A-Za-z\d]?\s?\d[A-Za-z]{2}$', entity_text.strip()))
                    if not (is_state_abbrev or is_zip or _is_known_city or _is_city_recognizer or _is_uk_postcode_short):
                        continue
                _compass_prefixes = {'north', 'south', 'east', 'west', 'new', 'old',
                                     'upper', 'lower', 'great', 'little', 'central',
                                     'inner', 'outer', 'mount', 'fort', 'port', 'san',
                                     'santa', 'saint', 'st', 'cape', 'el', 'la', 'las', 'los'}
                _has_geo_prefix = len(words) >= 2 and words[0].lower() in _compass_prefixes

                # Filter simple word pairs/triples without address indicators
                if len(words) <= 3 and all(w.isalpha() for w in words):
                    street_types = {'street', 'st', 'avenue', 'ave', 'road', 'rd', 'drive', 'dr',
                                   'lane', 'ln', 'blvd', 'boulevard', 'way', 'court', 'ct',
                                   'circle', 'cir', 'place', 'pl', 'terrace', 'ter', 'highway',
                                   'hwy', 'parkway', 'pkwy', 'trail', 'path', 'pike', 'alley',
                                   'square', 'plaza', 'crescent', 'close', 'gardens', 'grove',
                                   'row', 'mews', 'rise', 'walk', 'hill', 'green', 'commons'}
                    location_words = {'city', 'county', 'state', 'country', 'province', 'region',
                                      'district', 'township', 'village', 'town', 'borough', 'parish',
                                      'heights', 'crossing', 'point', 'springs', 'falls', 'bridge',
                                      'port', 'haven', 'bay', 'beach', 'lake', 'creek', 'valley',
                                      'mountain', 'ridge', 'island', 'harbour', 'harbor', 'hills',
                                      'park', 'forest', 'meadow', 'glen', 'shire', 'heath'}
                    has_street_type = any(w.lower() in street_types for w in words)
                    has_location_word = any(w.lower() in location_words for w in words)
                    if not has_street_type and not has_location_word:
                        # Allow US state abbreviations (e.g., "NC", "TX", "CA")
                        _us_state_set = {'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
                                         'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
                                         'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
                                         'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
                                         'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'}
                        _is_us_state = len(words) == 1 and words[0].upper() in _us_state_set
                        if not (_is_us_state or _is_known_city or _is_city_recognizer or _has_geo_prefix):
                            # Lower threshold for multi-word entities (2-3 words are more likely valid)
                            _threshold = 0.78 if len(words) >= 2 else 0.88
                            if entity.confidence < _threshold:
                                continue

                # Filter sentence-like patterns (contains common verbs)
                common_verbs = {'is', 'are', 'was', 'were', 'be', 'been', 'being',
                                'have', 'has', 'had', 'do', 'does', 'did', 'done',
                                'will', 'would', 'could', 'should', 'can', 'may', 'might',
                                'go', 'goes', 'went', 'gone', 'come', 'comes', 'came',
                                'make', 'makes', 'made', 'take', 'takes', 'took', 'taken',
                                'get', 'gets', 'got', 'give', 'gives', 'gave', 'given',
                                'see', 'sees', 'saw', 'seen', 'know', 'knows', 'knew', 'known',
                                'gathered', 'appreciate', 'enjoyed', 'attended', 'reviewed',
                                'said', 'told', 'asked', 'wanted', 'needed', 'tried',
                                'the', 'that', 'this', 'these', 'those'}
                if any(w.lower() in common_verbs for w in words):
                    if entity.confidence < 0.90:
                        continue

                # UK postcode bypass: distinctive format, skip LightGBM verifier
                _is_uk_postcode = bool(re.match(
                    r'^[A-Za-z]{1,2}\d[A-Za-z\d]?\s?\d[A-Za-z]{2}$', entity_text.strip()))
                # Unit/dwelling bypass: Suite/Flat/Office/Block/Basement + number
                _is_unit_pattern = bool(re.match(
                    r'^(?:Suite|Flat|Apt|Apartment|Unit|Office|Loft|Box|Room|Floor|Block|Basement|Level|Wing|Bay|Annex|Garage)\s+\d',
                    entity_text.strip(), re.IGNORECASE))
                # Known city/geo bypass: these fail libpostal validation (weight 0.3 < 0.7)
                _is_city_bypass = _is_known_city or _is_city_recognizer or _has_geo_prefix

                # LightGBM-based address verification (uses pre-loaded model)
                if ADDRESS_VERIFIER_AVAILABLE and not _is_uk_postcode and not _is_unit_pattern and not _is_city_bypass:
                    is_valid, adjusted_confidence = verify_address_detection(
                        text, entity_text, entity.start, entity.end, entity.confidence
                    )
                    if not is_valid:
                        continue
                    # Update confidence if adjusted
                    if adjusted_confidence != entity.confidence:
                        entity = PIIEntity(
                            entity_type=entity.entity_type,
                            text=entity.text,
                            start=entity.start,
                            end=entity.end,
                            confidence=adjusted_confidence,
                            pattern_name=entity.pattern_name,
                            locale=entity.locale,
                            recognition_metadata=entity.recognition_metadata,
                        )

            # Filter PERSON-specific false positives
            if entity.entity_type == "PERSON":
                # (Span cleanup already applied at top of loop)

                # Check if entity is from labeled detector (trusted source)
                _is_labeled = (entity.recognition_metadata or {}).get('recognizer_name') == 'LabeledPIIDetector'
                # Skip HTML closing tags and fragments (</, <div, etc.)
                if entity_text.startswith('<') or '</' in entity_text:
                    continue
                # Filter label-like text: "Name: X", but allow person-context labels
                # (Cardholder: Name, Contact: Name, etc. are valid PERSON detections)
                _label_match = re.match(r'^\s*(\w+):\s*\w', entity_text)
                if _label_match:
                    _label_word = _label_match.group(1).lower()
                    _person_labels = {'cardholder', 'contact', 'patient', 'client', 'customer',
                                      'applicant', 'employee', 'beneficiary', 'holder', 'owner',
                                      'recipient', 'sender', 'guardian', 'witness', 'spouse',
                                      'parent', 'agent', 'representative', 'name', 'person'}
                    if _label_word not in _person_labels:
                        continue
                # Filter social media/brand names mistaken for people
                brand_names = {'instagram', 'tiktok', 'youtube', 'snapchat', 'whatsapp', 'facebook', 'twitter', 'linkedin', 'pinterest', 'reddit'}
                if entity_text.lower() in brand_names:
                    continue
                # Filter single letter pairs with low confidence: "TL", "JF"
                if len(entity_text.replace(' ', '')) == 2 and entity_text.isupper():
                    if entity.confidence < 0.90:
                        continue
                # Cross-entity filtering: suppress PERSON contained within ADDRESS or COMPANY
                # Street names (e.g. "Johnson" in "819 Johnson Course") and company names
                # (e.g. "Taylor" in "Becker, Taylor and Davis") trigger false PERSON detections
                person_in_address = any(
                    a_start <= entity.start and entity.end <= a_end
                    for a_start, a_end in address_spans
                )
                person_in_company = any(
                    c_start <= entity.start and entity.end <= c_end
                    for c_start, c_end in company_spans
                )
                # Allow PERSON through if COMPANY span looks like a sentence, not a name
                # e.g., "Reach out to Shannon Matthews at Key Ltd" — the COMPANY span
                # swallows the person, but the text before the name has sentence words
                if person_in_company:
                    for c_start, c_end in company_spans:
                        if c_start <= entity.start and entity.end <= c_end:
                            before_person = text[c_start:entity.start].lower()
                            sentence_words = {'to', 'from', 'by', 'with', 'for',
                                              'contact', 'reach', 'ask', 'call',
                                              'email', 'notify', 'inform', 'dear'}
                            if any(w in before_person.split() for w in sentence_words):
                                person_in_company = False
                            break
                if (person_in_address or person_in_company) and not _is_labeled:
                    continue

                # --- Enhanced PERSON-in-address filtering ---
                # Addresses often detected as LOCATION (not ADDRESS), and PERSON
                # detections from street names/city names overlap with LOCATION spans.

                # Filter PERSON fully contained within a LOCATION span
                # Exception: labeled detections (XML tags, greetings) are trusted
                # Only suppress if LOCATION has >= confidence than the PERSON
                # (prevents low-confidence bogus LOCATIONs from killing good PERSONs)
                if not _is_labeled:
                    person_in_location = any(
                        l_start <= entity.start and entity.end <= l_end
                        and l_conf >= entity.confidence
                        for l_start, l_end, l_conf in location_spans
                    )
                    if person_in_location:
                        continue

                # Compute overlap between this PERSON and any LOCATION span
                _person_loc_overlap = 0
                for l_start, l_end, _l_conf in location_spans:
                    _ov_start = max(entity.start, l_start)
                    _ov_end = min(entity.end, l_end)
                    if _ov_end > _ov_start:
                        _person_loc_overlap += _ov_end - _ov_start
                _person_len = entity.end - entity.start
                _has_significant_loc_overlap = (
                    (_person_loc_overlap / _person_len > 0.5) if _person_len > 0 else False
                )

                _person_words = entity_text.split()
                _person_words_clean = [w.lower().rstrip('.,;:') for w in _person_words]

                # Street type words that are NOT common last names
                _safe_street_types = {
                    'course', 'drive', 'drives', 'road', 'roads', 'street',
                    'streets', 'avenue', 'avenues', 'way', 'ways', 'court',
                    'courts', 'boulevard', 'circle', 'circles', 'terrace',
                    'trail', 'trails', 'highway', 'parkway', 'plaza', 'spur',
                    'spurs', 'knoll', 'knolls', 'viaduct', 'motorway',
                    'crossing', 'passage', 'summit', 'overpass', 'underpass',
                    'turnpike', 'loop', 'alley', 'mews', 'skyway',
                    'stravenue', 'throughway', 'apt', 'apt.', 'suite', 'unit',
                }
                # Filter PERSON containing unambiguous street type words
                if any(w in _safe_street_types for w in _person_words_clean):
                    continue

                # Filter PERSON containing "Apt.", "Suite", or "Unit" (strong address signal)
                if re.search(r'\bApt\.?\b|\bSuite\b|\bUnit\b', entity_text, re.IGNORECASE):
                    continue

                # Filter "Name, XX" pattern (comma + 2-letter state abbreviation)
                if re.search(r',\s*[A-Z]{2}$', entity_text):
                    continue

                # Filter USNS/USS/USCGC prefix (military vessel names, not people)
                if re.match(r'^(USNS|USS|USCGC)\s', entity_text):
                    continue

                # Address-context words (geographic features used in street names)
                _address_context_words = {
                    'port', 'ports', 'ramp', 'forge', 'forges', 'key', 'keys',
                    'corner', 'corners', 'light', 'lights', 'plain', 'plains',
                    'valley', 'valleys', 'crest', 'meadow', 'meadows', 'creek',
                    'run', 'gateway', 'isle', 'neck', 'falls', 'spring',
                    'springs', 'orchard', 'inlet', 'fork', 'forks', 'bend',
                    'walk', 'walks', 'flat', 'flats', 'point', 'points',
                    'cape', 'mount', 'bluff', 'landing', 'hollow', 'bayou',
                    'trace', 'dam', 'locks', 'rapids', 'shoal', 'bar',
                    'estate', 'estates', 'manor', 'village', 'cove', 'island',
                    'islands', 'square', 'squares', 'tunnel', 'harbor',
                    'harbors', 'common', 'commons', 'stream', 'pine', 'pines',
                    'shore', 'shores', 'cliff', 'cliffs', 'mountain',
                    'mountains', 'green', 'greens', 'center', 'pass', 'mall',
                    'extension', 'extensions', 'well', 'wells', 'lake',
                }
                _has_addr_context = any(
                    w in _address_context_words for w in _person_words_clean
                )

                # Filter: address context word + text ends with comma (address fragment)
                if _has_addr_context and entity_text.strip().endswith(','):
                    continue

                # Filter: multiple commas in text (address pattern "Street, City, ST")
                if entity_text.count(',') >= 2:
                    continue

                # Filter: significant LOCATION overlap + trailing comma
                if _has_significant_loc_overlap and entity_text.strip().endswith(','):
                    continue

                # Filter: significant LOCATION overlap + address context word
                if _has_significant_loc_overlap and _has_addr_context:
                    continue

                # Filter common English words falsely detected as PERSON names
                # These are nouns, verbs, adjectives that happen to match name databases
                _person_text_clean = entity_text.strip(' ,.\'"')
                _person_text_lower = _person_text_clean.lower()
                _common_non_names = {
                    'agreement', 'contract', 'proposal', 'report', 'program',
                    'programs', 'scholarship', 'future', 'smart', 'fair',
                    'utilize', 'lentil', 'yogurt', 'parents', 'students',
                    'members', 'changes', 'monitoring', 'support', 'insights',
                    'skills', 'ensuring', 'addressing', 'knowledge', 'progress',
                    'emotional', 'financial', 'virtual', 'inclusive', 'digital',
                    'teaching', 'preferred', 'authorized', 'panelist', 'classrooms',
                    # Titles, ranks, roles (not personal names)
                    'esquire', 'countess', 'selectman', 'major', 'colonel',
                    'captain', 'sergeant', 'lieutenant', 'admiral', 'general',
                    'corporal', 'marshal', 'constable', 'magistrate', 'chancellor',
                    'infant', 'deacon', 'reverend', 'bishop', 'cardinal',
                    # Additional common words seen as FPs
                    'absence', 'badges', 'checking', 'cleaning', 'restocking',
                    'through', 'known', 'strategic', 'emphasizing', 'heritage',
                    'sterling', 'premium', 'platinum', 'bronze', 'diamond',
                    'liberty', 'justice', 'guardian', 'pioneer', 'frontier',
                    'summit', 'prospect', 'paramount', 'sovereign', 'dominion',
                    'legacy', 'genesis', 'triumph', 'victory', 'valor',
                    # Common English words misdetected as PERSON by NER
                    'stay', 'monitor', 'happy', 'wishing', 'cards', 'terms',
                    'department', 'university', 'important', 'updates',
                    'sure', 'filled', 'family', 'group', 'guardian',
                    'signature', 'welcome', 'dear', 'sincerely', 'regards',
                    'respectfully', 'greetings', 'attachment', 'enclosed',
                    'context', 'instructions', 'background', 'overview',
                    'guidelines', 'requirements', 'procedures', 'regulations',
                    'notification', 'conclusion', 'introduction', 'appendix',
                    'britain', 'scotland', 'ireland', 'wales', 'england',
                    # Common verbs/adjectives/nouns from ai4privacy FPs
                    'expected', 'step', 'pursue', 'pursue', 'applied', 'forward',
                    'pending', 'standard', 'resolve', 'navigate', 'achieve',
                    'maintain', 'establish', 'preserve', 'consider', 'proceed',
                    'resolve', 'request', 'approach', 'balance', 'review',
                }
                if _person_text_lower in _common_non_names:
                    continue
                # Filter multi-word PERSON detections that look like titles/headings
                _heading_words = {
                    'team', 'members', 'support', 'contact', 'time',
                    'report', 'proposal', 'changes', 'monitoring',
                    'approaches', 'misconceptions', 'classrooms',
                    'empowerment', 'inclusivity', 'methodology',
                    'security', 'manager', 'episode', 'podcast',
                    'financial', 'partnership', 'officer', 'research',
                    'virtual', 'digital', 'progress', 'strategic',
                    'director', 'coordinator', 'specialist', 'analyst',
                    'inspector', 'supervisor', 'administrator',
                    'assessment', 'development', 'engagement',
                    'evaluation', 'implementation', 'initiative',
                    'important', 'updates', 'group', 'guardian',
                    'authorized', 'preferred', 'department', 'university',
                    'cultural', 'exchange', 'programs', 'executive', 'summary',
                    'original', 'sender', 'signature', 'instructions',
                    'context', 'background', 'overview', 'guidelines',
                }
                if len(_person_words) > 2:
                    _heading_count = sum(1 for w in _person_words_clean if w in _heading_words)
                    if _heading_count > len(_person_words_clean) // 2:
                        continue
                # Also catch 2-word role/title phrases
                if len(_person_words) == 2:
                    if all(w in _heading_words for w in _person_words_clean):
                        continue
                # Filter PERSON detections with trailing punctuation (brackets, quotes)
                if entity_text.rstrip()[-1:] in (']', '}', ')', '"', "'", '*', ':'):
                    if entity.confidence < 0.95:
                        continue
                # Filter "Participant X", "Player X" patterns (not real names)
                if re.match(r'^(?:Participant|Player|Party|Team)\s+[A-Z]$', _person_text_clean):
                    continue
                # Filter email fragments detected as PERSON (e.g., "G@gmail.com")
                if '@' in entity_text:
                    continue
                # Filter possessive forms (e.g., "Britain's") - locations/nouns, not names
                if entity_text.rstrip().endswith("'s") or entity_text.rstrip().endswith("'s"):
                    continue

                # Filter labeled detector FPs: titles/roles detected via "Name: X" patterns
                if entity.pattern_name and 'labeled_' in entity.pattern_name:
                    _title_words = {
                        'esquire', 'countess', 'selectman', 'major', 'colonel',
                        'captain', 'sergeant', 'lieutenant', 'infant', 'deacon',
                        'reverend', 'bishop', 'cardinal', 'constable', 'marshal',
                        'mr', 'mrs', 'ms', 'dr', 'prof', 'sir', 'dame', 'lord',
                        'lady', 'hon', 'judge', 'justice', 'senator', 'governor',
                    }
                    _first_word = _person_text_clean.split()[0].lower() if _person_text_clean else ''
                    if _first_word in _title_words:
                        continue
                    # Filter all-caps "names" from labeled detector (likely abbreviations)
                    if _person_text_clean.isupper() and len(_person_text_clean) <= 5:
                        continue
                    # Filter single-word labeled results that end in common suffixes
                    if len(_person_words) == 1 and len(_person_text_clean) > 3:
                        _non_name_suffixes = ('ing', 'tion', 'ment', 'ness', 'ence', 'ance', 'ity')
                        if _person_text_lower.endswith(_non_name_suffixes):
                            continue

            # Filter NATIONAL_ID false positives
            if entity.entity_type == "NATIONAL_ID":
                # Skip HTML tags and fragments
                if '<' in entity_text or '>' in entity_text:
                    continue
                digits = re.sub(r'\D', '', entity_text)
                # Skip if it looks like a credit card (12-19 digits with card prefixes)
                # Maestro: 12-19 digits starting with 50, 56-69, 5018, 5020, 5038, 6304, 6759, 6761-6763
                if len(digits) >= 12 and len(digits) <= 19:
                    # Maestro/Debit prefixes (50, 56-69)
                    if digits[:2] in ('50', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69'):
                        continue
                    # Visa (4), Mastercard (5), Amex/Diners (3), Discover (6)
                    if len(digits) >= 15 and digits[0] in ('3', '4', '5', '6'):
                        continue
                    # Also check for Mastercard 2xxx range
                    if len(digits) >= 15 and digits[:4].isdigit() and 2221 <= int(digits[:4]) <= 2720:
                        continue
                # Skip if it has credit card format (4-4-4-4 with separators)
                if re.match(r'^\d{4}[\s-]\d{4}[\s-]\d{4}[\s-]?\d{0,4}$', entity_text):
                    continue
                # Skip if it looks like a phone number (10-11 digits with dashes/spaces)
                # But dot-separated NNN.NNN.NNNN is more characteristic of SSN/national ID
                if len(digits) == 10 or len(digits) == 11:
                    # Trust entities from labeled detector (explicit XML/JSON tag)
                    _recognizer = entity.recognition_metadata.get('recognizer_name', '') if entity.recognition_metadata else ''
                    if _recognizer != 'LabeledPIIDetector':
                        # Don't suppress if SSN/ID context keywords are present nearby
                        nid_ctx_start = max(0, entity.start - 50)
                        nid_ctx = text[nid_ctx_start:entity.start].lower()
                        nid_keywords = {'ssn', 'social security', 'national id', 'identification',
                                        'id number', 'tax id', 'tin', 'national insurance', 'nino',
                                        'id:', 'number:', 'social_number', 'social number'}
                        has_nid_context = any(kw in nid_ctx for kw in nid_keywords)
                        if not has_nid_context:
                            # Dot-separated NNN.NNN.NNNN: keep as ID (dots are SSN-characteristic)
                            _is_dot_separated = bool(re.match(r'^\d{3}\.\d{3}\.\d{4}$', entity_text))
                            if not _is_dot_separated:
                                # Phone patterns: (XXX) XXX-XXXX, XXX-XXX-XXXX, XXX XXX XXXX
                                # Only suppress if it looks like a valid NANP phone (area code 2XX-9XX)
                                _phone_match = re.match(r'^[\(\s]*(\d{3})[\)\s\-]*(\d{3})[\s\-]*\d{4}$', entity_text)
                                if _phone_match:
                                    _area = _phone_match.group(1)
                                    _exchange = _phone_match.group(2)
                                    # NANP: area code and exchange both start with 2-9
                                    # If either starts with 0 or 1, it's not a phone → keep as NID
                                    if _area[0] in '23456789' and _exchange[0] in '23456789':
                                        continue
                        # Canadian/Australian format: XXX XXX XXX or XXX-XXX-XXX (9 digits)
                        if len(digits) == 9 and re.match(r'^\d{3}[\s\-\.]\d{3}[\s\-\.]\d{3}$', entity_text):
                            continue
                # Skip if it looks like an IP address
                if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', entity_text):
                    continue
                # Skip if it matches UUID pattern (8-4-4-4-12 hex)
                if re.match(r'^[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}$', entity_text):
                    continue
                # For very generic digit patterns (12 digits with spaces - Japanese/Indian),
                # require context keywords OR high confidence to avoid random numbers
                # Only apply to the most generic patterns (digits-only, 12+ digits)
                if re.match(r'^[\d\s]+$', entity_text) and len(digits) >= 12:
                    if entity.confidence < 0.8:  # Allow high-confidence detections
                        context_start = max(0, entity.start - 80)
                        context = text[context_start:entity.start].lower()
                        id_keywords = {
                            'passport', 'national id', 'identity', 'identification', 'id number',
                            'aadhaar', 'mynumber', 'my number', 'tfn', 'tax file',
                            'sin', 'social insurance', 'nino', 'national insurance',
                            'steuer', 'cpr', 'personnummer', 'pesel', 'nie', 'dni',
                            'number:', 'id:', '#:',
                        }
                        if not any(kw in context for kw in id_keywords):
                            continue
                # Skip very low confidence
                if entity.confidence < 0.5:
                    continue

                # Skip short digit sequences that are likely IDs or order numbers
                if len(digits) <= 8 and entity.confidence < 0.75:
                    continue

                # Skip if it's just numbers without structure (likely random)
                # But keep if SSN/ID context is present nearby
                if re.match(r'^\d+$', entity_text) and len(digits) < 11:
                    if entity.confidence < 0.80:
                        bare_ctx_start = max(0, entity.start - 60)
                        bare_ctx = text[bare_ctx_start:entity.start].lower()
                        bare_id_kw = {'ssn', 'social security', 'social number', 'national id',
                                      'identification', 'tax id', 'id number', 'id card',
                                      'number:', 'id:'}
                        if not any(kw in bare_ctx for kw in bare_id_kw):
                            continue

                # Skip if context suggests it's NOT a national ID (order/invoice/account numbers)
                # But don't filter if SSN/ID-positive context is present
                context_start = max(0, entity.start - 60)
                context_end = min(len(text), entity.end + 30)
                surrounding = text[context_start:context_end].lower()
                id_positive_keywords = {
                    'ssn', 'social security', 'social number', 'national id',
                    'identification number', 'id card', 'identity', 'tax id', 'tin',
                    'national insurance', 'nino', 'passport', 'driver',
                    'id number', 'id:', 'personal id', 'citizen id', 'employee id',
                }
                has_id_context = any(kw in surrounding for kw in id_positive_keywords)
                if not has_id_context:
                    non_id_keywords = {
                        'order', 'invoice', 'account', 'reference', 'transaction', 'tracking',
                        'confirmation', 'receipt', 'payment', 'purchase', 'booking', 'reservation',
                        'ticket', 'serial', 'case', 'claim', 'policy', 'member', 'customer',
                        'employee', 'badge', 'file', 'record', 'entry', 'item', 'product', 'sku',
                        'batch', 'lot', 'job', '#:', 'no.', 'no:', 'number:', 'ref:',
                    }
                    if any(kw in surrounding for kw in non_id_keywords):
                        # Only filter if confidence is moderate
                        if entity.confidence < 0.85:
                            continue

                # Validate with stdnum (checksum validation for 35+ countries)
                # Boost confidence for valid IDs, reduce for invalid
                if STDNUM_AVAILABLE and len(digits) >= 8:
                    validation_result = self._validate_national_id_with_stdnum(entity_text)
                    if validation_result == "valid":
                        # Boost confidence for valid IDs
                        new_confidence = min(0.98, entity.confidence + 0.15)
                        entity = PIIEntity(
                            entity_type=entity.entity_type,
                            text=entity.text,
                            start=entity.start,
                            end=entity.end,
                            confidence=new_confidence,
                            pattern_name=entity.pattern_name,
                            locale=entity.locale,
                            recognition_metadata=entity.recognition_metadata,
                        )
                    elif validation_result == "possible":
                        # Small boost for Luhn-valid IDs
                        new_confidence = min(0.95, entity.confidence + 0.08)
                        entity = PIIEntity(
                            entity_type=entity.entity_type,
                            text=entity.text,
                            start=entity.start,
                            end=entity.end,
                            confidence=new_confidence,
                            pattern_name=entity.pattern_name,
                            locale=entity.locale,
                            recognition_metadata=entity.recognition_metadata,
                        )

            # Filter DRIVER_LICENSE - too many false positives from Presidio's default
            if entity.entity_type == "DRIVER_LICENSE":
                # Skip if too short (less than 6 chars) or low confidence
                if len(entity_text) < 6 or entity.confidence < 0.7:
                    continue
                # Skip if it's just digits (likely a phone/credit card fragment)
                if entity_text.replace("-", "").replace(" ", "").isdigit():
                    if len(entity_text.replace("-", "").replace(" ", "")) < 8:
                        continue

            # Filter DATE_TIME - reduce false positives from phone numbers and credit cards
            if entity.entity_type == "DATE_TIME":
                # Skip HTML tags and fragments (< and > are often detected as DATE_TIME)
                if '<' in entity_text or '>' in entity_text:
                    continue
                # Skip if it looks like an international phone number (+CC XXXXXXXXXX)
                if re.match(r'^\+\d{1,3}[\s-]?\d{7,14}$', entity_text):
                    continue
                # Skip if it looks like a phone number fragment (contains dashes in phone pattern)
                if re.match(r'^\d{3,4}[-.]?\d{4}$', entity_text):
                    continue
                # Skip if it looks like a credit card fragment (4 digit groups)
                if re.match(r'^\d{4}[\s-]?\d{4}', entity_text):
                    continue
                # Skip very short matches (allow times like "14:30" = 5 chars)
                if len(entity_text) < 4:
                    continue
                # Skip if it's just a street number pattern like "2901 E"
                if re.match(r'^\d+\s+[A-Z]$', entity_text):
                    continue
                # Skip if it's just numbers (no date separators or month names)
                if re.match(r'^\d+$', entity_text):
                    continue
                # Skip standalone years unless date/birth context is nearby
                if re.match(r'^(19|20)\d{2}$', entity_text):
                    ctx_start = max(0, entity.start - 50)
                    ctx = text[ctx_start:entity.start].lower()
                    year_keywords = {'born', 'birth', 'year', 'since', 'from', 'in', 'during',
                                     'until', 'between', 'circa', 'date', 'died', 'established'}
                    if not any(kw in ctx for kw in year_keywords):
                        continue
                # Skip very low confidence
                if entity.confidence < 0.5:
                    continue
                # Skip "X years" patterns - these are durations, not dates
                # Examples: "12 years", "over 10 years", "19 years", "18 years"
                if re.match(r'^(?:over\s+)?(\d+)\s*years?$', entity_text.lower()):
                    continue
                # Skip other duration patterns: hours, days, weeks, months, etc.
                # Examples: "48 hours", "3 months", "6 months", "60 credit hours", "7 business days"
                if re.match(r'^(?:over\s+)?(\d+)\s*(?:credit\s+)?(?:hours?|minutes?|seconds?|days?|weeks?|months?|business\s+days?)$', entity_text.lower()):
                    continue
                # Skip decade references like "the 1920s", "the 1990s"
                if re.match(r'^the\s+\d{4}s$', entity_text.lower()):
                    continue
                # Skip generic time-related words that aren't actual dates
                date_false_positives = {
                    'annual', 'annually', 'yearly', 'monthly', 'weekly', 'daily', 'quarterly',
                    'the year', 'the month', 'the date', 'the day', 'the week', 'the quarter',
                    'this year', 'last year', 'next year', 'each year', 'per year', 'every year',
                    'this month', 'last month', 'next month', 'each month', 'per month',
                    'fiscal year', 'calendar year', 'tax year', 'financial year',
                    'year ended', 'year ending', 'year end', 'year-end', 'year to date',
                    'schedule i', 'schedule ii', 'schedule iii', 'schedule iv', 'schedule v',
                    'part i', 'part ii', 'part iii', 'part iv', 'part v',
                    'section i', 'section ii', 'section iii', 'section iv', 'section v',
                    'the year ended on that date', 'during the year', 'for the year',
                }
                if entity_text.lower() in date_false_positives:
                    continue
                # Skip if it's a generic phrase without a specific date (no digits or month names)
                month_names = ['january', 'february', 'march', 'april', 'may', 'june',
                              'july', 'august', 'september', 'october', 'november', 'december',
                              'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
                has_digits = any(c.isdigit() for c in entity_text)
                has_month = any(month in entity_text.lower() for month in month_names)
                if not has_digits and not has_month:
                    continue

            # Filter BANK_NUMBER - many false positives
            if entity.entity_type == "BANK_NUMBER":
                # Keep bank accounts between 8-17 digits (covers most international formats)
                # 8-12 digits: US routing/account numbers
                # 13-17 digits: International BBAN, Asian bank accounts
                digits = re.sub(r'\D', '', entity_text)
                if len(digits) < 8 or len(digits) > 17:
                    continue
                # Skip if it matches credit card pattern (starts with 3, 4, 5, 6)
                if digits[0] in ('3', '4', '5', '6') and len(digits) >= 15:
                    continue
                # Skip if low confidence (unless it has "beneficiary" or "account" context)
                if entity.confidence < 0.6:
                    continue

            # Filter ITIN false positives (from SSN-like patterns)
            if entity.entity_type == "ITIN":
                # ITIN must start with 9 and have specific second digit patterns
                digits = re.sub(r'\D', '', entity_text)
                if len(digits) != 9 or not digits.startswith('9'):
                    continue
                # Second digit must be 7 or 8 (ITIN requirement)
                if digits[1] not in ('7', '8'):
                    continue

            # Filter MEDICAL_LICENSE false positives
            if entity.entity_type == "MEDICAL_LICENSE":
                # Skip if confidence is low
                if entity.confidence < 0.7:
                    continue

            # Filter PASSPORT false positives
            if entity.entity_type == "PASSPORT":
                # Skip common words that look like passport numbers
                passport_false_positives = {
                    'chartered', 'certified', 'registered', 'licensed', 'qualified',
                    'accredited', 'authorized', 'approved', 'verified', 'validated',
                }
                if entity_text.lower() in passport_false_positives:
                    continue
                # Passport numbers should be alphanumeric, not just letters
                if entity_text.isalpha() and len(entity_text) > 5:
                    continue
                # Skip if confidence is low
                if entity.confidence < 0.7:
                    continue

            # Filter URL false positives (from email domain parts)
            if entity.entity_type == "URL":
                # Skip if it's just a domain without protocol (likely email domain)
                if not entity_text.startswith(('http://', 'https://', 'www.')):
                    # Check if it's a bare domain (no path)
                    if re.match(r'^[a-z]+\.[a-z]{2,4}$', entity_text.lower()):
                        continue

            # Filter CREDIT_CARD false positives
            if entity.entity_type == "CREDIT_CARD":
                # Skip if it matches IMEI pattern (15 digits with TAC prefixes)
                digits = re.sub(r'\D', '', entity_text)
                if len(digits) == 15 and digits[:2] in ('35', '86', '49', '01', '45', '36', '87', '46'):
                    continue

            # Filter MEDICAL false positives (from name patterns and OCR artifacts)
            if entity.entity_type == "MEDICAL":
                # Skip OCR artifacts (random case, underscores, incomplete words)
                # Examples: "clTse", "Askreffect", "cqNtain", "Americsn", "CRuSe betetr"
                if self._is_ocr_artifact(entity_text):
                    continue
                # Skip if it looks like a person name (Title Case, no medical keywords)
                if entity.confidence < 0.6:
                    # Check if it's a person name pattern
                    if re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+$', entity_text):
                        continue

            # Filter PHONE_NUMBER false positives
            if entity.entity_type == "PHONE_NUMBER":
                # Labeled phone entities bypass most FP filters (trusted source)
                _phone_is_labeled = (entity.recognition_metadata or {}).get('recognizer_name') == 'LabeledPIIDetector'
                # Skip HTML tags and fragments
                if '<' in entity_text or '>' in entity_text:
                    continue
                digits = re.sub(r'\D', '', entity_text)
                # Strip extension digits before length checks (e.g., "(406)485-3615x30515")
                # Phone extensions use x, ext, #, etc. -- don't count extension digits
                # for IMEI/CC/IBAN length-based filters
                has_extension = bool(re.search(r'[xX#]|ext\.?|extension', entity_text))
                if has_extension:
                    # Extract only the base phone digits (before extension marker)
                    base_phone = re.split(r'[xX#]|ext\.?|extension', entity_text)[0]
                    base_digits = re.sub(r'\D', '', base_phone)
                else:
                    base_digits = digits
                # Skip if it's a credit card (16 digits in base number)
                if len(base_digits) == 16:
                    continue
                # Skip if it matches IBAN pattern (long alphanumeric with letters)
                # IBANs are 15-34 chars with letters, phone numbers are digits only
                # Use digit count instead of text length to avoid filtering OCR-spaced phones
                # Exclude extension markers (x/X) from the letter check
                non_ext_text = re.split(r'[xX#]|ext\.?|extension', entity_text)[0]
                if len(base_digits) > 15 and re.search(r'[A-Za-z]', non_ext_text):
                    continue
                # Skip if it looks like an IMEI (15 digits in base number)
                # But not for international-prefix numbers (00/+ add digits to real phones)
                # Also exempt numbers starting with 0 (UK/European local format - IMEIs never start with 0)
                if len(base_digits) == 15:
                    stripped_text = entity_text.strip()
                    if not stripped_text.startswith('+') and not stripped_text.startswith('00') and not stripped_text.startswith('0'):
                        continue
                # Skip if it looks like an IP address (4 octets separated by dots)
                if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', entity_text):
                    continue
                # Skip if it looks like GPS coordinates (decimal format)
                # Require comma or space separator to avoid matching phone numbers like 555.123.4567
                if re.match(r'^-?\d{1,3}\.\d+[,\s]+-?\d{1,3}\.\d+$', entity_text):
                    continue
                # Skip if it looks like a date format (YYYY-MM-DD or DD-MM-YYYY)
                if re.match(r'^\d{4}-\d{2}-\d{2}$', entity_text) or re.match(r'^\d{2}-\d{2}-\d{4}$', entity_text):
                    continue
                # Validate using phonenumbers library (Google's libphonenumber)
                # Returns "valid", "possible", "invalid", or "unknown"
                _phonenumbers_valid = False
                if PHONENUMBERS_AVAILABLE and len(digits) >= 7:
                    region = self._get_phone_region_from_context(text, entity.start, entity.end)
                    validation_result = self._validate_phone_with_phonenumbers(entity_text, region)
                    if validation_result == "valid":
                        _phonenumbers_valid = True
                        # Boost confidence for numbers that phonenumbers validates
                        # This significantly improves recall (87% -> 95%+ potential)
                        new_confidence = min(0.98, entity.confidence + 0.15)
                        entity = PIIEntity(
                            entity_type=entity.entity_type,
                            text=entity.text,
                            start=entity.start,
                            end=entity.end,
                            confidence=new_confidence,
                            pattern_name=entity.pattern_name,
                            locale=entity.locale,
                            recognition_metadata=entity.recognition_metadata
                        )
                    elif validation_result == "possible":
                        # Small boost for OCR-spaced patterns that pass "possible" check
                        # The spacing pattern is strong evidence of a phone number
                        new_confidence = min(0.90, entity.confidence + 0.08)
                        entity = PIIEntity(
                            entity_type=entity.entity_type,
                            text=entity.text,
                            start=entity.start,
                            end=entity.end,
                            confidence=new_confidence,
                            pattern_name=entity.pattern_name,
                            locale=entity.locale,
                            recognition_metadata=entity.recognition_metadata
                        )
                    elif validation_result == "invalid":
                        # Lower confidence for numbers that fail phonenumbers validation
                        # Use gentler penalty for numbers with strong phone formatting
                        _s = entity_text.strip()
                        _has_phone_format = bool(
                            re.match(r'\+\d{1,3}[\s\-\.\(]', _s) or  # intl prefix
                            re.match(r'\(\d{2,4}\)\s*\d', _s) or  # paren area code
                            re.match(r'\d{3,4}[\-\.]\d{3,4}[\-\.]\d{4}', _s) or  # separator format
                            re.match(r'00\d{2,4}[-.\s]', _s) or  # 00 prefix
                            re.match(r'0\d{3,4}[\s\-]\d{4,6}', _s) or  # local prefix
                            re.match(r'0\d{2,4}\.\d{3,4}\.\d{3,4}', _s)  # dot groups
                        )
                        _penalty = 0.90 if _has_phone_format else 0.75
                        entity = PIIEntity(
                            entity_type=entity.entity_type,
                            text=entity.text,
                            start=entity.start,
                            end=entity.end,
                            confidence=entity.confidence * _penalty,
                            pattern_name=entity.pattern_name,
                            locale=entity.locale,
                            recognition_metadata=entity.recognition_metadata
                        )

                # Labeled phone entities bypass context suppression and anchoring
                if not _phone_is_labeled:
                    # Context-based suppression: numeric IDs in labeled fields
                    # SSNs, passports, driver licenses share digit formats with phones
                    # Narrow window (30 before, 10 after) to avoid cross-contaminating nearby phones
                    phone_ctx_start = max(0, entity.start - 30)
                    phone_ctx_end = min(len(text), entity.end + 10)
                    phone_ctx = text[phone_ctx_start:phone_ctx_end].lower()
                    ssn_keywords = {'social security', 'social number', 'socialnumber',
                                    'ssn', '<socialnumber', 'national insurance',
                                    'national id', 'identification number', 'id number',
                                    'id card', 'tax id'}
                    passport_keywords = {'passport', '<passport'}
                    dl_keywords = {"driver's license", 'driver license', 'driverlicense',
                                   '<driverlicense', 'driving licence', 'dl number', 'dl:'}
                    if any(kw in phone_ctx for kw in ssn_keywords):
                        continue
                    if any(kw in phone_ctx for kw in passport_keywords):
                        continue
                    if any(kw in phone_ctx for kw in dl_keywords):
                        continue

                    # Context anchoring: phones without context keywords are likely FPs
                    # Only exempt numbers with strong phone formatting (international prefix)
                    stripped = entity_text.strip()
                    has_intl_prefix = bool(re.match(r'\+\d{1,3}[\s\-\.\(]', stripped))
                    has_paren_area_code = bool(re.match(r'\(\d{2,4}\)\s*\d', stripped))
                    # Also accept numbers with clear phone separator formatting (xxx-xxx-xxxx)
                    # Require 7+ digits total to avoid matching dates like 2026-02-08
                    has_separator_format = False
                    if re.match(r'\d{3,4}[\-\.]\d{3,4}[\-\.]\d{4}', stripped):
                        has_separator_format = True
                    has_00_prefix = bool(re.match(r'00\d{2,4}[-.\s]', stripped))
                    # Accept any mixed-separator format with enough digits
                    has_mixed_separators = bool(re.match(r'\d+[.\-]\d+[\s]\d+', stripped) or
                                                re.match(r'\d+[\s]\d+[.\-]\d+', stripped) or
                                                re.match(r'\d+[.\-]\d+[.\-]\d+[.\-]\d+', stripped))  # 4+ groups with dots/dashes
                    # Local format: 0XX-XXXXXXX through 0XXXXX-XXXXX (starting with 0 + separator)
                    has_local_prefix = bool(re.match(r'0\d{2,5}[\s\-\.]\d{4,10}', stripped))
                    # Dot-separated groups (European): 010.155.741.8175
                    has_dot_groups = bool(re.match(r'0\d{2,4}\.\d{3,4}\.\d{3,4}\.\d{3,4}', stripped))
                    has_strong_formatting = has_intl_prefix or has_paren_area_code or has_separator_format or has_00_prefix or has_mixed_separators or has_local_prefix or has_dot_groups or _phonenumbers_valid
                    if not has_strong_formatting:
                        ctx_start = max(0, entity.start - 100)
                        ctx_end = min(len(text), entity.end + 50)
                        surrounding = text[ctx_start:ctx_end].lower()
                        phone_keywords = {'phone', 'tel', 'mobile', 'cell', 'fax', 'contact',
                                          'call', 'dial', 'sms', 'whatsapp',
                                          'reach', 'hotline', 'helpline', 'ext', 'extension',
                                          'telephone', 'landline', 'cellular'}
                        if not any(kw in surrounding for kw in phone_keywords):
                            continue

            # Filter IP_ADDRESS false positives (version strings like "v1.2.3.4", "Chrome 110.0.5481.77")
            if entity.entity_type == "IP_ADDRESS":
                # Check for version context in preceding text (30 chars)
                context_start = max(0, entity.start - 30)
                preceding_text = text[context_start:entity.start].lower()

                version_indicators = [
                    'v', 'version', 'build', 'release', 'os', 'chrome', 'firefox',
                    'safari', 'edge', 'node', 'python', 'java', 'php', 'ruby',
                    'macos', 'windows', 'linux', 'ios', 'android', 'sdk', 'api',
                    'ver', 'rev', 'revision', 'update', 'patch', 'hotfix'
                ]

                # Check if preceded by version indicator
                preceding_words = preceding_text.split()[-2:] if preceding_text else []
                if any(indicator in preceding_words for indicator in version_indicators):
                    continue

                # Also check for version pattern immediately before (v1.2.3.4, version 1.2.3.4)
                if re.search(r'(?:^|[\s(])v?\d+\.\d+\.', preceding_text):
                    continue

                # Check for identical-octet patterns (e.g., 1.1.1.1, 0.0.0.0)
                # Filter if in tabular context (3+ IP-like patterns nearby)
                octets = entity_text.split('.')
                if len(octets) == 4 and len(set(octets)) == 1:
                    # All octets identical - check if in tabular/list context
                    window_start = max(0, entity.start - 200)
                    window_end = min(len(text), entity.end + 200)
                    window = text[window_start:window_end]
                    ip_count = len(re.findall(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', window))
                    if ip_count >= 3:
                        # Multiple similar patterns suggest tabular data or placeholders
                        continue

                # SSN disambiguation: IPv4-like patterns that are part of longer dot-separated
                # sequences (e.g., "27.01.06.52.N67.7" social security number format)
                if ':' not in entity_text:  # Only for IPv4 detections
                    post_end = min(len(text), entity.end + 10)
                    post_text = text[entity.end:post_end]
                    # If followed by .letter or .digit+letter, it's a longer ID, not an IP
                    if re.match(r'\.[A-Za-z]', post_text):
                        continue
                    if re.match(r'\.\d+[A-Za-z]', post_text):
                        continue
                    # Check what precedes: if digits. immediately before, part of longer sequence
                    pre_start = max(0, entity.start - 5)
                    pre_text = text[pre_start:entity.start]
                    if re.search(r'\d\.$', pre_text):
                        continue
                    # Social number context: "social", "ssn", "número", "numéro"
                    context_start = max(0, entity.start - 60)
                    context_before = text[context_start:entity.start].lower()
                    social_keywords = ['social number', 'social security', 'ssn', 'número', 'numéro',
                                       'social no', 'personal number', 'identification number',
                                       'identity number', 'national number', 'civic number',
                                       'citizen number', 'insurance number', 'pension number']
                    if any(kw in context_before for kw in social_keywords):
                        continue


            # Filter ADDRESS/LOCATION false positives
            if entity.entity_type in ("LOCATION", "ADDRESS"):
                # --- Span cleanup: strip "Address:" label prefix (second pass) ---
                _addr_lbl2 = re.match(r'^(?:Address|Location|Addr)\s*:\s*', entity_text, re.IGNORECASE)
                if _addr_lbl2:
                    _addr_rest2 = entity_text[_addr_lbl2.end():]
                    if len(_addr_rest2.strip()) >= 3:
                        entity = PIIEntity(
                            entity_type=entity.entity_type,
                            text=_addr_rest2,
                            start=entity.start + _addr_lbl2.end(),
                            end=entity.end,
                            confidence=entity.confidence,
                            pattern_name=entity.pattern_name,
                            locale=entity.locale,
                            recognition_metadata=entity.recognition_metadata,
                        )
                        entity_text = _addr_rest2
                # Filter "Employee Dr" type patterns (not addresses)
                if re.match(r'^(?:Employee|Customer|Patient|Client|Cardholder|Manager|Director)\s+Dr\.?$', entity_text, re.IGNORECASE):
                    continue
                # Labeled address entities bypass most FP filters (trusted source)
                _addr_is_labeled = (entity.recognition_metadata or {}).get('recognizer_name') == 'LabeledPIIDetector'
                # Skip OCR artifacts (random case, underscores, incomplete words)
                if self._is_ocr_artifact(entity_text) and not _addr_is_labeled:
                    continue
                # Skip very short text (less than 4 chars) - catches "in", "as", etc.
                # Allow labeled entities >= 2 chars (e.g., city "Ely", country "UK")
                # Allow US state abbreviations (e.g., "NC", "TX", "CA")
                if len(entity_text.strip()) < 4 and not _addr_is_labeled:
                    _short_us_states = {'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
                                        'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
                                        'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
                                        'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
                                        'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'}
                    if entity_text.strip().upper() not in _short_us_states:
                        continue
                if _addr_is_labeled and len(entity_text.strip()) < 2:
                    continue
                # Filter standalone US state abbreviations - require context
                us_state_codes = {'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
                                  'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
                                  'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
                                  'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
                                  'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'}
                if entity_text.upper() in us_state_codes:
                    # Require high confidence OR context keywords OR nearby ZIP/address data
                    if entity.confidence < 0.70:
                        # Wider context window (150 chars before and after)
                        context_start = max(0, entity.start - 150)
                        context_end = min(len(text), entity.end + 150)
                        context_around = text[context_start:context_end].lower()
                        has_keywords = any(kw in context_around for kw in ['state', 'from', 'to', 'city', 'address', 'located', ',', 'zip', 'postal', 'phone', 'name'])
                        # Check for nearby ZIP codes (5-digit numbers) indicating tabular address data
                        has_nearby_zip = bool(re.search(r'\b\d{5}\b', text[context_start:context_end]))
                        if not (has_keywords or has_nearby_zip):
                            continue
                # Skip SHORT fragments ending with prepositions (incomplete address phrases)
                # Examples: "was on", "such as", "niche in" - but not full addresses
                if len(entity_text) < 20 and re.search(r'\s(?:on|as|in|or|at|to|of|by)$', entity_text.lower()):
                    continue
                # Skip street-only patterns ending with "in" (incomplete context)
                # Example: "Tahrir Street in" (missing city/region)
                if re.search(r'(?:street|st|avenue|ave|road|rd|lane|ln|drive|dr|boulevard|blvd)\s+in$', entity_text.lower()):
                    continue
                # Skip common verb phrases mistaken as locations
                verb_phrases = {'resides in', 'focus on', 'depends on', 'based in',
                                'located in', 'situated in', 'works at', 'lives at',
                                'based on', 'rely on', 'built on', 'focused on',
                                'such as', 'focus on',
                                # Additional phrases from FP analysis
                                'also run', 'building yall', 'send to', 'ship to',
                                'travel to', 'work in', 'went to', 'came from',
                                'moved to', 'move to', 'going to', 'headed to',
                                'run in', 'go to', 'been to', 'live in'}
                if entity_text.lower().strip() in verb_phrases:
                    continue
                # Skip number+preposition fragments (not addresses)
                # Examples: "9083 is", "1234 to", "5678 at"
                if re.match(r'^\d+\s+(?:is|are|was|were|to|at|in|on|or)\b', entity_text.lower()):
                    continue
                # Skip "NUMBER ID" patterns (e.g., "1647 ID")
                if re.match(r'^\d+\s*ID$', entity_text.strip(), re.IGNORECASE):
                    continue
                # Skip ID number patterns starting with # (not apartment/unit numbers)
                # Examples: "#44925", "#97100", "#4942" - these are IDs, not addresses
                if re.match(r'^\s*#\d{4,}$', entity_text.strip()):
                    continue
                # Skip field labels (not actual addresses)
                address_field_labels = {
                    'street address', 'user id', 'tax id', 'post id', 'id or',
                    'email me', 'talk to me', 'the road', 'rating',
                }
                if entity_text.lower().strip() in address_field_labels:
                    continue
                # Skip "via email at" patterns
                if 'via email' in entity_text.lower() or 'email at' in entity_text.lower():
                    continue
                # Skip short fragments that end with "as" (common in sentences)
                # Examples: "such as", "well as", "ve done as"
                if len(entity_text) < 15 and entity_text.lower().strip().endswith(' as'):
                    continue
                # Skip numeric-only fragments that are likely IDs
                # Examples: "10000", "1991"
                # Exception: 5-digit US ZIP codes (e.g., "28074", "97201")
                if re.match(r'^\s*\d+\s*$', entity_text):
                    stripped = entity_text.strip()
                    if not re.match(r'^\d{5}(?:-\d{4})?$', stripped):
                        continue
                # Skip common short phrases that aren't locations
                location_short_false_positives = {
                    'in', 'at', 'to', 'on', 'of', 'as', 'by', 'or', 'an', 'is', 'it',
                    'claimed as', 'based on', 'delay in', 'delays in', 'org or',
                    'com or', 'net or', 'info', 'lakhs in', 'crores in',
                    'tds on', 'located in', 'situated at', 'found in',
                    # Additional common false positives from feedback
                    'role in', 're ever in', 'gov or', 'and as', 'for in',
                    'ever in', 'been in', 'here in', 'live in', 'was in',
                }
                if entity_text.lower().strip() in location_short_false_positives:
                    continue
                # Skip patterns like "XXXX or" (numbers followed by "or" - phone/contact fragments)
                if re.match(r'^\d+\s+or\b', entity_text.lower()):
                    continue
                # Skip if it looks like a URL (contains http, www, or common TLDs with path)
                if re.match(r'^(?:https?://|www\.)', entity_text.lower()):
                    continue
                # Skip if it looks like a bare domain with path
                if re.match(r'^[\w-]+\.(?:com|org|net|edu|gov|io|co|de|uk|jp|cn|ru|br|in)/\S*', entity_text.lower()):
                    continue
                # Skip if confidence is low (lowered to 0.50 for better recall)
                if entity.confidence < 0.50:
                    continue
                # Skip if text looks like disclaimer/description rather than address
                # Note: Reduced from large list to avoid filtering valid addresses
                # Only filter clear sentence-like patterns, not word matches
                text_lower = entity_text.lower()
                # Skip if text is clearly a sentence (has verbs/pronouns indicating narrative)
                sentence_indicators = ['i am', "i'm", 'we are', 'they are', 'you are', 'my name is', 'about me']
                if any(indicator in text_lower for indicator in sentence_indicators):
                    continue
                # Skip medium-length text (10-20 chars) that looks like a sentence fragment
                # These often lack address indicators but are picked up by NER
                # Examples: "focus on sales", "work in IT", "based in London" (without street/zip)
                if 10 <= len(entity_text) <= 20:
                    # Require at least one address-like indicator
                    has_number_prefix = re.match(r'^\d+\s+', entity_text)  # Street number
                    has_comma_separator = ',' in entity_text  # City, State format
                    has_street_suffix = re.search(r'\b(?:st|street|ave|avenue|rd|road|dr|drive|ln|lane|blvd|way|ct|court)\b', text_lower)
                    has_location_words = re.search(r'\b(?:road|lane|park|hill|town|village|green|bridge|heath|moor|field|bury|ham|stead|wick|ford|shire)\b', text_lower)
                    if not (has_number_prefix or has_comma_separator or has_street_suffix or has_location_words):
                        # This looks like a phrase, not an address
                        if entity.confidence < 0.65:
                            continue
                # Skip single words that are likely city/country names used in non-address contexts
                words = entity_text.split()
                if len(words) == 1 and len(entity_text) < 15:
                    # Single word locations need moderate confidence (lowered from 0.75)
                    if entity.confidence < 0.60:
                        continue
                    # Filter standalone country names (not PII without full address context)
                    standalone_countries = {
                        'india', 'china', 'japan', 'korea', 'australia', 'canada', 'mexico',
                        'brazil', 'argentina', 'france', 'germany', 'italy', 'spain', 'portugal',
                        'russia', 'ukraine', 'poland', 'netherlands', 'belgium', 'switzerland',
                        'austria', 'sweden', 'norway', 'denmark', 'finland', 'ireland', 'scotland',
                        'england', 'wales', 'britain', 'uk', 'usa', 'america', 'africa', 'asia',
                        'europe', 'singapore', 'malaysia', 'indonesia', 'thailand', 'vietnam',
                        'philippines', 'pakistan', 'bangladesh', 'egypt', 'turkey', 'greece',
                        'israel', 'dubai', 'qatar', 'kuwait', 'saudi', 'iran', 'iraq',
                    }
                    if entity_text.lower() in standalone_countries:
                        continue
                # Require at least one address indicator for texts > 20 chars
                # Note: Raised from >10 to avoid filtering valid short addresses
                if len(entity_text) > 20:
                    address_indicators = [
                        r'^\d+\s+\w',  # Street number at start (123 Main)
                        r'\b(?:st|street|ave|avenue|rd|road|dr|drive|ln|lane|ct|court|blvd|boulevard|way|place|pl|circle|cir|pkwy|parkway|hwy|highway|terrace|ter|crescent|cres|close|gardens|grove|mews|row|walk)\b',  # Street types (expanded)
                        r'\b(?:apt|apartment|suite|ste|unit|floor|fl|bldg|building|flat)\b',  # Unit indicators
                        r'\b[A-Z]{2}\s*\d{5}',  # State + ZIP (CA 90210)
                        r'\b\d{5}(?:-\d{4})?\b',  # ZIP code
                        r'\b[A-Z]\d[A-Z]\s*\d[A-Z]\d\b',  # Canadian postal code
                        r'\b[A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2}\b',  # UK postcode
                        r'\b\d{6}\b',  # Indian PIN code
                        r'\bP\.?O\.?\s*Box\b',  # PO Box
                        r'\b(?:rue|via|calle|strasse|platz|piazza|avenue|boulevard)\b',  # European street types
                        r',\s*[A-Z]{2,3}\s*\d{4,6}',  # City, STATE ZIP pattern
                    ]
                    has_address_indicator = any(re.search(pattern, entity_text, re.IGNORECASE) for pattern in address_indicators)
                    if not has_address_indicator:
                        continue
                # For longer addresses (>25 chars), require address indicator
                # Relaxed from >20 to allow more addresses through
                if len(entity_text) > 25:
                    has_street_number = re.match(r'^\d+\s+', entity_text)
                    has_po_box = re.search(r'\bP\.?O\.?\s*Box\b', entity_text, re.IGNORECASE)
                    has_zip = re.search(r'\b\d{5}(?:-\d{4})?\b', entity_text)
                    has_postal_code = re.search(r'\b[A-Z]\d[A-Z]\s*\d[A-Z]\d\b', entity_text)
                    has_uk_postcode = re.search(r'\b[A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2}\b', entity_text)
                    has_india_pin = re.search(r'\b[1-9]\d{5}\b', entity_text)
                    if not (has_street_number or has_po_box or has_zip or has_postal_code or has_uk_postcode or has_india_pin):
                        continue

            # Filter URL false positives
            if entity.entity_type == "URL":
                # Skip if it's an email address (contains @)
                if '@' in entity_text:
                    continue
                # Skip if it's just a domain without protocol (likely email domain)
                if not entity_text.startswith(('http://', 'https://', 'www.')):
                    # Check if it's a bare domain (no path)
                    if re.match(r'^[a-z]+\.[a-z]{2,4}$', entity_text.lower()):
                        continue
                # Skip truncated/incomplete URLs
                if entity_text.endswith('.') or entity_text.endswith('/'):
                    continue
                # Skip URLs with OCR artifacts (double dots, spaced dots)
                if re.search(r'\.\.+|\s+\.\s+', entity_text):
                    continue
                # Skip if URL has no TLD (protocol but no domain)
                if re.match(r'^https?://\w+$', entity_text):
                    continue
                # Skip URLs with invalid characters (likely OCR corruption)
                if re.search(r'[|\\<>{}]', entity_text):
                    continue

            # Filter GENDER false positives
            if entity.entity_type == "GENDER":
                gender_lower = entity_text.lower().strip()
                # Filter "Other" - extremely common word, requires very strong context
                if gender_lower == 'other':
                    ctx_start = max(0, entity.start - 80)
                    ctx_end = min(len(text), entity.end + 30)
                    ctx_text = text[ctx_start:ctx_end].lower()
                    strong_gender_ctx = {'gender', 'sex', 'male', 'female', 'm/f', 'f/m',
                                         'pronoun', 'identity', 'non-binary', 'transgender'}
                    if not any(kw in ctx_text for kw in strong_gender_ctx):
                        continue
                # Filter single-letter M/F without strong context
                if gender_lower in ('m', 'f'):
                    ctx_start = max(0, entity.start - 100)
                    ctx_end = min(len(text), entity.end + 40)
                    ctx_text = text[ctx_start:ctx_end].lower()
                    mf_context = {'gender', 'sex', 'male', 'female', 'm/f', 'f/m'}
                    if not any(kw in ctx_text for kw in mf_context):
                        continue
                # Common terms that need context validation
                common_binary = {'woman', 'man', 'boy', 'girl', 'female', 'male'}
                if gender_lower in common_binary:
                    # Check for gender-related context keywords (wide window for structured data)
                    context_start = max(0, entity.start - 150)
                    context_end = min(len(text), entity.end + 50)
                    context_text = text[context_start:context_end].lower()
                    gender_context = {'gender', 'sex', 'identity', 'pronoun', 'assigned', 'birth',
                                      'm/f', 'f/m', 'masculine', 'feminine', 'non-binary',
                                      'male / female', 'male/female', 'female / male',
                                      'female/male'}
                    has_context = any(kw in context_text for kw in gender_context)
                    # Also check for opposite gender term nearby (e.g., "MALE / FEMALE")
                    if not has_context:
                        opposite = {'female', 'woman', 'girl'} if gender_lower in ('male', 'man', 'boy') else {'male', 'man', 'boy'}
                        # Exclude self-match by checking context outside entity span
                        before = text[context_start:entity.start].lower()
                        after = text[entity.end:context_end].lower()
                        has_context = any(opp in before or opp in after for opp in opposite)
                    # Filter if no context and not high confidence
                    if not has_context:
                        if entity.confidence < 0.85:
                            continue
                # Filter compound words (saleswoman, businessman, etc.)
                compound_patterns = r'(sales|business|police|fire|chair|mail|sports)(wo)?man|woman-owned'
                if re.search(compound_patterns, entity_text.lower()):
                    continue
                # Filter OCR artifacts
                if re.search(r'[|*\[\]]', entity_text):
                    continue

            # Filter FINANCIAL false positives
            if entity.entity_type == "FINANCIAL":
                # Skip if it looks like an email address
                if '@' in entity_text:
                    continue
                # Skip if it looks like a URL
                if re.match(r'^(?:https?://|www\.)', entity_text.lower()):
                    continue
                # Skip if it looks like a person name (two capitalized words)
                if re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+$', entity_text):
                    continue
                # Skip if it's a company suffix appearing alone
                if entity_text.lower() in ('inc', 'inc.', 'llc', 'ltd', 'ltd.', 'corp', 'corp.', 's.a.', 'plc', 'pjsc'):
                    continue
                # Skip common English words that accidentally match SWIFT pattern (CUST+OM+ER = CUSTOMER)
                # SWIFT pattern is [A-Z]{4}[COUNTRY_CODE]{2}[A-Z0-9]{2,5} which matches many English words
                swift_false_positives = {
                    # -ER endings (CUST+OM+ER pattern)
                    'customer', 'customers', 'register', 'registers', 'chapter', 'chapters',
                    'picture', 'pictures', 'feature', 'features', 'measure', 'measures',
                    'treasure', 'treasures', 'pleasure', 'pleasures', 'capture', 'captures',
                    'venture', 'ventures', 'lecture', 'lectures', 'culture', 'cultures',
                    'texture', 'textures', 'mixture', 'mixtures', 'fixture', 'fixtures',
                    'denture', 'dentures', 'gesture', 'gestures', 'pasture', 'pastures',
                    'posture', 'postures', 'rupture', 'ruptures', 'suture', 'sutures',
                    'nature', 'natures', 'future', 'futures', 'torture', 'tortures',
                    'fracture', 'fractures', 'structure', 'structures', 'conjecture',
                    # -RY endings (INDUST+RY pattern)
                    'industry', 'industries', 'ministry', 'chemistry', 'geometry', 'symmetry',
                    'registry', 'forestry', 'ancestry', 'artistry', 'tapestry', 'infantry',
                    'poetry', 'poultry', 'country', 'countries', 'entry', 'entries',
                    'pantry', 'gentry', 'sentry', 'sultry', 'pastry', 'gastry',
                    # -CT endings (ABSTRA+CT, DISTRI+CT pattern)
                    'district', 'districts', 'abstract', 'abstracts', 'construct', 'constructs',
                    'instruct', 'instructs', 'contract', 'contracts', 'extract', 'extracts',
                    'contact', 'contacts', 'compact', 'compacts', 'impact', 'impacts',
                    # -ION endings (DECISI+ON, RECOGNITI+ON pattern)
                    'decision', 'decisions', 'recognition', 'recognitions', 'condition', 'conditions',
                    'position', 'positions', 'definition', 'definitions', 'composition', 'compositions',
                    'disposition', 'proposition', 'opposition', 'acquisition', 'requisition',
                    'transition', 'nutrition', 'coalition', 'partition', 'petition', 'ambition',
                    'admission', 'commission', 'permission', 'submission', 'emission', 'omission',
                    'transmission', 'intermission', 'remission', 'mission', 'missions',
                    'assumption', 'assumptions', 'consumption', 'presumption', 'resumption',
                    'exemption', 'redemption', 'preemption', 'option', 'options', 'adoption',
                    'caption', 'captions', 'fraction', 'fractions', 'action', 'actions',
                    'reaction', 'reactions', 'traction', 'attraction', 'distraction', 'extraction',
                    'satisfaction', 'infraction', 'refraction', 'interaction', 'transaction',
                    'section', 'sections', 'election', 'elections', 'selection', 'selections',
                    'collection', 'collections', 'direction', 'directions', 'protection', 'protections',
                    'connection', 'connections', 'correction', 'corrections', 'inspection', 'inspections',
                    'infection', 'infections', 'perfection', 'reflection', 'deflection', 'injection',
                    'projection', 'rejection', 'objection', 'subjection', 'detection', 'detections',
                    'function', 'functions', 'junction', 'junctions', 'sanction', 'sanctions',
                    'production', 'productions', 'reduction', 'reductions', 'deduction', 'deductions',
                    'introduction', 'reproduction', 'construction', 'destruction', 'instruction',
                    'obstruction', 'conduction', 'induction', 'seduction', 'abduction',
                    'version', 'versions', 'conversion', 'diversion', 'inversion', 'reversion',
                    'aversion', 'perversion', 'subversion', 'immersion', 'submersion', 'dispersion',
                    'excursion', 'incursion', 'recursion', 'tension', 'tensions', 'extension',
                    'dimension', 'dimensions', 'suspension', 'apprehension', 'comprehension',
                    'expansion', 'mansion', 'pension', 'pensions',
                    # -TY endings (PROPER+TY pattern)
                    'property', 'properties', 'poverty', 'novelty', 'penalty', 'penalties',
                    'faculty', 'difficulty', 'specialty', 'specialty', 'casualty', 'casualties',
                    'royalty', 'royalties', 'loyalty', 'cruelty', 'certainty', 'uncertainty',
                    'majority', 'minority', 'priority', 'priorities', 'authority', 'authorities',
                    'security', 'securities', 'publicity', 'electricity', 'simplicity', 'complexity',
                    'capacity', 'opacity', 'audacity', 'tenacity', 'veracity', 'vivacity',
                    'velocity', 'atrocity', 'ferocity', 'reciprocity', 'authenticity', 'domesticity',
                    'elasticity', 'plasticity', 'toxicity', 'publicity', 'complicity', 'duplicity',
                    'multiplicity', 'specificity', 'eccentricity', 'ethnicity', 'historicity',
                    # -AL endings
                    'approval', 'approvals', 'removal', 'removals', 'renewal', 'renewals',
                    'proposal', 'proposals', 'disposal', 'disposals', 'reversal', 'reversals',
                    'rehearsal', 'dispersal', 'dismissal', 'dismissals', 'appraisal', 'appraisals',
                    'survival', 'arrival', 'arrivals', 'revival', 'revivals', 'festival', 'festivals',
                    'interval', 'intervals', 'approval', 'disapproval', 'withdrawal', 'withdrawals',
                    'portrayal', 'betrayal', 'betrayals',
                    # Business/UI terms
                    'overview', 'overviews', 'services', 'business', 'businesses', 'process',
                    'processes', 'progress', 'address', 'addresses', 'success', 'successes',
                    'access', 'accesses', 'excess', 'excesses', 'recess', 'recesses',
                    'congress', 'progress', 'egress', 'ingress', 'regress', 'digress',
                    'compress', 'suppress', 'express', 'impress', 'depress', 'oppress',
                    'assess', 'possess', 'obsess', 'profess', 'confess', 'transgress',
                    # -NESS endings
                    'awareness', 'business', 'businesses', 'darkness', 'fairness', 'fitness',
                    'goodness', 'happiness', 'illness', 'kindness', 'madness', 'openness',
                    'readiness', 'sadness', 'weakness', 'wellness', 'witness', 'witnesses',
                    # -MENT endings
                    'assessment', 'assessments', 'assignment', 'assignments', 'agreement', 'agreements',
                    'statement', 'statements', 'treatment', 'treatments', 'movement', 'movements',
                    'department', 'departments', 'development', 'developments', 'government', 'governments',
                    'management', 'managements', 'investment', 'investments', 'environment', 'environments',
                    'requirement', 'requirements', 'improvement', 'improvements', 'achievement', 'achievements',
                    'establishment', 'establishments', 'entertainment', 'entertainments',
                    'adjustment', 'adjustments', 'arrangement', 'arrangements', 'attachment', 'attachments',
                    'commitment', 'commitments', 'employment', 'employments', 'equipment', 'equipments',
                    'measurement', 'measurements', 'replacement', 'replacements', 'settlement', 'settlements',
                    # Common nouns/verbs that match patterns
                    'property', 'properly', 'prospective', 'perspective', 'respective',
                    'objective', 'objectives', 'subjective', 'effective', 'detective', 'defective',
                    'selective', 'collective', 'corrective', 'protective', 'projective', 'directive',
                    'executive', 'executives', 'sensitive', 'intensive', 'extensive', 'expensive',
                    'defensive', 'offensive', 'comprehensive', 'apprehensive', 'reprehensive',
                    'progressive', 'aggressive', 'regressive', 'excessive', 'successive', 'impressive',
                    'expressive', 'depressive', 'oppressive', 'compressive', 'suppressive',
                    'inclusive', 'exclusive', 'conclusive', 'intrusive', 'obtrusive', 'protrusive',
                    'abusive', 'diffusive', 'effusive', 'infusive', 'profusive', 'confusive',
                    # Accounting/Finance terms (ironically false positives for FINANCIAL)
                    'amortisation', 'amortization', 'depreciation', 'appreciation', 'appropriation',
                    'capitalization', 'capitalisation', 'reconciliation', 'consolidation',
                    'liquidation', 'valuation', 'evaluation', 'devaluation', 'revaluation',
                    # Additional common words
                    'information', 'communication', 'organization', 'administration', 'registration',
                    'documentation', 'implementation', 'specification', 'classification', 'notification',
                    'verification', 'certification', 'identification', 'authorization', 'authentication',
                    'presentation', 'representation', 'demonstration', 'concentration', 'consideration',
                    'determination', 'discrimination', 'examination', 'imagination', 'investigation',
                    'negotiation', 'obligation', 'observation', 'operation', 'preparation', 'publication',
                    'recommendation', 'regulation', 'reservation', 'resignation', 'resolution', 'situation',
                    'specification', 'transportation', 'violation',
                }
                if entity_text.lower() in swift_false_positives:
                    continue
                # Additional check: if word is all letters and >6 chars, likely an English word not a SWIFT code
                # Real SWIFT codes are 8 or 11 chars with specific format and contain bank identifiers
                if entity_text.isalpha() and len(entity_text) > 6:
                    # SWIFT codes should not be common English words
                    continue

            # Filter MEDICAL false positives
            if entity.entity_type == "MEDICAL":
                # Skip OCR artifacts (random case, underscores, incomplete words)
                if self._is_ocr_artifact(entity_text):
                    continue
                # Skip if it looks like a MAC address (6 groups of 2 hex chars)
                if re.match(r'^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$', entity_text):
                    continue
                # Skip if it looks like a UUID
                if re.match(r'^[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}$', entity_text):
                    continue
                # Skip if confidence is low and looks like a person name
                if entity.confidence < 0.6:
                    if re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+$', entity_text):
                        continue
                # Skip if it looks like a postal code (STATE + ZIP pattern like "GA 22984")
                if re.match(r'^[A-Z]{2}\s*\d{5}(?:-\d{4})?$', entity_text):
                    continue
                # Skip if it looks like a reference number or ID code (short alphanumeric)
                if re.match(r'^[A-Z]{1,3}\s*\d{3,6}$', entity_text):
                    continue
                # Skip common technical/business terms incorrectly flagged as medical
                medical_false_positives = {
                    # Document terms
                    'schedule', 'schedules', 'section', 'sections', 'part', 'parts',
                    'item', 'items', 'note', 'notes', 'page', 'pages', 'form', 'forms',
                    'table', 'tables', 'figure', 'figures', 'appendix', 'exhibit',
                    'attachment', 'attachments', 'document', 'documents', 'report', 'reports',
                    # Common body words used in non-medical context
                    'heart', 'hearts', 'head', 'heads', 'hand', 'hands', 'arm', 'arms',
                    'leg', 'legs', 'foot', 'feet', 'eye', 'eyes', 'ear', 'ears',
                    'face', 'faces', 'skin', 'bone', 'bones', 'blood', 'brain', 'brains',
                    'tooth', 'teeth', 'tongue', 'tongues', 'finger', 'fingers',
                    # Common symptoms/conditions (generic usage)
                    'pain', 'pains', 'ache', 'aches', 'headache', 'headaches',
                    'toothache', 'toothaches', 'backache', 'backaches', 'stomachache',
                    'infection', 'infections', 'fever', 'fevers', 'cold', 'colds',
                    'cough', 'coughs', 'flu', 'allergy', 'allergies',
                    # Descriptive words that aren't medical identifiers
                    'healthy', 'unhealthy', 'sick', 'ill', 'well', 'better', 'worse',
                    'grandeur', 'stature', 'posture', 'gesture', 'gestures',
                    # Dental terms (common in general text)
                    'dental', 'dentist', 'cavity', 'cavities', 'filling', 'fillings',
                    'crown', 'crowns', 'brace', 'braces', 'gum', 'gums',
                    # Other PII labels (should not be flagged as MEDICAL)
                    'dob', 'ssn', 'ssn:', 'ssn#', 'social security', 'tin', 'ein',
                    # Tech terms
                    'gb ssd', 'gb ram', 'gb', 'ssd', 'ram', 'mca', 'hoa',
                    # Food/menu items (not medical)
                    'pancakes', 'pancake', 'chocolate cake', 'cake', 'salad',
                    'soup', 'steak', 'chicken', 'fish', 'pasta', 'pizza',
                    # Generic words
                    'truck', 'trucks', 'death', 'deaths', 'disability', 'disabilities',
                    'part-time', 'full-time', 'overtime',
                }
                if entity_text.lower().strip() in medical_false_positives:
                    continue
                # Skip short abbreviations that aren't medical (< 4 chars and uppercase)
                # Examples: "2SN", "S21", "MCA", "HOA"
                if len(entity_text.strip()) <= 4 and entity_text.strip().isupper():
                    continue

            # Filter PERSON false positives
            # Note: PERSON detection now uses context-aware patterns:
            # - Title + Name (Dr. Smith) - high confidence, always detected
            # - Labeled names (Name: John Smith) - require explicit label
            # This filter handles edge cases from spaCy NER
            if entity.entity_type == "PERSON":
                # (Span cleanup already applied at top of loop)

                # Check if entity is from labeled detector (trusted source)
                _is_labeled2 = (entity.recognition_metadata or {}).get('recognizer_name') == 'LabeledPIIDetector'
                # Skip OCR artifacts (random case, underscores, incomplete words)
                if self._is_ocr_artifact(entity_text):
                    continue
                # Skip form labels ending with colon (e.g., " Name: L", "Authorized Signature:")
                if entity_text.strip().endswith(':') and len(entity_text) < 30:
                    continue
                # Skip parenthesized common words (e.g., "Guardian(s)")
                _clean_parens = re.sub(r'\([^)]*\)', '', entity_text).strip()
                if _clean_parens.lower() in {'guardian', 'parent', 'sponsor', 'supervisor',
                                              'applicant', 'participant', 'employer', 'employee'}:
                    continue
                # Skip patterns with pipe characters (table fragments)
                if '|' in entity_text:
                    continue
                # Skip text with spaces in the middle of words (OCR artifacts like "mu Nt")
                if re.search(r'\b[a-z]\s[a-z]\b', entity_text.lower()):
                    continue
                # Skip very short text (allow 3-letter names like Roy, Tom, Amy)
                if len(entity_text) <= 2:
                    continue
                # Skip if contains numbers (names don't have digits)
                if any(c.isdigit() for c in entity_text):
                    continue
                # Skip if less than 50% alphabetic characters (extraction artifacts)
                alpha_count = sum(1 for c in entity_text if c.isalpha())
                if alpha_count < len(entity_text.strip()) * 0.5:
                    continue
                # Skip very long text (likely phrases, not names)
                if len(entity_text) > 35:
                    continue
                # For spaCy NER detections (lower confidence), require moderate threshold
                # Our pattern recognizers have scores 0.85-0.95
                # Lowered to 0.50 to match ENTITY_THRESHOLDS (was 0.70, caused recall drop)
                if entity.confidence < 0.50:
                    continue
                # Require higher confidence for very short entities (<=4 chars)
                # Short words like "Jan", "May" often trigger as person names
                if len(entity_text.strip()) <= 4 and entity.confidence < 0.80:
                    continue
                # Require higher confidence for single-word entities
                # Single words from spaCy NER with moderate confidence are often FPs
                # Labeled entities (XML tags, bold names, etc.) bypass these checks
                words_in_name = entity_text.strip().split()
                if len(words_in_name) == 1 and entity.confidence < 0.55 and not _is_labeled2:
                    continue
                # For single-word PERSON with confidence < 0.58, require name context
                # Lowered from 0.80→0.72→0.68→0.64→0.62→0.58
                # Common-word filter (common_non_names) provides FP safety net
                if len(words_in_name) == 1 and entity.confidence < 0.58 and not _is_labeled2:
                    ctx_start = max(0, entity.start - 60)
                    ctx_end = min(len(text), entity.end + 30)
                    ctx_text = text[ctx_start:ctx_end].lower()
                    name_indicators = {'name', 'mr', 'mrs', 'ms', 'dr', 'prof',
                                       'dear', 'sincerely', 'regards', 'attn',
                                       'author', 'by ', 'from ', 'signed',
                                       'participant', 'applicant', 'student',
                                       'contact', 'username', 'email',
                                       'member', 'staff', 'personnel', 'account',
                                       'leader', 'manager', 'officer', 'director',
                                       'patient', 'client', 'customer', 'user',
                                       'owner', 'holder', 'sender', 'recipient',
                                       'employee', 'colleague', 'representative',
                                       'agent', 'witness', 'resident', 'tenant',
                                       'candidate', 'volunteer', 'intern',
                                       'hi ', 'hello', 'hey ', 'to ', 'cc ',
                                       'advisor', 'consultant', 'specialist',
                                       'interviewer', 'respondent', 'subscriber'}
                    if not any(kw in ctx_text for kw in name_indicators):
                        continue
                # Skip entities containing HTML tags or markup artifacts
                # Examples: "<p>", "s:</b>", "et: A", HTML fragments from text extraction
                if '<' in entity_text or '>' in entity_text:
                    continue
                if '&lt;' in entity_text or '&gt;' in entity_text or '&amp;' in entity_text:
                    continue
                # Skip entities containing markdown/formatting artifacts
                # Examples: "Report**", "Literacy**", "Users:**", "Areas","
                stripped_name = entity_text.strip()
                if '**' in stripped_name or stripped_name.endswith(':**'):
                    continue
                # Strip trailing/leading punctuation to check the clean name
                clean_name = re.sub(r'[*:,"\'\s!?\[\]()]+$', '', stripped_name)
                clean_name = re.sub(r'^[*:,"\'\s!?\[\]()]+', '', clean_name)
                if len(clean_name) < 2:
                    continue
                # Skip if clean name still contains colon (field labels like "Contact Information:")
                # Use clean_name (not entity_text) to tolerate span-bleed trailing colons
                # e.g., "Scott Powell, SSN:" → clean_name "Scott Powell, SSN" (no colon → ok)
                if ':' in clean_name:
                    continue
                # Skip if contains exclamation mark (names never have !)
                if '!' in entity_text:
                    continue
                # Skip common English words that aren't names (targeted set)
                # Only words that are NEVER person names
                common_non_names = {
                    'comment', 'areas', 'literacy', 'additionally', 'workshop',
                    'incorporating', 'academic', 'truancy', 'strategy', 'rooms',
                    'program', 'users', 'feedback', 'ensure', 'compliance',
                    'control', 'safety', 'security', 'access', 'records',
                    'information', 'assessment', 'results', 'system', 'tracking',
                    'completion', 'readiness', 'education', 'experts', 'events',
                    'schedule', 'training', 'matters', 'consent', 'review',
                    'confirmation', 'background', 'policy', 'platform',
                    'inspections', 'measures', 'coordinator', 'forums',
                    'guidelines', 'requirements', 'procedures', 'regulations',
                    'overview', 'introduction', 'conclusion', 'appendix',
                    'notification', 'alert', 'warning', 'notice',
                    # Gender words (not person names)
                    'masculine', 'feminine', 'nonbinary', 'transgender',
                    'female', 'male',
                    # Form/document labels
                    'admission', 'sample', 'form', 'draft', 'template',
                    'attachment', 'document', 'report', 'memo', 'receipt',
                    # Common words from ai4privacy FP analysis
                    'contact', 'party', 'prefer', 'other', 'both',
                    'absolutely', 'team', 'steps', 'next', 'maintain',
                    'participate', 'changes', 'please', 'counseling',
                    'sessions', 'practice', 'mindfulness', 'video', 'short',
                    'dear', 'wellbeing', 'mentorship', 'password', 'palace',
                    'residence', 'read', 'clubs', 'student', 'records',
                    'update', 'signed', 'immerse', 'additionally',
                    'incorporating', 'music', 'arts', 'performing',
                    # Action/descriptive words
                    'important', 'special', 'general', 'primary', 'secondary',
                    'advanced', 'basic', 'standard', 'custom', 'initial',
                    # Single words from ai4privacy FP analysis (never names)
                    'office', 'visionary', 'reality', 'link', 'curriculum',
                    'technology', 'thread', 'messaging', 'masterclass',
                    'emergency', 'postcode', 'forums',
                    # Form field labels (ai4privacy FP analysis 2026-02-08)
                    'email', 'username', 'sex', 'subject', 'description',
                    'birthdate', 'telephone', 'birth', 'details', 'time',
                    # Role/descriptor words (never person names)
                    'participant', 'applicant', 'candidate', 'tutor',
                    'recipient', 'individual', 'user', 'participants',
                    # Sentence starters / common words (never person names)
                    'the', 'your', 'thank', 'let', 'welcome', 'this',
                    'looking', 'warm', 'with', 'should', 'our', 'here',
                    # Location terms
                    'country', 'state', 'street', 'city', 'location',
                    # Equipment/item terms
                    'equipment', 'material', 'type', 'item', 'component',
                    # Document/credential terms
                    'driver', 'license', 'passport', 'card', 'certificate',
                    'citizenship', 'record', 'application', 'confirmed',
                    # Non-name months (keep May/June/March/April/August - valid names)
                    'september', 'october', 'november', 'december',
                    'february', 'january',
                    # Additional common nouns (2026-02-08 FP analysis)
                    'detail', 'achievement', 'methodology', 'quantity',
                    'operational', 'deposit', 'headquarters', 'exhibit',
                    'issuer', 'delivery', 'provider', 'analysis', 'buyer',
                    'total', 'acquisition', 'employee', 'today',
                    # Common words and locations (2026-02-10 FP analysis)
                    'context', 'instructions', 'guardian', 'sender',
                    'britain', 'scotland', 'ireland', 'wales', 'england',
                    'summary', 'exchange', 'cultural', 'executive', 'original',
                    # Capitalized English words detected as PERSON (2026-02-11 FP analysis)
                    'rating', 'reviewed', 'attend', 'highlight', 'birthday',
                    'planning', 'event', 'excellence', 'beginner', 'wrapping',
                    'banana', 'prevent', 'bullying', 'assembly', 'booking',
                }
                if clean_name.lower() in common_non_names:
                    continue
                # Skip words with English-word suffixes that NEVER appear in real names
                # Carefully curated: excludes -ance (Constance), -ence (Lawrence),
                # -ment (Clement), -ity (Trinity), -ive (Clive), -ward (Edward),
                # -ling (Sterling), -ing (Manning), -ly (Kelly)
                _clean_lower = clean_name.lower()
                _safe_non_name_suffixes = (
                    'tion', 'sion',     # information, decision (never a name)
                    'ness',             # readiness, wellness
                    'ous', 'ious', 'eous',  # various, serious, gorgeous
                    'ical',             # medical, political
                    'ally',             # especially, additionally
                    'ible',             # possible, terrible
                    'ful',              # beautiful, helpful
                    'less',             # careless, useless
                    'ism',              # capitalism, marxism
                )
                # For single words, check suffix directly (min 6 chars to avoid short names)
                if len(words_in_name) == 1 and len(_clean_lower) > 5:
                    if any(_clean_lower.endswith(sfx) for sfx in _safe_non_name_suffixes):
                        continue
                # For multi-word, if ANY word has a non-name suffix, suppress
                if len(words_in_name) >= 2:
                    _has_non_name_word = False
                    for w in words_in_name:
                        wl = w.lower().rstrip('*:,"!.')
                        if len(wl) > 5 and any(wl.endswith(sfx) for sfx in _safe_non_name_suffixes):
                            _has_non_name_word = True
                            break
                    if _has_non_name_word:
                        continue
                # Skip multi-word form field labels ending with generic label words
                # Examples: "Donor Name", "Part Number", "Delivery Date", "School Form"
                if len(words_in_name) >= 2:
                    last_word = words_in_name[-1].lower().rstrip('*:,"')
                    field_labels = {'name', 'number', 'date', 'form', 'report',
                                    'address', 'phone', 'email', 'details'}
                    if last_word in field_labels:
                        continue
                # Skip multi-word phrases containing institutional/organizational keywords
                # These are titles, headers, or org names - not person names
                if len(words_in_name) >= 2:
                    lower_words = {w.lower().rstrip('*:,"') for w in words_in_name}
                    # Skip phrases with clearly institutional/organizational words
                    org_keywords = {
                        'board', 'institute', 'academy', 'system', 'coordinator',
                        'department', 'committee', 'program', 'service', 'center',
                        'centre', 'foundation', 'authority', 'commission', 'bureau',
                        'management', 'information', 'certification',
                        'completion', 'password', 'address', 'secondary', 'assessment',
                        'competition', 'tracking', 'education', 'community',
                        'platform', 'learning', 'mentorship', 'clubs', 'practice',
                        'sessions', 'counseling', 'performing', 'arts', 'music',
                        'compliance', 'records', 'confirmation', 'steps', 'video',
                        'skills', 'training', 'student', 'results', 'review',
                        # Document structure / course titles (ai4privacy FP analysis)
                        'messaging', 'admissions', 'course', 'learner', 'needs',
                        'engaging', 'content', 'practices', 'understanding',
                        'creating', 'sustainable', 'goals', 'emergency',
                        'response', 'plan', 'thread', 'masterclass', 'feedback',
                        'request', 'forums', 'strategy', 'curriculum',
                        'technology', 'reality', 'development',
                        # Additional document/form terms (2026-02-08 FP analysis)
                        'labor', 'health', 'general', 'public', 'agreement',
                        'responsibilities', 'purchase', 'power', 'notary',
                        'inspection', 'diagnosis', 'discharge', 'donor',
                        'supplier', 'technician', 'task', 'vehicle',
                    }
                    if lower_words & org_keywords:
                        continue
                    # Skip multi-word phrases where ALL words are common English
                    # (no word looks like it could be a proper name)
                    all_common = all(
                        w.lower().rstrip('*:,"') in common_non_names or
                        w.lower().rstrip('*:,"') in org_keywords
                        for w in words_in_name
                    )
                    if all_common:
                        continue
                # Skip "patient" prefix phrases (medical context, not person names)
                # Examples: "patient records", "patient was an elderly woman"
                if entity_text.lower().startswith('patient'):
                    continue
                # Skip "signed to" phrases (document context, not person names)
                # Examples: "signed to maintain", "signed to ensure"
                if entity_text.lower().startswith('signed to'):
                    continue
                # Skip nationalities/demonyms (not person names)
                nationalities = {
                    'american', 'british', 'canadian', 'australian', 'german', 'french',
                    'italian', 'spanish', 'chinese', 'japanese', 'korean', 'indian',
                    'russian', 'brazilian', 'mexican', 'egyptian', 'yemeni', 'israeli',
                    'turkish', 'greek', 'polish', 'dutch', 'swedish', 'norwegian',
                    'danish', 'finnish', 'irish', 'scottish', 'welsh', 'african',
                    'asian', 'european', 'latin', 'hispanic', 'caribbean', 'arab',
                    'persian', 'vietnamese', 'thai', 'filipino', 'indonesian',
                    'malaysian', 'singaporean', 'pakistani', 'bangladeshi', 'sri lankan',
                }
                if clean_name.lower() in nationalities:
                    continue
                # Skip country/region names that are never person names
                country_names = {
                    'nederland', 'deutschland', 'schweiz', 'brasil', 'españa',
                    'britain', 'great britain', 'united kingdom', 'united states',
                    'australia', 'new zealand', 'south africa', 'saudi arabia',
                    'netherlands', 'belgium', 'switzerland', 'germany', 'france',
                    'spain', 'portugal', 'italy', 'italia', 'austria', 'romania', 'bulgaria',
                    'croatia', 'slovenia', 'serbia', 'bosnia', 'montenegro',
                    'norway', 'sweden', 'denmark', 'finland', 'iceland',
                    'russia', 'ukraine', 'poland', 'czechia', 'slovakia',
                    'hungary', 'greece', 'turkey', 'egypt', 'morocco',
                    'nigeria', 'kenya', 'ghana', 'ethiopia', 'tanzania',
                    'brazil', 'argentina', 'colombia', 'peru', 'chile',
                    'mexico', 'venezuela', 'ecuador', 'uruguay', 'paraguay',
                    'japan', 'korea', 'china', 'taiwan', 'vietnam',
                    'thailand', 'indonesia', 'malaysia', 'singapore', 'philippines',
                    'pakistan', 'bangladesh', 'iran', 'iraq', 'israel',
                    'canada', 'ireland', 'scotland', 'wales', 'england',
                }
                if clean_name.lower() in country_names:
                    continue
                # Skip entities preceded by field labels (structured data context)
                # "City: Newton Abbot", "Country: Italia", "Sex: Female", "Street: Via dell'Industria"
                field_ctx_start = max(0, entity.start - 40)
                field_ctx = text[field_ctx_start:entity.start].lower()
                field_labels = ['city:', 'country:', 'street:', 'state:', 'sex:',
                                'gender:', 'event:', 'postcode:', 'zip:', 'address:',
                                'building:', 'province:', 'region:', 'district:',
                                'second address:', 'secondary address:', 'location:',
                                'nationality:', 'citizenship:', 'born in:', 'city/town:',
                                'subject:', 'session:', 'topic:', 'scope:', 'details:',
                                'course:', 'module:', 'title:', 'description:']
                if any(field_ctx.rstrip().endswith(lbl) or
                       field_ctx.rstrip().endswith(lbl.rstrip(':'))
                       for lbl in field_labels):
                    continue
                # Skip entities containing conversational contractions
                # "Amy, I'm", "It's", "I've been" - these are sentence fragments
                if re.search(r"\b(?:I'm|I've|I'll|I'd|it's|he's|she's|we're|they're|you're|isn't|aren't|wasn't|weren't|don't|doesn't|didn't|can't|won't|wouldn't|shouldn't|couldn't)\b", entity_text, re.IGNORECASE):
                    continue
                # Skip food/menu items (commonly flagged as names)
                food_items = {
                    'caesar', 'caesar salad', 'juicy', 'crispy', 'grilled',
                    'roasted', 'baked', 'fried', 'steamed', 'fresh',
                }
                if clean_name.lower() in food_items:
                    continue
                # Skip generic role words (not person names)
                generic_roles = {
                    'member', 'members', 'recipient', 'recipients', 'authorized personnel',
                    'employee', 'employees', 'customer', 'customers', 'client', 'clients',
                    'user', 'users', 'patient', 'patients', 'subscriber', 'subscribers',
                    'applicant', 'applicants', 'candidate', 'candidates', 'participant',
                    'billing card', 'employee name', 'employee id',
                }
                if clean_name.lower() in generic_roles:
                    continue
                # Skip phrases starting with "member" or "recipient"
                if entity_text.lower().startswith(('member ', 'recipient ', 'reviewed ')):
                    continue
                # Skip UI/navigation phrases commonly detected as person names
                # These are app interface elements, not actual names
                person_ui_phrases = {
                    # QuickBooks/accounting app UI elements
                    'get paid', 'customer agent', 'expenses & bills', 'sales & get paid',
                    'turn leads', 'my apps', 'unbilled income', 'overdue invoice',
                    'recently paid', 'responses to help', 'prioritizes follow',
                    # App names and product UI terms
                    'creo', 'pro tools', 'quick books', 'sage', 'xero',
                    # Generic UI/document phrases
                    'customer agent spots', 'turn leads into sales', 'spots or hides',
                    'view all', 'see all', 'show more', 'load more', 'read more',
                    'learn more', 'get started', 'sign up', 'log in', 'sign in',
                    # Header/title phrases
                    'dear customer', 'dear user', 'dear member', 'dear client',
                    'hello there', 'hi there', 'welcome back', 'good morning',
                    # Status/label phrases
                    'not available', 'coming soon', 'in progress', 'on hold',
                    'pending review', 'awaiting approval', 'under review',
                }
                if entity_text.lower() in person_ui_phrases:
                    continue
                # Skip if text contains common UI indicator words (limited to clear UI patterns)
                # Note: Removed 'into', 'paid' as they appeared in legitimate names
                ui_indicator_words = ['&', 'spots', 'hides', 'leads', 'income', 'invoice', 'apps']
                text_lower = entity_text.lower()
                # Only filter if the word is a complete word match (not part of a name)
                if any(f' {word} ' in f' {text_lower} ' for word in ui_indicator_words):
                    continue
                # Skip document header phrases (commonly flagged as person names)
                document_header_phrases = {
                    'patient demographics', 'beneficiary number', 'patient responsibility',
                    'customer service', 'customer support', 'customer care',
                    'account holder', 'policy holder', 'card holder', 'cardholder',
                    'authorized user', 'primary contact', 'emergency contact',
                    'billing address', 'shipping address', 'mailing address',
                    'date of birth', 'place of birth', 'social security',
                    'identification number', 'reference number', 'confirmation number',
                    # Medical form sections
                    'reason for visit', 'treatment plan', 'discharge summary',
                    'patient history', 'medical history', 'family history',
                    'review of systems', 'physical examination', 'assessment and plan',
                    'vital signs', 'lab results', 'imaging results',
                    'chief complaint', 'present illness', 'past medical history',
                    'surgical history', 'social history', 'allergies',
                    'current medications', 'immunizations', 'follow up',
                    # Legal document phrases
                    'pursuant to', 'in accordance with', 'subject to', 'with respect to',
                    'notwithstanding', 'hereinafter', 'whereas', 'wherefore',
                }
                if text_lower in document_header_phrases:
                    continue
                # Skip "customer can be reached at" type phrases
                if 'can be reached' in text_lower or 'may be contacted' in text_lower:
                    continue
                # Skip contact instruction patterns
                if re.search(r'(?:available|reachable|contacted?)\s+(?:at|via|by|through)', text_lower):
                    continue
                # Skip "signed to prevent/ensure/maintain" type phrases
                if re.match(r'^signed\s+to\s+\w+', text_lower):
                    continue
                # Skip job titles and occupations (not person names)
                occupation_phrases = {
                    'real estate broker', 'real estate agent', 'sales agent',
                    'janitor', 'building cleaner', 'office manager', 'project manager',
                    'software engineer', 'data analyst', 'financial advisor',
                    'legal counsel', 'general manager', 'executive director',
                    # Additional occupations
                    'bank manager', 'branch manager', 'store manager', 'product manager',
                    'senior developer', 'junior developer', 'lead engineer', 'tech lead',
                    'sales manager', 'sales representative', 'sales executive',
                    'operations manager', 'hr manager', 'marketing manager', 'finance manager',
                    'accountant', 'auditor', 'controller', 'paralegal', 'attorney',
                    'nurse practitioner', 'physician assistant', 'medical assistant',
                    'graphic designer', 'web designer', 'ui designer', 'ux designer',
                    'product owner', 'scrum master', 'business analyst', 'systems analyst',
                }
                if text_lower in occupation_phrases:
                    continue
                # Skip sentence-like phrases containing auxiliary verb + noun/verb patterns
                # Examples: "customer was notified", "patient is scheduled", "applicant has submitted"
                # These are sentences describing actions, not person names
                # Pattern: word + (is|are|was|were|has|have|had|will|would|can|could) + word
                if re.search(r'\b(?:is|are|was|were|has|have|had|will|would|can|could|should|must|may|might)\s+(?:not\s+)?\w+', text_lower):
                    # Only filter if there are 3+ words (to avoid filtering "Smith was" type partial names)
                    if len(entity_text.split()) >= 3:
                        continue
                # Skip text starting with common article + noun patterns (not names)
                # Examples: "the customer", "a patient", "the applicant", "an employee"
                if re.match(r'^(?:the|a|an)\s+(?:customer|patient|applicant|employee|user|client|member|recipient|subscriber|participant|beneficiary|holder|owner|borrower|tenant|landlord|vendor|supplier|buyer|seller)s?\b', text_lower):
                    continue
                # Skip phrases ending with preposition (incomplete sentence fragments)
                # Examples: "responded to", "applied for", "submitted by"
                if len(entity_text.split()) >= 2 and re.search(r'\s(?:to|for|by|with|from|in|at|on|of|about|after|before|during|through|between|among|under|over|into|onto|upon)$', text_lower):
                    continue
                # Skip patterns ending with company suffixes (these are companies, not person names)
                # Examples: "Swift Logistics", "Smith Inc", "Johnson LLC"
                company_suffixes = (
                    ' inc', ' llc', ' ltd', ' corp', ' co', ' group', ' solutions',
                    ' logistics', ' services', ' consulting', ' enterprises', ' systems',
                    ' technologies', ' partners', ' associates', ' holdings', ' international',
                    ' foundation', ' institute', ' agency', ' network', ' labs', ' studio',
                )
                if any(text_lower.endswith(suffix) for suffix in company_suffixes):
                    continue
                # Skip brand/product names that are commonly detected as PERSON
                brand_names = {
                    'instagram', 'facebook', 'twitter', 'tiktok', 'snapchat', 'whatsapp',
                    'linkedin', 'pinterest', 'youtube', 'netflix', 'spotify', 'slack',
                    'zoom', 'figma', 'canva', 'adobe', 'microsoft', 'google', 'apple',
                    'amazon', 'walmart', 'target', 'costco', 'starbucks', 'mcdonalds',
                }
                if text_lower.strip() in brand_names:
                    continue
                # Skip ALL-CAPS text with 2+ words (likely company names or acronyms)
                # Examples: "ACME CORP", "ABC HOLDINGS", "JOHN DOE LLC"
                if entity_text.isupper() and len(entity_text.split()) >= 2:
                    continue
                # Skip single-word ALL-CAPS (likely acronyms, not names)
                if entity_text.isupper() and len(entity_text) <= 6:
                    continue

            # Filter ID false positives (generic alphanumeric patterns)
            if entity.entity_type == "ID":
                stripped = entity_text.strip()
                # Require minimum 6 characters
                if len(stripped) < 6:
                    continue
                # Require both letters and digits (alphanumeric mix)
                has_letters = any(c.isalpha() for c in stripped)
                has_digits = any(c.isdigit() for c in stripped)
                if not (has_letters and has_digits):
                    if entity.confidence < 0.90:
                        continue
                # Short letter+digit patterns (e.g., AB1234) need context keywords
                if re.match(r'^[A-Z]{2}\d{4,5}$', stripped):
                    context_start = max(0, entity.start - 60)
                    context = text[context_start:entity.start].lower()
                    id_keywords = {'id', 'card', 'reference', 'number', 'identifier',
                                   'registration', 'account', 'customer', 'case', 'ticket'}
                    if not any(kw in context for kw in id_keywords):
                        continue

            # Filter COORDINATES false positives
            if entity.entity_type == "COORDINATES":
                # Skip if it looks like an IP address fragment
                if re.match(r'^\d{1,3}\.\d{1,3}$', entity_text):
                    continue

            # Filter NATIONAL_ID false positives (SSN, passport, etc.)
            if entity.entity_type == "NATIONAL_ID":
                # Skip incomplete patterns (less than 9 digits total)
                digits = re.sub(r'\D', '', entity_text)
                if len(digits) < 9:
                    continue
                # Skip table/ASCII artifacts (multiple consecutive dashes or pipes)
                if re.search(r'[|\-]{3,}', entity_text):
                    continue
                # Skip patterns with adjacent table markers in surrounding context
                context_start = max(0, entity.start - 5)
                context_end = min(len(text), entity.end + 5)
                context = text[context_start:context_end]
                if '|' in context:
                    continue
                # Skip patterns that look like phone numbers (area code in parens)
                if re.match(r'\(\d{3}\)', entity_text):
                    continue
                # Skip patterns that are clearly dates (month-day-year patterns)
                if re.match(r'^\d{2}[-/]\d{2}[-/]\d{4}$', entity_text):
                    continue
                # Skip IP address fragments (3 dot-separated octets like "213.237.252")
                if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}$', entity_text):
                    continue
                # Skip if context indicates IP address
                natl_ctx_start = max(0, entity.start - 40)
                natl_ctx = text[natl_ctx_start:entity.start].lower()
                if 'ip address' in natl_ctx or 'ip:' in natl_ctx or 'ipv' in natl_ctx:
                    continue
                # Skip if context indicates phone number (preceded by + country code)
                natl_ctx_before = text[max(0, entity.start - 15):entity.start]
                if re.search(r'\+\d{1,3}[\s-]*$', natl_ctx_before):
                    continue
                # Skip if "phone" or "telephone" in nearby context
                if any(kw in natl_ctx for kw in ['phone', 'tel:', 'telephone', 'mobile', 'fax']):
                    continue

            # Filter COMPANY false positives
            if entity.entity_type == "COMPANY":
                # Skip OCR artifacts (random case, underscores, incomplete words)
                if self._is_ocr_artifact(entity_text):
                    continue
                # Skip IP address label patterns: "IPV4_29.217.74.44", "IP: 192.168.1.1"
                if re.match(r'^(?:IPV?[46]?[_:\s]|IP[_:\s])', entity_text, re.IGNORECASE):
                    continue
                # Skip sentences (contain period followed by space and capital or end of text)
                if re.search(r'\.\s+[A-Z]', entity_text) or entity_text.count('.') > 1:
                    continue
                # Skip patterns with pipe characters (table fragments)
                if '|' in entity_text:
                    continue
                # Skip number-ID patterns: "897-1647 ID", "200 annually"
                # Pattern: numbers followed by ID/annually/monthly/weekly/daily
                if re.match(r'^[\d\-]+\s+(?:ID|annually|monthly|weekly|daily|yearly|quarterly|semi-annually)\b', entity_text, re.IGNORECASE):
                    continue
                # Skip education unit patterns: "60 credit hours", "120 semester hours"
                if re.match(r'^\d+\s+(?:credit|semester|quarter)\s+hours?\b', entity_text, re.IGNORECASE):
                    continue
                # Skip OCR-corrupted prescription text: "10-16 Prsecriptiun foR"
                # Pattern: text with unusual capitalization in middle of words (OCR artifact)
                if re.search(r'[a-z][A-Z][a-z]', entity_text):  # lowercase-UPPER-lowercase
                    continue
                # Skip text fragments with numbers followed by "or" (contact info fragments)
                # Examples: "51412 or by sending an email", "2095 or you can reach me"
                if re.match(r'^\d+\s+or\s+', entity_text.lower()):
                    continue
                # Skip "X years of experience" patterns
                if re.search(r'\d+\s+years?\s+of\s+experience', entity_text.lower()):
                    continue
                # Skip text fragments that are partial sentences with numbers
                # Examples: "4502239 and the profile", "51412 or by sending"
                if re.match(r'^\d+\s+(?:and|or|with|for|to|by|in|at|on)\s+', entity_text.lower()):
                    continue
                # Skip phrases that describe activities, not companies
                if re.search(r'(?:surge|increase|decrease|rise|fall|growth|decline)\s+in\s+', entity_text.lower()):
                    continue
                # Skip common hyphenated adjectives/phrases that aren't company names
                company_hyphen_false_positives = {
                    'cross-verified', 'cross-checked', 'cross-referenced', 'cross-border',
                    'high-value', 'high-risk', 'high-quality', 'high-level', 'high-profile',
                    'low-cost', 'low-risk', 'low-value', 'low-level',
                    'carry-forward', 'carry-over', 'carry-back',
                    'self-employed', 'self-assessment', 'self-certification', 'self-declaration',
                    'well-known', 'well-established', 'well-documented', 'well-defined',
                    'non-compliance', 'non-resident', 'non-taxable', 'non-deductible',
                    'tax-related', 'tax-exempt', 'tax-free', 'tax-deductible',
                    'long-term', 'short-term', 'mid-term', 'near-term',
                    'real-time', 'full-time', 'part-time', 'one-time',
                    'year-end', 'year-over-year', 'month-end', 'quarter-end',
                    'pre-tax', 'post-tax', 'after-tax', 'before-tax',
                    'inter-company', 'intra-company', 'inter-bank', 'intra-group',
                }
                text_lower = entity_text.lower()
                if text_lower in company_hyphen_false_positives:
                    continue
                # Skip accounting/financial phrases that look like company names but aren't
                accounting_phrases = {
                    'health & safety', 'profit & loss', 'assets & liabilities',
                    'receivables & payables', 'revenue & expenses',
                    'debits & credits', 'income & expenses', 'cash & equivalents',
                    'sales & marketing', 'research & development', 'mergers & acquisitions',
                }
                if text_lower in accounting_phrases:
                    continue

                # Skip UI/menu navigation phrases falsely detected as companies
                ui_menu_false_positives = {
                    'expenses & bills', 'sales & get paid', 'reports & insights',
                    'settings & preferences', 'help & support', 'billing & payments',
                    'accounts & settings', 'profile & settings', 'search & filter',
                    'import & export', 'copy & paste', 'cut & paste',
                    'terms & conditions', 'privacy & security', 'rules & policies',
                }
                if text_lower in ui_menu_false_positives:
                    continue

                # Skip phrases that start with common words (e.g., "The company", "A company")
                if re.match(r'^(?:the|a|an|this|that|our|their|your|its)\s+', text_lower):
                    continue
                # Skip if it's a gerund phrase with hyphen (e.g., "self-assessing")
                # But NOT if it's a hyphenated proper name (e.g., "Carr-Archer", "Bryant-Meyer")
                if re.match(r'^[a-z]+-[a-z]+(?:ing|ed|er|tion)\b', text_lower):
                    if not re.match(r'^[A-Z][a-z]+-[A-Z][a-z]+$', entity_text):
                        continue
                # For hyphenated names, require higher confidence when not followed by suffix
                # BUT allow through if company context words are nearby (headquarters, at X via, etc.)
                if '-' in entity_text and entity.confidence < 0.7:
                    # Check if NOT followed by a company suffix
                    if not re.search(r'\b(?:Inc|LLC|Ltd|Corp|Company|Group|Partners)\b', entity_text, re.IGNORECASE):
                        # Check for company context: "X headquarters", "at X via", "at X,"
                        has_company_context = False
                        try:
                            entity_end_pos = entity.end
                            entity_start_pos = entity.start
                            # Check text after the entity for "headquarters"
                            after_text = text[entity_end_pos:min(len(text), entity_end_pos + 30)].lower()
                            if after_text.lstrip().startswith('headquarters'):
                                has_company_context = True
                            # Check text before the entity for "at ", "for ", "from ", "with "
                            before_text = text[max(0, entity_start_pos - 10):entity_start_pos].lower()
                            if re.search(r'\b(?:at|for|from|with)\s+$', before_text):
                                has_company_context = True
                        except (AttributeError, IndexError):
                            pass
                        if not has_company_context:
                            continue
                # Skip very short company names (less than 4 chars)
                if len(entity_text.strip()) < 4:
                    continue
                # Skip very long "company names" (likely sentences/phrases, not company names)
                if len(entity_text.strip()) > 50:
                    continue
                # Skip if it looks like a descriptive phrase (contains clear sentence indicators)
                # Note: Reduced list to avoid filtering valid company names
                # Removed 'has been', 'have been', 'was ', 'were ', 'will be' as too broad
                company_phrase_indicators = [
                    'verified against', 'claims were', 'purchased by',
                    'in accordance with', 'in compliance with', 'in relation to',
                    # Clear accounting phrases (not company names)
                    'property, plant and equipment', 'assets and liabilities',
                ]
                if any(indicator in text_lower for indicator in company_phrase_indicators):
                    continue
                # Skip department/division names that aren't company names
                department_names = {
                    'research & development', 'research and development', 'r&d',
                    'human resources', 'hr', 'finance', 'accounting', 'marketing',
                    'sales', 'operations', 'it', 'information technology', 'legal',
                    'compliance', 'audit', 'internal audit', 'external audit',
                    'customer service', 'customer support', 'technical support',
                    'quality assurance', 'qa', 'quality control', 'qc',
                    'supply chain', 'logistics', 'procurement', 'purchasing',
                    'engineering', 'product development', 'business development',
                }
                # Skip generic service/system names (not company names)
                generic_service_names = {
                    # Service patterns
                    'support services', 'title services', 'affected services',
                    'affected systems', 'ventilation systems', 'management services',
                    'network management', 'identity management', 'compare products',
                    'all products', 'groundbreaking research',
                    # Policies and practices
                    'good manufacturing practices', 'good manufacturing practices in pharmaceuticals',
                    'regular maintenance of heating and cooling systems',
                    'ensure all electrical systems',
                }
                if text_lower in generic_service_names:
                    continue
                # Skip generic business suffix words when appearing alone
                # These are common in navigation/headers, not company names
                generic_business_words = {
                    'international', 'global', 'solutions', 'services', 'group',
                    'holdings', 'management', 'consulting', 'worldwide', 'systems',
                    'technologies', 'partners', 'associates', 'enterprises', 'network',
                    'agency', 'foundation', 'institute', 'corporation', 'company',
                }
                if text_lower.strip() in generic_business_words:
                    continue
                # Skip patterns with "X business days", "X months", "X years", "X hours"
                # Examples: "7 business days", "14 business days", "36 months", "4 hours"
                if re.match(r'^\d+\s+(?:business\s+)?(?:days?|months?|years?|hours?|minutes?|weeks?)\b', text_lower):
                    continue
                # Skip patterns with "X stars", "X each", "X per", "X and"
                if re.match(r'^\d+\s+(?:stars?|each|per|and|or|mm|cm|kg|lb)\b', text_lower):
                    continue
                if text_lower in department_names:
                    continue
                # Skip UI navigation/menu patterns that aren't company names
                ui_navigation_patterns = {
                    'expenses & bills', 'expenses and bills', 'expenses',
                    'sales & get paid', 'sales and get paid', 'get paid',
                    'banking & cash', 'banking and cash', 'banking',
                    'reports & insights', 'reports and insights', 'insights',
                    'settings & preferences', 'settings and preferences', 'preferences',
                    'accounts & sync', 'accounts and sync', 'sync',
                    'help & support', 'help and support', 'support',
                    'tools & templates', 'tools and templates', 'templates',
                    'import & export', 'import and export', 'export',
                    'billing & payments', 'billing and payments', 'payments',
                    'invoices & estimates', 'invoices and estimates', 'estimates',
                    'time & projects', 'time and projects', 'projects',
                    'payroll & employees', 'payroll and employees', 'payroll',
                    'taxes & forms', 'taxes and forms', 'forms',
                }
                if text_lower in ui_navigation_patterns:
                    continue

                # LightGBM-based company verification (if available)
                # This uses the trained ORGANIZATION classifier to filter false positives
                if COMPANY_VERIFIER_AVAILABLE:
                    is_valid, adjusted_confidence = verify_company_detection(
                        text, entity_text, entity.start, entity.end, entity.confidence
                    )
                    if not is_valid:
                        continue
                    # Update confidence if adjusted
                    if adjusted_confidence != entity.confidence:
                        entity = PIIEntity(
                            entity_type=entity.entity_type,
                            text=entity.text,
                            start=entity.start,
                            end=entity.end,
                            confidence=adjusted_confidence,
                            pattern_name=entity.pattern_name,
                            locale=entity.locale,
                            recognition_metadata=entity.recognition_metadata,
                        )

            # Filter DATE_TIME false positives (dates are not PII in most contexts)
            if entity.entity_type == "DATE_TIME":
                # Skip OCR artifacts with pipe characters (table fragments)
                if '|' in entity_text:
                    continue
                # Skip patterns with excessive whitespace (table artifacts)
                if re.search(r'\s{3,}', entity_text):
                    continue
                # Skip generic date phrases (fiscal year, month names, etc.)
                date_false_positives = {
                    'the fiscal year', 'fiscal year ending', 'fiscal year ended',
                    'the month of', 'month of july', 'month of january', 'month of december',
                    'year ended', 'year ending', 'quarter ended', 'quarter ending',
                    'as at', 'as of', 'march 31', 'december 31', 'june 30', 'september 30',
                }
                text_lower = entity_text.lower()
                if any(fp in text_lower for fp in date_false_positives):
                    continue
                # Skip standalone month names
                month_names = {'january', 'february', 'march', 'april', 'may', 'june',
                               'july', 'august', 'september', 'october', 'november', 'december'}
                if text_lower.strip() in month_names:
                    continue
                # Skip if confidence is low
                if entity.confidence < 0.75:
                    continue

            # Filter FINANCIAL false positives
            if entity.entity_type == "FINANCIAL":
                # Skip OCR artifacts with pipe characters (table fragments)
                if '|' in entity_text:
                    continue
                # Skip OCR artifacts (random case, underscores, incomplete words)
                if self._is_ocr_artifact(entity_text):
                    continue
                # Skip standalone negative numbers without currency symbol (e.g., "-50", "-6286")
                # These are typically IDs, offsets, or numeric data, not financial amounts
                if re.match(r'^-\d+$', entity_text):
                    continue
                # Skip negative numbers with decimals but no currency (e.g., "-50.00")
                if re.match(r'^-\d+\.\d+$', entity_text):
                    continue
                # Skip round currency amounts that are likely prices/ranges, not personal financial data
                # These are common in documents for prices, salaries, limits, etc.
                round_amount_pattern = r'^\s*[\$€£¥]\s*[\d,]+(?:\.00)?\s*$'
                if re.match(round_amount_pattern, entity_text):
                    # Extract numeric value
                    numeric_value = re.sub(r'[^\d.]', '', entity_text)
                    try:
                        amount = float(numeric_value) if numeric_value else 0
                        # Skip very common round amounts (likely prices, not personal data)
                        common_amounts = {0, 50, 100, 200, 250, 500, 1000, 1200, 1500, 2000, 2500,
                                         5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000}
                        if amount in common_amounts:
                            continue
                        # Skip "$0.00" specifically (common placeholder)
                        if amount == 0:
                            continue
                    except ValueError:
                        pass

            # Filter EMAIL_ADDRESS false positives
            if entity.entity_type == "EMAIL_ADDRESS":
                # Fix email boundaries - Presidio sometimes captures surrounding noise
                # or truncates the domain when text has escape sequences
                email_re = re.compile(r'[\w.+-]+@[\w.-]+\.\w{2,}')
                # Search in a wider window around the detection to find the real email
                search_start = max(0, entity.start - 5)
                search_end = min(len(text), entity.end + 10)
                search_text = text[search_start:search_end]
                email_match = email_re.search(search_text)
                if email_match:
                    # Adjust entity boundaries to the actual email
                    entity.start = search_start + email_match.start()
                    entity.end = search_start + email_match.end()
                    entity_text = text[entity.start:entity.end]
                else:
                    # No valid email in or near the span - skip
                    continue

                # Suppress email-like patterns embedded within HTTP URLs
                # e.g., https://example.com/user@domain.com/profile
                # But keep mailto: emails (those are real emails)
                pre_start = max(0, entity.start - 30)
                pre_text = text[pre_start:entity.start].lower()
                post_end = min(len(text), entity.end + 5)
                post_text = text[entity.end:post_end]
                if '://' in pre_text and 'mailto:' not in pre_text:
                    # Only suppress if followed by path (part of URL)
                    if post_text.startswith('/'):
                        continue

            filtered.append(entity)

        return filtered

    def analyze_text(
        self,
        text: str,
        language: str = "en",
        locales: Optional[List[str]] = None,
        auto_detect_locale: bool = False
    ) -> List[PIIEntity]:
        """
        Analyze text for PII entities with multi-pass detection.

        Pass 1: Run Presidio detection
        Pass 2: Apply locale-based confidence adjustments
        Pass 3: Resolve conflicts (e.g., NHS vs phone numbers)
        Pass 4: Filter false positives
        Pass 5: Deduplicate overlapping entities

        Args:
            text: Input text to analyze
            language: Language code (default: "en")
            locales: List of ISO locale codes (e.g., ["en-US", "es-MX"]).
                     If None, no locale-based filtering is applied.
                     For international spreadsheets, pass None.
            auto_detect_locale: If True and locales is None, attempt to
                                auto-detect the document locale from content.

        Returns:
            List of detected PII entities with locale information
        """
        # Auto-detect locale if requested and not provided
        effective_locales = locales
        if auto_detect_locale and locales is None:
            effective_locales = detect_document_locale(text)

        # Normalize text for evasion defense (fullwidth chars, zero-width spaces)
        if TEXT_NORMALIZER_AVAILABLE:
            original_text = text
            text = normalize_text(text)

            # Also scan for encoded PII (Base64, URL-encoded, Hex)
            encoded_findings = decode_and_scan(text)
        else:
            original_text = text
            encoded_findings = []

        results = self.analyzer.analyze(
            text=text,
            language=language,
            entities=None,  # Detect all entity types
            return_decision_process=True  # Get pattern names
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

            # Extract pattern name from recognition metadata
            pattern_name = None
            if hasattr(r, 'recognition_metadata') and r.recognition_metadata:
                pattern_name = r.recognition_metadata.get('pattern_name')
            elif hasattr(r, 'analysis_explanation') and r.analysis_explanation:
                # Try to get pattern name from analysis explanation
                if hasattr(r.analysis_explanation, 'pattern_name'):
                    pattern_name = r.analysis_explanation.pattern_name

            # Calculate locale boost and determine entity locale
            entity_locale = None
            adjusted_confidence = r.score

            if pattern_name and pattern_name in PATTERN_LOCALE_MAP:
                mapping = PATTERN_LOCALE_MAP[pattern_name]
                # Get the primary locale for this pattern
                if mapping.locales:
                    entity_locale = next(iter(mapping.locales), None)

                # Apply locale-based confidence adjustment
                locale_boost = calculate_locale_boost(pattern_name, effective_locales)
                adjusted_confidence = min(1.0, max(0.0, r.score + locale_boost))

            # Capture recognition_metadata if present (e.g., detection_source from PersonRecognizer)
            recognition_metadata = getattr(r, 'recognition_metadata', None)

            entities.append(PIIEntity(
                entity_type=entity_type,
                text=entity_text,
                start=r.start,
                end=r.end,
                confidence=adjusted_confidence,
                pattern_name=pattern_name,
                locale=entity_locale,
                recognition_metadata=recognition_metadata
            ))

        # Pass 2.5: Detect PII in labeled fields (XML tags, JSON keys, template vars)
        labeled_entities = self._detect_labeled_pii(text)
        entities.extend(labeled_entities)

        # Pass 3: Resolve NHS/phone conflicts
        entities = self._resolve_nhs_phone_conflicts(entities, text)

        # Pass 4: Filter out common false positives
        entities = self._filter_false_positives(entities, text)

        # Pass 5: Deduplicate overlapping entities (keep highest confidence)
        entities = self._deduplicate_entities(entities)

        # Pass 6: Validate entities and adjust confidence (IBAN, phone, credit card, etc.)
        entities = self._validate_entities(entities, effective_locales)

        # Pass 6.5: Heuristic verification for uncertain high-FP entities
        # Re-enabled with minimal rules: only rejects exact form labels
        if HEURISTIC_VERIFIER_AVAILABLE:
            entities = filter_with_heuristics(entities, text)

        # Pass 7: Scan encoded content for hidden PII (Base64, URL-encoded, Hex)
        if encoded_findings:
            for encoding_type, decoded_text, start_pos, end_pos in encoded_findings:
                # Recursively scan decoded content for PII
                decoded_entities = self.analyzer.analyze(
                    text=decoded_text,
                    language=language,
                    entities=None
                )
                for r in decoded_entities:
                    if r.score >= 0.5:  # Only report medium-high confidence findings
                        entities.append(PIIEntity(
                            entity_type=r.entity_type,
                            text=decoded_text[r.start:r.end],
                            start=start_pos,  # Map to original encoded position
                            end=end_pos,
                            confidence=r.score * 0.9,  # Slightly lower confidence for encoded PII
                            pattern_name=f"encoded_{encoding_type}",
                            locale=None,
                            recognition_metadata=getattr(r, 'recognition_metadata', None)
                        ))

        # Pass 8: Filter by minimum confidence score threshold
        # Entity-specific thresholds - calibrated using Yellowbrick PR analysis (2026-02-07)
        # PersonRecognizer: single-model 0.60-0.85, multi-model 0.55-0.92
        # AddressVerifier: weighted component scoring 0.50-1.10
        ENTITY_THRESHOLDS = {
            'PERSON': 0.50,       # Balanced: lower adds marginal recall but significant FP processing cost
            'ADDRESS': 0.50,      # Lowered from 0.60 for partial address component recall
            'LOCATION': 0.50,     # Same as ADDRESS - LOCATION maps to ADDRESS in benchmark
            'SSN': 0.60,          # Balanced
            'CREDIT_CARD': 0.55,  # Calibrated: high precision recognizer
            'PHONE_NUMBER': 0.55, # Lowered from 0.65 for international format recall
            'CREDENTIAL': 0.80,   # Lowered from 0.85 for better recall on short special-char passwords
            'USERNAME': 0.55,     # Balanced: requires context for simple patterns to avoid FPs
            'COMPANY': 0.60,      # Balanced for suffix validation
            'ID': 0.75,           # Lowered from 0.80 to improve recall on alphanumeric codes
            'NATIONAL_ID': 0.40,  # Lowered from 0.55→0.50→0.45→0.40 for more international ID formats
            'VEHICLE': 0.78,      # Keep
            'VEHICLE_ID': 0.90,   # Keep
            'MEDICAL': 0.65,      # Moderate - reduce FPs
            'PASSPORT': 0.50,     # Aligned with NATIONAL_ID - passports are valid national IDs
            'UK_NHS': 0.65,       # Balanced
            'DEVICE_ID': 0.99,    # Keep - spurious
            'BANK_NUMBER': 0.65,  # Balanced
            'DRIVERS_LICENSE': 0.65,  # Lowered from 0.99 - driver's licenses are valid national IDs
            'IP_ADDRESS': 0.60,   # Lowered from 0.65 to improve recall for valid IPs
            'FINANCIAL': 0.65,    # Balanced
            'COORDINATES': 0.85,  # High: single decimal_coords (0.70) creates massive FPs
            'AGE': 0.80,          # Raised from default 0.65 to reduce false positives (e.g., "28M" room codes)
            'GENDER': 0.70,       # Raised from default 0.65 to reduce false positives from loose context matches
        }
        DEFAULT_THRESHOLD = 0.65  # Balanced default
        entities = [
            e for e in entities
            if e.confidence >= ENTITY_THRESHOLDS.get(e.entity_type, DEFAULT_THRESHOLD)
        ]

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

    def _validate_entities(
        self,
        entities: List[PIIEntity],
        locales: Optional[List[str]] = None
    ) -> List[PIIEntity]:
        """
        Validate detected entities using external libraries and adjust confidence.

        Uses python-stdnum and phonenumbers for validation of:
        - IBAN (checksum validation)
        - Credit cards (Luhn algorithm)
        - Phone numbers (international validation)
        - National IDs (country-specific validation)

        Args:
            entities: List of detected entities
            locales: Document locales for context

        Returns:
            List of entities with adjusted confidence scores
        """
        try:
            from .validators import validate_detected_entity
        except ImportError:
            # Validators not available - return entities unchanged
            return entities

        validated = []
        for entity in entities:
            # Get locale for validation context
            locale = entity.locale
            if not locale and locales:
                locale = locales[0] if locales else None

            # Validate and get confidence adjustment
            is_valid, confidence_adj = validate_detected_entity(
                entity.entity_type,
                entity.text,
                locale
            )

            # Apply confidence adjustment
            new_confidence = min(1.0, max(0.0, entity.confidence + confidence_adj))

            # Skip entities that fail validation with very low confidence
            if not is_valid and new_confidence < 0.3:
                continue

            # Create updated entity with new confidence (preserve recognition_metadata)
            validated.append(PIIEntity(
                entity_type=entity.entity_type,
                text=entity.text,
                start=entity.start,
                end=entity.end,
                confidence=new_confidence,
                pattern_name=entity.pattern_name,
                locale=entity.locale,
                recognition_metadata=entity.recognition_metadata
            ))

        return validated

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        sample_size: int = 20,
        locales: Optional[List[str]] = None,
        auto_detect_locale: bool = False
    ) -> Dict[str, List[PIIEntity]]:
        """
        Analyze a DataFrame for PII, using sampling for efficiency.

        For international spreadsheets containing data from multiple countries,
        pass locales=None to use all pattern types without locale filtering.

        Args:
            df: Input DataFrame
            sample_size: Number of rows to sample per column
            locales: List of ISO locale codes (e.g., ["en-US", "ja-JP"]).
                     Pass None for international data with multiple locales.
            auto_detect_locale: If True, attempt to detect locale from content.
                                Not recommended for international datasets.

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
            entities = self.analyze_text(
                sample_text,
                locales=locales,
                auto_detect_locale=auto_detect_locale
            )

            if entities:
                results[column] = entities

        return results

    def analyze_text_with_debug(
        self,
        text: str,
        language: str = "en",
        locales: Optional[List[str]] = None,
        auto_detect_locale: bool = False
    ) -> Dict[str, any]:
        """
        Analyze text for PII with detailed decision explanations.

        This method provides insight into why detections are made, including:
        - Which recognizer fired for each detection
        - Original score vs context-boosted score
        - What context words triggered score boosts
        - Recognition metadata (detection sources, engine counts)

        Useful for:
        - Debugging false positives
        - Understanding detection behavior
        - Tuning detection thresholds
        - Training data analysis

        Args:
            text: Input text to analyze
            language: Language code (default: "en")
            locales: List of ISO locale codes (e.g., ["en-US", "es-MX"])
            auto_detect_locale: If True, attempt to auto-detect locale

        Returns:
            Dictionary with keys:
            - 'entities': List of PIIEntity objects (final filtered results)
            - 'raw_detections': List of raw Presidio detections with explanations
            - 'filtered_count': Number of detections filtered by post-processing
            - 'by_recognizer': Detections grouped by recognizer name
            - 'context_boosted': Detections that received context boost

        Example:
            >>> detector = PIIDetector()
            >>> result = detector.analyze_text_with_debug("Dr. John Smith works at Apple")
            >>> for d in result['raw_detections']:
            ...     print(f"{d['entity_type']}: {d['text']} (recognizer: {d['recognizer']})")
        """
        # Auto-detect locale if requested
        effective_locales = locales
        if auto_detect_locale and locales is None:
            effective_locales = detect_document_locale(text)

        # Normalize text for evasion defense
        if TEXT_NORMALIZER_AVAILABLE:
            original_text = text
            text = normalize_text(text)
        else:
            original_text = text

        # Get raw Presidio results with decision process
        raw_results = self.analyzer.analyze(
            text=text,
            language=language,
            entities=None,
            return_decision_process=True
        )

        # Build raw detections with explanations
        raw_detections = []
        by_recognizer = {}
        context_boosted = []

        for r in raw_results:
            entity_text = text[r.start:r.end]

            # Extract analysis explanation
            exp = getattr(r, 'analysis_explanation', None)
            original_score = r.score
            context_boost = 0.0
            recognizer_name = "unknown"
            pattern_name = None
            supportive_context = None
            textual_explanation = None

            if exp:
                recognizer_name = getattr(exp, 'recognizer', 'unknown')
                pattern_name = getattr(exp, 'pattern_name', None)
                original_score = getattr(exp, 'original_score', r.score)
                context_boost = getattr(exp, 'score_context_improvement', 0.0)
                supportive_context = getattr(exp, 'supportive_context_word', None)
                textual_explanation = getattr(exp, 'textual_explanation', None)

            # Extract recognition metadata
            recognition_metadata = getattr(r, 'recognition_metadata', None)
            if recognition_metadata and hasattr(recognition_metadata, 'items'):
                recognition_metadata = dict(recognition_metadata)

            detection_info = {
                'entity_type': r.entity_type,
                'text': entity_text,
                'start': r.start,
                'end': r.end,
                'final_score': r.score,
                'original_score': original_score,
                'context_boost': context_boost,
                'recognizer': recognizer_name,
                'pattern_name': pattern_name,
                'supportive_context': supportive_context,
                'textual_explanation': textual_explanation,
                'recognition_metadata': recognition_metadata,
            }

            raw_detections.append(detection_info)

            # Group by recognizer
            if recognizer_name not in by_recognizer:
                by_recognizer[recognizer_name] = []
            by_recognizer[recognizer_name].append(detection_info)

            # Track context-boosted detections
            if context_boost > 0.05:
                context_boosted.append(detection_info)

        # Get final filtered entities using normal pipeline
        final_entities = self.analyze_text(
            text=original_text,
            language=language,
            locales=locales,
            auto_detect_locale=auto_detect_locale
        )

        # Calculate filtered count
        filtered_count = len(raw_detections) - len(final_entities)

        return {
            'entities': final_entities,
            'raw_detections': raw_detections,
            'filtered_count': filtered_count,
            'by_recognizer': by_recognizer,
            'context_boosted': context_boosted,
            'raw_count': len(raw_detections),
            'final_count': len(final_entities),
        }

    def get_registered_recognizers(self) -> List[str]:
        """
        Get list of all registered recognizer names.

        Returns:
            Sorted list of recognizer names
        """
        recognizers = self.analyzer.registry.recognizers
        return sorted(set(r.name for r in recognizers))

    def get_recognizers_for_entity(self, entity_type: str) -> List[str]:
        """
        Get recognizers that support a specific entity type.

        Args:
            entity_type: Entity type (e.g., "PERSON", "LOCATION")

        Returns:
            List of recognizer names that can detect the given entity type
        """
        recognizers = self.analyzer.registry.recognizers
        return sorted(set(
            r.name for r in recognizers
            if entity_type in r.supported_entities
        ))
