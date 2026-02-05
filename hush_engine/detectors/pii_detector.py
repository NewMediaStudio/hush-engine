"""
PII Detection using Microsoft Presidio

Supports locale-aware detection for international documents.
"""

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

# LLM Verifier for precision improvement (Apple Silicon only)
try:
    from hush_engine.detectors.llm_verifier import get_verifier, LLMVerifier
    LLM_VERIFIER_AVAILABLE = True
except ImportError:
    try:
        from .llm_verifier import get_verifier, LLMVerifier
        LLM_VERIFIER_AVAILABLE = True
    except ImportError:
        LLM_VERIFIER_AVAILABLE = False
        get_verifier = None
        LLMVerifier = None

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
    """
    entity_type: str  # e.g., "EMAIL_ADDRESS", "PHONE_NUMBER", "AWS_KEY"
    text: str
    start: int
    end: int
    confidence: float
    pattern_name: Optional[str] = None
    locale: Optional[str] = None


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
                context_prefix_count=5,
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
        # Add scispaCy-based medical NER (broader coverage using biomedical models)
        self._add_scispacy_medical_recognizer()
        # Add phone number recognizers with NA area code validation
        self._add_phone_recognizers()
        # Add ID document recognizers (passports, driver's licenses)
        self._add_id_document_recognizers()
        # Add vehicle recognizers (VIN)
        self._add_vehicle_recognizers()
        # Add device ID recognizers (IMEI, MAC address, UUID)
        self._add_device_recognizers()
        # Add IP address recognizers (IPv4, IPv6)
        self._add_ip_address_recognizers()
        # Add SSN recognizer for test/fake SSNs with reserved area numbers
        self._add_ssn_recognizers()
        # Add international national ID recognizers
        self._add_international_id_recognizers()
        # Add person name recognizers (title + name patterns)
        self._add_person_recognizers()
        # Add URL recognizers (http, https, www, subdomains)
        self._add_url_recognizers()
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

        # US state abbreviations (all 50 states + DC + territories)
        us_states = r"(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY|DC|PR|VI|GU|AS|MP)"

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

        # 10. Standalone apartment/unit: "APT 1104", "SUITE 200", "UNIT 5A", "FL 3"
        standalone_unit = PatternRecognizer(
            supported_entity="LOCATION",
            patterns=[
                Pattern(
                    name="standalone_apt",
                    # Matches: Apt/Unit/Suite/Floor + number (with optional letter)
                    regex=r"\b(?:APT|Apt\.?|APARTMENT|Apartment|UNIT|Unit|SUITE|Suite|STE|Ste\.?|FL|Floor|FLOOR|RM|Room|ROOM)\s+[0-9]+[A-Za-z]?\b",
                    score=0.75,
                )
            ],
            context=["address", "mail", "deliver", "ship", "building", "floor"]
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
                    regex=r"\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b",
                    score=0.85,
                ),
            ],
            context=["postcode", "post code", "uk", "england", "scotland", "wales", "london"]
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
                    regex=r"\bBANGLADESH\b",
                    score=0.80,
                ),
                Pattern(
                    name="bangladesh_cities",
                    regex=r"\b(?:DHAKA|Dhaka|CHITTAGONG|Chittagong|KHULNA|Khulna|RAJSHAHI|Rajshahi|SYLHET|Sylhet|COMILLA|Comilla)\b",
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
        self.analyzer.registry.add_recognizer(street_address_with_number)
        self.analyzer.registry.add_recognizer(street_address_no_number)
        self.analyzer.registry.add_recognizer(european_street_address)
        self.analyzer.registry.add_recognizer(po_box_address)
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
            ],
        )

        # Building/floor/room patterns: "Bldg A", "Tower 2", "Wing B"
        building_patterns = PatternRecognizer(
            supported_entity="LOCATION",
            patterns=[
                Pattern(
                    name="building_wing",
                    regex=r"\b(?:Bldg\.?|Building|Tower|Wing|Block|Annex)\s+[A-Z0-9]+\b",
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

        # Register enhanced address patterns
        self.analyzer.registry.add_recognizer(hash_unit)
        self.analyzer.registry.add_recognizer(care_of_address)
        self.analyzer.registry.add_recognizer(rural_route)
        self.analyzer.registry.add_recognizer(building_patterns)
        self.analyzer.registry.add_recognizer(address_continuation)

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

        # Signed amounts without currency symbol (common in financial tables)
        # Handles: - 1,027.00, + 1,749.00, -500.00, -500, +1000 (decimals now optional)
        signed_amount_regex = r"\b[+\-]\s?\d+(?:,\d{3})*(?:\.\d{1,2})?\b"

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
                    score=0.8,
                ),
                Pattern(
                    name="signed_amount",
                    regex=signed_amount_regex,
                    score=0.70,  # Lower score for signed amounts without currency symbol
                ),
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
                    score=0.85,
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
        """
        try:
            from .person_recognizer import get_person_recognizer, is_person_ner_available

            # Use accurate mode: patterns + name-dataset + spaCy + GLiNER + Flair
            person_recognizer = get_person_recognizer(mode="accurate")
            self.analyzer.registry.add_recognizer(person_recognizer)

            if is_person_ner_available():
                import sys
                sys.stderr.write("[PIIDetector] PersonRecognizer loaded with NER cascade\n")
            else:
                sys.stderr.write("[PIIDetector] PersonRecognizer loaded (patterns only)\n")

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
            r"Foundation", r"Trust",
            r"Solutions", r"Services",
            r"Management", r"Consulting",
            r"International", r"Worldwide", r"Global",
            r"Brands", r"Products",
            r"Systems", r"Technologies", r"Tech",
            r"Research", r"Pharmaceuticals",
            r"Entertainment", r"Media",
            r"Realty", r"Properties", r"Investments",
            r"Clinic", r"Hospital", r"Medical",
            r"Law\s+Firm", r"Legal",
            r"House", r"Press", r"Publishing",
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
        all_caps_company_regex = rf"\b[A-Z]{{2,}}(?:\s+[A-Z]{{2,}})*\s+(?:{designations_pattern.upper()}|COMPANY|CORPORATION|GROUP|SOLUTIONS|SERVICES|INTERNATIONAL)\b"

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
                    name="company_hyphenated",
                    regex=hyphenated_company_regex,
                    score=0.55,  # Reduced from 0.65 - too many false positives
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
            ],
            context=["company", "inc", "ltd", "corp", "limited", "firm", "business", "employer", "organization", "corporation", "enterprise", "client", "vendor"]
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

        self.analyzer.registry.add_recognizer(medical_recognizer)
        self.analyzer.registry.add_recognizer(condition_recognizer)
        self.analyzer.registry.add_recognizer(lab_recognizer)
        self.analyzer.registry.add_recognizer(body_part_recognizer)

    def _add_scispacy_medical_recognizer(self):
        """
        Add scispaCy-based medical NER recognizer for broader medical term coverage.

        Uses biomedical NER models trained on PubMed/clinical text (BC5CDR corpus)
        to detect diseases, drugs, and medical conditions without maintaining
        hardcoded lists.

        Falls back gracefully if scispaCy or models are not installed.
        """
        try:
            from .medical_recognizer import get_medical_recognizer, is_scispacy_available

            if is_scispacy_available():
                recognizer = get_medical_recognizer()
                if recognizer:
                    self.analyzer.registry.add_recognizer(recognizer)
                    import sys
                    sys.stderr.write("[PIIDetector] Added scispaCy medical recognizer\n")
            else:
                import sys
                sys.stderr.write("[PIIDetector] scispaCy not available, using pattern-based medical detection only\n")
        except ImportError as e:
            import sys
            sys.stderr.write(f"[PIIDetector] Could not load scispaCy medical recognizer: {e}\n")

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

        # NA phone with dashes: 416-770-4541
        na_phone_dashes = PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=[
                Pattern(
                    name="na_phone_dashes",
                    regex=r"\b\d{3}-\d{3}-\d{4}\b",
                    score=0.95,  # Boosted for better recall
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
                    regex=r"\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b",
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

        # UK phone: +44 XXXX XXXXXX, 07XXX XXXXXX
        uk_phone = PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=[
                Pattern(
                    name="uk_mobile",
                    # UK mobile: 07XXX XXXXXX
                    regex=r"\b07\d{3}\s?\d{6}\b",
                    score=0.90,
                ),
                Pattern(
                    name="uk_intl",
                    # +44 XXXX XXXXXX
                    regex=r"\+44\s?\d{4}\s?\d{6}\b",
                    score=0.92,
                ),
            ],
            context=["phone", "tel", "mobile", "cell", "contact", "uk"]
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
                # XXX-XXX-XXXX x NNNNN - dashes with extension
                Pattern(
                    name="phone_ext_dashes",
                    regex=r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\s?(?:x|ext\.?|extension)\s?\d{1,6}\b",
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
        # Visa: starts with 4, 16 digits
        # Mastercard: starts with 51-55 or 2221-2720, 16 digits
        # American Express: starts with 34 or 37, 15 digits
        # Discover: starts with 6011, 622126-622925, 644-649, or 65, 16 digits
        # JCB: starts with 3528-3589, 16 digits
        # Diners Club: starts with 300-305, 36, 38, 14-16 digits
        # UnionPay: starts with 62, 16-19 digits

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
                # JCB (3528-3589) - 15 or 16 digits
                Pattern(
                    name="jcb_16",
                    regex=r"\b35(?:2[89]|[3-8]\d)[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
                    score=0.9
                ),
                Pattern(
                    name="jcb_15",
                    regex=r"\b35(?:2[89]|[3-8]\d)[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{3}\b",
                    score=0.9
                ),
                # Diners Club (300-305, 36, 38) - 14 digits
                Pattern(
                    name="diners_14",
                    regex=r"\b(?:30[0-5]\d|36\d{2}|38\d{2})[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{2}\b",
                    score=0.9
                ),
                # UnionPay (62) - 16-19 digits
                Pattern(
                    name="unionpay_16",
                    regex=r"\b62\d{2}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
                    score=0.85
                ),
                Pattern(
                    name="unionpay_19",
                    regex=r"\b62\d{2}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{3}\b",
                    score=0.85
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
                # Generic 15-digit pattern (fallback for Amex/JCB)
                Pattern(
                    name="generic_15",
                    regex=r"\b\d{4}[\s-]?\d{6}[\s-]?\d{5}\b",
                    score=0.6
                ),
                # Maestro cards (12-19 digits, starts with 50, 56-69, 5018, 5020, 5038, 6304, 6759, 6761-6763)
                Pattern(
                    name="maestro_12_13",
                    regex=r"\b(?:50|5[6-9]|6[0-9])\d{10,11}\b",
                    score=0.75
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
        generic_credential_recognizer = PatternRecognizer(
            supported_entity="CREDENTIAL",
            patterns=[
                Pattern(
                    name="bearer_token",
                    # Bearer tokens in Authorization headers
                    regex=r"Bearer\s+[A-Za-z0-9_\-\.]{20,}",
                    score=0.85,
                ),
                Pattern(
                    name="api_key_labeled",
                    # Labeled API keys: api_key=xxx, apiKey: xxx
                    regex=r"(?:api[_\-]?key|apikey)\s*[=:]\s*['\"]?[A-Za-z0-9_\-]{20,}['\"]?",
                    score=0.85,
                ),
                Pattern(
                    name="secret_labeled",
                    # Labeled secrets: secret=xxx, SECRET_KEY: xxx
                    regex=r"(?:secret|SECRET)[_\-]?(?:key|KEY)?\s*[=:]\s*['\"]?[A-Za-z0-9_\-]{16,}['\"]?",
                    score=0.80,
                ),
                Pattern(
                    name="password_labeled",
                    # Labeled passwords (not the word "password" alone)
                    regex=r"(?:password|passwd|pwd)\s*[=:]\s*['\"]?[^\s'\"\n]{8,}['\"]?",
                    score=0.75,
                ),
                Pattern(
                    name="token_labeled",
                    # Labeled tokens: token=xxx, access_token: xxx
                    regex=r"(?:access[_\-]?token|auth[_\-]?token|TOKEN)\s*[=:]\s*['\"]?[A-Za-z0-9_\-\.]{20,}['\"]?",
                    score=0.80,
                ),
            ],
            context=["api", "key", "token", "secret", "password", "credential", "auth", "bearer"]
        )

        # Slack Tokens
        slack_recognizer = PatternRecognizer(
            supported_entity="CREDENTIAL",
            patterns=[
                Pattern(
                    name="slack_bot_token",
                    regex=r"xoxb-[0-9]{10,13}-[0-9]{10,13}-[a-zA-Z0-9]{24}",
                    score=0.95,
                ),
                Pattern(
                    name="slack_user_token",
                    regex=r"xoxp-[0-9]{10,13}-[0-9]{10,13}-[0-9]{10,13}-[a-f0-9]{32}",
                    score=0.95,
                ),
                Pattern(
                    name="slack_workspace_token",
                    regex=r"xoxa-[0-9]{10,13}-[0-9]{10,13}-[a-zA-Z0-9]{24}",
                    score=0.95,
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
                    score=0.95,
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

        self.analyzer.registry.add_recognizer(aws_recognizer)
        self.analyzer.registry.add_recognizer(stripe_recognizer)
        self.analyzer.registry.add_recognizer(github_recognizer)
        self.analyzer.registry.add_recognizer(google_recognizer)
        self.analyzer.registry.add_recognizer(generic_credential_recognizer)
        self.analyzer.registry.add_recognizer(slack_recognizer)
        self.analyzer.registry.add_recognizer(npm_recognizer)
        self.analyzer.registry.add_recognizer(private_key_recognizer)

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

        generic_id_recognizer = PatternRecognizer(
            supported_entity="ID",
            patterns=[
                Pattern(name="id_letter_dash_digits", regex=id_letter_dash_digits, score=0.85),
                Pattern(name="id_letters_digits", regex=id_letters_digits, score=0.80),
                Pattern(name="id_mixed_pattern", regex=id_mixed_pattern, score=0.80),
            ],
            context=["id", "reference", "number", "account", "customer", "client",
                     "case", "ticket", "order", "tracking", "confirmation", "ref",
                     "member", "subscriber", "policy", "claim"]
        )

        self.analyzer.registry.add_recognizer(generic_id_recognizer)

    def _add_vehicle_recognizers(self):
        """
        Add pattern recognizers for vehicle identification.

        Detects:
        - VIN (Vehicle Identification Number) - ISO 3779 standard
        - License plates (limited - highly variable by jurisdiction)
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
        # Simplified pattern for common formats
        ipv6_full = r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"
        # Compressed IPv6 with ::
        ipv6_compressed = r"\b(?:[0-9a-fA-F]{1,4}:)*:(?::[0-9a-fA-F]{1,4})*\b"

        ip_recognizer = PatternRecognizer(
            supported_entity="IP_ADDRESS",
            patterns=[
                Pattern(name="ipv4_standard", regex=ipv4_pattern, score=0.85),
                Pattern(name="ipv4_with_port", regex=ipv4_with_port, score=0.90),
                Pattern(name="ipv6_full", regex=ipv6_full, score=0.95),
                Pattern(name="ipv6_compressed", regex=ipv6_compressed, score=0.85),
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

                        if has_street or (has_location and num_components >= 2):
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

                        # Require at least one meaningful component
                        meaningful = {'house_number', 'road', 'house', 'city', 'suburb',
                                     'state', 'postcode', 'country', 'unit', 'po_box'}

                        if component_types & meaningful:
                            boost = min(0.35, len(component_types & meaningful) * 0.08)
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
        # Format: XXX-XX-XXXX or XXX XX XXXX or XXXXXXXXX
        ssn_all_areas = r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"

        ssn_recognizer = PatternRecognizer(
            supported_entity="SSN",
            patterns=[
                Pattern(name="ssn_all_areas", regex=ssn_all_areas, score=0.85),
            ],
            context=["ssn", "social security", "social security number", "ss#", "ss #",
                     "tax id", "tin", "taxpayer"]
        )

        # SSN with spaces or no separators (needs stronger context)
        # Format: XXX XXX XXXX (3-3-4 with spaces) or XXXXXXXXX (9 digits no sep)
        ssn_alternative = PatternRecognizer(
            supported_entity="SSN",
            patterns=[
                # "755 979 3272" - 3-3-4 with spaces (unusual but found in some docs)
                Pattern(
                    name="ssn_3_3_4_spaces",
                    regex=r"\b\d{3}\s+\d{3}\s+\d{4}\b",
                    score=0.65,  # Lower score - conflicts with phone format
                ),
                # "044034803" - 9 digits no separators (very generic, needs context)
                Pattern(
                    name="ssn_no_sep",
                    regex=r"\b\d{9}\b",
                    score=0.45,  # Very low - highly ambiguous without context
                ),
            ],
            context=["ssn", "social security", "social security number", "ss#", "ss #",
                     "social", "tax id", "tin", "taxpayer", "employee id"]
        )

        self.analyzer.registry.add_recognizer(ssn_recognizer)
        self.analyzer.registry.add_recognizer(ssn_alternative)

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
                Pattern(name="canadian_sin", regex=canadian_sin, score=0.75),
            ],
            context=["sin", "social insurance", "social insurance number"]
        )

        german_id_recognizer = PatternRecognizer(
            supported_entity="NATIONAL_ID",
            patterns=[
                Pattern(name="german_steuer", regex=german_steuer, score=0.70),
            ],
            context=["steuer", "tax", "steuer-id", "steuernummer", "identifikationsnummer"]
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
                Pattern(name="japanese_mynumber", regex=japanese_mynumber, score=0.70),
            ],
            context=["my number", "マイナンバー", "individual number", "kojin bango"]
        )

        indian_aadhaar_recognizer = PatternRecognizer(
            supported_entity="NATIONAL_ID",
            patterns=[
                Pattern(name="indian_aadhaar", regex=indian_aadhaar, score=0.70),
            ],
            context=["aadhaar", "aadhar", "uid", "unique identification"]
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

    def _validate_phone_with_phonenumbers(self, phone_text: str, default_region: str = "US") -> str:
        """
        Validate a phone number using Google's libphonenumber library.

        Args:
            phone_text: The phone number text to validate
            default_region: Default region code for parsing (e.g., "US", "GB", "CA")

        Returns:
            "valid" if the phone number is definitely valid (boost confidence)
            "invalid" if the phone number is definitely invalid (reduce confidence)
            "unknown" if validation is unavailable or inconclusive (keep as is)
        """
        if not PHONENUMBERS_AVAILABLE:
            return "unknown"  # Fall back to pattern-only validation if library unavailable

        try:
            # Try to parse the phone number
            parsed = phonenumbers.parse(phone_text, default_region)

            # Check if it's a valid number
            if phonenumbers.is_valid_number(parsed):
                return "valid"  # Definitely valid - boost confidence

            # Check if it's a possible number (less strict check)
            if not phonenumbers.is_possible_number(parsed):
                return "invalid"  # Definitely invalid - reduce confidence

            return "unknown"  # Possible but not confirmed valid
        except NumberParseException:
            # If parsing fails, it's likely not a valid phone number format
            return "invalid"

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
                # Check for random caps in middle of word
                mid = word[1:-1]
                if mid and any(c.isupper() for c in mid) and any(c.islower() for c in mid):
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

        for entity in entities:
            entity_text = entity.text.strip()

            # Filter using negative gazetteer (common words, brand names, etc.)
            # This catches high-frequency false positives for PERSON, COMPANY, LOCATION
            if entity.entity_type in ("PERSON", "COMPANY", "LOCATION", "ADDRESS"):
                if is_negative_match(entity_text, entity.entity_type):
                    continue
                # For COMPANY: filter single common words without corporate suffix
                if is_single_common_word(entity_text, entity.entity_type):
                    continue

            # Filter NATIONAL_ID false positives
            if entity.entity_type == "NATIONAL_ID":
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
                # Skip if it looks like a phone number (10-11 digits with dashes)
                if len(digits) == 10 or len(digits) == 11:
                    if re.match(r'^\d{3}[-.\s]?\d{3}[-.\s]?\d{4}$', entity_text):
                        continue
                # Skip if it looks like an IP address
                if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', entity_text):
                    continue
                # Skip if it matches UUID pattern (8-4-4-4-12 hex)
                if re.match(r'^[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}$', entity_text):
                    continue
                # Skip very low confidence
                if entity.confidence < 0.5:
                    continue

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
                # Skip if it looks like an international phone number (+CC XXXXXXXXXX)
                if re.match(r'^\+\d{1,3}[\s-]?\d{7,14}$', entity_text):
                    continue
                # Skip if it looks like a phone number fragment (contains dashes in phone pattern)
                if re.match(r'^\d{3,4}[-.]?\d{4}$', entity_text):
                    continue
                # Skip if it looks like a credit card fragment (4 digit groups)
                if re.match(r'^\d{4}[\s-]?\d{4}', entity_text):
                    continue
                # Skip very short matches
                if len(entity_text) < 6:
                    continue
                # Skip if it's just a street number pattern like "2901 E"
                if re.match(r'^\d+\s+[A-Z]$', entity_text):
                    continue
                # Skip if it's just numbers (no date separators or month names)
                if re.match(r'^\d+$', entity_text):
                    continue
                # Skip if it looks like a year only (4 digits)
                if re.match(r'^(19|20)\d{2}$', entity_text):
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
                digits = re.sub(r'\D', '', entity_text)
                # Skip if it's a credit card (16 digits)
                if len(digits) == 16:
                    continue
                # Skip if it matches IBAN pattern (long alphanumeric)
                if len(entity_text) > 20:
                    continue
                # Skip if it looks like an IMEI (15 digits)
                if len(digits) == 15:
                    continue
                # Skip if it looks like an IP address (4 octets separated by dots)
                if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', entity_text):
                    continue
                # Skip if it looks like GPS coordinates (decimal format with minus signs)
                if re.match(r'^-?\d{1,3}\.\d+,?\s*-?\d{1,3}\.\d+$', entity_text):
                    continue
                # Skip if it looks like a date format (YYYY-MM-DD or DD-MM-YYYY)
                if re.match(r'^\d{4}-\d{2}-\d{2}$', entity_text) or re.match(r'^\d{2}-\d{2}-\d{4}$', entity_text):
                    continue
                # Validate using phonenumbers library (Google's libphonenumber)
                # Returns "valid", "invalid", or "unknown"
                if PHONENUMBERS_AVAILABLE and len(digits) >= 7:
                    region = self._get_phone_region_from_context(text, entity.start, entity.end)
                    validation_result = self._validate_phone_with_phonenumbers(entity_text, region)
                    if validation_result == "valid":
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
                            locale=entity.locale
                        )
                    elif validation_result == "invalid":
                        # Lower confidence for numbers that fail phonenumbers validation
                        entity = PIIEntity(
                            entity_type=entity.entity_type,
                            text=entity.text,
                            start=entity.start,
                            end=entity.end,
                            confidence=entity.confidence * 0.5,  # Reduce confidence
                            pattern_name=entity.pattern_name,
                            locale=entity.locale
                        )

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

            # Filter ADDRESS/LOCATION false positives
            if entity.entity_type in ("LOCATION", "ADDRESS"):
                # Skip OCR artifacts (random case, underscores, incomplete words)
                if self._is_ocr_artifact(entity_text):
                    continue
                # Skip very short text (less than 4 chars) - catches "in", "as", "WY", etc.
                if len(entity_text.strip()) < 4:
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
                                'such as', 'focus on'}
                if entity_text.lower().strip() in verb_phrases:
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
                if re.match(r'^\s*\d+\s*$', entity_text):
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
                    if not (has_number_prefix or has_comma_separator or has_street_suffix):
                        # This looks like a phrase, not an address
                        if entity.confidence < 0.70:
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
                # Skip OCR artifacts (random case, underscores, incomplete words)
                if self._is_ocr_artifact(entity_text):
                    continue
                # Skip form labels ending with colon (e.g., " Name: L", "Patient Name:")
                if entity_text.strip().endswith(':') and len(entity_text) < 20:
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
                # Skip very long text (likely phrases, not names)
                if len(entity_text) > 35:
                    continue
                # For spaCy NER detections (lower confidence), require moderate threshold
                # Our pattern recognizers have scores 0.85-0.95
                # Lowered to 0.55 to avoid filtering valid names (was 0.70, caused recall drop)
                if entity.confidence < 0.55:
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
                if entity_text.lower().strip() in nationalities:
                    continue
                # Skip food/menu items (commonly flagged as names)
                food_items = {
                    'caesar', 'caesar salad', 'juicy', 'crispy', 'grilled',
                    'roasted', 'baked', 'fried', 'steamed', 'fresh',
                }
                if entity_text.lower().strip() in food_items:
                    continue
                # Skip generic role words (not person names)
                generic_roles = {
                    'member', 'members', 'recipient', 'recipients', 'authorized personnel',
                    'employee', 'employees', 'customer', 'customers', 'client', 'clients',
                    'user', 'users', 'patient', 'patients', 'subscriber', 'subscribers',
                    'applicant', 'applicants', 'candidate', 'candidates', 'participant',
                    'billing card', 'employee name', 'employee id',
                }
                if entity_text.lower().strip() in generic_roles:
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

            # Filter COORDINATES false positives
            if entity.entity_type == "COORDINATES":
                # Skip if it looks like an IP address fragment
                if re.match(r'^\d{1,3}\.\d{1,3}$', entity_text):
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
                if re.match(r'^[a-z]+-[a-z]+(?:ing|ed|er|tion)\b', text_lower):
                    continue
                # For hyphenated names, require higher confidence when not followed by suffix
                if '-' in entity_text and entity.confidence < 0.7:
                    # Check if NOT followed by a company suffix
                    if not re.search(r'\b(?:Inc|LLC|Ltd|Corp|Company|Group|Partners)\b', entity_text, re.IGNORECASE):
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

            entities.append(PIIEntity(
                entity_type=entity_type,
                text=entity_text,
                start=r.start,
                end=r.end,
                confidence=adjusted_confidence,
                pattern_name=pattern_name,
                locale=entity_locale
            ))

        # Pass 3: Resolve NHS/phone conflicts
        entities = self._resolve_nhs_phone_conflicts(entities, text)

        # Pass 4: Filter out common false positives
        entities = self._filter_false_positives(entities, text)

        # Pass 5: Deduplicate overlapping entities (keep highest confidence)
        entities = self._deduplicate_entities(entities)

        # Pass 6: Validate entities and adjust confidence (IBAN, phone, credit card, etc.)
        entities = self._validate_entities(entities, effective_locales)

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
                            locale=None
                        ))

        # Pass 8: LLM verification for precision (if enabled)
        entities = self._verify_with_llm(entities, text)

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

    def _verify_with_llm(
        self,
        entities: List[PIIEntity],
        text: str
    ) -> List[PIIEntity]:
        """
        Verify mid-confidence detections using local LLM (Apple Silicon only).

        Uses MLX-based Llama model to assess surrounding context and filter
        false positives. Only verifies detections in the 0.40-0.80 confidence
        range to balance precision gains with latency.

        Args:
            entities: List of detected PII entities
            text: Original text for context extraction

        Returns:
            List of entities with false positives removed
        """
        if not LLM_VERIFIER_AVAILABLE:
            return entities

        # Check if LLM verification is enabled in config
        config = get_config()
        if not config.is_integration_enabled("mlx_verifier"):
            return entities

        try:
            verifier = get_verifier(enabled=True)
            if not verifier.is_available():
                return entities
        except Exception:
            return entities

        verified_entities = []
        for entity in entities:
            # High confidence (>=0.85): skip verification, keep entity
            if entity.confidence >= 0.85:
                verified_entities.append(entity)
                continue

            # Low confidence (<0.40): skip verification, keep entity
            # (already filtered by threshold)
            if entity.confidence < 0.40:
                verified_entities.append(entity)
                continue

            # Mid-range confidence (0.40-0.85): verify with LLM
            # Extract 5-word context on each side
            context = self._extract_context(text, entity.start, entity.end, window=5)

            try:
                is_confirmed = verifier.verify_pii(
                    candidate_text=entity.text,
                    entity_type=entity.entity_type,
                    context=context
                )

                if is_confirmed:
                    verified_entities.append(entity)
                # else: drop the entity (LLM said NO)

            except Exception:
                # On error, keep the entity
                verified_entities.append(entity)

        return verified_entities

    def _extract_context(
        self,
        text: str,
        start: int,
        end: int,
        window: int = 5
    ) -> str:
        """
        Extract surrounding context words for LLM verification.

        Args:
            text: Full text
            start: Entity start position
            end: Entity end position
            window: Number of words on each side

        Returns:
            Context string with entity highlighted
        """
        # Get text before and after entity
        before_text = text[:start]
        after_text = text[end:]
        entity_text = text[start:end]

        # Extract last N words before
        before_words = before_text.split()[-window:] if before_text else []

        # Extract first N words after
        after_words = after_text.split()[:window] if after_text else []

        # Build context with entity in brackets
        context_parts = []
        if before_words:
            context_parts.append(" ".join(before_words))
        context_parts.append(f"[{entity_text}]")
        if after_words:
            context_parts.append(" ".join(after_words))

        return " ".join(context_parts)

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

            # Create updated entity with new confidence
            validated.append(PIIEntity(
                entity_type=entity.entity_type,
                text=entity.text,
                start=entity.start,
                end=entity.end,
                confidence=new_confidence,
                pattern_name=entity.pattern_name,
                locale=entity.locale
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
