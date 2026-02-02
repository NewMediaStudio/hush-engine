"""
PII Detection using Microsoft Presidio

Supports locale-aware detection for international documents.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
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

# Package version
__version__ = "1.1.1"

# Import locale support
from .locale import (
    Locale, calculate_locale_boost, detect_document_locale,
    get_locale_from_string, PATTERN_LOCALE_MAP
)


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
        # Add SSN recognizer for test/fake SSNs with reserved area numbers
        self._add_ssn_recognizers()
        # Add international national ID recognizers
        self._add_international_id_recognizers()
        # Add person name recognizers (title + name patterns)
        self._add_person_recognizers()

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
                    score=0.5,
                )
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
                    score=0.50,  # Low score due to collision with US ZIP
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
                    score=0.55,
                ),
            ],
            context=["pin", "pincode", "pin code", "india", "delhi", "mumbai", "bangalore"]
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
                    score=0.50,
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
                    score=0.50,
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
                    score=0.50,
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

        # Register all recognizers
        self.analyzer.registry.add_recognizer(canadian_address_full)
        self.analyzer.registry.add_recognizer(canadian_address_prov_postal)
        self.analyzer.registry.add_recognizer(canadian_city_prov)
        self.analyzer.registry.add_recognizer(canadian_postal_only)
        self.analyzer.registry.add_recognizer(us_address_full)
        self.analyzer.registry.add_recognizer(us_city_state)
        self.analyzer.registry.add_recognizer(us_state_zip)
        self.analyzer.registry.add_recognizer(us_zip_only)
        self.analyzer.registry.add_recognizer(us_state_only)
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
        self.analyzer.registry.add_recognizer(brazil_cep)
        self.analyzer.registry.add_recognizer(korea_postal)
        self.analyzer.registry.add_recognizer(china_postal)
        self.analyzer.registry.add_recognizer(russia_postal)
        self.analyzer.registry.add_recognizer(netherlands_postal)
        self.analyzer.registry.add_recognizer(australia_postal)

    def _add_datetime_recognizers(self):
        """Add pattern recognizers for international date formats."""
        # DD/MM/YYYY format (European, UK, Australia, most of the world)
        # Examples: 04/01/1945, 12/06/2033, 31/12/2024
        date_dmy_slash = PatternRecognizer(
            supported_entity="DATE_TIME",
            patterns=[
                Pattern(
                    name="date_dmy_slash",
                    # DD/MM/YYYY - day 01-31, month 01-12, year 1900-2099
                    regex=r"\b(?:0[1-9]|[12][0-9]|3[01])/(?:0[1-9]|1[0-2])/(?:19|20)\d{2}\b",
                    score=0.85,
                ),
                Pattern(
                    name="date_dmy_dash",
                    # DD-MM-YYYY
                    regex=r"\b(?:0[1-9]|[12][0-9]|3[01])-(?:0[1-9]|1[0-2])-(?:19|20)\d{2}\b",
                    score=0.85,
                ),
                Pattern(
                    name="date_dmy_dot",
                    # DD.MM.YYYY (German format)
                    regex=r"\b(?:0[1-9]|[12][0-9]|3[01])\.(?:0[1-9]|1[0-2])\.(?:19|20)\d{2}\b",
                    score=0.85,
                ),
            ],
        )

        # YYYY/MM/DD format (ISO-like, Asian countries)
        date_ymd = PatternRecognizer(
            supported_entity="DATE_TIME",
            patterns=[
                Pattern(
                    name="date_ymd_slash",
                    regex=r"\b(?:19|20)\d{2}/(?:0[1-9]|1[0-2])/(?:0[1-9]|[12][0-9]|3[01])\b",
                    score=0.85,
                ),
                Pattern(
                    name="date_ymd_dash",
                    regex=r"\b(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])\b",
                    score=0.90,  # ISO 8601 format - higher confidence
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

        self.analyzer.registry.add_recognizer(date_dmy_slash)
        self.analyzer.registry.add_recognizer(date_ymd)
        self.analyzer.registry.add_recognizer(date_with_prefix)
        self.analyzer.registry.add_recognizer(date_birth)
        self.analyzer.registry.add_recognizer(date_partial)

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

        self.analyzer.registry.add_recognizer(age_recognizer)

    def _add_financial_recognizers(self):
        """Add pattern recognizers for financial data (currency, SWIFT codes, etc.)."""
        # Generic currency regex: symbol followed by numbers (with or without commas)
        # Supports $, £, €, ¥, ₹, ₽, 元 and handles optional space
        # Now handles: $1234.56, $1,234.56, €3750.25, ¥450000
        currency_regex = r"[\$£€¥₹₽]\s?\d{1,3}(?:,?\d{3})*(?:\.\d{2})?\b"

        # Word-based currency regex: USD, CAD, EUR, INR, etc. followed by numbers
        word_currency_regex = r"\b(?:USD|CAD|EUR|GBP|JPY|AUD|CNY|INR|CHF|SGD|HKD|NZD|SEK|NOK|DKK|MXN|BRL|ZAR|KRW|THB|MYR|PHP|IDR|VND)\s?\d{1,3}(?:,?\d{3})*(?:\.\d{2})?\b"

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
            context=["swift", "bic", "bank", "transfer", "wire", "iban", "routing", "branch", "inr", "rupee", "crore", "lakh"]
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

        self.analyzer.registry.add_recognizer(currency_recognizer)
        self.analyzer.registry.add_recognizer(iban_recognizer)
        self.analyzer.registry.add_recognizer(labeled_bank_account)

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

            # Use balanced mode: patterns + name-dataset + spaCy
            person_recognizer = get_person_recognizer(mode="balanced")
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
            ],
            context=["company", "inc", "ltd", "corp", "limited", "firm", "business", "employer", "organization", "corporation", "enterprise", "client", "vendor"]
        )

        self.analyzer.registry.add_recognizer(company_recognizer)

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
        decimal_coords = r"\b-?\d{1,3}\.\d{3,8}°?\s*[NSEW]?\b"

        # Decimal degree pairs (latitude, longitude)
        decimal_pair = r"\b-?\d{1,3}\.\d{3,8}\s*,\s*-?\d{1,3}\.\d{3,8}\b"

        # Degrees, minutes, seconds: 40°42'46"N
        dms_coord = r"\b\d{1,3}°\s*\d{1,2}['\u2032]\s*\d{1,2}(?:\.\d+)?[\"″\u2033]?\s*[NSEW]\b"

        # Degrees decimal minutes: N 40° 42.767'
        ddm_coord = r"\b[NSEW]\s*\d{1,3}°\s*\d{1,2}\.\d+['\u2032]\b"

        # Full coordinate pair with directions
        coord_pair_directions = r"\b\d{1,3}°\s*\d{1,2}['\u2032]\s*\d{1,2}(?:\.\d+)?[\"″\u2033]?\s*[NS]\s*,?\s*\d{1,3}°\s*\d{1,2}['\u2032]\s*\d{1,2}(?:\.\d+)?[\"″\u2033]?\s*[EW]\b"

        # GPS format: GPS: 40.7128, -74.0060
        gps_labeled = r"\bGPS[:\s]+[-\d.,\s°NSEW]+\b"

        # Lat/Long labeled: Lat: 40.7128, Long: -74.0060
        latlong_labeled = r"\b(?:Lat(?:itude)?|Long(?:itude)?)[:\s]+-?\d{1,3}\.\d{3,8}\b"

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

        self.analyzer.registry.add_recognizer(na_phone_parens)
        self.analyzer.registry.add_recognizer(na_phone_dashes)
        self.analyzer.registry.add_recognizer(na_phone_dots)
        self.analyzer.registry.add_recognizer(intl_phone)
        self.analyzer.registry.add_recognizer(vanity_phone)

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

        # TODO: Add more recognizers (GitHub tokens, Google API keys, etc.)

        self.analyzer.registry.add_recognizer(aws_recognizer)
        self.analyzer.registry.add_recognizer(stripe_recognizer)

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

        self.analyzer.registry.add_recognizer(ssn_recognizer)

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

    def _validate_phone_with_phonenumbers(self, phone_text: str, default_region: str = "US") -> bool:
        """
        Validate a phone number using Google's libphonenumber library.

        Args:
            phone_text: The phone number text to validate
            default_region: Default region code for parsing (e.g., "US", "GB", "CA")

        Returns:
            True if the phone number is valid, False otherwise
        """
        if not PHONENUMBERS_AVAILABLE:
            return True  # Fall back to pattern-only validation if library unavailable

        try:
            # Try to parse the phone number
            parsed = phonenumbers.parse(phone_text, default_region)

            # Check if it's a valid number
            if not phonenumbers.is_valid_number(parsed):
                return False

            # Check if it's a possible number (less strict check)
            if not phonenumbers.is_possible_number(parsed):
                return False

            return True
        except NumberParseException:
            # If parsing fails, fall back to pattern validation
            return True  # Allow it through to pattern-based validation

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

            # Filter MEDICAL false positives (from name patterns)
            if entity.entity_type == "MEDICAL":
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
                if PHONENUMBERS_AVAILABLE and len(digits) >= 7:
                    region = self._get_phone_region_from_context(text, entity.start, entity.end)
                    if not self._validate_phone_with_phonenumbers(entity_text, region):
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

            # Filter LOCATION false positives
            if entity.entity_type == "LOCATION":
                # Skip very short text (less than 4 chars) - catches "in", "as", "WY", etc.
                if len(entity_text.strip()) < 4:
                    continue
                # Skip common short phrases that aren't locations
                location_short_false_positives = {
                    'in', 'at', 'to', 'on', 'of', 'as', 'by', 'or', 'an', 'is', 'it',
                    'claimed as', 'based on', 'delay in', 'delays in', 'org or',
                    'com or', 'net or', 'info', 'lakhs in', 'crores in',
                    'tds on', 'located in', 'situated at', 'found in',
                }
                if entity_text.lower().strip() in location_short_false_positives:
                    continue
                # Skip if it looks like a URL (contains http, www, or common TLDs with path)
                if re.match(r'^(?:https?://|www\.)', entity_text.lower()):
                    continue
                # Skip if it looks like a bare domain with path
                if re.match(r'^[\w-]+\.(?:com|org|net|edu|gov|io|co|de|uk|jp|cn|ru|br|in)/\S*', entity_text.lower()):
                    continue
                # Skip if confidence is low (increased threshold from 0.55 to 0.65)
                if entity.confidence < 0.65:
                    continue
                # Skip if text contains common non-address words (disclaimers, explanations)
                non_address_words = {
                    # Common function words that appear in non-address contexts
                    'connected', 'info', 'information', 'may be', 'network', 'billing', 'purposes',
                    'used for', 'this', 'that', 'which', 'help', 'your', 'business', 'please',
                    'contact', 'visit', 'report', 'report on', 'assessed', 'controls', 'performed',
                    'tests of', 'details on', 'timeliness', 'decision', 'include', 'assumptions',
                    'recognition', 'capitalization', 'assets', 'the following', 'in place',
                    'accordance', 'compliance', 'creating', 'beautiful', 'jewelry', 'spending',
                    'enjoy', 'love', 'sharing', 'knowledge', 'connecting', 'passionate', 'art form',
                    'also', 'family', 'exploring', 'places', 'learn', 'more about', 'feel free',
                    'website', 'studio', 'located at', 'remember', 'unique', 'challenging', 'project',
                    'work on', 'customer', 'approached', 'precious', 'heirloom', 'diamond', 'necklace',
                    'passed down', 'generations', 'unfortunately', 'condition', 'loose', 'broken',
                    'restore', 'glory', 'ordinary', 'repair', 'specialized', 'tools', 'techniques',
                    'delicate', 'task', 'dismantling', 'carefully', 'removed', 'setting', 'damaged',
                    'disassembled', 'meticulously', 'cleaned', 'inspected', 'damage', 'fortunately',
                    'cracks', 'chips', 'soldered', 'pieces', 'together', 'ensuring', 'sturdy', 'secure',
                    'repaired', 'process', 'reassembling', 'placed', 'polished', 'sparkled', 'presented',
                    'restored', 'overjoyed', 'believe', 'bring', 'life', 'looked', 'beautiful', 'created',
                    'thrilled', 'possession', 'discuss', 'forward', 'hearing', 'when', "i'm", 'not',
                    'about me', 'my name', 'i am', 'years of experience', 'last year',
                }
                text_lower = entity_text.lower()
                if any(word in text_lower for word in non_address_words):
                    continue
                # Skip single words that are likely city/country names used in non-address contexts
                words = entity_text.split()
                if len(words) == 1 and len(entity_text) < 15:
                    # Single word locations need high confidence
                    if entity.confidence < 0.75:
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
                # Require at least one address indicator for texts > 10 chars
                if len(entity_text) > 10:
                    address_indicators = [
                        r'^\d+\s+\w',  # Street number at start (123 Main)
                        r'\b(?:st|street|ave|avenue|rd|road|dr|drive|ln|lane|ct|court|blvd|boulevard|way|place|pl|circle|cir|pkwy|parkway|hwy|highway)\b',  # Street types
                        r'\b(?:apt|apartment|suite|ste|unit|floor|fl|bldg|building)\b',  # Unit indicators
                        r'\b[A-Z]{2}\s*\d{5}',  # State + ZIP (CA 90210)
                        r'\b\d{5}(?:-\d{4})?\b',  # ZIP code
                        r'\b[A-Z]\d[A-Z]\s*\d[A-Z]\d\b',  # Canadian postal code
                        r'\bP\.?O\.?\s*Box\b',  # PO Box
                        r'\b(?:rue|via|calle|strasse|platz|piazza|avenue|boulevard)\b',  # European street types
                    ]
                    has_address_indicator = any(re.search(pattern, entity_text, re.IGNORECASE) for pattern in address_indicators)
                    if not has_address_indicator:
                        continue
                # For longer addresses (>20 chars), require a street number at the start
                # to filter out descriptive phrases that look like addresses
                if len(entity_text) > 20:
                    has_street_number = re.match(r'^\d+\s+', entity_text)
                    has_po_box = re.search(r'\bP\.?O\.?\s*Box\b', entity_text, re.IGNORECASE)
                    has_zip = re.search(r'\b\d{5}(?:-\d{4})?\b', entity_text)
                    has_postal_code = re.search(r'\b[A-Z]\d[A-Z]\s*\d[A-Z]\d\b', entity_text)
                    if not (has_street_number or has_po_box or has_zip or has_postal_code):
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
                    'schedule', 'schedules', 'section', 'sections', 'part', 'parts',
                    'item', 'items', 'note', 'notes', 'page', 'pages', 'form', 'forms',
                    'table', 'tables', 'figure', 'figures', 'appendix', 'exhibit',
                    'attachment', 'attachments', 'document', 'documents', 'report', 'reports',
                }
                if entity_text.lower() in medical_false_positives:
                    continue

            # Filter PERSON false positives
            # Note: PERSON detection now uses context-aware patterns:
            # - Title + Name (Dr. Smith) - high confidence, always detected
            # - Labeled names (Name: John Smith) - require explicit label
            # This filter handles edge cases from spaCy NER
            if entity.entity_type == "PERSON":
                # Skip if contains newlines or excessive whitespace (OCR artifact)
                if '\n' in entity_text or '\r' in entity_text or '  ' in entity_text:
                    continue
                # Skip very short text (likely abbreviations)
                if len(entity_text) <= 3:
                    continue
                # Skip if contains numbers (names don't have digits)
                if any(c.isdigit() for c in entity_text):
                    continue
                # Skip very long text (likely phrases, not names)
                if len(entity_text) > 35:
                    continue
                # For spaCy NER detections (lower confidence), require higher threshold
                # Our pattern recognizers have scores 0.85-0.95
                if entity.confidence < 0.80:
                    continue

            # Filter COORDINATES false positives
            if entity.entity_type == "COORDINATES":
                # Skip if it looks like an IP address fragment
                if re.match(r'^\d{1,3}\.\d{1,3}$', entity_text):
                    continue

            # Filter COMPANY false positives
            if entity.entity_type == "COMPANY":
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
                # Skip if it looks like a descriptive phrase (contains common non-company words)
                company_phrase_indicators = [
                    'verified against', 'claims were', 'related to', 'purchased by',
                    'provided by', 'submitted by', 'reviewed by', 'approved by',
                    'has been', 'have been', 'was ', 'were ', 'will be',
                    'in accordance', 'in compliance', 'in relation', 'in connection',
                    # Accounting terms and partial sentences
                    'property, plant', 'plant and equipment', 'namely ', 'one branch',
                    'assets and', 'and liabilities', 'revenue and', 'income and',
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

            # Note: Currency amounts ARE now detected as FINANCIAL (user feedback)
            # Previously filtered out, but users want to redact financial data

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
