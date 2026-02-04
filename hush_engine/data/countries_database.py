"""
Countries database for LOCATION detection.

Provides a comprehensive list of country names and variations for
detecting country mentions in text as LOCATION entities.
"""

from typing import Set, Optional, Tuple

# All countries with common variations and abbreviations
COUNTRIES = {
    # A
    "afghanistan", "albania", "algeria", "andorra", "angola", "antigua", "argentina",
    "armenia", "australia", "austria", "azerbaijan",
    # B
    "bahamas", "bahrain", "bangladesh", "barbados", "belarus", "belgium", "belize",
    "benin", "bhutan", "bolivia", "bosnia", "botswana", "brazil", "brunei", "bulgaria",
    "burkina faso", "burundi",
    # C
    "cambodia", "cameroon", "canada", "cape verde", "central african republic", "chad",
    "chile", "china", "colombia", "comoros", "congo", "costa rica", "croatia", "cuba",
    "cyprus", "czechia", "czech republic",
    # D
    "denmark", "djibouti", "dominica", "dominican republic",
    # E
    "ecuador", "egypt", "el salvador", "equatorial guinea", "eritrea", "estonia",
    "eswatini", "ethiopia",
    # F
    "fiji", "finland", "france",
    # G
    "gabon", "gambia", "georgia", "germany", "ghana", "greece", "grenada", "guatemala",
    "guinea", "guinea-bissau", "guyana",
    # H
    "haiti", "honduras", "hungary",
    # I
    "iceland", "india", "indonesia", "iran", "iraq", "ireland", "israel", "italy",
    "ivory coast",
    # J
    "jamaica", "japan", "jordan",
    # K
    "kazakhstan", "kenya", "kiribati", "kosovo", "kuwait", "kyrgyzstan",
    # L
    "laos", "latvia", "lebanon", "lesotho", "liberia", "libya", "liechtenstein",
    "lithuania", "luxembourg",
    # M
    "madagascar", "malawi", "malaysia", "maldives", "mali", "malta", "marshall islands",
    "mauritania", "mauritius", "mexico", "micronesia", "moldova", "monaco", "mongolia",
    "montenegro", "morocco", "mozambique", "myanmar",
    # N
    "namibia", "nauru", "nepal", "netherlands", "new zealand", "nicaragua", "niger",
    "nigeria", "north korea", "north macedonia", "norway",
    # O
    "oman",
    # P
    "pakistan", "palau", "palestine", "panama", "papua new guinea", "paraguay", "peru",
    "philippines", "poland", "portugal",
    # Q
    "qatar",
    # R
    "romania", "russia", "rwanda",
    # S
    "saint kitts", "saint lucia", "saint vincent", "samoa", "san marino",
    "sao tome", "saudi arabia", "senegal", "serbia", "seychelles", "sierra leone",
    "singapore", "slovakia", "slovenia", "solomon islands", "somalia", "south africa",
    "south korea", "south sudan", "spain", "sri lanka", "sudan", "suriname", "sweden",
    "switzerland", "syria",
    # T
    "taiwan", "tajikistan", "tanzania", "thailand", "timor-leste", "togo", "tonga",
    "trinidad", "tunisia", "turkey", "turkmenistan", "tuvalu",
    # U
    "uganda", "ukraine", "united arab emirates", "united kingdom", "united states",
    "uruguay", "uzbekistan",
    # V
    "vanuatu", "vatican", "venezuela", "vietnam",
    # Y
    "yemen",
    # Z
    "zambia", "zimbabwe",
}

# Common abbreviations and alternate names
COUNTRY_ABBREVIATIONS = {
    # Common abbreviations
    "usa": "united states",
    "us": "united states",
    "u.s.": "united states",
    "u.s.a.": "united states",
    "uk": "united kingdom",
    "u.k.": "united kingdom",
    "uae": "united arab emirates",
    "u.a.e.": "united arab emirates",
    "prc": "china",
    "roc": "taiwan",
    "dprk": "north korea",
    "rok": "south korea",
    "ussr": "russia",

    # Full names that map to short names in COUNTRIES
    "united states of america": "united states",
    "great britain": "united kingdom",
    "england": "united kingdom",
    "scotland": "united kingdom",
    "wales": "united kingdom",
    "northern ireland": "united kingdom",
    "holland": "netherlands",
    "republic of ireland": "ireland",
    "eire": "ireland",
    "ivory coast": "ivory coast",
    "cote d'ivoire": "ivory coast",
    "burma": "myanmar",
    "persia": "iran",
    "siam": "thailand",
    "formosa": "taiwan",
    "zaire": "congo",
    "ceylon": "sri lanka",
    "rhodesia": "zimbabwe",
    "abyssinia": "ethiopia",
}

# Nationalities/demonyms (optional, for context boosting)
NATIONALITIES = {
    "american", "british", "canadian", "australian", "french", "german", "italian",
    "spanish", "portuguese", "dutch", "belgian", "swiss", "austrian", "polish",
    "russian", "ukrainian", "chinese", "japanese", "korean", "indian", "pakistani",
    "bangladeshi", "thai", "vietnamese", "indonesian", "malaysian", "filipino",
    "singaporean", "brazilian", "mexican", "argentinian", "colombian", "peruvian",
    "chilean", "venezuelan", "cuban", "jamaican", "egyptian", "moroccan", "algerian",
    "tunisian", "libyan", "south african", "nigerian", "kenyan", "ethiopian",
    "ghanaian", "tanzanian", "ugandan", "congolese", "cameroonian", "senegalese",
    "turkish", "iranian", "iraqi", "saudi", "emirati", "qatari", "kuwaiti",
    "israeli", "lebanese", "syrian", "jordanian", "greek", "swedish", "norwegian",
    "danish", "finnish", "icelandic", "irish", "scottish", "welsh", "new zealander",
}


class CountriesDatabase:
    """Database for country name lookups."""

    _instance = None

    def __new__(cls):
        """Singleton pattern for memory efficiency."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Build lookup sets
        self._countries: Set[str] = set(COUNTRIES)
        self._abbreviations: dict = dict(COUNTRY_ABBREVIATIONS)
        self._nationalities: Set[str] = set(NATIONALITIES)

        # Add uppercase versions
        self._countries_upper: Set[str] = {c.upper() for c in self._countries}
        self._abbreviations_upper: dict = {k.upper(): v for k, v in self._abbreviations.items()}

        self._initialized = True

    def is_country(self, text: str) -> bool:
        """Check if text is a country name."""
        text_lower = text.lower().strip()
        text_upper = text.upper().strip()

        # Direct match
        if text_lower in self._countries:
            return True

        # Abbreviation match
        if text_lower in self._abbreviations or text_upper in self._abbreviations_upper:
            return True

        return False

    def is_nationality(self, text: str) -> bool:
        """Check if text is a nationality/demonym."""
        return text.lower().strip() in self._nationalities

    def normalize_country(self, text: str) -> Optional[str]:
        """Normalize country name to standard form."""
        text_lower = text.lower().strip()
        text_upper = text.upper().strip()

        if text_lower in self._countries:
            return text_lower

        if text_lower in self._abbreviations:
            return self._abbreviations[text_lower]

        if text_upper in self._abbreviations_upper:
            return self._abbreviations_upper[text_upper]

        return None

    def get_confidence(self, text: str) -> float:
        """Get confidence score for country match."""
        text_lower = text.lower().strip()
        text_upper = text.upper().strip()

        # Exact country name match - high confidence
        if text_lower in self._countries:
            return 0.90

        # Abbreviation - slightly lower (could be ambiguous)
        if text_lower in self._abbreviations or text_upper in self._abbreviations_upper:
            # Common abbreviations like USA, UK get high confidence
            if text_upper in {"USA", "UK", "UAE", "US", "U.S.", "U.S.A.", "U.K."}:
                return 0.90
            return 0.80

        return 0.0

    def find_countries_in_text(self, text: str) -> list:
        """Find all country mentions in text.

        Returns:
            List of (country_name, start, end, confidence) tuples
        """
        import re

        results = []
        text_lower = text.lower()

        # Check for multi-word countries first (longer matches take priority)
        multi_word_countries = [c for c in self._countries if ' ' in c or '-' in c]
        for country in sorted(multi_word_countries, key=len, reverse=True):
            pattern = re.compile(rf'\b{re.escape(country)}\b', re.IGNORECASE)
            for match in pattern.finditer(text):
                results.append((match.group(), match.start(), match.end(), 0.90))

        # Check for single-word countries
        single_word_countries = [c for c in self._countries if ' ' not in c and '-' not in c]
        word_pattern = re.compile(r'\b([A-Za-z]+)\b')

        for match in word_pattern.finditer(text):
            word = match.group(1).lower()
            if word in single_word_countries:
                # Check not already covered by multi-word match
                start, end = match.start(), match.end()
                if not any(r[1] <= start < r[2] or r[1] < end <= r[2] for r in results):
                    results.append((match.group(1), start, end, 0.90))

        # Check for abbreviations (case-sensitive for uppercase abbreviations)
        for abbrev in self._abbreviations:
            if len(abbrev) <= 4:  # Short abbreviations like USA, UK
                pattern = re.compile(rf'\b{re.escape(abbrev.upper())}\b')
                for match in pattern.finditer(text):
                    start, end = match.start(), match.end()
                    if not any(r[1] <= start < r[2] or r[1] < end <= r[2] for r in results):
                        results.append((match.group(), start, end, 0.85))

        return results

    def stats(self) -> dict:
        """Return database statistics."""
        return {
            "countries": len(self._countries),
            "abbreviations": len(self._abbreviations),
            "nationalities": len(self._nationalities),
        }


# Singleton instance
_db: Optional[CountriesDatabase] = None


def get_countries_database() -> CountriesDatabase:
    """Get the singleton CountriesDatabase instance."""
    global _db
    if _db is None:
        _db = CountriesDatabase()
    return _db
