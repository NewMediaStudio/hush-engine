#!/usr/bin/env python3
"""
Feature Extractor for Lightweight NER

Extracts numeric features from text tokens for use with gradient boosting
classifiers (LightGBM/XGBoost). This enables fast, low-memory NER without
heavy transformer models.

Privacy: This module does NOT store or transmit any PII.
All processing is 100% local.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Lazy-loaded database instances
_names_db = None
_cities_db = None
_countries_db = None
_companies_db = None


def _get_names_db():
    """Lazy load names database."""
    global _names_db
    if _names_db is None:
        try:
            from hush_engine.data.names_database import NamesDatabase
            _names_db = NamesDatabase()
        except ImportError:
            logger.warning("NamesDatabase not available")
            _names_db = False
    return _names_db if _names_db else None


def _get_cities_db():
    """Lazy load cities database."""
    global _cities_db
    if _cities_db is None:
        try:
            from hush_engine.data.cities_database import CitiesDatabase
            _cities_db = CitiesDatabase()
        except ImportError:
            logger.warning("CitiesDatabase not available")
            _cities_db = False
    return _cities_db if _cities_db else None


def _get_countries_db():
    """Lazy load countries database."""
    global _countries_db
    if _countries_db is None:
        try:
            from hush_engine.data.countries_database import CountriesDatabase
            _countries_db = CountriesDatabase()
        except ImportError:
            logger.warning("CountriesDatabase not available")
            _countries_db = False
    return _countries_db if _countries_db else None


def _get_companies_db():
    """Lazy load companies database."""
    global _companies_db
    if _companies_db is None:
        try:
            from hush_engine.data.companies_database import CompaniesDatabase
            _companies_db = CompaniesDatabase()
        except ImportError:
            logger.debug("CompaniesDatabase not available")
            _companies_db = False
    return _companies_db if _companies_db else None


# Common patterns
TITLE_PATTERN = re.compile(r'^(Mr|Mrs|Ms|Miss|Dr|Prof|Sir|Rev|Hon)\.?$', re.IGNORECASE)
EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
PHONE_PATTERN = re.compile(r'^[\d\-\(\)\s\+\.]{7,}$')
DATE_PATTERN = re.compile(r'^\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}$')
URL_PATTERN = re.compile(r'^https?://|^www\.|\.com$|\.org$|\.net$', re.IGNORECASE)
CURRENCY_PATTERN = re.compile(r'^[$€£¥₹]?\d+[,.]?\d*$')

# Word shape patterns
SHAPE_PATTERNS = {
    'Xxxxx': re.compile(r'^[A-Z][a-z]+$'),           # Capitalized word
    'XXXXX': re.compile(r'^[A-Z]+$'),                # All caps
    'xxxxx': re.compile(r'^[a-z]+$'),                # All lowercase
    'Xx.': re.compile(r'^[A-Z][a-z]?\.$'),           # Initial (J. or Jr.)
    '99999': re.compile(r'^\d+$'),                   # All digits
    '99-99': re.compile(r'^\d+[\-/\.]\d+'),          # Date-like
    'Xxxxx-Xxxxx': re.compile(r'^[A-Z][a-z]+-[A-Z][a-z]+$'),  # Hyphenated name
}

# Common English stop words
STOP_WORDS = frozenset({
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
    'this', 'that', 'these', 'those', 'it', 'its', 'i', 'you', 'he',
    'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
    'his', 'our', 'their', 'what', 'which', 'who', 'whom', 'when', 'where',
    'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
    'so', 'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there',
})

# Context indicators for entity types
PERSON_CONTEXT = frozenset({
    'name', 'patient', 'client', 'customer', 'employee', 'dr', 'mr', 'mrs',
    'ms', 'miss', 'prof', 'dear', 'hi', 'hello', 'sincerely', 'regards',
    'signed', 'by', 'from', 'to', 'cc', 'attention', 'attn', 'contact',
})

LOCATION_CONTEXT = frozenset({
    'address', 'street', 'st', 'ave', 'avenue', 'road', 'rd', 'blvd',
    'drive', 'dr', 'lane', 'ln', 'city', 'state', 'country', 'zip',
    'postal', 'located', 'location', 'from', 'to', 'in', 'at', 'near',
})

ORG_CONTEXT = frozenset({
    'inc', 'corp', 'corporation', 'llc', 'ltd', 'company', 'co', 'org',
    'organization', 'institute', 'university', 'college', 'school',
    'hospital', 'bank', 'group', 'foundation', 'association', 'agency',
})

# Company suffixes (for detecting end of company names)
COMPANY_SUFFIXES = frozenset({
    'inc', 'inc.', 'corp', 'corp.', 'corporation', 'llc', 'llc.',
    'ltd', 'ltd.', 'limited', 'plc', 'plc.', 'co', 'co.',
    'company', 'companies', 'group', 'holdings', 'partners',
    'associates', 'enterprises', 'industries', 'international',
    'gmbh', 'ag', 's.a.', 'sa', 'nv', 'bv', 'pty',
})


@dataclass
class TokenFeatures:
    """Features extracted from a single token."""
    token: str
    position: int

    # Character-level features
    is_capitalized: bool
    is_all_caps: bool
    is_all_lower: bool
    has_digits: bool
    has_punctuation: bool
    has_hyphen: bool
    char_length: int
    vowel_ratio: float

    # Word shape
    word_shape: str
    prefix_3: str
    suffix_3: str

    # Dictionary features
    is_first_name: bool
    is_last_name: bool
    is_any_name: bool
    is_city: bool
    is_country: bool
    is_stop_word: bool
    is_company_suffix: bool
    is_known_company: bool

    # Pattern features
    is_title: bool
    looks_like_email: bool
    looks_like_phone: bool
    looks_like_date: bool
    looks_like_url: bool
    looks_like_currency: bool

    # Context features (filled by extract_features_with_context)
    prev_is_title: bool = False
    prev_is_capitalized: bool = False
    prev_token_lower: str = ""
    next_is_capitalized: bool = False
    next_token_lower: str = ""
    has_person_context: bool = False
    has_location_context: bool = False
    has_org_context: bool = False
    sentence_position: str = "middle"  # "start", "middle", "end"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for model input."""
        return {
            'is_capitalized': int(self.is_capitalized),
            'is_all_caps': int(self.is_all_caps),
            'is_all_lower': int(self.is_all_lower),
            'has_digits': int(self.has_digits),
            'has_punctuation': int(self.has_punctuation),
            'has_hyphen': int(self.has_hyphen),
            'char_length': self.char_length,
            'vowel_ratio': self.vowel_ratio,
            'word_shape_Xxxxx': int(self.word_shape == 'Xxxxx'),
            'word_shape_XXXXX': int(self.word_shape == 'XXXXX'),
            'word_shape_xxxxx': int(self.word_shape == 'xxxxx'),
            'word_shape_digits': int(self.word_shape == '99999'),
            'is_first_name': int(self.is_first_name),
            'is_last_name': int(self.is_last_name),
            'is_any_name': int(self.is_any_name),
            'is_city': int(self.is_city),
            'is_country': int(self.is_country),
            'is_stop_word': int(self.is_stop_word),
            'is_company_suffix': int(self.is_company_suffix),
            'is_known_company': int(self.is_known_company),
            'is_title': int(self.is_title),
            'looks_like_email': int(self.looks_like_email),
            'looks_like_phone': int(self.looks_like_phone),
            'looks_like_date': int(self.looks_like_date),
            'looks_like_url': int(self.looks_like_url),
            'looks_like_currency': int(self.looks_like_currency),
            'prev_is_title': int(self.prev_is_title),
            'prev_is_capitalized': int(self.prev_is_capitalized),
            'next_is_capitalized': int(self.next_is_capitalized),
            'has_person_context': int(self.has_person_context),
            'has_location_context': int(self.has_location_context),
            'has_org_context': int(self.has_org_context),
            'position_start': int(self.sentence_position == 'start'),
            'position_end': int(self.sentence_position == 'end'),
        }


def _get_word_shape(token: str) -> str:
    """Determine the shape pattern of a token."""
    for shape_name, pattern in SHAPE_PATTERNS.items():
        if pattern.match(token):
            return shape_name
    # Default: mixed
    return 'mixed'


def _count_vowels(token: str) -> float:
    """Calculate vowel ratio in token."""
    if not token:
        return 0.0
    vowels = sum(1 for c in token.lower() if c in 'aeiou')
    return vowels / len(token)


def extract_token_features(token: str, position: int = 0) -> TokenFeatures:
    """
    Extract features from a single token.

    Args:
        token: The text token to analyze
        position: Position in the sequence (0-indexed)

    Returns:
        TokenFeatures dataclass with all extracted features
    """
    token_lower = token.lower()
    token_stripped = token.strip()

    # Character-level features
    is_capitalized = len(token_stripped) > 0 and token_stripped[0].isupper()
    is_all_caps = token_stripped.isupper() and len(token_stripped) > 1
    is_all_lower = token_stripped.islower()
    has_digits = any(c.isdigit() for c in token)
    has_punctuation = any(c in '.,;:!?()[]{}' for c in token)
    has_hyphen = '-' in token

    # Dictionary lookups
    names_db = _get_names_db()
    cities_db = _get_cities_db()
    countries_db = _get_countries_db()
    companies_db = _get_companies_db()

    is_first_name = names_db.is_first_name(token_lower) if names_db else False
    is_last_name = names_db.is_last_name(token_lower) if names_db else False
    is_any_name = names_db.is_name(token_lower) if names_db else False
    is_city = cities_db.is_city(token_lower) if cities_db else False
    is_country = countries_db.is_country(token_lower) if countries_db else False
    is_company_suffix = token_lower.rstrip('.') in COMPANY_SUFFIXES
    is_known_company = companies_db.is_company(token_lower) if companies_db else False

    # Pattern matching
    is_title = bool(TITLE_PATTERN.match(token_stripped))
    looks_like_email = bool(EMAIL_PATTERN.match(token_stripped))
    looks_like_phone = bool(PHONE_PATTERN.match(token_stripped))
    looks_like_date = bool(DATE_PATTERN.match(token_stripped))
    looks_like_url = bool(URL_PATTERN.search(token_stripped))
    looks_like_currency = bool(CURRENCY_PATTERN.match(token_stripped))

    return TokenFeatures(
        token=token,
        position=position,
        is_capitalized=is_capitalized,
        is_all_caps=is_all_caps,
        is_all_lower=is_all_lower,
        has_digits=has_digits,
        has_punctuation=has_punctuation,
        has_hyphen=has_hyphen,
        char_length=len(token_stripped),
        vowel_ratio=_count_vowels(token_stripped),
        word_shape=_get_word_shape(token_stripped),
        prefix_3=token_lower[:3] if len(token_lower) >= 3 else token_lower,
        suffix_3=token_lower[-3:] if len(token_lower) >= 3 else token_lower,
        is_first_name=is_first_name,
        is_last_name=is_last_name,
        is_any_name=is_any_name,
        is_city=is_city,
        is_country=is_country,
        is_stop_word=token_lower in STOP_WORDS,
        is_company_suffix=is_company_suffix,
        is_known_company=is_known_company,
        is_title=is_title,
        looks_like_email=looks_like_email,
        looks_like_phone=looks_like_phone,
        looks_like_date=looks_like_date,
        looks_like_url=looks_like_url,
        looks_like_currency=looks_like_currency,
    )


def tokenize(text: str) -> List[Tuple[str, int, int]]:
    """
    Simple whitespace tokenizer with character offsets.

    Args:
        text: Input text

    Returns:
        List of (token, start_offset, end_offset) tuples
    """
    tokens = []
    current_pos = 0

    for match in re.finditer(r'\S+', text):
        token = match.group()
        start = match.start()
        end = match.end()
        tokens.append((token, start, end))

    return tokens


def extract_features_with_context(
    text: str,
    window_size: int = 2
) -> List[TokenFeatures]:
    """
    Extract features for all tokens in text, including context features.

    Args:
        text: Input text to process
        window_size: Number of tokens to consider for context (default: 2)

    Returns:
        List of TokenFeatures for each token
    """
    tokens = tokenize(text)
    if not tokens:
        return []

    # First pass: extract basic features
    features_list = []
    for i, (token, start, end) in enumerate(tokens):
        features = extract_token_features(token, i)
        features_list.append(features)

    # Second pass: add context features
    for i, features in enumerate(features_list):
        # Previous token context
        if i > 0:
            prev = features_list[i - 1]
            features.prev_is_title = prev.is_title
            features.prev_is_capitalized = prev.is_capitalized
            features.prev_token_lower = prev.token.lower()

        # Next token context
        if i < len(features_list) - 1:
            next_f = features_list[i + 1]
            features.next_is_capitalized = next_f.is_capitalized
            features.next_token_lower = next_f.token.lower()

        # Check context window for entity type indicators
        context_tokens = set()
        for j in range(max(0, i - window_size), min(len(features_list), i + window_size + 1)):
            if j != i:
                context_tokens.add(features_list[j].token.lower())

        features.has_person_context = bool(context_tokens & PERSON_CONTEXT)
        features.has_location_context = bool(context_tokens & LOCATION_CONTEXT)
        features.has_org_context = bool(context_tokens & ORG_CONTEXT)

        # Sentence position (simplified)
        if i == 0:
            features.sentence_position = "start"
        elif i == len(features_list) - 1:
            features.sentence_position = "end"
        else:
            features.sentence_position = "middle"

    return features_list


def features_to_matrix(features_list: List[TokenFeatures]) -> List[Dict[str, Any]]:
    """
    Convert list of TokenFeatures to list of feature dictionaries.

    Suitable for creating a DataFrame or numpy array for model input.

    Args:
        features_list: List of TokenFeatures

    Returns:
        List of feature dictionaries
    """
    return [f.to_dict() for f in features_list]


def extract_ngram_features(
    tokens: List[str],
    position: int,
    n: int = 2
) -> Dict[str, bool]:
    """
    Extract n-gram features around a position.

    Args:
        tokens: List of tokens
        position: Current token position
        n: N-gram size (default: 2 for bigrams)

    Returns:
        Dictionary of n-gram presence features
    """
    features = {}

    # Get surrounding tokens
    start = max(0, position - n + 1)
    end = min(len(tokens), position + n)

    # Create n-grams
    for i in range(start, end - n + 1):
        ngram = "_".join(t.lower() for t in tokens[i:i + n])
        features[f"ngram_{ngram}"] = True

    return features


# Precomputed feature names for model compatibility
FEATURE_NAMES = [
    'is_capitalized', 'is_all_caps', 'is_all_lower', 'has_digits',
    'has_punctuation', 'has_hyphen', 'char_length', 'vowel_ratio',
    'word_shape_Xxxxx', 'word_shape_XXXXX', 'word_shape_xxxxx', 'word_shape_digits',
    'is_first_name', 'is_last_name', 'is_any_name', 'is_city', 'is_country',
    'is_stop_word', 'is_company_suffix', 'is_known_company',
    'is_title', 'looks_like_email', 'looks_like_phone',
    'looks_like_date', 'looks_like_url', 'looks_like_currency',
    'prev_is_title', 'prev_is_capitalized', 'next_is_capitalized',
    'has_person_context', 'has_location_context', 'has_org_context',
    'position_start', 'position_end',
]
