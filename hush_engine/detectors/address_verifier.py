#!/usr/bin/env python3
"""
LightGBM-based ADDRESS Verification

Uses trained LightGBM classifier to filter false positive ADDRESS detections.
Similar approach to COMPANY verification - uses LOCATION classifier to evaluate
whether a detected span is likely a real address.

IMPORTANT: The LightGBM model is loaded early in PersonRecognizer to avoid
OpenMP library conflicts with spaCy/presidio on macOS. This module uses the
pre-loaded model via get_address_lgbm_model().
"""

import logging
import re
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Try to import numpy (needed for predictions)
try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    np = None
    _NUMPY_AVAILABLE = False


# Model directory (for fallback path reference)
MODEL_DIR = Path(__file__).parent.parent / "models" / "lgbm"


def _get_preloaded_address_model():
    """Get pre-loaded ADDRESS model from lgbm_preloader.

    The model is loaded at import time in lgbm_preloader to avoid OpenMP conflicts.
    """
    try:
        from hush_engine.detectors.lgbm_preloader import get_address_model, is_preload_successful
        if is_preload_successful():
            return get_address_model()
    except ImportError:
        pass

    # Fallback: try PersonRecognizer (legacy path)
    try:
        from hush_engine.detectors.person_recognizer import (
            get_address_lgbm_model,
            is_address_lgbm_available,
        )
        if is_address_lgbm_available():
            return get_address_lgbm_model()
    except ImportError:
        pass

    return None

# Common false positive patterns for ADDRESS
ADDRESS_FALSE_POSITIVE_PATTERNS = frozenset({
    # Verb phrases
    'also run', 'based on', 'depends on', 'focus on', 'rely on',
    'built on', 'based in', 'located in', 'situated in', 'found in',
    'works at', 'lives at', 'resides in', 'travel to', 'went to',
    'send to', 'ship to', 'move to', 'going to', 'headed to',
    'came from', 'go to', 'come from', 'moved from', 'taken from',

    # Form labels
    'street address', 'home address', 'work address', 'mailing address',
    'billing address', 'shipping address', 'physical address',
    'address line', 'address 1', 'address 2', 'enter address',

    # Generic words
    'the road', 'the street', 'the way', 'the path', 'the building',
    'in the', 'at the', 'on the', 'to the', 'from the',

    # OCR artifacts and fragments
    'user id', 'tax id', 'post id', 'id or', 'email at',

    # Patterns from feedback analysis
    'building yall', 'political view', 'building plans',
})

# Minimum length for valid address (characters)
# Note: Keep this low to allow short addresses like "NY 10001" or "CA 90210"
MIN_ADDRESS_LENGTH = 6

# Words that when followed by non-address content indicate false positive
NON_ADDRESS_CONTEXT_WORDS = frozenset({
    'is', 'are', 'was', 'were', 'will', 'would', 'could', 'should',
    'can', 'may', 'might', 'must', 'have', 'has', 'had', 'do', 'does',
    'did', 'been', 'being', 'run', 'runs', 'running', 'view', 'views',
    'plan', 'plans', 'yall', 'y\'all', 'gonna', 'wanna',
})

# Street type indicators (strong address signal)
STREET_TYPES = frozenset({
    'street', 'st', 'avenue', 'ave', 'road', 'rd', 'drive', 'dr',
    'boulevard', 'blvd', 'lane', 'ln', 'way', 'court', 'ct', 'circle',
    'place', 'pl', 'terrace', 'ter', 'trail', 'highway', 'hwy',
    'parkway', 'pkwy', 'expressway', 'freeway', 'route', 'rte',
})

# Famous street names that don't need a suffix (e.g., "789 Broadway", "101 Main")
FAMOUS_STREET_NAMES = frozenset({
    'broadway', 'main', 'park', 'fifth', 'madison', 'lexington',
    'wall', 'market', 'king', 'queen', 'first', 'second', 'third',
    'fourth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth',
})

# Minimum address components for validation
REQUIRED_ADDRESS_COMPONENTS = 2  # Reduced from 3 for better recall

# Component weights for weighted scoring (replaces strict count)
# Higher weight = stronger address signal
ADDRESS_COMPONENT_WEIGHTS = {
    'house_number': 0.5,   # Strong signal - street numbers are definitive
    'postcode': 0.4,       # Strong signal - zip codes are definitive
    'road': 0.4,           # Strong signal - street names are key
    'city': 0.3,           # Medium signal - city names help
    'state': 0.2,          # Weaker signal - state alone is weak
    'country': 0.15,       # Weak signal - country alone is very weak
    'suburb': 0.25,        # Medium signal - similar to city
    'state_district': 0.15,
    'county': 0.2,
    'unit': 0.2,           # Unit numbers help confirm address
    'level': 0.15,
    'entrance': 0.1,
}

# Minimum cumulative weight to accept an address
MIN_ADDRESS_WEIGHT = 0.7  # e.g., house_number(0.5) + road(0.4) = 0.9 >= 0.7 passes

VALID_ADDRESS_COMPONENT_TYPES = frozenset({
    'house_number', 'road', 'city', 'postcode', 'state', 'country',
    'suburb', 'state_district', 'county', 'unit', 'level', 'entrance',
})

# Try to import libpostal for component validation
try:
    from postal.parser import parse_address as _parse_address_raw
    from functools import lru_cache

    @lru_cache(maxsize=2000)
    def parse_address(text: str):
        """Cached version of libpostal parse_address."""
        return _parse_address_raw(text)

    LIBPOSTAL_AVAILABLE = True
except ImportError:
    LIBPOSTAL_AVAILABLE = False

    def parse_address(text: str):
        """Stub when libpostal not available."""
        return []


def _is_valid_component_value(value: str, label: str) -> bool:
    """
    Check if a parsed component value is meaningful (not a common word or fragment).

    Args:
        value: The parsed component value
        label: The component type (house_number, road, city, etc.)

    Returns:
        True if the value appears to be a valid address component
    """
    value_lower = value.lower().strip()

    # Reject empty or very short values (except house_number which can be short)
    if label != 'house_number' and len(value_lower) < 2:
        return False

    # House numbers should be numeric or alphanumeric (e.g., "123", "221B")
    if label == 'house_number':
        if not re.search(r'\d', value):
            return False
        # Reject if it's just a common word
        if value_lower in NON_ADDRESS_CONTEXT_WORDS:
            return False

    # Postcodes should have at least one digit
    if label == 'postcode':
        if not re.search(r'\d', value):
            return False

    # Roads should not be just common words
    if label == 'road':
        words = value_lower.split()
        # Single word roads must end with street type or be proper nouns
        if len(words) == 1 and words[0] not in STREET_TYPES:
            # Allow if it looks like a street name (capitalized)
            if not value[0].isupper():
                return False
            # Reject common non-road words
            if value_lower in NON_ADDRESS_CONTEXT_WORDS:
                return False

    # Cities should be capitalized and not common words
    if label == 'city':
        if not value[0].isupper() and not value.isupper():
            return False
        if value_lower in NON_ADDRESS_CONTEXT_WORDS:
            return False

    return True


def validate_address_components(text: str) -> tuple[bool, float, int]:
    """
    Validate address using WEIGHTED component scoring instead of strict count.

    Uses weighted scoring where each component type has a weight:
    - house_number: 0.5 (strong signal)
    - postcode: 0.4 (strong signal)
    - road: 0.4 (strong signal)
    - city: 0.3 (medium signal)
    - state: 0.2 (weak signal)

    Accept if cumulative weight >= MIN_ADDRESS_WEIGHT (0.7).
    Example: "123 Main St" has house_number(0.5) + road(0.4) = 0.9 >= 0.7, passes.

    Also requires structural validation:
    - (house_number OR postcode) AND (road OR city OR state)

    Args:
        text: Address text to validate

    Returns:
        Tuple of (is_valid, confidence_multiplier, component_count)
        - is_valid: True if address has enough weighted components
        - confidence_multiplier: Multiplier to apply to confidence (1.0 = full, 0.0 = reject)
        - component_count: Number of valid components found
    """
    # Minimum length check - reject very short strings
    text_stripped = text.strip()
    if len(text_stripped) < MIN_ADDRESS_LENGTH:
        return False, 0.0, 0

    # Check for non-address context words at end of text
    words = text_stripped.lower().split()
    if words and words[-1] in NON_ADDRESS_CONTEXT_WORDS:
        return False, 0.0, 0

    # Check for pattern "number + non-address word" like "9083 is"
    if len(words) >= 2 and words[0].isdigit() and words[1] in NON_ADDRESS_CONTEXT_WORDS:
        return False, 0.0, 0

    if not LIBPOSTAL_AVAILABLE:
        # Can't validate without libpostal - return neutral
        return True, 1.0, 0

    try:
        components = parse_address(text)
    except Exception:
        return True, 1.0, 0

    # Count valid component types (unique) with VALUE validation
    valid_types = set()
    component_values = {}  # Store values for additional checks
    for value, label in components:
        if label in VALID_ADDRESS_COMPONENT_TYPES and value.strip():
            # Validate the component value, not just the type
            if _is_valid_component_value(value, label):
                valid_types.add(label)
                component_values[label] = value

    component_count = len(valid_types)

    # Calculate cumulative weight
    cumulative_weight = sum(
        ADDRESS_COMPONENT_WEIGHTS.get(comp_type, 0.1)
        for comp_type in valid_types
    )

    # Structural validation: require (house_number OR postcode) AND (road OR city OR state)
    has_number_component = 'house_number' in valid_types or 'postcode' in valid_types
    has_location_component = 'road' in valid_types or 'city' in valid_types or 'state' in valid_types

    # Primary check: weighted scoring with structural requirement
    if cumulative_weight >= MIN_ADDRESS_WEIGHT and has_number_component and has_location_component:
        # Strong weighted address - confidence based on weight
        if cumulative_weight >= 1.0:
            return True, 1.1, component_count  # Very strong (e.g., house+road+city)
        elif cumulative_weight >= 0.85:
            return True, 1.0, component_count  # Strong (e.g., house+road)
        else:
            return True, 0.9, component_count  # Acceptable
    elif cumulative_weight >= MIN_ADDRESS_WEIGHT:
        # Meets weight threshold but missing structural requirement
        # Still accept but with reduced confidence
        return True, 0.7, component_count
    elif cumulative_weight >= 0.5 and has_number_component and has_location_component:
        # Below weight threshold but has key structural components
        # Accept with reduced confidence (e.g., postcode + state = 0.6)
        return True, 0.65, component_count
    elif component_count >= REQUIRED_ADDRESS_COMPONENTS:
        # Fallback: 2+ components but low weight - accept with low confidence
        return True, 0.55, component_count
    elif component_count == 1 and cumulative_weight >= 0.4:
        # Single strong component (house_number or postcode) - marginal
        # Accept only if text has other address-like characteristics
        if has_number_component:
            return True, 0.5, component_count
        return False, 0.0, component_count
    else:
        # Insufficient weight and components - reject
        return False, 0.0, component_count


class AddressVerifier:
    """
    Verifies ADDRESS detections using LightGBM classifier.

    Uses the ADDRESS model to score address candidates
    and filter out false positives.

    NOTE: The LightGBM model is loaded early in PersonRecognizer to avoid
    OpenMP library conflicts with spaCy/presidio. This class uses the
    pre-loaded model via _get_preloaded_address_model().
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        threshold: float = 0.40,
        min_confidence_boost: float = 0.1,
    ):
        """
        Initialize the address verifier.

        Args:
            model_path: Path to the ADDRESS classifier model (unused - uses pre-loaded)
            threshold: Minimum LightGBM score to keep a detection
            min_confidence_boost: Minimum boost for high-confidence predictions
        """
        self.model_path = model_path or MODEL_DIR / "address_classifier.txt"
        self.threshold = threshold
        self.min_confidence_boost = min_confidence_boost
        self._model = None
        self._loaded = False

    @property
    def is_available(self) -> bool:
        """Check if verifier is available."""
        # Check for pre-loaded model first
        preloaded = _get_preloaded_address_model()
        if preloaded is not None:
            return _NUMPY_AVAILABLE

        # Fallback: check if model file exists (but loading may cause crash)
        return _NUMPY_AVAILABLE and self.model_path.exists()

    def load(self) -> bool:
        """Load the classifier model.

        Uses pre-loaded model from PersonRecognizer to avoid OpenMP conflicts.
        """
        if self._loaded:
            return True

        # Try to get pre-loaded model first (safe - avoids OpenMP conflict)
        preloaded = _get_preloaded_address_model()
        if preloaded is not None:
            self._model = preloaded
            self._loaded = True
            logger.info("Using pre-loaded ADDRESS classifier (OpenMP-safe)")
            return True

        # Fallback: model not pre-loaded
        # This path may cause segfaults on macOS if spaCy is already loaded
        logger.warning(
            "ADDRESS model not pre-loaded. Ensure PersonRecognizer is "
            "initialized before spaCy/presidio to avoid OpenMP conflicts."
        )
        return False

    def verify_address(
        self,
        text: str,
        address_text: str,
        start: int,
        end: int,
        original_confidence: float,
    ) -> Tuple[bool, float]:
        """
        Verify if an ADDRESS detection is likely valid.

        Args:
            text: Full document text
            address_text: The detected address text
            start: Start position in document
            end: End position in document
            original_confidence: Original detection confidence

        Returns:
            Tuple of (is_valid, adjusted_confidence)
        """
        address_lower = address_text.lower().strip()
        address_stripped = address_text.strip()

        # Skip known false positive patterns
        if address_lower in ADDRESS_FALSE_POSITIVE_PATTERNS:
            return False, 0.0

        # Check for sentence structure (not an address)
        if '. ' in address_text and address_text.count('.') > 1:
            return False, 0.0

        # Minimum length requirement
        if len(address_stripped) < MIN_ADDRESS_LENGTH:
            return False, 0.0

        words = address_lower.split()

        # Check for non-address patterns: "number + is/are/was/etc."
        if len(words) >= 2:
            first_word = words[0].rstrip('.,')
            second_word = words[1].rstrip('.,')
            # Pattern like "9083 is", "123 was", etc.
            if first_word.isdigit() and second_word in NON_ADDRESS_CONTEXT_WORDS:
                return False, 0.0
            # Pattern ending with non-address context word
            if words[-1].rstrip('.,') in NON_ADDRESS_CONTEXT_WORDS:
                return False, 0.0

        # Reject patterns with hash + alphanumeric that look like IDs (e.g., "#93208r")
        if re.match(r'^\s*#\d+[a-zA-Z]', address_text):
            return False, 0.0

        # Very short without street type - check for state+zip pattern or famous streets
        if len(address_stripped) < 10 and len(words) <= 2:
            has_street = any(w.rstrip('.,') in STREET_TYPES for w in words)
            has_famous = any(w.rstrip('.,') in FAMOUS_STREET_NAMES for w in words)
            # Allow state abbreviation + zip code patterns (e.g., "NY 10001", "CA 90210")
            has_state_zip = bool(re.match(r'^[A-Z]{2}\s+\d{5}', address_text.strip()))
            if not has_street and not has_famous and not has_state_zip:
                # Check for multi-word capitalized names (e.g., "Los Angeles", "New York")
                if len(words) == 2 and all(w[0].isupper() for w in address_text.split() if w):
                    pass  # Allow city names
                else:
                    return False, original_confidence * 0.5

        # Very long - likely a phrase, not address
        if len(address_stripped) > 100:
            return False, 0.0

        # Libpostal component validation (require 3+ components for high confidence)
        is_valid_components, component_multiplier, component_count = validate_address_components(address_text)
        if not is_valid_components:
            # No valid address components - reject
            return False, 0.0

        # Check for strong address indicators
        has_street_type = any(w.rstrip('.,') in STREET_TYPES for w in words)
        has_famous_street = any(w.rstrip('.,') in FAMOUS_STREET_NAMES for w in words)
        has_street_indicator = has_street_type or has_famous_street
        has_numbers = bool(re.search(r'\d', address_text))
        has_postal = bool(re.search(r'\b\d{5}(-\d{4})?\b|\b[A-Z]\d[A-Z]\s*\d[A-Z]\d\b', address_text, re.I))

        # Strong address pattern - boost confidence (apply component multiplier)
        if has_street_indicator and has_numbers:
            adjusted = min(1.0, original_confidence + 0.15) * component_multiplier
            return True, min(1.0, adjusted)

        if has_postal:
            adjusted = min(1.0, original_confidence + 0.20) * component_multiplier
            return True, min(1.0, adjusted)

        # If component validation passed with good confidence, accept without LightGBM
        # This improves recall for valid addresses that LightGBM might reject
        if component_multiplier >= 0.65:
            # Good component validation - accept with multiplier-adjusted confidence
            adjusted = original_confidence * component_multiplier
            return True, min(1.0, adjusted)

        # Use LightGBM for nuanced verification of marginal cases
        if not self._loaded and not self.load():
            return self._heuristic_verify(address_text, original_confidence, has_street_indicator, has_numbers, component_multiplier)

        try:
            lgbm_score = self._get_lgbm_score(text, address_text, start, end)

            if lgbm_score < self.threshold:
                # Below threshold - likely false positive
                # Allow if component validation passed OR has both street indicator and numbers
                if component_multiplier >= 0.5:
                    return True, original_confidence * component_multiplier
                if has_street_indicator and has_numbers:
                    return True, original_confidence * 0.7
                # Neither - filter it
                return False, 0.0

            # Above threshold - likely valid
            if lgbm_score > 0.6:
                adjusted = min(1.0, original_confidence + self.min_confidence_boost)
            else:
                adjusted = original_confidence

            return True, adjusted

        except Exception as e:
            logger.debug(f"LightGBM verification failed: {e}")
            return self._heuristic_verify(address_text, original_confidence, has_street_indicator, has_numbers, component_multiplier)

    def _get_lgbm_score(
        self,
        text: str,
        address_text: str,
        start: int,
        end: int
    ) -> float:
        """Get LightGBM score for an address span."""
        from .feature_extractor import (
            extract_features_with_context,
            tokenize,
            features_to_matrix,
            FEATURE_NAMES,
        )

        tokens = tokenize(text)
        if not tokens:
            return 0.5

        features_list = extract_features_with_context(text)
        feature_dicts = features_to_matrix(features_list)

        # Find tokens that overlap with the address span
        span_indices = []
        for i, (token, tok_start, tok_end) in enumerate(tokens):
            if tok_start < end and tok_end > start:
                span_indices.append(i)

        if not span_indices:
            return 0.5

        # Build feature matrix for span tokens
        X = np.zeros((len(span_indices), len(FEATURE_NAMES)))
        for idx, span_idx in enumerate(span_indices):
            feat_dict = feature_dicts[span_idx]
            for j, name in enumerate(FEATURE_NAMES):
                X[idx, j] = feat_dict.get(name, 0)

        # Predict
        probs = self._model.predict(X)

        return float(np.mean(probs))

    def _heuristic_verify(
        self,
        address_text: str,
        original_confidence: float,
        has_street_indicator: bool,
        has_numbers: bool,
        component_multiplier: float = 1.0
    ) -> Tuple[bool, float]:
        """Fallback heuristic verification - balanced for precision and recall."""
        words = address_text.split()
        address_lower = address_text.lower()

        # Check for non-address context words at end (strong false positive signal)
        word_list = [w.lower().rstrip('.,') for w in words]
        if word_list and word_list[-1] in NON_ADDRESS_CONTEXT_WORDS:
            return False, 0.0

        # Check for "number + non-address word" pattern like "9083 is"
        if len(word_list) >= 2 and word_list[0].isdigit() and word_list[1] in NON_ADDRESS_CONTEXT_WORDS:
            return False, 0.0

        # Minimum length check
        if len(address_text.strip()) < MIN_ADDRESS_LENGTH:
            return False, 0.0

        # If component validation passed with good confidence, trust it
        if component_multiplier >= 0.65:
            return True, original_confidence * component_multiplier

        # Has street indicator + numbers - likely valid
        if has_street_indicator and has_numbers:
            return True, original_confidence

        # Has just street indicator - reduce confidence slightly
        if has_street_indicator:
            return True, original_confidence * 0.90

        # Has just numbers with 2+ words - likely valid (e.g., "NY 10001", "CA 90210")
        if has_numbers and len(words) >= 2:
            return True, original_confidence * 0.85

        # If component multiplier indicates marginal address, accept with adjusted confidence
        if component_multiplier >= 0.5:
            return True, original_confidence * component_multiplier

        # Multi-word with capitalization and comma - might be valid location
        if len(words) >= 2 and ',' in address_text:
            cap_count = sum(1 for w in words if w and w[0].isupper())
            if cap_count >= 2:
                return True, original_confidence * 0.80

        # Multi-word with capitalization (3+ words, 2+ caps) - location phrase
        if len(words) >= 3:
            cap_count = sum(1 for w in words if w and w[0].isupper())
            if cap_count >= 2:
                return True, original_confidence * 0.75

        # Two word with both capitalized - might be city name (e.g., "Los Angeles")
        if len(words) == 2:
            if all(w[0].isupper() for w in words if w):
                return True, original_confidence * 0.70

        # Single word with no indicators - likely false positive
        if len(words) == 1:
            return False, 0.0

        # Default: reduce confidence for uncertain cases
        return True, original_confidence * 0.60


# Global singleton instance
_address_verifier: Optional[AddressVerifier] = None


def get_address_verifier() -> AddressVerifier:
    """Get the global address verifier instance."""
    global _address_verifier
    if _address_verifier is None:
        _address_verifier = AddressVerifier()
    return _address_verifier


def verify_address_detection(
    text: str,
    address_text: str,
    start: int,
    end: int,
    original_confidence: float,
) -> Tuple[bool, float]:
    """
    Convenience function to verify an ADDRESS detection.

    Args:
        text: Full document text
        address_text: The detected address
        start: Start position
        end: End position
        original_confidence: Original detection confidence

    Returns:
        Tuple of (is_valid, adjusted_confidence)
    """
    verifier = get_address_verifier()
    return verifier.verify_address(
        text, address_text, start, end, original_confidence
    )


def is_address_verifier_available() -> bool:
    """Check if address verifier is available."""
    verifier = get_address_verifier()
    return verifier.is_available
