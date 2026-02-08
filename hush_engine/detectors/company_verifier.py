#!/usr/bin/env python3
"""
LightGBM-based COMPANY Verification

Uses trained LightGBM classifier to filter false positive COMPANY detections.
The classifier evaluates whether a detected span is likely a real company name
based on token-level features and context.

Similar approach to PERSON verification, but optimized for company patterns.
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Try to import LightGBM
_lgbm = None
_LGBM_AVAILABLE = False

try:
    import lightgbm as lgbm
    _lgbm = lgbm
    _LGBM_AVAILABLE = True
except ImportError:
    logger.debug("LightGBM not installed for company verification")

# Try to import numpy
try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    np = None
    _NUMPY_AVAILABLE = False


# Model directory
MODEL_DIR = Path(__file__).parent.parent / "models" / "lgbm"

# Common false positive patterns for COMPANY
COMPANY_FALSE_POSITIVE_PATTERNS = frozenset({
    # UI/Navigation elements
    'expenses & bills', 'sales & get paid', 'reports & insights',
    'settings & preferences', 'help & support', 'billing & payments',
    'accounts & settings', 'profile & settings', 'search & filter',
    'import & export', 'copy & paste', 'terms & conditions',

    # Accounting/Business phrases
    'health & safety', 'profit & loss', 'assets & liabilities',
    'receivables & payables', 'revenue & expenses',
    'research & development', 'mergers & acquisitions',

    # Department names
    'human resources', 'customer service', 'technical support',
    'quality assurance', 'supply chain', 'business development',

    # Hyphenated adjectives
    'cross-verified', 'high-value', 'low-cost', 'self-employed',
    'well-known', 'non-compliance', 'tax-exempt', 'long-term',
    'real-time', 'full-time', 'year-end', 'pre-tax',
})

# Words that commonly appear in company names (corporate suffixes)
COMPANY_INDICATOR_WORDS = frozenset({
    'inc', 'inc.', 'corp', 'corp.', 'corporation', 'llc', 'ltd', 'ltd.',
    'limited', 'plc', 'company', 'co', 'co.', 'group', 'holdings',
    'partners', 'associates', 'enterprises', 'industries', 'international',
    'global', 'solutions', 'services', 'technologies', 'systems',
})

# Corporate suffixes for suffix validation (normalized)
CORPORATE_SUFFIXES = frozenset({
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'ltd', 'limited',
    'plc', 'gmbh', 'ag', 'sa', 'nv', 'bv', 'pty', 'co', 'company',
    'group', 'holdings', 'partners', 'associates', 'international', 'intl',
    'enterprises', 'industries', 'solutions', 'services', 'technologies',
})

# Common single English words that MUST have a corporate suffix to be valid
AMBIGUOUS_COMPANY_WORDS = frozenset({
    # Tech/brand terms often misdetected
    'apple', 'amazon', 'google', 'microsoft', 'facebook', 'twitter', 'instagram',
    'oracle', 'swift', 'python', 'java', 'ruby', 'rust', 'go', 'kotlin',
    'target', 'focus', 'impact', 'vision', 'summit', 'pioneer', 'frontier',
    'horizon', 'spectrum', 'nexus', 'apex', 'zenith', 'matrix', 'vector',
    'delta', 'omega', 'alpha', 'beta', 'sigma', 'gamma',
    # Common nouns that could be brands
    'pioneer', 'gateway', 'compass', 'bridge', 'catalyst', 'fusion', 'synergy',
    'velocity', 'quantum', 'vertex', 'prism', 'pulse', 'echo', 'wave', 'spark',
})


def has_corporate_suffix(text: str) -> bool:
    """Check if text contains a corporate suffix."""
    text_lower = text.lower()
    words = text_lower.split()

    # Check if any word is a corporate suffix
    for word in words:
        # Strip common punctuation
        word_clean = word.rstrip('.,;:')
        if word_clean in CORPORATE_SUFFIXES:
            return True

    return False


def has_corporate_suffix_in_context(company_text: str, full_text: str, window: int = 50) -> bool:
    """
    Check if corporate suffix exists within token window of company detection.

    Args:
        company_text: The detected company text
        full_text: Full document text
        window: Number of characters to search around detection

    Returns:
        True if corporate suffix found nearby
    """
    # First check if suffix is part of the detection itself
    if has_corporate_suffix(company_text):
        return True

    # Find position of company text in full text
    text_lower = company_text.lower()
    full_lower = full_text.lower()

    try:
        text_pos = full_lower.find(text_lower)
        if text_pos < 0:
            return False

        # Get surrounding context
        start = max(0, text_pos - window)
        end = min(len(full_text), text_pos + len(company_text) + window)
        surrounding = full_lower[start:end]

        # Check for corporate suffix in context
        for suffix in CORPORATE_SUFFIXES:
            if suffix in surrounding:
                return True

    except Exception:
        pass

    return False


def requires_suffix_validation(text: str) -> bool:
    """
    Check if a company detection needs suffix validation.

    Single ambiguous words that could be common English words
    require a corporate suffix nearby to be valid company names.
    """
    words = text.strip().lower().split()

    # Only single words need validation
    if len(words) != 1:
        return False

    # Check if it's an ambiguous word
    word = words[0].rstrip('.,;:')
    return word in AMBIGUOUS_COMPANY_WORDS


def validate_company_suffix(company_text: str, full_text: str) -> tuple[bool, float]:
    """
    Validate company detection based on suffix rules.

    Args:
        company_text: Detected company text
        full_text: Full document text

    Returns:
        Tuple of (is_valid, confidence_multiplier)
    """
    # Check if this detection requires suffix validation
    if not requires_suffix_validation(company_text):
        # Multi-word or non-ambiguous - allow with full confidence
        return True, 1.0

    # Single ambiguous word - must have suffix nearby
    if has_corporate_suffix_in_context(company_text, full_text):
        # Has suffix - valid company
        return True, 1.0
    else:
        # No suffix - likely false positive
        return False, 0.0


class CompanyVerifier:
    """
    Verifies COMPANY detections using LightGBM classifier.

    Uses the ORGANIZATION model to score company name candidates
    and filter out false positives.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        threshold: float = 0.4,  # Conservative - we want to filter FPs
        min_confidence_boost: float = 0.1,
    ):
        """
        Initialize the company verifier.

        Args:
            model_path: Path to the ORGANIZATION classifier model
            threshold: Minimum LightGBM score to keep a detection
            min_confidence_boost: Minimum boost for high-confidence predictions
        """
        self.model_path = model_path or MODEL_DIR / "organization_classifier.txt"
        self.threshold = threshold
        self.min_confidence_boost = min_confidence_boost
        self._model = None
        self._loaded = False

    @property
    def is_available(self) -> bool:
        """Check if verifier is available."""
        return _LGBM_AVAILABLE and _NUMPY_AVAILABLE and self.model_path.exists()

    def load(self) -> bool:
        """Load the classifier model."""
        if self._loaded:
            return True

        if not self.is_available:
            logger.debug("Company verifier not available")
            return False

        try:
            self._model = _lgbm.Booster(model_file=str(self.model_path))
            self._loaded = True
            logger.info("Loaded ORGANIZATION classifier for company verification")
            return True
        except Exception as e:
            logger.error(f"Failed to load company verifier: {e}")
            return False

    def verify_company(
        self,
        text: str,
        company_text: str,
        start: int,
        end: int,
        original_confidence: float,
    ) -> Tuple[bool, float]:
        """
        Verify if a COMPANY detection is likely valid.

        Args:
            text: Full document text
            company_text: The detected company name text
            start: Start position in document
            end: End position in document
            original_confidence: Original detection confidence

        Returns:
            Tuple of (is_valid, adjusted_confidence)
        """
        # Quick heuristic checks first
        company_lower = company_text.lower().strip()

        # Skip known false positive patterns
        if company_lower in COMPANY_FALSE_POSITIVE_PATTERNS:
            return False, 0.0

        # Very short names without corporate suffix are suspicious
        if len(company_text.strip()) < 4:
            if not any(word in company_lower for word in COMPANY_INDICATOR_WORDS):
                return False, original_confidence * 0.5

        # Very long "names" are likely phrases
        if len(company_text.strip()) > 60:
            return False, 0.0

        # Contains sentence structure indicators
        if '. ' in company_text or company_text.count('.') > 2:
            return False, 0.0

        # Suffix validation for ambiguous single-word company names
        # (e.g., "Apple" needs "Inc." nearby to be valid)
        is_valid_suffix, suffix_multiplier = validate_company_suffix(company_text, text)
        if not is_valid_suffix:
            return False, 0.0

        # Use LightGBM for more nuanced verification
        if not self._loaded and not self.load():
            # Fallback to heuristics only
            return self._heuristic_verify(company_text, original_confidence)

        try:
            lgbm_score = self._get_lgbm_score(text, company_text, start, end)

            # Combine original confidence with LightGBM score
            # LightGBM is used as a filter - low scores reject
            if lgbm_score < self.threshold:
                # Below threshold - likely false positive
                # But allow if has strong company indicators
                if self._has_strong_company_indicators(company_text):
                    return True, original_confidence * 0.8
                return False, 0.0

            # Above threshold - likely valid
            # Boost confidence for high LightGBM scores
            if lgbm_score > 0.7:
                adjusted = min(1.0, original_confidence + self.min_confidence_boost)
            else:
                adjusted = original_confidence

            return True, adjusted

        except Exception as e:
            logger.debug(f"LightGBM verification failed: {e}")
            return self._heuristic_verify(company_text, original_confidence)

    def _get_lgbm_score(
        self,
        text: str,
        company_text: str,
        start: int,
        end: int
    ) -> float:
        """
        Get LightGBM score for a company name span.

        Extracts features for tokens in the span and returns
        the average prediction probability.
        """
        from .feature_extractor import (
            extract_features_with_context,
            tokenize,
            features_to_matrix,
            FEATURE_NAMES,
        )

        # Tokenize and extract features
        tokens = tokenize(text)
        if not tokens:
            return 0.5

        features_list = extract_features_with_context(text)
        feature_dicts = features_to_matrix(features_list)

        # Find tokens that overlap with the company span
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

        # Return average score across tokens
        return float(np.mean(probs))

    def _heuristic_verify(
        self,
        company_text: str,
        original_confidence: float
    ) -> Tuple[bool, float]:
        """
        Fallback heuristic verification when LightGBM not available.
        """
        company_lower = company_text.lower()

        # Has corporate suffix - likely valid
        if self._has_strong_company_indicators(company_text):
            return True, original_confidence

        # Single word without suffix - suspicious
        if ' ' not in company_text.strip() and len(company_text) < 10:
            return True, original_confidence * 0.7

        # Multi-word capitalized - likely valid
        words = company_text.split()
        if len(words) >= 2 and all(w[0].isupper() for w in words if w):
            return True, original_confidence

        return True, original_confidence * 0.9

    def _has_strong_company_indicators(self, company_text: str) -> bool:
        """Check if text has strong company name indicators."""
        company_lower = company_text.lower()

        # Check for corporate suffixes
        for suffix in COMPANY_INDICATOR_WORDS:
            if company_lower.endswith(suffix) or f' {suffix}' in company_lower:
                return True

        # Check for known company patterns
        # "X & Y" pattern with capitalized words
        if re.match(r'^[A-Z][a-z]+\s*&\s*[A-Z][a-z]+', company_text):
            return True

        return False


# Global singleton instance
_company_verifier: Optional[CompanyVerifier] = None


def get_company_verifier() -> CompanyVerifier:
    """Get the global company verifier instance."""
    global _company_verifier
    if _company_verifier is None:
        _company_verifier = CompanyVerifier()
    return _company_verifier


def verify_company_detection(
    text: str,
    company_text: str,
    start: int,
    end: int,
    original_confidence: float,
) -> Tuple[bool, float]:
    """
    Convenience function to verify a COMPANY detection.

    Args:
        text: Full document text
        company_text: The detected company name
        start: Start position
        end: End position
        original_confidence: Original detection confidence

    Returns:
        Tuple of (is_valid, adjusted_confidence)
    """
    verifier = get_company_verifier()
    return verifier.verify_company(
        text, company_text, start, end, original_confidence
    )


def is_company_verifier_available() -> bool:
    """Check if company verifier is available."""
    verifier = get_company_verifier()
    return verifier.is_available
