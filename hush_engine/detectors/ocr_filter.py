"""
OCR Artifact Filtering using LARVPC Rules

Filters garbage text from OCR output that triggers false positives.
LARVPC = Length, Alphanumeric, Ratio, Vowel, Punctuation, Complexity

These rules target common OCR artifacts:
- Corrupted text with low alphanumeric density
- Consonant-heavy strings (OCR misreads)
- Excessive punctuation (background noise)
- IP address patterns detected as entities
"""

import re
from typing import Set

# Common IP address pattern (avoid detecting as COMPANY/PERSON)
IP_PATTERN = re.compile(r'^IPV[46]?[-_]?\d{1,3}\.', re.IGNORECASE)

# Common single English words that shouldn't be COMPANY names
COMMON_ENGLISH_WORDS: Set[str] = {
    # Tech/brand terms often misdetected
    'apple', 'amazon', 'google', 'microsoft', 'facebook', 'twitter', 'instagram',
    'oracle', 'swift', 'python', 'java', 'ruby', 'rust', 'go', 'kotlin',
    # Common nouns
    'target', 'focus', 'impact', 'vision', 'summit', 'pioneer', 'frontier',
    'horizon', 'spectrum', 'nexus', 'apex', 'zenith', 'matrix', 'vector',
    # Business terms
    'customer', 'service', 'support', 'sales', 'marketing', 'finance',
    'operations', 'strategy', 'growth', 'innovation', 'solutions',
    # UI/Navigation
    'menu', 'dashboard', 'overview', 'settings', 'profile', 'home',
    'search', 'filter', 'sort', 'export', 'import', 'save', 'cancel',
}


def is_ocr_garbage(text: str) -> bool:
    """
    LARVPC rules to filter OCR artifacts.

    Returns True if text appears to be OCR garbage that should be rejected.

    Rules:
    - L: Length < 2 characters
    - A: Alphanumeric density < 50%
    - R: (reserved for future use)
    - V: Vowel ratio < 10% for alphabetic strings
    - P: More than 2 distinct punctuation types
    - C: (reserved for complexity checks)
    """
    if not text:
        return True

    text = text.strip()

    # Rule L: Length check
    if len(text) < 2:
        return True

    # Rule A: Alphanumeric density < 50%
    alphanumeric = sum(c.isalnum() for c in text)
    if alphanumeric / len(text) < 0.5:
        return True

    # Rule V: Vowel ratio < 10% for alphabetic strings
    alpha_chars = [c for c in text if c.isalpha()]
    if len(alpha_chars) >= 3:  # Only check strings with 3+ letters
        vowels = sum(c.lower() in 'aeiou' for c in alpha_chars)
        vowel_ratio = vowels / len(alpha_chars)
        if vowel_ratio < 0.10:
            return True

    # Rule P: More than 2 distinct punctuation types
    punct_types = set(c for c in text if not c.isalnum() and not c.isspace())
    if len(punct_types) > 2:
        return True

    return False


def is_ip_address_artifact(text: str) -> bool:
    """
    Check if text looks like an IP address artifact.

    These often appear as "IPV4_192.168.1.1" or "IPv4:10.0.0.1"
    and trigger false COMPANY/LOCATION detections.
    """
    return bool(IP_PATTERN.match(text.strip()))


def is_single_common_word(text: str) -> bool:
    """
    Check if text is a single common English word.

    Single common words like "Apple", "Target" need additional
    context (like "Inc.", "Corp.") to be valid company names.
    """
    words = text.strip().lower().split()
    return len(words) == 1 and words[0] in COMMON_ENGLISH_WORDS


def filter_ocr_artifacts(text: str, entity_type: str) -> tuple[bool, str]:
    """
    Filter OCR artifacts for specific entity types.

    Args:
        text: Detected text
        entity_type: Entity type (COMPANY, PERSON, ADDRESS, etc.)

    Returns:
        (should_keep, reason) - Whether to keep the detection and why
    """
    text_stripped = text.strip()

    # General LARVPC check
    if is_ocr_garbage(text_stripped):
        return False, "ocr_garbage"

    # IP address check for COMPANY/PERSON
    if entity_type in {'COMPANY', 'PERSON', 'LOCATION'}:
        if is_ip_address_artifact(text_stripped):
            return False, "ip_artifact"

    # Single common word check for COMPANY
    if entity_type == 'COMPANY':
        if is_single_common_word(text_stripped):
            return False, "common_word_needs_suffix"

    return True, "ok"


# For backwards compatibility with existing code
def is_likely_ocr_noise(text: str) -> bool:
    """Alias for is_ocr_garbage for compatibility."""
    return is_ocr_garbage(text)
