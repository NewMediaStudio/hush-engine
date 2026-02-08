"""
Rule-based heuristic verification for high-precision PII filtering.

Zero-dependency approach using pattern matching and linguistic rules
to filter false positives for uncertain detections.

Strategy:
- Only verifies entities with uncertain confidence (0.45-0.80)
- Targets high-FP entity types: ADDRESS, COMPANY, LOCATION, ID
- Uses pattern matching, capitalization, and structural checks
"""

import re
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class VerificationResult:
    """Result of heuristic verification."""
    is_pii: bool
    confidence_adjustment: float  # Positive = boost, negative = reduce
    reason: str


# Entity types that benefit from verification
VERIFIABLE_ENTITIES = {"ADDRESS", "COMPANY", "LOCATION", "ID", "ORGANIZATION"}

# Confidence range for uncertain detections
# Narrowed range - only verify truly uncertain entities
UNCERTAIN_LOW = 0.40
UNCERTAIN_HIGH = 0.65

# Common form labels and headers (false positives)
FORM_LABELS = {
    "name", "address", "city", "state", "zip", "country", "phone", "email",
    "company", "organization", "employer", "business", "street", "apt",
    "suite", "unit", "floor", "building", "po box", "postal code",
    "first name", "last name", "full name", "middle name", "maiden name",
    "date of birth", "dob", "ssn", "social security", "tax id",
    "account number", "id number", "customer id", "order number",
    "billing address", "shipping address", "mailing address", "home address",
    "work address", "office address", "headquarters", "location",
    "enter your", "please enter", "type your", "your name", "your address",
    "n/a", "none", "not applicable", "unknown", "tbd", "pending",
}

# Street type indicators (strong ADDRESS signal)
STREET_TYPES = {
    "street", "st", "avenue", "ave", "road", "rd", "drive", "dr",
    "boulevard", "blvd", "lane", "ln", "way", "court", "ct", "circle",
    "place", "pl", "terrace", "ter", "trail", "highway", "hwy",
    "parkway", "pkwy", "expressway", "freeway", "route", "rte",
}

# Corporate suffixes (strong COMPANY signal)
CORPORATE_SUFFIXES = {
    "inc", "inc.", "llc", "llc.", "ltd", "ltd.", "corp", "corp.",
    "co", "co.", "plc", "corporation", "incorporated", "limited",
    "company", "gmbh", "ag", "sa", "nv", "bv", "pty", "& co",
}

# Common words that are often false positives for COMPANY/LOCATION
GENERIC_WORDS = {
    "the", "a", "an", "and", "or", "but", "for", "with", "from", "to",
    "at", "in", "on", "by", "of", "as", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "must", "shall",
    "this", "that", "these", "those", "it", "its", "they", "them",
    "we", "us", "our", "you", "your", "he", "she", "him", "her",
    "all", "any", "some", "many", "few", "more", "most", "other",
    "new", "old", "good", "bad", "first", "last", "next", "previous",
}


def should_verify(entity_type: str, confidence: float) -> bool:
    """Determine if an entity should be verified."""
    if entity_type not in VERIFIABLE_ENTITIES:
        return False
    if confidence < UNCERTAIN_LOW or confidence > UNCERTAIN_HIGH:
        return False
    return True


def verify_address(text: str, context: str = "") -> VerificationResult:
    """Verify if text is a real address vs form label.

    MINIMAL VERSION: Only reject exact form label matches.
    """
    text_lower = text.lower().strip()

    # ONLY reject exact form label matches
    if text_lower in FORM_LABELS:
        return VerificationResult(False, -0.5, "Form label pattern")

    # Check for street type indicators - boost confidence if found
    words = text_lower.split()
    has_street_type = any(w.rstrip('.,') in STREET_TYPES for w in words)
    has_numbers = bool(re.search(r'\d', text))

    if has_street_type and has_numbers:
        return VerificationResult(True, 0.10, "Street + number pattern")

    # Default: accept (do not penalize)
    return VerificationResult(True, 0.0, "Accepted")


def verify_company(text: str, context: str = "") -> VerificationResult:
    """Verify if text is a real company name vs generic word.

    MINIMAL VERSION: Only reject exact form label matches.
    """
    text_lower = text.lower().strip()
    words = text_lower.split()

    # ONLY reject exact form label matches
    if text_lower in FORM_LABELS:
        return VerificationResult(False, -0.5, "Form label pattern")

    # Boost confidence for corporate suffixes
    has_suffix = any(w.rstrip('.,') in CORPORATE_SUFFIXES for w in words)
    if has_suffix:
        return VerificationResult(True, 0.15, "Corporate suffix detected")

    # Default: accept (do not penalize)
    return VerificationResult(True, 0.0, "Accepted")


def verify_location(text: str, context: str = "") -> VerificationResult:
    """Verify if text is a real location vs generic term.

    MINIMAL VERSION: Only reject exact form label matches.
    """
    text_lower = text.lower().strip()

    # ONLY reject exact form label matches
    if text_lower in FORM_LABELS:
        return VerificationResult(False, -0.5, "Form label pattern")

    # Boost confidence for comma-separated locations (City, State)
    if ',' in text:
        return VerificationResult(True, 0.10, "Comma-separated location")

    # Default: accept (do not penalize)
    return VerificationResult(True, 0.0, "Accepted")


def verify_id(text: str, context: str = "") -> VerificationResult:
    """Verify if text is a real ID vs form label or placeholder.

    MINIMAL VERSION: Only reject exact form label matches and obvious placeholders.
    """
    text_lower = text.lower().strip()

    # ONLY reject exact form label matches
    if text_lower in FORM_LABELS:
        return VerificationResult(False, -0.5, "Form label pattern")

    # Reject obvious placeholders
    placeholders = {"xxx", "xxxx", "0000", "1234", "test", "sample", "demo"}
    if text_lower in placeholders:
        return VerificationResult(False, -0.40, "Placeholder pattern")

    # Default: accept (do not penalize)
    return VerificationResult(True, 0.0, "Accepted")


def verify_entity(
    text: str,
    entity_type: str,
    context: str = ""
) -> VerificationResult:
    """
    Verify a potential PII entity using heuristic rules.

    Args:
        text: The detected text span
        entity_type: Type of PII (ADDRESS, COMPANY, etc.)
        context: Surrounding text for context

    Returns:
        VerificationResult with is_pii, confidence_adjustment, and reason
    """
    entity_type_upper = entity_type.upper()

    if entity_type_upper == "ADDRESS":
        return verify_address(text, context)
    elif entity_type_upper in ("COMPANY", "ORGANIZATION"):
        return verify_company(text, context)
    elif entity_type_upper == "LOCATION":
        return verify_location(text, context)
    elif entity_type_upper == "ID":
        return verify_id(text, context)
    else:
        return VerificationResult(True, 0.0, "No verification rules")


def filter_with_heuristics(
    entities,  # List[PIIEntity]
    text: str,
    min_confidence: float = 0.40
) -> list:
    """
    Filter false positives using heuristic verification.

    Args:
        entities: List of PIIEntity objects
        text: Original text
        min_confidence: Minimum confidence after adjustment to keep

    Returns:
        Filtered list of PIIEntity objects with adjusted confidence
    """
    from hush_engine.detectors.pii_detector import PIIEntity

    filtered = []

    for entity in entities:
        # Check if this entity should be verified
        if not should_verify(entity.entity_type, entity.confidence):
            filtered.append(entity)
            continue

        # Verify with heuristics
        result = verify_entity(entity.text, entity.entity_type)

        # Adjust confidence
        new_confidence = entity.confidence + result.confidence_adjustment

        # Filter if below minimum or explicitly marked as not PII
        if not result.is_pii or new_confidence < min_confidence:
            continue

        # Create updated entity with adjusted confidence
        filtered.append(PIIEntity(
            entity_type=entity.entity_type,
            text=entity.text,
            start=entity.start,
            end=entity.end,
            confidence=min(1.0, max(0.0, new_confidence)),
            pattern_name=entity.pattern_name,
            locale=entity.locale,
            recognition_metadata=entity.recognition_metadata
        ))

    return filtered
