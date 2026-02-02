"""
PII Detection module - Presidio-based entity recognition

Supports locale-aware detection for international documents.
"""

from .pii_detector import PIIDetector, PIIEntity
from .table_detector import (
    TableDetector,
    TableCell,
    TableColumn,
    DetectedTable,
    ContextAwarePIIDetector,
    HEADER_ENTITY_MAP,
)
from .locale import (
    Locale,
    LocalePatternMapping,
    PATTERN_LOCALE_MAP,
    get_locale_from_string,
    get_locales_for_country,
    get_locales_for_language,
    calculate_locale_boost,
    detect_document_locale,
)
from .validators import (
    validate_iban,
    validate_bic,
    validate_phone,
    validate_credit_card,
    validate_national_id,
    validate_south_african_id,
    validate_detected_entity,
    get_supported_countries,
    # Checksum algorithms
    luhn_checksum,
    luhn_validate,
    verhoeff_checksum,
    verhoeff_validate,
    mod11_checksum,
    mod11_validate,
    mod97_validate,
    # Pattern data
    IBAN_COUNTRY_SPECS,
    PHONE_PATTERNS,
)

__all__ = [
    # Detection classes
    "PIIDetector",
    "PIIEntity",
    "TableDetector",
    "TableCell",
    "TableColumn",
    "DetectedTable",
    "ContextAwarePIIDetector",
    "HEADER_ENTITY_MAP",
    # Locale support
    "Locale",
    "LocalePatternMapping",
    "PATTERN_LOCALE_MAP",
    "get_locale_from_string",
    "get_locales_for_country",
    "get_locales_for_language",
    "calculate_locale_boost",
    "detect_document_locale",
    # Validators
    "validate_iban",
    "validate_bic",
    "validate_phone",
    "validate_credit_card",
    "validate_national_id",
    "validate_south_african_id",
    "validate_detected_entity",
    "get_supported_countries",
    # Checksum algorithms
    "luhn_checksum",
    "luhn_validate",
    "verhoeff_checksum",
    "verhoeff_validate",
    "mod11_checksum",
    "mod11_validate",
    "mod97_validate",
    # Pattern data
    "IBAN_COUNTRY_SPECS",
    "PHONE_PATTERNS",
]
