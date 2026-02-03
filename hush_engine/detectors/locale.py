"""
ISO Locale Support for PII Detection

This module provides locale-aware detection capabilities, allowing the engine
to prioritize patterns based on document locale context.

Locale Format: ISO 639-1 language code + ISO 3166-1 alpha-2 country code
Examples: en-US, en-GB, de-DE, ja-JP, zh-CN, pt-BR

Usage:
    detector = PIIDetector()

    # Single locale document
    entities = detector.analyze_text(text, locales=["en-US"])

    # Multi-locale document (e.g., passport, translation)
    entities = detector.analyze_text(text, locales=["en-US", "es-MX"])

    # International spreadsheet (no locale restriction)
    entities = detector.analyze_text(text, locales=None)  # Uses all patterns
"""

from dataclasses import dataclass
from typing import List, Set, Optional, Dict
from enum import Enum


class Locale(Enum):
    """
    ISO locales supported by the detection engine.

    Format: language_COUNTRY (ISO 639-1 + ISO 3166-1)
    """
    # North America
    EN_US = "en-US"  # United States
    EN_CA = "en-CA"  # Canada (English)
    FR_CA = "fr-CA"  # Canada (French)
    ES_MX = "es-MX"  # Mexico

    # Europe - Western
    EN_GB = "en-GB"  # United Kingdom
    EN_IE = "en-IE"  # Ireland
    DE_DE = "de-DE"  # Germany
    DE_AT = "de-AT"  # Austria
    DE_CH = "de-CH"  # Switzerland (German)
    FR_FR = "fr-FR"  # France
    FR_BE = "fr-BE"  # Belgium (French)
    FR_CH = "fr-CH"  # Switzerland (French)
    IT_IT = "it-IT"  # Italy
    ES_ES = "es-ES"  # Spain
    PT_PT = "pt-PT"  # Portugal
    NL_NL = "nl-NL"  # Netherlands
    NL_BE = "nl-BE"  # Belgium (Dutch)

    # Europe - Nordic
    SV_SE = "sv-SE"  # Sweden
    NO_NO = "no-NO"  # Norway
    DA_DK = "da-DK"  # Denmark
    FI_FI = "fi-FI"  # Finland

    # Europe - Eastern
    PL_PL = "pl-PL"  # Poland
    RU_RU = "ru-RU"  # Russia
    UK_UA = "uk-UA"  # Ukraine
    CS_CZ = "cs-CZ"  # Czech Republic
    SK_SK = "sk-SK"  # Slovakia
    HU_HU = "hu-HU"  # Hungary
    RO_RO = "ro-RO"  # Romania

    # Asia - East
    JA_JP = "ja-JP"  # Japan
    ZH_CN = "zh-CN"  # China (Simplified)
    ZH_TW = "zh-TW"  # Taiwan (Traditional)
    ZH_HK = "zh-HK"  # Hong Kong
    KO_KR = "ko-KR"  # South Korea

    # Asia - South/Southeast
    HI_IN = "hi-IN"  # India (Hindi)
    EN_IN = "en-IN"  # India (English)
    BN_BD = "bn-BD"  # Bangladesh (Bengali)
    EN_BD = "en-BD"  # Bangladesh (English)
    TH_TH = "th-TH"  # Thailand
    VI_VN = "vi-VN"  # Vietnam
    ID_ID = "id-ID"  # Indonesia
    MS_MY = "ms-MY"  # Malaysia
    EN_SG = "en-SG"  # Singapore
    EN_PH = "en-PH"  # Philippines

    # Middle East
    AR_SA = "ar-SA"  # Saudi Arabia
    AR_AE = "ar-AE"  # UAE
    HE_IL = "he-IL"  # Israel
    TR_TR = "tr-TR"  # Turkey

    # Oceania
    EN_AU = "en-AU"  # Australia
    EN_NZ = "en-NZ"  # New Zealand

    # South America
    PT_BR = "pt-BR"  # Brazil
    ES_AR = "es-AR"  # Argentina
    ES_CL = "es-CL"  # Chile
    ES_CO = "es-CO"  # Colombia

    # Africa
    EN_ZA = "en-ZA"  # South Africa
    AR_EG = "ar-EG"  # Egypt
    EN_NG = "en-NG"  # Nigeria
    EN_KE = "en-KE"  # Kenya


@dataclass
class LocalePatternMapping:
    """Maps entity patterns to their associated locales."""
    pattern_name: str
    entity_type: str
    locales: Set[str]
    confidence_boost: float = 0.1  # Boost when locale matches


# Pattern to locale mappings
# This maps specific pattern names to their applicable locales
PATTERN_LOCALE_MAP: Dict[str, LocalePatternMapping] = {
    # === NATIONAL IDs ===
    "uk_nino": LocalePatternMapping(
        "uk_nino", "NATIONAL_ID", {Locale.EN_GB.value}, 0.15
    ),
    "canadian_sin": LocalePatternMapping(
        "canadian_sin", "NATIONAL_ID", {Locale.EN_CA.value, Locale.FR_CA.value}, 0.15
    ),
    "german_steuer": LocalePatternMapping(
        "german_steuer", "NATIONAL_ID", {Locale.DE_DE.value, Locale.DE_AT.value, Locale.DE_CH.value}, 0.15
    ),
    "french_insee": LocalePatternMapping(
        "french_insee", "NATIONAL_ID", {Locale.FR_FR.value, Locale.FR_BE.value}, 0.15
    ),
    "japanese_mynumber": LocalePatternMapping(
        "japanese_mynumber", "NATIONAL_ID", {Locale.JA_JP.value}, 0.15
    ),
    "indian_aadhaar": LocalePatternMapping(
        "indian_aadhaar", "NATIONAL_ID", {Locale.HI_IN.value, Locale.EN_IN.value}, 0.15
    ),
    "brazilian_cpf": LocalePatternMapping(
        "brazilian_cpf", "NATIONAL_ID", {Locale.PT_BR.value}, 0.15
    ),
    "korean_rrn": LocalePatternMapping(
        "korean_rrn", "NATIONAL_ID", {Locale.KO_KR.value}, 0.15
    ),
    "italian_cf": LocalePatternMapping(
        "italian_cf", "NATIONAL_ID", {Locale.IT_IT.value}, 0.15
    ),
    "spanish_dni": LocalePatternMapping(
        "spanish_dni", "NATIONAL_ID", {Locale.ES_ES.value}, 0.15
    ),
    "spanish_nie": LocalePatternMapping(
        "spanish_nie", "NATIONAL_ID", {Locale.ES_ES.value}, 0.15
    ),
    "polish_pesel": LocalePatternMapping(
        "polish_pesel", "NATIONAL_ID", {Locale.PL_PL.value}, 0.15
    ),
    "swedish_pn": LocalePatternMapping(
        "swedish_pn", "NATIONAL_ID", {Locale.SV_SE.value}, 0.15
    ),
    "danish_cpr": LocalePatternMapping(
        "danish_cpr", "NATIONAL_ID", {Locale.DA_DK.value}, 0.15
    ),
    "norwegian_fn": LocalePatternMapping(
        "norwegian_fn", "NATIONAL_ID", {Locale.NO_NO.value}, 0.15
    ),
    "singapore_nric": LocalePatternMapping(
        "singapore_nric", "NATIONAL_ID", {Locale.EN_SG.value}, 0.15
    ),
    "irish_pps": LocalePatternMapping(
        "irish_pps", "NATIONAL_ID", {Locale.EN_IE.value}, 0.15
    ),
    "chinese_id": LocalePatternMapping(
        "chinese_id", "NATIONAL_ID", {Locale.ZH_CN.value}, 0.15
    ),
    "taiwan_id": LocalePatternMapping(
        "taiwan_id", "NATIONAL_ID", {Locale.ZH_TW.value}, 0.15
    ),
    "uae_eid": LocalePatternMapping(
        "uae_eid", "NATIONAL_ID", {Locale.AR_AE.value}, 0.15
    ),

    # === DRIVER'S LICENSES ===
    "uk_dl": LocalePatternMapping(
        "uk_dl", "DRIVERS_LICENSE", {Locale.EN_GB.value}, 0.15
    ),
    "de_dl": LocalePatternMapping(
        "de_dl", "DRIVERS_LICENSE", {Locale.DE_DE.value, Locale.DE_AT.value}, 0.15
    ),
    "fl_dl": LocalePatternMapping(
        "fl_dl", "DRIVERS_LICENSE", {Locale.EN_US.value}, 0.10
    ),
    "il_dl": LocalePatternMapping(
        "il_dl", "DRIVERS_LICENSE", {Locale.EN_US.value}, 0.10
    ),
    "ca_dl": LocalePatternMapping(
        "ca_dl", "DRIVERS_LICENSE", {Locale.EN_US.value}, 0.10
    ),
    "on_dl": LocalePatternMapping(
        "on_dl", "DRIVERS_LICENSE", {Locale.EN_CA.value, Locale.FR_CA.value}, 0.15
    ),
    "au_qld_dl": LocalePatternMapping(
        "au_qld_dl", "DRIVERS_LICENSE", {Locale.EN_AU.value}, 0.15
    ),
    "au_letter_dl": LocalePatternMapping(
        "au_letter_dl", "DRIVERS_LICENSE", {Locale.EN_AU.value}, 0.10
    ),

    # === POSTAL CODES ===
    "us_zip_only": LocalePatternMapping(
        "us_zip_only", "LOCATION", {Locale.EN_US.value}, 0.15
    ),
    "us_city_state_zip": LocalePatternMapping(
        "us_city_state_zip", "LOCATION", {Locale.EN_US.value}, 0.10
    ),
    "canadian_postal_only": LocalePatternMapping(
        "canadian_postal_only", "LOCATION", {Locale.EN_CA.value, Locale.FR_CA.value}, 0.15
    ),
    "japan_postal": LocalePatternMapping(
        "japan_postal", "LOCATION", {Locale.JA_JP.value}, 0.15
    ),
    "uk_postcode": LocalePatternMapping(
        "uk_postcode", "LOCATION", {Locale.EN_GB.value}, 0.15
    ),
    "germany_plz": LocalePatternMapping(
        "germany_plz", "LOCATION", {Locale.DE_DE.value, Locale.DE_AT.value, Locale.DE_CH.value}, 0.15
    ),
    "india_pin": LocalePatternMapping(
        "india_pin", "LOCATION", {Locale.HI_IN.value, Locale.EN_IN.value}, 0.15
    ),
    "india_pin_labeled": LocalePatternMapping(
        "india_pin_labeled", "LOCATION", {Locale.HI_IN.value, Locale.EN_IN.value}, 0.20
    ),
    "india_pin_with_state": LocalePatternMapping(
        "india_pin_with_state", "LOCATION", {Locale.HI_IN.value, Locale.EN_IN.value}, 0.20
    ),
    "indian_city": LocalePatternMapping(
        "indian_city", "LOCATION", {Locale.HI_IN.value, Locale.EN_IN.value}, 0.15
    ),
    "indian_state": LocalePatternMapping(
        "indian_state", "LOCATION", {Locale.HI_IN.value, Locale.EN_IN.value}, 0.15
    ),
    "indian_locality_state": LocalePatternMapping(
        "indian_locality_state", "LOCATION", {Locale.HI_IN.value, Locale.EN_IN.value}, 0.20
    ),
    "bangladesh_country": LocalePatternMapping(
        "bangladesh_country", "LOCATION", {Locale.BN_BD.value, Locale.EN_BD.value}, 0.15
    ),
    "bangladesh_cities": LocalePatternMapping(
        "bangladesh_cities", "LOCATION", {Locale.BN_BD.value, Locale.EN_BD.value}, 0.15
    ),
    "brazil_cep": LocalePatternMapping(
        "brazil_cep", "LOCATION", {Locale.PT_BR.value}, 0.15
    ),
    "korea_postal": LocalePatternMapping(
        "korea_postal", "LOCATION", {Locale.KO_KR.value}, 0.15
    ),
    "china_postal": LocalePatternMapping(
        "china_postal", "LOCATION", {Locale.ZH_CN.value}, 0.15
    ),
    "russia_postal": LocalePatternMapping(
        "russia_postal", "LOCATION", {Locale.RU_RU.value}, 0.15
    ),
    "netherlands_postal": LocalePatternMapping(
        "netherlands_postal", "LOCATION", {Locale.NL_NL.value, Locale.NL_BE.value}, 0.15
    ),
    "australia_postal": LocalePatternMapping(
        "australia_postal", "LOCATION", {Locale.EN_AU.value, Locale.EN_NZ.value}, 0.15
    ),

    # === PHONE NUMBERS ===
    "na_phone_parens": LocalePatternMapping(
        "na_phone_parens", "PHONE_NUMBER", {Locale.EN_US.value, Locale.EN_CA.value}, 0.10
    ),
    "na_phone_dashes": LocalePatternMapping(
        "na_phone_dashes", "PHONE_NUMBER", {Locale.EN_US.value, Locale.EN_CA.value}, 0.10
    ),
    "intl_phone_plus": LocalePatternMapping(
        "intl_phone_plus", "PHONE_NUMBER", set(), 0.0  # Universal pattern
    ),
    "uk_nhs_spaces": LocalePatternMapping(
        "uk_nhs_spaces", "UK_NHS", {Locale.EN_GB.value}, 0.15
    ),

    # === SSN ===
    "ssn_dashes": LocalePatternMapping(
        "ssn_dashes", "SSN", {Locale.EN_US.value}, 0.15
    ),
    "ssn_spaces": LocalePatternMapping(
        "ssn_spaces", "SSN", {Locale.EN_US.value}, 0.15
    ),

    # === PASSPORTS ===
    "us_passport_new": LocalePatternMapping(
        "us_passport_new", "PASSPORT", {Locale.EN_US.value}, 0.10
    ),
    "ca_passport": LocalePatternMapping(
        "ca_passport", "PASSPORT", {Locale.EN_CA.value, Locale.FR_CA.value}, 0.10
    ),
    "de_passport": LocalePatternMapping(
        "de_passport", "PASSPORT", {Locale.DE_DE.value}, 0.10
    ),
    "fr_passport": LocalePatternMapping(
        "fr_passport", "PASSPORT", {Locale.FR_FR.value}, 0.10
    ),

    # === IBAN_CODE (International Bank Account Number) ===
    "iban_de": LocalePatternMapping(
        "iban_de", "IBAN_CODE", {Locale.DE_DE.value, Locale.DE_AT.value, Locale.DE_CH.value}, 0.15
    ),
    "iban_gb": LocalePatternMapping(
        "iban_gb", "IBAN_CODE", {Locale.EN_GB.value}, 0.15
    ),
    "iban_fr": LocalePatternMapping(
        "iban_fr", "IBAN_CODE", {Locale.FR_FR.value, Locale.FR_BE.value, Locale.FR_CH.value}, 0.15
    ),
    "iban_nl": LocalePatternMapping(
        "iban_nl", "IBAN_CODE", {Locale.NL_NL.value, Locale.NL_BE.value}, 0.15
    ),
    "iban_es": LocalePatternMapping(
        "iban_es", "IBAN_CODE", {Locale.ES_ES.value}, 0.15
    ),
    "iban_it": LocalePatternMapping(
        "iban_it", "IBAN_CODE", {Locale.IT_IT.value}, 0.15
    ),
    "iban_spaced": LocalePatternMapping(
        "iban_spaced", "IBAN_CODE", set(), 0.0  # Universal pattern - no locale boost
    ),
    "iban_compact": LocalePatternMapping(
        "iban_compact", "IBAN_CODE", set(), 0.0  # Universal pattern - no locale boost
    ),

    # === SOUTH AFRICAN ID ===
    "za_id": LocalePatternMapping(
        "za_id", "NATIONAL_ID", {Locale.EN_ZA.value}, 0.15
    ),
}


def get_locale_from_string(locale_str: str) -> Optional[Locale]:
    """
    Convert a locale string to a Locale enum.

    Args:
        locale_str: Locale string (e.g., "en-US", "ja-JP")

    Returns:
        Locale enum or None if not found
    """
    try:
        # Try direct match
        for locale in Locale:
            if locale.value == locale_str:
                return locale
        # Try case-insensitive match
        locale_str_upper = locale_str.upper().replace("-", "_")
        return Locale[locale_str_upper]
    except (KeyError, ValueError):
        return None


def get_locales_for_country(country_code: str) -> List[Locale]:
    """
    Get all locales for a given ISO 3166-1 alpha-2 country code.

    Args:
        country_code: Two-letter country code (e.g., "US", "CA", "DE")

    Returns:
        List of Locale enums for that country
    """
    country_code = country_code.upper()
    return [
        locale for locale in Locale
        if locale.value.endswith(f"-{country_code}")
    ]


def get_locales_for_language(language_code: str) -> List[Locale]:
    """
    Get all locales for a given ISO 639-1 language code.

    Args:
        language_code: Two-letter language code (e.g., "en", "de", "zh")

    Returns:
        List of Locale enums for that language
    """
    language_code = language_code.lower()
    return [
        locale for locale in Locale
        if locale.value.startswith(f"{language_code}-")
    ]


def calculate_locale_boost(
    pattern_name: str,
    document_locales: Optional[List[str]]
) -> float:
    """
    Calculate confidence boost for a pattern based on document locales.

    Args:
        pattern_name: Name of the detection pattern
        document_locales: List of locale strings for the document, or None for no locale

    Returns:
        Confidence boost value (0.0 if no match, up to 0.15 for strong match)
    """
    if document_locales is None:
        # No locale specified - no boost or penalty
        return 0.0

    mapping = PATTERN_LOCALE_MAP.get(pattern_name)
    if mapping is None:
        # Pattern not in mapping - no boost
        return 0.0

    if not mapping.locales:
        # Universal pattern - no locale-specific boost
        return 0.0

    # Check if any document locale matches pattern locales
    for doc_locale in document_locales:
        if doc_locale in mapping.locales:
            return mapping.confidence_boost

    # No match - apply a small penalty for locale mismatch
    # This helps reduce false positives when locale is known
    return -0.05


def detect_document_locale(text: str) -> List[str]:
    """
    Attempt to auto-detect document locale from content.

    This is a heuristic based on:
    - Character sets (CJK, Cyrillic, Arabic, etc.)
    - Common words/phrases
    - Date/number formats

    Args:
        text: Document text to analyze

    Returns:
        List of likely locale strings, ordered by confidence
    """
    detected = []
    text_lower = text.lower()

    # Character set detection
    import unicodedata

    # Count character categories
    cjk_count = 0
    cyrillic_count = 0
    arabic_count = 0
    latin_count = 0

    for char in text[:5000]:  # Sample first 5000 chars
        try:
            name = unicodedata.name(char, '')
            if 'CJK' in name or 'HIRAGANA' in name or 'KATAKANA' in name:
                cjk_count += 1
            elif 'CYRILLIC' in name:
                cyrillic_count += 1
            elif 'ARABIC' in name:
                arabic_count += 1
            elif 'LATIN' in name:
                latin_count += 1
        except ValueError:
            pass

    # Japanese detection (hiragana/katakana mixed with kanji)
    if 'の' in text or 'は' in text or 'を' in text:
        detected.append(Locale.JA_JP.value)

    # Chinese detection (only CJK, no kana)
    elif cjk_count > 50 and '的' in text:
        if '繁體' in text or '臺灣' in text:
            detected.append(Locale.ZH_TW.value)
        else:
            detected.append(Locale.ZH_CN.value)

    # Korean detection
    if any('\uac00' <= c <= '\ud7a3' for c in text):  # Hangul syllables
        detected.append(Locale.KO_KR.value)

    # Cyrillic - Russian or Ukrainian
    if cyrillic_count > 50:
        if 'і' in text_lower or 'є' in text_lower:  # Ukrainian specific
            detected.append(Locale.UK_UA.value)
        else:
            detected.append(Locale.RU_RU.value)

    # Arabic
    if arabic_count > 50:
        detected.append(Locale.AR_SA.value)

    # Latin-based languages
    if latin_count > 100:
        # German
        if 'ß' in text or 'straße' in text_lower or 'und' in text_lower:
            detected.append(Locale.DE_DE.value)

        # French
        if 'être' in text_lower or 'avoir' in text_lower or 'c\'est' in text_lower:
            detected.append(Locale.FR_FR.value)

        # Spanish
        if 'está' in text_lower or 'ñ' in text:
            detected.append(Locale.ES_ES.value)

        # Portuguese
        if 'ção' in text_lower or 'não' in text_lower:
            if 'você' in text_lower:
                detected.append(Locale.PT_BR.value)
            else:
                detected.append(Locale.PT_PT.value)

        # Italian
        if 'è' in text and 'che' in text_lower:
            detected.append(Locale.IT_IT.value)

        # Dutch
        if 'ij' in text_lower and ('het' in text_lower or 'van' in text_lower):
            detected.append(Locale.NL_NL.value)

        # Default to English if no specific detection
        if not detected:
            # Try to distinguish US vs UK
            if 'colour' in text_lower or 'honour' in text_lower or '£' in text:
                detected.append(Locale.EN_GB.value)
            elif '$' in text:
                detected.append(Locale.EN_US.value)
            else:
                detected.append(Locale.EN_US.value)  # Default

    return detected if detected else [Locale.EN_US.value]  # Fallback to US English
