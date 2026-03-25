"""Robust parser for LLM PII detection output."""

import json
import re


# Map non-standard entity type names back to benchmark taxonomy
ENTITY_TYPE_ALIASES = {
    # Phone variations
    "PHONE_NUMBER": "PHONE",
    "TELEPHONE": "PHONE",
    "MOBILE": "PHONE",
    "FAX": "PHONE",
    # Email variations
    "EMAIL_ADDRESS": "EMAIL",
    "E_MAIL": "EMAIL",
    # Name variations
    "NAME": "PERSON",
    "FULL_NAME": "PERSON",
    "FIRST_NAME": "PERSON",
    "LAST_NAME": "PERSON",
    "PERSON_NAME": "PERSON",
    # Address/Location variations
    "LOCATION": "ADDRESS",
    "STREET_ADDRESS": "ADDRESS",
    "MAILING_ADDRESS": "ADDRESS",
    "CITY": "ADDRESS",
    "STATE": "ADDRESS",
    "ZIP_CODE": "ADDRESS",
    "POSTAL_CODE": "ADDRESS",
    "COUNTRY": "ADDRESS",
    # ID variations
    "SSN": "NATIONAL_ID",
    "SOCIAL_SECURITY": "NATIONAL_ID",
    "SOCIAL_SECURITY_NUMBER": "NATIONAL_ID",
    "PASSPORT": "NATIONAL_ID",
    "PASSPORT_NUMBER": "NATIONAL_ID",
    "DRIVERS_LICENSE": "NATIONAL_ID",
    "DRIVER_LICENSE": "NATIONAL_ID",
    "TAX_ID": "NATIONAL_ID",
    # Date variations
    "DATE": "DATE_TIME",
    "TIME": "DATE_TIME",
    "DOB": "DATE_TIME",
    "DATE_OF_BIRTH": "DATE_TIME",
    "BIRTHDAY": "DATE_TIME",
    # Financial variations
    "CREDIT_CARD_NUMBER": "CREDIT_CARD",
    "BANK_ACCOUNT": "FINANCIAL",
    "IBAN": "FINANCIAL",
    "SWIFT": "FINANCIAL",
    "ROUTING_NUMBER": "FINANCIAL",
    # Organization variations
    "ORGANIZATION": "COMPANY",
    "ORG": "COMPANY",
    "COMPANY_NAME": "COMPANY",
    # Network variations
    "MAC_ADDRESS": "NETWORK",
    "DEVICE_ID": "NETWORK",
    "IMEI": "NETWORK",
    # Medical variations
    "HEALTH": "MEDICAL",
    "DIAGNOSIS": "MEDICAL",
    "MEDICATION": "MEDICAL",
    # Other
    "PASSWORD": "CREDENTIAL",
    "API_KEY": "CREDENTIAL",
    "USERNAME": "PERSON",
    "LICENSE_PLATE": "VEHICLE",
    "VIN": "VEHICLE",
    "GPS": "COORDINATES",
    "LATITUDE": "COORDINATES",
    "LONGITUDE": "COORDINATES",
}

# Valid entity types in benchmark taxonomy
VALID_TYPES = {
    "PERSON", "EMAIL", "PHONE", "ADDRESS", "DATE_TIME",
    "CREDIT_CARD", "NATIONAL_ID", "URL", "IP_ADDRESS",
    "COMPANY", "FINANCIAL", "MEDICAL", "AGE", "GENDER",
    "CREDENTIAL", "COORDINATES", "VEHICLE", "NETWORK",
    "BIOMETRIC", "ID",
}


def normalize_entity_type(raw_type: str) -> str | None:
    """Normalize an entity type string to benchmark taxonomy."""
    if not raw_type:
        return None
    cleaned = raw_type.strip().upper().replace(" ", "_").replace("-", "_")
    if cleaned in VALID_TYPES:
        return cleaned
    return ENTITY_TYPE_ALIASES.get(cleaned)


def parse_llm_pii_output(raw_response: str) -> dict:
    """Parse LLM response into {entity_type: [{'text': ..., 'confidence': 1.0}]}.

    Handles:
    1. Clean JSON array
    2. JSON wrapped in markdown ```json ... ``` blocks
    3. JSON with trailing commas or single quotes
    4. Partial/truncated JSON
    5. Line-by-line JSON objects
    6. Complete failure -> empty dict

    Returns:
        dict mapping entity types to lists of detection dicts
    """
    if not raw_response or not raw_response.strip():
        return {}

    text = raw_response.strip()
    items = None

    # Strategy 1: Direct JSON parse
    items = _try_parse_json_array(text)

    # Strategy 2: Extract from markdown fences
    if items is None:
        items = _try_extract_from_markdown(text)

    # Strategy 3: Find JSON array in text
    if items is None:
        items = _try_find_json_array(text)

    # Strategy 4: Fix common issues and retry
    if items is None:
        items = _try_fix_and_parse(text)

    # Strategy 5: Parse individual JSON objects line by line
    if items is None:
        items = _try_line_by_line(text)

    if not items:
        return {}

    return _items_to_detections(items)


def _try_parse_json_array(text: str) -> list | None:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _try_extract_from_markdown(text: str) -> list | None:
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        return _try_parse_json_array(match.group(1).strip())
    return None


def _try_find_json_array(text: str) -> list | None:
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        return _try_parse_json_array(text[start:end + 1])
    return None


def _try_fix_and_parse(text: str) -> list | None:
    # Extract array portion
    start = text.find("[")
    end = text.rfind("]")
    if start == -1:
        return None
    fragment = text[start:end + 1] if end > start else text[start:] + "]"
    # Fix trailing commas
    fragment = re.sub(r",\s*([}\]])", r"\1", fragment)
    # Fix single quotes to double quotes
    fragment = fragment.replace("'", '"')
    return _try_parse_json_array(fragment)


def _try_line_by_line(text: str) -> list | None:
    items = []
    for match in re.finditer(r"\{[^{}]+\}", text):
        try:
            obj = json.loads(match.group())
            items.append(obj)
        except (json.JSONDecodeError, ValueError):
            # Try fixing quotes
            fixed = match.group().replace("'", '"')
            try:
                obj = json.loads(fixed)
                items.append(obj)
            except (json.JSONDecodeError, ValueError):
                continue
    return items if items else None


def _items_to_detections(items: list) -> dict:
    """Convert parsed JSON items to detection dict format."""
    detections = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        text = item.get("text", "").strip()
        raw_type = item.get("type", "") or item.get("entity_type", "") or item.get("label", "")
        if not text or not raw_type:
            continue
        entity_type = normalize_entity_type(raw_type)
        if entity_type is None:
            continue
        if entity_type not in detections:
            detections[entity_type] = []
        detections[entity_type].append({
            "text": text,
            "confidence": 1.0,
        })
    return detections
