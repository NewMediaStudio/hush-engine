"""
Text normalization and encoded PII detection for evasion defense.

This module provides preprocessing functions to handle:
- Unicode normalization (fullwidth characters, zero-width spaces)
- Encoded PII detection (Base64, URL-encoded, Hex-encoded strings)
"""

import unicodedata
import re
import base64
from urllib.parse import unquote
from typing import List, Tuple, Optional


def normalize_text(text: str) -> str:
    """
    Normalize text for PII detection.

    Handles common evasion techniques:
    - Fullwidth characters (＠ → @, ０ → 0)
    - Zero-width characters that break patterns
    - Inconsistent whitespace

    Args:
        text: Raw input text

    Returns:
        Normalized text ready for PII detection
    """
    if not text:
        return text

    # NFKC normalization: converts fullwidth to ASCII equivalents
    # e.g., ＠ → @, ０ → 0, ｅｍａｉｌ → email
    text = unicodedata.normalize('NFKC', text)

    # Remove zero-width characters that can break regex patterns
    # U+200B: Zero Width Space
    # U+200C: Zero Width Non-Joiner
    # U+200D: Zero Width Joiner
    # U+200E: Left-to-Right Mark
    # U+200F: Right-to-Left Mark
    # U+2060: Word Joiner
    # U+FEFF: Zero Width No-Break Space (BOM)
    text = re.sub(r'[\u200b-\u200f\u2060\ufeff]', '', text)

    # Normalize various dash/hyphen characters to standard hyphen
    # U+2010: Hyphen
    # U+2011: Non-Breaking Hyphen
    # U+2012: Figure Dash
    # U+2013: En Dash
    # U+2014: Em Dash
    # U+2015: Horizontal Bar
    text = re.sub(r'[\u2010-\u2015]', '-', text)

    # Normalize whitespace (tabs, multiple spaces) but preserve newlines
    text = re.sub(r'[^\S\n]+', ' ', text)

    return text


def decode_and_scan(text: str) -> List[Tuple[str, str, int, int]]:
    """
    Find and decode potential encoded strings in text.

    Detects PII hidden in:
    - Base64 encoded strings
    - URL-encoded strings (percent encoding)
    - Hex-encoded strings

    Args:
        text: Text to scan for encoded content

    Returns:
        List of tuples: (encoding_type, decoded_text, start_pos, end_pos)
    """
    findings = []

    if not text:
        return findings

    # Base64 pattern: minimum 20 chars to avoid false positives
    # Must be valid base64 alphabet with optional padding
    base64_pattern = re.compile(r'[A-Za-z0-9+/]{20,}={0,2}')
    for match in base64_pattern.finditer(text):
        decoded = _try_decode_base64(match.group())
        if decoded:
            findings.append(('base64', decoded, match.start(), match.end()))

    # URL-encoded pattern: sequences of %XX (at least 3 to be meaningful)
    url_pattern = re.compile(r'(?:%[0-9A-Fa-f]{2}){3,}')
    for match in url_pattern.finditer(text):
        decoded = _try_decode_url(match.group())
        if decoded:
            findings.append(('url', decoded, match.start(), match.end()))

    # Hex pattern: continuous hex digits (min 16 for 8 bytes)
    # Optional 0x prefix
    hex_pattern = re.compile(r'(?:0x)?([0-9A-Fa-f]{16,})')
    for match in hex_pattern.finditer(text):
        decoded = _try_decode_hex(match.group(1))
        if decoded:
            findings.append(('hex', decoded, match.start(), match.end()))

    return findings


def _try_decode_base64(encoded: str) -> Optional[str]:
    """
    Attempt to decode a Base64 string.

    Args:
        encoded: Potential Base64 string

    Returns:
        Decoded string if valid UTF-8, None otherwise
    """
    try:
        # Add padding if needed
        padding = 4 - (len(encoded) % 4)
        if padding != 4:
            encoded += '=' * padding

        decoded_bytes = base64.b64decode(encoded, validate=True)
        decoded = decoded_bytes.decode('utf-8', errors='strict')

        # Sanity check: must be printable and meaningful
        if len(decoded) > 4 and decoded.isprintable():
            return decoded
    except Exception:
        pass

    return None


def _try_decode_url(encoded: str) -> Optional[str]:
    """
    Attempt to decode a URL-encoded string.

    Args:
        encoded: Potential percent-encoded string

    Returns:
        Decoded string if different from input, None otherwise
    """
    try:
        decoded = unquote(encoded)
        # Only return if decoding actually changed something
        if decoded != encoded and len(decoded) > 2:
            return decoded
    except Exception:
        pass

    return None


def _try_decode_hex(hex_string: str) -> Optional[str]:
    """
    Attempt to decode a hex string to UTF-8.

    Args:
        hex_string: Hex string (without 0x prefix)

    Returns:
        Decoded string if valid UTF-8, None otherwise
    """
    try:
        # Must have even number of characters
        if len(hex_string) % 2 != 0:
            return None

        decoded_bytes = bytes.fromhex(hex_string)
        decoded = decoded_bytes.decode('utf-8', errors='strict')

        # Sanity check: must be printable and meaningful
        if len(decoded) > 4 and decoded.isprintable():
            return decoded
    except Exception:
        pass

    return None


def normalize_and_decode(text: str) -> Tuple[str, List[Tuple[str, str, int, int]]]:
    """
    Full preprocessing pipeline: normalize text and find encoded PII.

    Args:
        text: Raw input text

    Returns:
        Tuple of (normalized_text, encoded_findings)
    """
    normalized = normalize_text(text)
    encoded = decode_and_scan(normalized)
    return normalized, encoded
