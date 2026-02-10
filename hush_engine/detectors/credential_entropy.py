"""
Shannon entropy-based credential verification for high-precision secret detection.

This module implements CONTEXT-ANCHORED ENTROPY filtering to separate real secrets
from false positives. Key insight: high-entropy strings are only credentials if
they appear near credential-related context words (trigger words).

Based on industry research:
- GitGuardian uses 4.5-6.0 entropy range for secrets
- TruffleHog uses character set analysis + entropy
- Real secrets typically have entropy 4.0-6.0 for passwords, 4.5-5.8 for tokens

Shannon entropy formula: H(X) = -SUM(p(x) * log2(p(x)))
Where p(x) is the probability of character x in the string.

CONTEXT-ANCHORED DETECTION
==========================

The key innovation is requiring "trigger words" within 5 tokens of the credential:
- Trigger words: key, secret, token, password, passwd, pwd, auth, api, bearer, credential

This dramatically reduces false positives from random high-entropy strings while
maintaining recall for actual credentials that appear in context.

Entropy ranges for different string types:
- English words: 2.0-3.5 bits/char
- Random lowercase: 4.7 bits/char (theoretical)
- Random alphanumeric: 5.2 bits/char (theoretical)
- Base64: 5.5-6.0 bits/char (theoretical max 6.0)
- Hex: 4.0 bits/char (theoretical)
- Real passwords: 4.0-5.5 bits/char (depending on complexity)
- API tokens: 4.5-6.0 bits/char

Character class analysis:
- Good secrets have mixed character classes (upper, lower, digit, special)
- Low-entropy false positives often lack diversity

ENABLING CREDENTIAL DETECTION
=============================

CREDENTIAL detection is enabled in pii_detector.py (threshold=0.85).

The context-anchored entropy filter will:
- Accept known secret prefixes (ghp_, sk_live_, AKIA, Bearer, etc.) - always valid
- Accept labeled credentials (password=, api_key:, etc.) - have built-in context
- For unlabeled high-entropy strings, require trigger word within 5 tokens
- Accept entropy range: 4.0 <= entropy <= 6.5 (stricter minimum)
- Reject known false positive patterns (password, test123, your_api_key, etc.)
- Reject strings that are too short (<12 chars unlabeled, <6 chars labeled)
- Reject strings with low character class diversity (unlabeled only)

TUNING THRESHOLDS
=================

The ENTROPY_THRESHOLDS dict can be adjusted for different use cases:

- Increase min_unlabeled (4.0 -> 4.5) for higher precision
- Decrease min_unlabeled (4.0 -> 3.5) for higher recall
- Adjust ideal_low/ideal_high to change confidence boosting range

Add patterns to FALSE_POSITIVE_PATTERNS set to filter specific strings.
Add prefixes to HIGH_CONFIDENCE_PREFIXES set for new API key formats.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class EntropyResult:
    """Result of entropy-based credential analysis."""
    is_credential: bool
    entropy: float
    char_classes: int  # Number of character classes present
    confidence_adjustment: float  # Positive = boost, negative = reduce
    reason: str


# Entropy thresholds based on industry research
# These are tuned for precision - requiring higher entropy for unlabeled credentials
ENTROPY_THRESHOLDS = {
    # Minimum entropy for unlabeled credentials
    # Lowered from 4.0 to 3.5 for better recall on shorter passwords/PINs
    "min_unlabeled": 3.5,

    # Minimum entropy for labeled credentials (password=, api_key=, etc.)
    # Lower threshold since labels provide strong context
    "min_labeled": 2.5,

    # Maximum entropy (above this is likely random noise or encoding)
    # Raised from 6.5 to 7.0 to catch more base64 tokens
    "max_valid": 7.0,

    # Ideal range for high-confidence secrets
    "ideal_low": 4.0,
    "ideal_high": 6.0,
}

# =============================================================================
# CONTEXT-ANCHORED DETECTION - Trigger Words
# =============================================================================

# Trigger words that indicate a credential is nearby
# If a high-entropy string is found, it must be within 5 tokens of one of these
CREDENTIAL_TRIGGER_WORDS = {
    # Primary trigger words (strongest signal)
    "key", "secret", "token", "password", "passwd", "pwd", "auth", "api",
    "bearer", "credential", "credentials", "apikey", "secretkey", "accesskey",
    # Secondary trigger words (moderate signal)
    "access", "private", "oauth", "jwt", "authorization", "authenticate",
    "encryption", "decrypt", "encrypt", "ssh", "rsa", "pem", "certificate",
    # Login/account context
    "login", "signin", "signup", "account", "user", "username", "pin",
    "passcode", "passphrase", "hash", "salt", "cipher", "hmac", "digest",
    # Configuration context
    "config", "env", "environment", "variable", "setting", "connection",
    "database", "db", "redis", "mongo", "mysql", "postgres",
    # Platform-specific triggers
    "stripe", "aws", "github", "slack", "npm", "google", "firebase", "azure",
    "heroku", "twilio", "sendgrid", "mailgun", "openai", "anthropic",
    "docker", "kubernetes", "webhook", "endpoint",
}

# Maximum distance (in tokens) between credential and trigger word
# Widened from 5 to 10 for better recall on credentials in longer contexts
CONTEXT_WINDOW_TOKENS = 10

# Minimum length for unlabeled credentials (lowered from 12 for shorter secrets)
MIN_UNLABELED_LENGTH = 8

# Minimum length for labeled credentials
MIN_LABELED_LENGTH = 4

# Known credential label patterns
CREDENTIAL_LABELS = re.compile(
    r'(?:password|passwd|pwd|pass|secret|api[_\-]?key|apikey|'
    r'bearer|token|access[_\-]?token|auth[_\-]?token|'
    r'private[_\-]?key|secret[_\-]?key|client[_\-]?secret|'
    r'credential|ssh[_\-]?key|encryption[_\-]?key)'
    r'\s*[:=]',
    re.IGNORECASE
)

# Known credential prefixes (high confidence patterns)
HIGH_CONFIDENCE_PREFIXES = {
    # AWS
    "AKIA", "ABIA", "ACCA", "ASIA",  # AWS access keys
    # GitHub
    "ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_",
    # Stripe
    "sk_live_", "pk_live_", "sk_test_", "pk_test_", "rk_live_", "rk_test_",
    # Slack
    "xoxb-", "xoxp-", "xoxa-", "xoxr-",
    # NPM
    "npm_",
    # Generic API key prefixes
    "api_key_", "apikey_", "api-key-",
    # Google
    "AIza",  # Google API keys
    # Twilio
    "SK",  # Twilio API keys (SK followed by 32 hex chars)
    # SendGrid
    "SG.",  # SendGrid API keys
    # Mailgun
    "key-",  # Mailgun API keys
    # Generic Bearer token
    "Bearer ",
}

# JWT pattern - header.payload.signature format
JWT_PATTERN = re.compile(
    r'^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$'
)

# Explicit credential format patterns (regex for common formats)
CREDENTIAL_FORMAT_PATTERNS = [
    # API key formats: key=xxx, api_key=xxx
    re.compile(r'(?:api[_-]?key|apikey)\s*[=:]\s*["\']?([A-Za-z0-9_\-]{16,})["\']?', re.IGNORECASE),
    # Password formats: password=xxx, passwd=xxx, pwd=xxx
    re.compile(r'(?:password|passwd|pwd)\s*[=:]\s*["\']?([^\s"\',]{8,})["\']?', re.IGNORECASE),
    # Token formats: token=xxx, access_token=xxx
    re.compile(r'(?:access[_-]?token|auth[_-]?token|token)\s*[=:]\s*["\']?([A-Za-z0-9_\-\.]{16,})["\']?', re.IGNORECASE),
    # Secret formats: secret=xxx, client_secret=xxx
    re.compile(r'(?:secret|client[_-]?secret|secret[_-]?key)\s*[=:]\s*["\']?([A-Za-z0-9_\-]{16,})["\']?', re.IGNORECASE),
]

# Common false positive patterns (low entropy but match credential regex)
FALSE_POSITIVE_PATTERNS = {
    # Common placeholder values
    "password", "PASSWORD", "secret", "SECRET", "token", "TOKEN",
    "apikey", "APIKEY", "api_key", "API_KEY",
    "your_api_key", "your_secret", "your_password", "your_token",
    "xxx", "XXXX", "xxxx", "****", "____",
    "test", "TEST", "demo", "DEMO", "sample", "SAMPLE",
    "changeme", "changeit", "replace", "REPLACE",
    "null", "NULL", "none", "NONE", "undefined", "UNDEFINED",
    "placeholder", "PLACEHOLDER", "example", "EXAMPLE",
    # Common test values
    "password123", "test123", "admin123", "12345678", "123456789",
    "qwerty", "qwerty123", "abc123", "letmein", "welcome",
    # Config examples
    "your-api-key-here", "insert-key-here", "put-your-key-here",
}

# Words that reduce credential likelihood when found
LOW_CONFIDENCE_WORDS = {
    "the", "and", "for", "with", "from", "this", "that", "your",
    "enter", "type", "input", "field", "value", "string",
    "example", "sample", "test", "demo", "fake", "mock",
}


def calculate_shannon_entropy(text: str) -> float:
    """
    Calculate Shannon entropy of a string.

    Formula: H(X) = -SUM(p(x) * log2(p(x)))

    Args:
        text: String to analyze

    Returns:
        Entropy in bits per character (0.0 to ~6.5 for printable ASCII)
    """
    if not text:
        return 0.0

    # Count character frequencies
    freq = Counter(text)
    length = len(text)

    # Calculate entropy
    entropy = -sum(
        (count / length) * math.log2(count / length)
        for count in freq.values()
    )

    return entropy


def count_character_classes(text: str) -> int:
    """
    Count the number of character classes present in text.

    Classes: lowercase, uppercase, digits, special characters

    Args:
        text: String to analyze

    Returns:
        Number of character classes (0-4)
    """
    classes = 0

    if re.search(r'[a-z]', text):
        classes += 1
    if re.search(r'[A-Z]', text):
        classes += 1
    if re.search(r'\d', text):
        classes += 1
    if re.search(r'[^a-zA-Z0-9]', text):
        classes += 1

    return classes


def has_credential_label(text: str) -> bool:
    """Check if text contains a credential label (password=, api_key:, etc.)."""
    return bool(CREDENTIAL_LABELS.search(text))


def is_jwt_token(text: str) -> bool:
    """
    Check if text matches JWT format (header.payload.signature).

    JWT tokens have three base64url-encoded parts separated by dots.
    """
    if not JWT_PATTERN.match(text):
        return False

    # Additional validation: each part should be reasonably sized
    parts = text.split('.')
    if len(parts) != 3:
        return False

    # JWT parts are typically: header (short), payload (medium), signature (long)
    # Minimum sizes: header ~20, payload ~20, signature ~40
    if len(parts[0]) < 10 or len(parts[1]) < 10 or len(parts[2]) < 20:
        return False

    return True


def matches_credential_format(text: str) -> bool:
    """
    Check if text matches explicit credential format patterns.

    This catches labeled credentials like password=xxx, api_key=xxx, etc.
    """
    for pattern in CREDENTIAL_FORMAT_PATTERNS:
        if pattern.search(text):
            return True
    return False


def has_high_confidence_prefix(text: str) -> bool:
    """
    Check if text starts with a known secret prefix or matches high-confidence patterns.

    This includes:
    - Known API key prefixes (sk_live_, ghp_, AKIA, etc.)
    - JWT tokens (header.payload.signature)
    - Labeled credentials (password=xxx, api_key=xxx)
    """
    # Check static prefixes
    if any(text.startswith(prefix) for prefix in HIGH_CONFIDENCE_PREFIXES):
        return True

    # Check JWT format
    if is_jwt_token(text):
        return True

    # Check labeled credential formats
    if matches_credential_format(text):
        return True

    return False


def is_known_false_positive(text: str) -> bool:
    """Check if text matches known false positive patterns."""
    # Exact match
    if text.lower() in FALSE_POSITIVE_PATTERNS:
        return True

    # Check for placeholder patterns
    text_lower = text.lower()
    for fp in FALSE_POSITIVE_PATTERNS:
        if text_lower == fp:
            return True

    return False


def extract_credential_value(text: str) -> str:
    """
    Extract the actual credential value from a labeled credential.

    Examples:
        "password=mysecret123" -> "mysecret123"
        "api_key: sk_live_abc123" -> "sk_live_abc123"

    Args:
        text: Full credential text (may include label)

    Returns:
        Extracted credential value
    """
    # Try to extract value after = or :
    match = re.search(r'[:=]\s*["\']?([^\s"\',]+)', text)
    if match:
        return match.group(1).strip('"\'')

    return text


def has_trigger_word_nearby(
    full_text: str,
    credential_start: int,
    credential_end: int,
    window_tokens: int = CONTEXT_WINDOW_TOKENS
) -> Tuple[bool, Optional[str]]:
    """
    Check if a credential trigger word is within N tokens of the detected credential.

    This is the core of context-anchored detection. A high-entropy string is only
    considered a credential if it appears near a trigger word like "key", "secret",
    "password", "token", etc.

    Args:
        full_text: The complete text being analyzed
        credential_start: Start position of the detected credential
        credential_end: End position of the detected credential
        window_tokens: Maximum number of tokens to look before/after (default 5)

    Returns:
        Tuple of (has_trigger, trigger_word_found)
    """
    if not full_text:
        return False, None

    # Get text before and after the credential
    # Use character window approximation: ~6 chars per token average
    char_window = window_tokens * 8  # Slightly larger to be safe

    before_start = max(0, credential_start - char_window)
    after_end = min(len(full_text), credential_end + char_window)

    context_before = full_text[before_start:credential_start].lower()
    context_after = full_text[credential_end:after_end].lower()

    # Tokenize (simple word split)
    tokens_before = re.findall(r'\b\w+\b', context_before)[-window_tokens:]
    tokens_after = re.findall(r'\b\w+\b', context_after)[:window_tokens]

    # Check for trigger words
    all_tokens = tokens_before + tokens_after
    for token in all_tokens:
        if token in CREDENTIAL_TRIGGER_WORDS:
            return True, token

    return False, None


def check_context_anchor(
    credential_text: str,
    full_text: str,
    credential_start: int,
    credential_end: int
) -> Tuple[bool, str]:
    """
    Check if a potential credential has proper context anchoring.

    This combines multiple signals:
    1. Known prefixes (sk_live_, ghp_, etc.) - always valid
    2. Labeled credentials (password=, api_key:) - have built-in context
    3. Trigger word proximity for unlabeled high-entropy strings

    Args:
        credential_text: The detected credential text
        full_text: Complete document text
        credential_start: Start position in full_text
        credential_end: End position in full_text

    Returns:
        Tuple of (is_anchored, reason)
    """
    # Check for known high-confidence prefixes
    if has_high_confidence_prefix(credential_text):
        return True, "Known secret prefix (always valid)"

    # Check for credential labels (password=, api_key:, etc.)
    if has_credential_label(credential_text):
        return True, "Labeled credential (has built-in context)"

    # For unlabeled credentials, require trigger word nearby
    has_trigger, trigger_word = has_trigger_word_nearby(
        full_text, credential_start, credential_end
    )

    if has_trigger:
        return True, f"Trigger word '{trigger_word}' found nearby"

    return False, "No context anchor (no trigger word within 5 tokens)"


def analyze_credential_entropy(
    text: str,
    context: str = "",
    strict_mode: bool = True,
    full_text: str = "",
    position_start: int = -1,
    position_end: int = -1
) -> EntropyResult:
    """
    Analyze a potential credential using Shannon entropy with context anchoring.

    This is the main entry point for credential verification. It implements
    context-anchored detection: high-entropy strings are only valid credentials
    if they appear near trigger words (key, secret, token, password, etc.).

    Args:
        text: The detected credential text
        context: Surrounding text for additional context (deprecated, use full_text)
        strict_mode: If True, use stricter thresholds (fewer false positives)
        full_text: Complete document text for context anchor checking
        position_start: Start position of credential in full_text
        position_end: End position of credential in full_text

    Returns:
        EntropyResult with verdict and details
    """
    original_text = text

    # Check for known high-confidence prefixes first (always valid)
    if has_high_confidence_prefix(text):
        entropy = calculate_shannon_entropy(text)
        return EntropyResult(
            is_credential=True,
            entropy=entropy,
            char_classes=count_character_classes(text),
            confidence_adjustment=0.20,
            reason=f"Known secret prefix detected (entropy={entropy:.2f})"
        )

    # Check if this is a labeled credential (has built-in context)
    is_labeled = has_credential_label(text)

    # Extract the actual value if labeled
    if is_labeled:
        value = extract_credential_value(text)
        text_to_analyze = value
    else:
        text_to_analyze = text

    # Check for known false positives
    if is_known_false_positive(text_to_analyze):
        return EntropyResult(
            is_credential=False,
            entropy=0.0,
            char_classes=0,
            confidence_adjustment=-0.50,
            reason="Known false positive pattern"
        )

    # Calculate entropy
    entropy = calculate_shannon_entropy(text_to_analyze)
    char_classes = count_character_classes(text_to_analyze)

    # Get thresholds based on whether this is labeled
    if is_labeled:
        min_entropy = ENTROPY_THRESHOLDS["min_labeled"]
        min_length = MIN_LABELED_LENGTH
    else:
        min_entropy = ENTROPY_THRESHOLDS["min_unlabeled"]
        min_length = MIN_UNLABELED_LENGTH

    # Length check
    if len(text_to_analyze) < min_length:
        return EntropyResult(
            is_credential=False,
            entropy=entropy,
            char_classes=char_classes,
            confidence_adjustment=-0.30,
            reason=f"Too short: {len(text_to_analyze)} < {min_length} chars"
        )

    # Entropy too low - likely natural language or placeholder
    if entropy < min_entropy:
        return EntropyResult(
            is_credential=False,
            entropy=entropy,
            char_classes=char_classes,
            confidence_adjustment=-0.40,
            reason=f"Low entropy: {entropy:.2f} < {min_entropy}"
        )

    # Entropy too high - likely random noise or binary data
    if entropy > ENTROPY_THRESHOLDS["max_valid"]:
        return EntropyResult(
            is_credential=False,
            entropy=entropy,
            char_classes=char_classes,
            confidence_adjustment=-0.30,
            reason=f"Entropy too high: {entropy:.2f} > {ENTROPY_THRESHOLDS['max_valid']}"
        )

    # CONTEXT-ANCHORED CHECK for unlabeled credentials
    # This is the key precision improvement: require trigger word nearby
    if not is_labeled and full_text and position_start >= 0:
        is_anchored, anchor_reason = check_context_anchor(
            text, full_text, position_start, position_end
        )
        if not is_anchored:
            return EntropyResult(
                is_credential=False,
                entropy=entropy,
                char_classes=char_classes,
                confidence_adjustment=-0.45,
                reason=f"No context anchor: {anchor_reason}"
            )

    # Check character class diversity for unlabeled credentials
    if not is_labeled and char_classes < 2:
        return EntropyResult(
            is_credential=False,
            entropy=entropy,
            char_classes=char_classes,
            confidence_adjustment=-0.35,
            reason=f"Low character diversity: {char_classes} classes"
        )

    # Check for low-confidence words in unlabeled credentials
    if not is_labeled:
        text_lower = text_to_analyze.lower()
        for word in LOW_CONFIDENCE_WORDS:
            if word in text_lower:
                return EntropyResult(
                    is_credential=False,
                    entropy=entropy,
                    char_classes=char_classes,
                    confidence_adjustment=-0.25,
                    reason=f"Contains low-confidence word: '{word}'"
                )

    # Calculate confidence adjustment based on entropy quality
    ideal_low = ENTROPY_THRESHOLDS["ideal_low"]
    ideal_high = ENTROPY_THRESHOLDS["ideal_high"]

    if ideal_low <= entropy <= ideal_high:
        # Ideal entropy range - boost confidence
        adjustment = 0.15 + (char_classes * 0.05)  # Up to +0.35
        reason = f"Ideal entropy range: {entropy:.2f}"
    else:
        # Acceptable but not ideal
        adjustment = 0.05
        if entropy < ideal_low:
            reason = f"Acceptable entropy (low side): {entropy:.2f}"
        else:
            reason = f"Acceptable entropy (high side): {entropy:.2f}"

    # Bonus for labeled credentials
    if is_labeled:
        adjustment += 0.10
        reason += " + labeled credential bonus"

    return EntropyResult(
        is_credential=True,
        entropy=entropy,
        char_classes=char_classes,
        confidence_adjustment=adjustment,
        reason=reason
    )


def filter_credential_by_entropy(
    text: str,
    base_confidence: float,
    context: str = "",
    full_text: str = "",
    position_start: int = -1,
    position_end: int = -1
) -> Tuple[bool, float, str]:
    """
    Filter a credential detection using entropy analysis with context anchoring.

    Convenience function that returns (should_keep, new_confidence, reason).

    Args:
        text: The detected credential text
        base_confidence: Original confidence from pattern matching
        context: Surrounding text for context (deprecated)
        full_text: Complete document text for context anchor checking
        position_start: Start position of credential in full_text
        position_end: End position of credential in full_text

    Returns:
        Tuple of (should_keep, adjusted_confidence, reason)
    """
    result = analyze_credential_entropy(
        text, context,
        full_text=full_text,
        position_start=position_start,
        position_end=position_end
    )

    new_confidence = base_confidence + result.confidence_adjustment
    new_confidence = max(0.0, min(1.0, new_confidence))

    return (result.is_credential, new_confidence, result.reason)


# =============================================================================
# Batch Analysis for Benchmarking
# =============================================================================

def analyze_credential_batch(credentials: list) -> dict:
    """
    Analyze a batch of credential candidates for benchmarking.

    Args:
        credentials: List of dicts with 'text' and optionally 'is_true_positive'

    Returns:
        Summary statistics and per-credential results
    """
    results = []
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for cred in credentials:
        text = cred.get('text', '')
        is_true = cred.get('is_true_positive', None)

        result = analyze_credential_entropy(text)
        results.append({
            'text': text,
            'is_true_positive': is_true,
            'predicted_credential': result.is_credential,
            'entropy': result.entropy,
            'char_classes': result.char_classes,
            'reason': result.reason,
        })

        if is_true is not None:
            if is_true and result.is_credential:
                true_positives += 1
            elif not is_true and not result.is_credential:
                true_negatives += 1
            elif not is_true and result.is_credential:
                false_positives += 1
            else:
                false_negatives += 1

    total = len(credentials)
    labeled = sum(1 for c in credentials if c.get('is_true_positive') is not None)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'total': total,
        'labeled': labeled,
        'true_positives': true_positives,
        'true_negatives': true_negatives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'results': results,
    }


# =============================================================================
# Entropy Distribution Analysis
# =============================================================================

def entropy_distribution_report(texts: list) -> str:
    """
    Generate a report on entropy distribution for a list of texts.

    Useful for tuning thresholds based on real data.

    Args:
        texts: List of potential credential strings

    Returns:
        Formatted report string
    """
    entropies = [(t, calculate_shannon_entropy(t)) for t in texts]
    entropies.sort(key=lambda x: x[1])

    report = []
    report.append("=" * 60)
    report.append("ENTROPY DISTRIBUTION REPORT")
    report.append("=" * 60)
    report.append(f"Total samples: {len(entropies)}")

    if entropies:
        values = [e for _, e in entropies]
        report.append(f"Min entropy: {min(values):.2f}")
        report.append(f"Max entropy: {max(values):.2f}")
        report.append(f"Mean entropy: {sum(values)/len(values):.2f}")

        # Histogram buckets
        buckets = {}
        for _, e in entropies:
            bucket = int(e)
            buckets[bucket] = buckets.get(bucket, 0) + 1

        report.append("\nEntropy distribution:")
        for bucket in sorted(buckets.keys()):
            bar = "#" * (buckets[bucket] * 40 // len(entropies))
            report.append(f"  {bucket}.0-{bucket}.9: {buckets[bucket]:4d} {bar}")

        report.append("\nSample low-entropy texts:")
        for text, entropy in entropies[:5]:
            report.append(f"  [{entropy:.2f}] {text[:50]}...")

        report.append("\nSample high-entropy texts:")
        for text, entropy in entropies[-5:]:
            report.append(f"  [{entropy:.2f}] {text[:50]}...")

    report.append("=" * 60)
    return "\n".join(report)
