#!/usr/bin/env python3
"""
MLX-based LLM Verifier for PII Detection Precision

This module provides a local LLM verification step to reduce false positives
in PII detection by assessing surrounding context.

Security guarantees:
- Model runs entirely locally via MLX (no network calls)
- No data persistence - all processing in-memory only
- Context window limited to minimize exposure
- Can be disabled via config toggle

Usage:
    verifier = LLMVerifier()
    is_pii = verifier.verify_pii("John Smith", "PERSON", "Contact John Smith at...")
"""

import logging
import re
from typing import List, Tuple, Optional, Any

logger = logging.getLogger(__name__)

# MLX availability flag
MLX_AVAILABLE = False
try:
    import mlx_lm
    MLX_AVAILABLE = True
except ImportError:
    logger.debug("mlx_lm not available - LLM verification disabled")


class LLMVerifier:
    """
    Local LLM-based verifier for PII detection precision.

    Uses MLX (Apple Silicon optimized) for fast, local inference.
    Targets false positives by assessing surrounding context.
    """

    # Default model - small, fast, good for classification tasks
    DEFAULT_MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"

    def __init__(
        self,
        model_name: str = None,
        enabled: bool = True,
        min_confidence_threshold: float = 0.45,
        max_confidence_skip: float = 0.75
    ):
        """
        Initialize the LLM Verifier.

        Args:
            model_name: MLX model to use (default: Llama-3.2-1B-Instruct-4bit)
            enabled: Whether verification is enabled
            min_confidence_threshold: Only verify candidates above this confidence
            max_confidence_skip: Skip verification for candidates above this confidence
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.enabled = enabled and MLX_AVAILABLE
        self.min_confidence_threshold = min_confidence_threshold
        self.max_confidence_skip = max_confidence_skip

        self._model = None
        self._tokenizer = None
        self._loaded = False

        if self.enabled:
            logger.info(f"[LLMVerifier] Initialized with model: {self.model_name}")
        else:
            if not MLX_AVAILABLE:
                logger.info("[LLMVerifier] Disabled - mlx_lm not installed")
            else:
                logger.info("[LLMVerifier] Disabled by configuration")

    def _ensure_loaded(self) -> bool:
        """Lazy load the model on first use."""
        if self._loaded:
            return True

        if not self.enabled:
            return False

        try:
            logger.info(f"[LLMVerifier] Loading model: {self.model_name}")
            self._model, self._tokenizer = mlx_lm.load(self.model_name)
            self._loaded = True
            logger.info("[LLMVerifier] Model loaded successfully")
            return True
        except Exception as e:
            logger.warning(f"[LLMVerifier] Failed to load model: {e}")
            self.enabled = False
            return False

    def verify_pii(
        self,
        candidate_text: str,
        entity_type: str,
        context: str
    ) -> bool:
        """
        Verify if detected text is actually PII.

        Uses a strict YES/NO prompt to determine if the detected text
        is genuinely the specified entity type given the surrounding context.

        Args:
            candidate_text: The detected PII text
            entity_type: Type of PII (PERSON, ADDRESS, EMAIL, etc.)
            context: 5-10 surrounding words for context

        Returns:
            True if confirmed as PII, False if likely false positive
        """
        if not self._ensure_loaded():
            # If model not available, default to accepting the detection
            return True

        # Build a strict verification prompt
        prompt = self._build_verification_prompt(candidate_text, entity_type, context)

        try:
            response = mlx_lm.generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=150,  # Increased for Chain-of-Thought reasoning
                temp=0.0  # Deterministic output
            )

            # Parse CoT response - look for final YES/NO verdict
            is_confirmed = self._parse_cot_response(response)

            logger.debug(
                f"[LLMVerifier] '{candidate_text}' as {entity_type}: "
                f"{'CONFIRMED' if is_confirmed else 'REJECTED'} (response: {response_clean[:20]})"
            )

            return is_confirmed

        except Exception as e:
            logger.warning(f"[LLMVerifier] Verification failed: {e}")
            # On error, default to accepting the detection
            return True

    def _parse_cot_response(self, response: str) -> bool:
        """
        Parse Chain-of-Thought response, looking for final YES/NO verdict.

        Handles responses that include reasoning before the final answer.
        """
        response_lower = response.lower().strip()

        # Look for explicit verdict at the end
        if response_lower.endswith('yes') or response_lower.endswith('yes.'):
            return True
        if response_lower.endswith('no') or response_lower.endswith('no.'):
            return False

        # Look for "Answer: YES/NO" pattern
        answer_match = re.search(r'answer[:\s]+\**(yes|no)\**', response_lower)
        if answer_match:
            return answer_match.group(1) == 'yes'

        # Fall back to counting keywords (if only one appears)
        has_yes = 'yes' in response_lower
        has_no = 'no' in response_lower

        if has_yes and not has_no:
            return True
        if has_no and not has_yes:
            return False

        # Ambiguous - default to accepting detection (conservative)
        return True

    def _build_verification_prompt(
        self,
        candidate_text: str,
        entity_type: str,
        context: str
    ) -> str:
        """
        Build a Chain-of-Thought verification prompt for the LLM.

        Uses step-by-step reasoning to improve verification accuracy.
        """
        # Entity-specific CoT prompts
        cot_prompts = {
            "PERSON": f'''Is "{candidate_text}" a person's name?

Context: "{context}"

Think step by step:
1. Is this a common first name, last name, or full name?
2. Could this be a brand name, product, or organization?
3. Could this be a UI element, button, or form label?
4. Does the context suggest a real person?

Answer YES or NO.''',

            "COMPANY": f'''Is "{candidate_text}" a company or organization name?

Context: "{context}"

Think step by step:
1. Does this look like a registered business name?
2. Could this be a generic term or product name?
3. Is there organizational context (Inc, LLC, Corp)?
4. Does the context suggest an actual company?

Answer YES or NO.''',

            "ADDRESS": f'''Is "{candidate_text}" a specific physical address?

Context: "{context}"

Think step by step:
1. Does this contain street number, street name, or building identifier?
2. Does this include city, state/province, or postal code?
3. Could this be directions, a general location, or a landmark?
4. Is this specific enough to identify a physical location?

Answer YES or NO.''',

            "LOCATION": f'''Is "{candidate_text}" a specific geographic location?

Context: "{context}"

Think step by step:
1. Is this a city, country, region, or landmark name?
2. Could this be a generic place word (here, there, location)?
3. Is it part of an address or just a place reference?
4. Does the context suggest a real, specific location?

Answer YES or NO.''',

            "EMAIL": f'''Is "{candidate_text}" a valid email address?

Context: "{context}"

Think step by step:
1. Does it have the format user@domain?
2. Does the domain look real (not example.com or placeholder)?
3. Could this be a partial email or typo?

Answer YES or NO.''',

            "PHONE": f'''Is "{candidate_text}" a telephone number?

Context: "{context}"

Think step by step:
1. Does it have the format of a phone number?
2. Could it be an ID number, reference number, or date?
3. Does the context suggest it's used for calling?

Answer YES or NO.''',

            "DATE_TIME": f'''Is "{candidate_text}" a specific date or time?

Context: "{context}"

Think step by step:
1. Does this represent a specific date, time, or both?
2. Could this be a duration, period, or recurring event?
3. Is it a generic time reference (annual, monthly)?

Answer YES or NO.''',

            "MEDICAL": f'''Is "{candidate_text}" medical/health information?

Context: "{context}"

Think step by step:
1. Is this a diagnosis, condition, medication, or treatment?
2. Could this be a general health term (not specific to a person)?
3. Does the context suggest it's about a specific patient?

Answer YES or NO.''',

            "FINANCIAL": f'''Is "{candidate_text}" specific financial account information?

Context: "{context}"

Think step by step:
1. Is this an account number, routing number, or card number?
2. Could this be a general amount, price, or ID?
3. Does it identify a specific financial account?

Answer YES or NO.''',
        }

        # Use CoT prompt if available, otherwise generic
        if entity_type in cot_prompts:
            return cot_prompts[entity_type]

        # Generic fallback prompt with reasoning
        return f'''Is "{candidate_text}" definitely a {entity_type}?

Context: "{context}"

Think about what type of data this is and whether it matches {entity_type}.

Answer YES or NO.'''

    def verify_batch(
        self,
        candidates: List[Tuple[str, str, str, float]],
    ) -> List[Tuple[str, str, str, float, bool]]:
        """
        Verify a batch of PII candidates.

        Only verifies candidates in the mid-confidence range
        (between min_confidence_threshold and max_confidence_skip).

        Args:
            candidates: List of (text, entity_type, context, confidence)

        Returns:
            List of (text, entity_type, context, confidence, verified)
        """
        if not self.enabled:
            # If disabled, accept all candidates
            return [(t, e, c, conf, True) for t, e, c, conf in candidates]

        results = []

        for text, entity_type, context, confidence in candidates:
            # Skip verification for high-confidence detections
            if confidence >= self.max_confidence_skip:
                results.append((text, entity_type, context, confidence, True))
                continue

            # Skip verification for low-confidence (already filtered by threshold)
            if confidence < self.min_confidence_threshold:
                results.append((text, entity_type, context, confidence, False))
                continue

            # Verify mid-range confidence candidates
            verified = self.verify_pii(text, entity_type, context)
            results.append((text, entity_type, context, confidence, verified))

        return results

    def is_available(self) -> bool:
        """Check if the verifier is available and enabled."""
        return self.enabled and MLX_AVAILABLE

    def get_stats(self) -> dict:
        """Get verifier statistics."""
        return {
            "enabled": self.enabled,
            "mlx_available": MLX_AVAILABLE,
            "model": self.model_name,
            "loaded": self._loaded,
            "min_confidence_threshold": self.min_confidence_threshold,
            "max_confidence_skip": self.max_confidence_skip,
        }


# Global instance for convenience
_verifier_instance: Optional[LLMVerifier] = None


def get_verifier(enabled: bool = True) -> LLMVerifier:
    """Get the global LLM verifier instance."""
    global _verifier_instance
    if _verifier_instance is None:
        _verifier_instance = LLMVerifier(enabled=enabled)
    return _verifier_instance


def verify_pii_candidate(
    candidate_text: str,
    entity_type: str,
    context: str,
    confidence: float
) -> bool:
    """
    Convenience function to verify a single PII candidate.

    Args:
        candidate_text: The detected PII text
        entity_type: Type of PII
        context: Surrounding context
        confidence: Detection confidence

    Returns:
        True if confirmed as PII or verification skipped
    """
    verifier = get_verifier()

    # Skip for high confidence
    if confidence >= verifier.max_confidence_skip:
        return True

    # Skip for low confidence
    if confidence < verifier.min_confidence_threshold:
        return False

    return verifier.verify_pii(candidate_text, entity_type, context)
