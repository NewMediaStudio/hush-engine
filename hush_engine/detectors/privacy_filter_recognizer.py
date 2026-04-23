"""
OpenAI Privacy Filter integration (add-on backend).

Wraps the Apache-2.0 token-classification model released 2026-04-22
(https://huggingface.co/openai/privacy-filter) and exposes:

- A cached HF pipeline with lazy first-use loading.
- `detect_with_privacy_filter` — the shared span query used by both
  `PersonRecognizer` (PERSON cascade) and `PrivacyFilterRecognizer`
  (the six non-PERSON span classes).
- `PrivacyFilterRecognizer` — a Presidio `EntityRecognizer` that maps the
  non-PERSON OpenAI labels to hush entity types:

      private_email   -> EMAIL_ADDRESS
      private_phone   -> PHONE_NUMBER
      private_address -> LOCATION
      private_url     -> URL
      private_date    -> DATE_TIME
      account_number  -> FINANCIAL
      secret          -> CREDENTIAL

  `private_person` is handled separately by `PersonRecognizer` so the
  engine's existing ensemble-voting / verifier pipeline can still weigh
  or override the hit depending on `privacy_filter_authoritative`.

Opt-in: requires `pip install hush-engine[privacy-filter]` and toggling
`DetectionConfig.set_enabled_integration("openai_privacy_filter", True)`.
Silent import failure keeps the engine usable without the extra installed.
"""

import sys
from typing import Dict, List, Optional, Tuple

from presidio_analyzer import AnalysisExplanation, EntityRecognizer, RecognizerResult

# Default model id — override via HUSH_PRIVACY_FILTER_MODEL env var if you
# have a fine-tuned checkpoint on disk.
_DEFAULT_MODEL_ID = "openai/privacy-filter"

# OpenAI label -> hush entity type.
# `private_person` is intentionally absent: PERSON is routed through the
# PersonRecognizer cascade so authoritative/candidate gating applies.
OPENAI_LABEL_TO_HUSH = {
    "private_email": "EMAIL_ADDRESS",
    "private_phone": "PHONE_NUMBER",
    "private_address": "LOCATION",
    "private_url": "URL",
    "private_date": "DATE_TIME",
    "account_number": "FINANCIAL",
    "secret": "CREDENTIAL",
}

# Module-level singleton + availability flag (same pattern used by
# _load_flair / _load_transformers_ner in person_recognizer.py).
_privacy_filter_pipeline = None
PRIVACY_FILTER_AVAILABLE = False


def _load_privacy_filter(model_id: Optional[str] = None) -> None:
    """Lazy-load the OpenAI Privacy Filter HF pipeline on first use."""
    global _privacy_filter_pipeline, PRIVACY_FILTER_AVAILABLE

    if _privacy_filter_pipeline is not None:
        return

    import os
    model_id = model_id or os.environ.get("HUSH_PRIVACY_FILTER_MODEL", _DEFAULT_MODEL_ID)

    try:
        from transformers import pipeline
    except ImportError:
        sys.stderr.write(
            "[PrivacyFilter] transformers not installed. "
            "Run: pip install hush-engine[privacy-filter]\n"
        )
        return

    try:
        # aggregation_strategy="simple" collapses BIOES token tags into whole
        # spans and is what the HF model card example uses.
        _privacy_filter_pipeline = pipeline(
            task="token-classification",
            model=model_id,
            aggregation_strategy="simple",
            device=-1,  # CPU; transformers auto-upgrades to MPS/CUDA if available elsewhere
        )
        PRIVACY_FILTER_AVAILABLE = True
        sys.stderr.write(f"[PrivacyFilter] Loaded {model_id}\n")
    except Exception as e:
        sys.stderr.write(f"[PrivacyFilter] Load failed ({model_id}): {e}\n")


def is_privacy_filter_available() -> bool:
    """True iff the pipeline loaded successfully (does NOT trigger a load)."""
    return PRIVACY_FILTER_AVAILABLE


def detect_with_privacy_filter(
    text: str,
    target_labels: Optional[List[str]] = None,
) -> List[Tuple[str, int, int, float, str]]:
    """Run Privacy Filter over `text` and return raw span hits.

    Args:
        text: Input text.
        target_labels: Optional OpenAI label allow-list (e.g. ["private_person"]
            when called from the PERSON cascade). If None, all labels are
            returned so the caller can map them.

    Returns:
        List of (text, start, end, score, openai_label) tuples.
    """
    _load_privacy_filter()

    if not PRIVACY_FILTER_AVAILABLE or _privacy_filter_pipeline is None:
        return []

    if not text:
        return []

    results: List[Tuple[str, int, int, float, str]] = []
    try:
        entities = _privacy_filter_pipeline(text)
    except Exception as e:
        sys.stderr.write(f"[PrivacyFilter] inference error: {e}\n")
        return results

    for ent in entities:
        label = ent.get("entity_group") or ent.get("entity") or ""
        if target_labels and label not in target_labels:
            continue

        score = float(ent.get("score", 0.0))
        start = int(ent.get("start", 0))
        end = int(ent.get("end", 0))
        span_text = ent.get("word", text[start:end])

        # HF pipeline sometimes prepends a leading space to the aggregated
        # word; trim it and fix up `start` to match the real span.
        if span_text.startswith(" ") and start < end:
            span_text = span_text[1:]
            start += 1
        if end <= start:
            continue

        results.append((span_text, start, end, score, label))

    return results


def detect_persons_with_privacy_filter(text: str) -> List[Tuple[str, int, int, float]]:
    """Convenience wrapper for the PERSON cascade.

    Returns only the `private_person` hits as (text, start, end, score) tuples,
    shaped to match the other `detect_with_*` functions in person_recognizer.py.
    """
    hits = detect_with_privacy_filter(text, target_labels=["private_person"])
    return [(span, start, end, score) for span, start, end, score, _label in hits]


class PrivacyFilterRecognizer(EntityRecognizer):
    """Presidio recognizer covering the non-PERSON Privacy Filter labels.

    PERSON is intentionally excluded so the existing `PersonRecognizer`
    cascade retains ownership of that entity type (and so the
    `privacy_filter_authoritative` toggle can gate it without double-counting).
    """

    ENTITIES = sorted(set(OPENAI_LABEL_TO_HUSH.values()))

    def __init__(
        self,
        supported_language: str = "en",
        supported_entities: Optional[List[str]] = None,
        min_confidence: float = 0.55,
    ):
        supported_entities = supported_entities or self.ENTITIES
        self.min_confidence = min_confidence
        super().__init__(
            supported_entities=supported_entities,
            supported_language=supported_language,
            name="PrivacyFilterRecognizer",
        )

    def load(self) -> None:
        _load_privacy_filter()

    def analyze(
        self,
        text: str,
        entities: List[str],
        nlp_artifacts=None,
    ) -> List[RecognizerResult]:
        # Only run if at least one requested entity is one we produce.
        wanted = set(entities) & set(self.ENTITIES)
        if not wanted:
            return []

        # Only ask the model for OpenAI labels that map to wanted entities.
        target_labels = [
            lbl for lbl, hush in OPENAI_LABEL_TO_HUSH.items() if hush in wanted
        ]
        hits = detect_with_privacy_filter(text, target_labels=target_labels)

        results: List[RecognizerResult] = []
        for span_text, start, end, score, openai_label in hits:
            if score < self.min_confidence:
                continue
            hush_type = OPENAI_LABEL_TO_HUSH.get(openai_label)
            if not hush_type or hush_type not in wanted:
                continue

            explanation = AnalysisExplanation(
                recognizer=self.name,
                original_score=score,
                pattern_name=f"privacy_filter_{openai_label}",
                pattern=None,
                validation_result=None,
            )
            results.append(
                RecognizerResult(
                    entity_type=hush_type,
                    start=start,
                    end=end,
                    score=score,
                    analysis_explanation=explanation,
                    recognition_metadata={
                        "recognizer_name": self.name,
                        "detection_source": "openai_privacy_filter",
                        "openai_label": openai_label,
                    },
                )
            )
        return results


def get_privacy_filter_recognizer() -> Optional["PrivacyFilterRecognizer"]:
    """Return a ready recognizer instance, or None if the model is unavailable."""
    _load_privacy_filter()
    if not PRIVACY_FILTER_AVAILABLE:
        return None
    return PrivacyFilterRecognizer()


__all__ = [
    "OPENAI_LABEL_TO_HUSH",
    "PrivacyFilterRecognizer",
    "detect_persons_with_privacy_filter",
    "detect_with_privacy_filter",
    "get_privacy_filter_recognizer",
    "is_privacy_filter_available",
]
