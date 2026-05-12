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
from typing import List, Optional, Tuple

from presidio_analyzer import AnalysisExplanation, EntityRecognizer, RecognizerResult

# Default model id — override via HUSH_PRIVACY_FILTER_MODEL env var if you
# have a fine-tuned checkpoint on disk.
_DEFAULT_MODEL_ID = "openai/privacy-filter"

# Pinned commit on huggingface.co/openai/privacy-filter as of 2026-05-12.
# Mitigates supply-chain risk: a malicious typosquat (Open-OSS/privacy-filter)
# briefly hit HF's trending list with 244K downloads before takedown. Pinning
# a known-good SHA closes the door on a future repo compromise. Overridable
# via HUSH_PRIVACY_FILTER_REVISION (set to empty string to disable pinning).
# Only applied when loading the canonical Hub model; user-supplied paths
# bypass the pin.
_PINNED_REVISION = "7ffa9a043d54d1be65afb281eddf0ffbe629385b"

# ONNX file inside the repo when HUSH_PRIVACY_FILTER_BACKEND=onnx. The
# quantized variant is ~4x smaller than BF16 and runs without torch.
_DEFAULT_ONNX_FILE = "onnx/model_quantized.onnx"

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
    """Lazy-load the OpenAI Privacy Filter HF pipeline on first use.

    Backend selection via ``HUSH_PRIVACY_FILTER_BACKEND``:
      - ``torch`` (default): the original transformers pipeline path.
      - ``onnx``: load via ``optimum.onnxruntime`` for browser/edge deployments
        where torch is too heavy. Requires ``pip install hush-engine[privacy-filter-onnx]``.
        File inside the repo is controlled by ``HUSH_PRIVACY_FILTER_ONNX_FILE``
        (default ``onnx/model_quantized.onnx``; other options published by
        OpenAI: ``onnx/model.onnx``, ``onnx/model_fp16.onnx``,
        ``onnx/model_q4.onnx``, ``onnx/model_q4f16.onnx``).
    """
    global _privacy_filter_pipeline, PRIVACY_FILTER_AVAILABLE

    if _privacy_filter_pipeline is not None:
        return

    import os
    resolved_model = (
        model_id
        or os.environ.get("HUSH_PRIVACY_FILTER_MODEL")
        or _DEFAULT_MODEL_ID
    )

    # Apply the pinned revision only when fetching the canonical Hub model.
    # User-supplied paths/fine-tunes bypass the pin so local checkpoints work.
    revision_env = os.environ.get("HUSH_PRIVACY_FILTER_REVISION")
    if revision_env is not None:
        revision = revision_env or None  # empty string disables pinning
    elif resolved_model == _DEFAULT_MODEL_ID:
        revision = _PINNED_REVISION
    else:
        revision = None

    backend = os.environ.get("HUSH_PRIVACY_FILTER_BACKEND", "torch").lower()

    try:
        from transformers import pipeline
    except ImportError:
        sys.stderr.write(
            "[PrivacyFilter] transformers not installed. "
            "Run: pip install hush-engine[privacy-filter] (or [privacy-filter-onnx])\n"
        )
        return

    try:
        if backend == "onnx":
            try:
                from optimum.onnxruntime import ORTModelForTokenClassification
                from transformers import AutoTokenizer
            except ImportError:
                sys.stderr.write(
                    "[PrivacyFilter] optimum[onnxruntime] not installed. "
                    "Run: pip install hush-engine[privacy-filter-onnx]\n"
                )
                return
            onnx_file = os.environ.get(
                "HUSH_PRIVACY_FILTER_ONNX_FILE", _DEFAULT_ONNX_FILE
            )
            tokenizer = AutoTokenizer.from_pretrained(resolved_model, revision=revision)
            ort_model = ORTModelForTokenClassification.from_pretrained(
                resolved_model, file_name=onnx_file, revision=revision
            )
            _privacy_filter_pipeline = pipeline(
                task="token-classification",
                model=ort_model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
            )
        else:
            # aggregation_strategy="simple" collapses BIOES token tags into
            # whole spans and is what the HF model card example uses.
            _privacy_filter_pipeline = pipeline(
                task="token-classification",
                model=resolved_model,
                revision=revision,
                aggregation_strategy="simple",
                device=-1,  # CPU; transformers auto-upgrades to MPS/CUDA if available elsewhere
            )
        PRIVACY_FILTER_AVAILABLE = True
        rev_tag = f", rev={revision[:7]}" if revision else ""
        sys.stderr.write(
            f"[PrivacyFilter] Loaded {resolved_model} (backend={backend}{rev_tag})\n"
        )
    except Exception as e:
        sys.stderr.write(f"[PrivacyFilter] Load failed ({resolved_model}): {e}\n")


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

    The `excluded_entities` parameter removes specific hush-mapped entity types
    from the output even when PF detects them. Default shipping exclude list
    is ["PHONE_NUMBER"] because the 2026-04-23 Kaggle ablation showed PF's
    phone spans cost 5.71 pp of PHONE F1 versus Hush's libphonenumber-validated
    output. Set to an empty iterable to let PF contribute everywhere.
    """

    ENTITIES = sorted(set(OPENAI_LABEL_TO_HUSH.values()))

    def __init__(
        self,
        supported_language: str = "en",
        supported_entities: Optional[List[str]] = None,
        min_confidence: float = 0.55,
        excluded_entities: Optional[List[str]] = None,
    ):
        supported_entities = supported_entities or self.ENTITIES
        self.min_confidence = min_confidence
        self.excluded_entities = set(excluded_entities or [])
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
        # Honor the excluded_entities filter: PF won't run at all if every
        # requested entity type is excluded.
        wanted = wanted - self.excluded_entities
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


def get_privacy_filter_recognizer(
    excluded_entities: Optional[List[str]] = None,
) -> Optional["PrivacyFilterRecognizer"]:
    """Return a ready recognizer instance, or None if the model is unavailable.

    Args:
        excluded_entities: Hush entity types to suppress in PF output. Callers
            that read `DetectionConfig.get_privacy_filter_excluded_entities()`
            pass the list here.
    """
    _load_privacy_filter()
    if not PRIVACY_FILTER_AVAILABLE:
        return None
    return PrivacyFilterRecognizer(excluded_entities=excluded_entities)


__all__ = [
    "OPENAI_LABEL_TO_HUSH",
    "PrivacyFilterRecognizer",
    "detect_persons_with_privacy_filter",
    "detect_with_privacy_filter",
    "get_privacy_filter_recognizer",
    "is_privacy_filter_available",
]
