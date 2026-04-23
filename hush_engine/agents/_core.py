"""
Shared helpers for all agent transports (MCP, Claude Agent SDK, OpenAI Agents SDK).

Every transport under `hush_engine.agents.*` wraps the same two operations:

- `detect_pii_json(text, entity_types)`      -> list[dict], JSON-serializable.
- `redact_text_inline(text, entity_types)`   -> str with PII replaced inline.

Transport modules stay as thin as possible and delegate here so changes to the
detector contract flow through one place.

The detector is constructed once per process with libpostal disabled. Agents
care about throughput and latency; libpostal adds ~30–40 % on every document
for a LOCATION lift the average agent task does not need.
"""

from __future__ import annotations

from threading import Lock
from typing import List, Optional

_DETECTOR = None
_DETECTOR_LOCK = Lock()


def _get_detector():
    """Lazy singleton. Skipped at import time so the process doesn't spin up
    Presidio + LightGBM just because one of the transport modules got
    imported (e.g. during `pip install` metadata introspection)."""
    global _DETECTOR
    if _DETECTOR is not None:
        return _DETECTOR
    with _DETECTOR_LOCK:
        if _DETECTOR is None:
            from hush_engine.detectors.pii_detector import PIIDetector
            _DETECTOR = PIIDetector(enable_libpostal=False)
    return _DETECTOR


def detect_pii_json(
    text: str,
    entity_types: Optional[List[str]] = None,
) -> List[dict]:
    """Run Hush detection on `text` and return a JSON-serializable list.

    Each detection is a dict with keys:
        entity_type  str   e.g. "PERSON", "EMAIL_ADDRESS"
        text         str   the exact substring the detector matched
        start        int   character offset, inclusive
        end          int   character offset, exclusive
        confidence   float 0.0 – 1.0

    Args:
        text: The text to analyze. Empty or missing returns an empty list.
        entity_types: Allow-list of entity types to keep. None keeps all.

    Returns:
        A list of dicts sorted by `start`.
    """
    if not text:
        return []

    detections = _get_detector().analyze_text(text)
    out: List[dict] = []
    for d in detections:
        if entity_types and d.entity_type not in entity_types:
            continue
        out.append(
            {
                "entity_type": d.entity_type,
                "text": d.text,
                "start": int(d.start),
                "end": int(d.end),
                "confidence": float(d.confidence),
            }
        )
    out.sort(key=lambda e: e["start"])
    return out


def redact_text_inline(
    text: str,
    entity_types: Optional[List[str]] = None,
    mask: str = "[{entity_type}]",
) -> str:
    """Return `text` with detected PII replaced inline.

    Default mask tags each span with its type, so `John Doe` becomes
    `[PERSON]`. Pass `mask="***"` to blank every span uniformly, or any
    other template with `{entity_type}` to customize.

    Spans are replaced from the end of the text to the start so offsets
    stay valid as edits accumulate.
    """
    if not text:
        return text
    detections = detect_pii_json(text, entity_types=entity_types)
    if not detections:
        return text
    result = text
    for d in reversed(detections):
        replacement = mask.format(entity_type=d["entity_type"])
        result = result[: d["start"]] + replacement + result[d["end"] :]
    return result


__all__ = ["detect_pii_json", "redact_text_inline"]
