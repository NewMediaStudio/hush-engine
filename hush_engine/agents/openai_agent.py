"""
Hush Engine tools for the OpenAI Agents SDK (`openai-agents`).

Install:   pip install hush-engine[agent-openai]

Usage:
    from agents import Agent, Runner
    from hush_engine.agents.openai_agent import detect_pii_tool, redact_text_tool

    agent = Agent(
        name="SafeAssistant",
        instructions="Redact PII before answering questions about user data.",
        tools=[detect_pii_tool, redact_text_tool],
    )
    result = await Runner.run(agent, "Summarize this: My name is John Doe, email john@x.com.")
    print(result.final_output)
"""

from __future__ import annotations

from typing import List, Optional

try:
    from agents import function_tool
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The OpenAI Agents SDK tool requires the `openai-agents` package. "
        "Install with: pip install hush-engine[agent-openai]"
    ) from exc

from ._core import detect_pii_json, redact_text_inline


@function_tool
def detect_pii_tool(
    text: str,
    entity_types: Optional[List[str]] = None,
) -> List[dict]:
    """Detect personally identifiable information in text using Hush Engine.

    Returns a list of detections, each with entity_type (e.g. "PERSON",
    "EMAIL_ADDRESS"), the matched text, start/end character offsets, and
    a 0.0–1.0 confidence score. Hush covers 27 entity types including
    Luhn-validated credit cards, IBAN, SSN and passport across 35+
    countries, AWS and Stripe keys, medical codes, and more.

    Args:
        text: The text to scan for PII.
        entity_types: Optional allow-list of entity types to return, for
            example ["PERSON", "EMAIL_ADDRESS"]. If omitted, all 27 types
            are reported.
    """
    return detect_pii_json(text, entity_types=entity_types)


@function_tool
def redact_text_tool(
    text: str,
    entity_types: Optional[List[str]] = None,
) -> str:
    """Redact personally identifiable information in text.

    Replaces each detected PII span with a tag derived from the entity
    type, so "Call John at 555-1234" becomes "Call [PERSON] at
    [PHONE_NUMBER]". Useful as a pre-processing step before sending user
    content to an LLM that should not see raw PII.

    Args:
        text: The text to redact.
        entity_types: Optional allow-list of entity types to redact. If
            omitted, every detected PII span is replaced.
    """
    return redact_text_inline(text, entity_types=entity_types)


__all__ = ["detect_pii_tool", "redact_text_tool"]
