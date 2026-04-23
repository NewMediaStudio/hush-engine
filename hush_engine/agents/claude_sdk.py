"""
Hush Engine tools for the Claude Agent SDK (`claude-agent-sdk`).

Builds an in-process MCP server so Claude agents can call Hush without
spawning a subprocess.

Install:   pip install hush-engine[agent-claude]

Usage:
    from claude_agent_sdk import ClaudeAgentOptions, query
    from hush_engine.agents.claude_sdk import hush_server

    options = ClaudeAgentOptions(
        mcp_servers={"hush": hush_server},
        allowed_tools=[
            "mcp__hush__detect_pii",
            "mcp__hush__redact_text",
        ],
    )
    async for message in query(prompt="Redact PII then summarize: ...", options=options):
        print(message)
"""

from __future__ import annotations

import json
from typing import Any

try:
    from claude_agent_sdk import create_sdk_mcp_server, tool
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The Claude Agent SDK helper requires the `claude-agent-sdk` package. "
        "Install with: pip install hush-engine[agent-claude]"
    ) from exc

from ._core import detect_pii_json, redact_text_inline


@tool(
    "detect_pii",
    "Detect personally identifiable information in text. Returns a JSON array "
    "where each item has entity_type, text, start, end, confidence.",
    {"text": str},
)
async def _detect_pii(args: dict[str, Any]) -> dict[str, Any]:
    detections = detect_pii_json(args["text"])
    return {
        "content": [
            {"type": "text", "text": json.dumps(detections, ensure_ascii=False)}
        ]
    }


@tool(
    "redact_text",
    "Redact personally identifiable information in text. Returns the text with "
    "each PII span replaced by a tag like [PERSON] or [EMAIL_ADDRESS].",
    {"text": str},
)
async def _redact_text(args: dict[str, Any]) -> dict[str, Any]:
    redacted = redact_text_inline(args["text"])
    return {"content": [{"type": "text", "text": redacted}]}


# Ready-to-use in-process MCP server. Agents register it via
# ClaudeAgentOptions(mcp_servers={"hush": hush_server}).
hush_server = create_sdk_mcp_server(
    name="hush",
    version="1.0.0",
    tools=[_detect_pii, _redact_text],
)


__all__ = ["hush_server"]
