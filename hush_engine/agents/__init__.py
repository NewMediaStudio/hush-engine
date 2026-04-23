"""
Agent integrations for Hush Engine.

Thin wrappers that expose the detector as tools in the three current
agent transports:

- `hush_engine.agents.mcp_server`   — MCP stdio server (Claude Code, Cursor,
  Zed, Cline, Windsurf, Continue). Console script: `hush-mcp`. Install with
  `pip install hush-engine[mcp]`.
- `hush_engine.agents.claude_sdk`   — in-process MCP server for the
  Claude Agent SDK. Install with `pip install hush-engine[agent-claude]`.
- `hush_engine.agents.openai_agent` — @function_tool wrappers for the OpenAI
  Agents SDK. Install with `pip install hush-engine[agent-openai]`.

Every transport shares the helpers in `_core.py` so behavior stays consistent
and the detector is constructed once per process.
"""

from ._core import detect_pii_json, redact_text_inline

__all__ = ["detect_pii_json", "redact_text_inline"]
