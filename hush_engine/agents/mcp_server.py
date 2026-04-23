"""
MCP server exposing Hush Engine as tools for any MCP-aware client.

Tested with:
- Claude Code (`claude mcp add hush`)
- Cursor (Settings -> MCP -> New Server, command `hush-mcp`)
- Any other stdio MCP client

Tools:
- detect_pii(text, entity_types=None)   -> list of detection dicts
- redact_text(text, entity_types=None, mask="[{entity_type}]") -> redacted string

Install:   pip install hush-engine[mcp]
Run:       hush-mcp                 (stdio transport, blocks on stdin)
"""

from __future__ import annotations

from typing import List, Optional

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The MCP server requires the `mcp` package. "
        "Install with: pip install hush-engine[mcp]"
    ) from exc

from ._core import detect_pii_json, redact_text_inline

mcp = FastMCP("hush-engine")


@mcp.tool()
def detect_pii(
    text: str,
    entity_types: Optional[List[str]] = None,
) -> List[dict]:
    """Detect personally identifiable information in text.

    Returns a list of detections with the keys `entity_type`, `text`,
    `start`, `end`, `confidence`. Pass `entity_types` to filter to a
    specific subset, for example `["PERSON", "EMAIL_ADDRESS"]`.

    Hush covers 27 entity types including checksum-validated credit cards
    (Luhn), IBAN (mod-97), SSN / passport / driver's license across
    35+ countries, AWS and Stripe keys, medical codes, and faces / QR
    codes in images.
    """
    return detect_pii_json(text, entity_types=entity_types)


@mcp.tool()
def redact_text(
    text: str,
    entity_types: Optional[List[str]] = None,
    mask: str = "[{entity_type}]",
) -> str:
    """Redact personally identifiable information in text.

    Replaces each detected span with a placeholder derived from `mask`.
    The default mask uses the entity type as a label, so `"John Doe's
    email is john@example.com"` becomes
    `"[PERSON]'s email is [EMAIL_ADDRESS]"`. Pass `mask="***"` or any
    other template to customize the replacement.
    """
    return redact_text_inline(text, entity_types=entity_types, mask=mask)


def main() -> None:
    """Console script entry point. Runs the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
