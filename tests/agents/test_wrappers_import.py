"""
Import-level smoke tests for the three agent wrappers.

These tests verify the ImportError messages — they do NOT require the
third-party SDKs to be installed. When an SDK is not present, importing
the wrapper must raise ImportError with a message pointing at the
correct `pip install hush-engine[...]` extra.

When an SDK IS present (developer machine), the import succeeds and the
test just confirms the expected symbols are exported.
"""

from __future__ import annotations

import importlib
import unittest


class WrapperImportTests(unittest.TestCase):
    def _check(self, module_name: str, extra_name: str, expected_symbol: str):
        try:
            mod = importlib.import_module(module_name)
        except ImportError as exc:
            self.assertIn(extra_name, str(exc))
            return
        self.assertTrue(
            hasattr(mod, expected_symbol),
            f"{module_name} imported but missing `{expected_symbol}`",
        )

    def test_mcp_server_module(self):
        # `main` is the console-script entrypoint
        self._check("hush_engine.agents.mcp_server", "hush-engine[mcp]", "main")

    def test_claude_sdk_module(self):
        # `hush_server` is the ready-to-use in-process MCP server
        self._check(
            "hush_engine.agents.claude_sdk",
            "hush-engine[agent-claude]",
            "hush_server",
        )

    def test_openai_agent_module(self):
        self._check(
            "hush_engine.agents.openai_agent",
            "hush-engine[agent-openai]",
            "detect_pii_tool",
        )


if __name__ == "__main__":
    unittest.main()
