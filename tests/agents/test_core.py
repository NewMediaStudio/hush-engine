"""
Unit tests for `hush_engine.agents._core`.

These tests run without any third-party agent SDK installed. They exercise
the shared helpers directly so the CI lane for the base install catches any
regression in the detector contract (e.g. a rename from `confidence` ->
`score`) before it hits users of the MCP / Claude / OpenAI wrappers.
"""

from __future__ import annotations

import unittest

from hush_engine.agents._core import detect_pii_json, redact_text_inline


class CoreTests(unittest.TestCase):
    SAMPLE = "John Doe's email is john.doe@example.com and phone is 555-867-5309."

    def test_empty_text_returns_empty_list(self):
        self.assertEqual(detect_pii_json(""), [])
        self.assertEqual(detect_pii_json(None), [])  # type: ignore[arg-type]

    def test_detect_returns_sorted_jsonable_dicts(self):
        detections = detect_pii_json(self.SAMPLE)
        # There should be at least one detection; the exact count depends on
        # which optional backends are installed, so we stay loose.
        self.assertGreater(len(detections), 0)

        # Every detection is JSON-serializable with the documented shape.
        import json
        json.dumps(detections)  # must not raise

        for d in detections:
            self.assertIn("entity_type", d)
            self.assertIn("text", d)
            self.assertIn("start", d)
            self.assertIn("end", d)
            self.assertIn("confidence", d)
            self.assertIsInstance(d["entity_type"], str)
            self.assertIsInstance(d["start"], int)
            self.assertIsInstance(d["end"], int)
            self.assertIsInstance(d["confidence"], float)

        # Sorted by start offset.
        starts = [d["start"] for d in detections]
        self.assertEqual(starts, sorted(starts))

    def test_entity_types_filter(self):
        # When we allow only EMAIL_ADDRESS, nothing else comes back.
        emails = detect_pii_json(self.SAMPLE, entity_types=["EMAIL_ADDRESS"])
        self.assertTrue(all(d["entity_type"] == "EMAIL_ADDRESS" for d in emails))

    def test_redact_replaces_pii_with_tags(self):
        redacted = redact_text_inline(self.SAMPLE)
        # The raw email string must be gone.
        self.assertNotIn("john.doe@example.com", redacted)
        # The tag format is `[{entity_type}]` by default.
        self.assertIn("[EMAIL_ADDRESS]", redacted)

    def test_redact_custom_mask(self):
        redacted = redact_text_inline(self.SAMPLE, mask="***")
        self.assertNotIn("john.doe@example.com", redacted)
        self.assertIn("***", redacted)

    def test_redact_empty_text_is_identity(self):
        self.assertEqual(redact_text_inline(""), "")


if __name__ == "__main__":
    unittest.main()
