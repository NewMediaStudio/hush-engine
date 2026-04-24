"""
Unit tests for Privacy Filter cascade modes (1.12.0+).

These tests monkey-patch `detect_persons_with_privacy_filter` in
`hush_engine.detectors.person_recognizer` so they run without the
~3 GB PF model loaded. They verify:

- `_effective_privacy_filter_mode` resolves correctly from legacy booleans
  and explicit mode overrides.
- `DetectionConfig` round-trips the three new settings and enforces the
  contested-band invariants.
- `_has_contested_span` fires on band hits, skips on early-exit winners,
  skips on no-hit cases.
- `_apply_veto` drops low-confidence Hush spans when PF is silent, keeps
  high-confidence spans, keeps low-confidence spans that PF confirms.
- `PrivacyFilterRecognizer.excluded_entities` filters the analyze() output.
- The arbiter callback fires on veto-mode disagreements and its return value
  is honored.

No network, no model download, no Presidio instantiation inside tests that
mock PF.
"""

from __future__ import annotations

import unittest
from unittest import mock


class ConfigModeResolutionTests(unittest.TestCase):
    def test_derive_defaults_to_off_when_bools_false(self):
        from hush_engine.detection_config import _derive_privacy_filter_mode
        self.assertEqual(
            _derive_privacy_filter_mode("", enabled=False, authoritative=False),
            "off",
        )

    def test_derive_candidate_when_enabled_bool_only(self):
        from hush_engine.detection_config import _derive_privacy_filter_mode
        self.assertEqual(
            _derive_privacy_filter_mode("", enabled=True, authoritative=False),
            "candidate",
        )

    def test_derive_authoritative_when_both_bools_true(self):
        from hush_engine.detection_config import _derive_privacy_filter_mode
        self.assertEqual(
            _derive_privacy_filter_mode("", enabled=True, authoritative=True),
            "authoritative",
        )

    def test_explicit_mode_takes_precedence_over_bools(self):
        from hush_engine.detection_config import _derive_privacy_filter_mode
        # Even though enabled=False (legacy says "off"), explicit mode wins.
        self.assertEqual(
            _derive_privacy_filter_mode("tiebreaker", enabled=False, authoritative=False),
            "tiebreaker",
        )
        self.assertEqual(
            _derive_privacy_filter_mode("veto", enabled=True, authoritative=True),
            "veto",
        )

    def test_empty_mode_string_uses_bool_fallback(self):
        from hush_engine.detection_config import _derive_privacy_filter_mode
        # Explicit "" means "don't override" - use legacy bools.
        self.assertEqual(
            _derive_privacy_filter_mode("", enabled=True, authoritative=False),
            "candidate",
        )


class DetectionConfigRoundtripTests(unittest.TestCase):
    """Verify the three new fields persist and validate correctly."""

    def setUp(self):
        import tempfile
        from hush_engine.detection_config import DetectionConfig
        self.tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        self.tmp.close()
        self.cfg = DetectionConfig(config_path=self.tmp.name)

    def tearDown(self):
        import os
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    def test_default_contested_band(self):
        self.assertEqual(self.cfg.get_privacy_filter_contested_band(), [0.45, 0.75])

    def test_default_excluded_entities_has_phone(self):
        # Default ships with PHONE_NUMBER excluded because the Kaggle ablation
        # showed PF regressing phone F1 by 5.71 pp.
        excluded = self.cfg.get_privacy_filter_excluded_entities()
        self.assertIn("PHONE_NUMBER", excluded)

    def test_default_mode_empty_falls_back_to_off(self):
        # Both legacy booleans are False in DEFAULT_INTEGRATIONS, and
        # privacy_filter_mode is "" by default, so resolved = "off".
        self.assertEqual(self.cfg.get_privacy_filter_mode(), "off")

    def test_set_mode_persists(self):
        self.cfg.set_privacy_filter_mode("tiebreaker")
        self.assertEqual(self.cfg.get_privacy_filter_mode(), "tiebreaker")

    def test_set_mode_rejects_unknown(self):
        with self.assertRaises(ValueError):
            self.cfg.set_privacy_filter_mode("nonsense")

    def test_set_excluded_entities_roundtrip(self):
        self.cfg.set_privacy_filter_excluded_entities(["EMAIL_ADDRESS", "URL"])
        self.assertEqual(
            self.cfg.get_privacy_filter_excluded_entities(),
            ["EMAIL_ADDRESS", "URL"],
        )

    def test_set_excluded_entities_empty(self):
        self.cfg.set_privacy_filter_excluded_entities([])
        self.assertEqual(self.cfg.get_privacy_filter_excluded_entities(), [])

    def test_set_contested_band_roundtrip(self):
        self.cfg.set_privacy_filter_contested_band([0.3, 0.8])
        self.assertEqual(self.cfg.get_privacy_filter_contested_band(), [0.3, 0.8])

    def test_set_contested_band_rejects_low_gte_high(self):
        with self.assertRaises(ValueError):
            self.cfg.set_privacy_filter_contested_band([0.6, 0.6])
        with self.assertRaises(ValueError):
            self.cfg.set_privacy_filter_contested_band([0.8, 0.4])

    def test_set_contested_band_clamps_to_unit_interval(self):
        self.cfg.set_privacy_filter_contested_band([-0.1, 1.5])
        self.assertEqual(self.cfg.get_privacy_filter_contested_band(), [0.0, 1.0])

    def test_update_all_accepts_all_three_fields(self):
        self.cfg.update_all(
            privacy_filter_mode="veto",
            privacy_filter_excluded_entities=["URL"],
            privacy_filter_contested_band=[0.4, 0.7],
        )
        self.assertEqual(self.cfg.get_privacy_filter_mode(), "veto")
        self.assertEqual(self.cfg.get_privacy_filter_excluded_entities(), ["URL"])
        self.assertEqual(self.cfg.get_privacy_filter_contested_band(), [0.4, 0.7])


class PersonRecognizerModeTests(unittest.TestCase):
    """Exercise the mode resolution + cascade helpers without running any model."""

    def _make(self, **kwargs):
        from hush_engine.detectors.person_recognizer import PersonRecognizer
        # Disable all heavy models so __init__ + load() stay fast.
        defaults = dict(
            use_lgbm_ner=False,
            use_nltagger=False,
            use_spacy=False,
            use_gliner=False,
            use_flair=False,
            use_transformers=False,
            use_name_dataset=False,
            use_patterns=False,
        )
        defaults.update(kwargs)
        return PersonRecognizer.__new__(PersonRecognizer).__class__(**defaults)

    def test_mode_off_when_all_flags_off(self):
        pr = self._make()
        self.assertEqual(pr._effective_privacy_filter_mode(), "off")

    def test_mode_candidate_from_legacy_bool(self):
        pr = self._make(use_privacy_filter=True)
        self.assertEqual(pr._effective_privacy_filter_mode(), "candidate")

    def test_mode_authoritative_from_legacy_bools(self):
        pr = self._make(use_privacy_filter=True, privacy_filter_authoritative=True)
        self.assertEqual(pr._effective_privacy_filter_mode(), "authoritative")

    def test_explicit_mode_wins_over_bools(self):
        pr = self._make(
            use_privacy_filter=True,
            privacy_filter_authoritative=True,
            privacy_filter_mode="tiebreaker",
        )
        self.assertEqual(pr._effective_privacy_filter_mode(), "tiebreaker")

    def test_has_contested_span_skips_on_high_conf(self):
        pr = self._make(privacy_filter_contested_band=(0.45, 0.75))
        detections = [("X", 0, 1, 0.9, "lgbm")]  # above early exit
        self.assertFalse(pr._has_contested_span(detections, found_high_conf=True))

    def test_has_contested_span_fires_on_band(self):
        pr = self._make(privacy_filter_contested_band=(0.45, 0.75))
        detections = [("X", 0, 1, 0.6, "lgbm")]  # in band
        self.assertTrue(pr._has_contested_span(detections, found_high_conf=False))

    def test_has_contested_span_false_below_band(self):
        pr = self._make(privacy_filter_contested_band=(0.45, 0.75))
        detections = [("X", 0, 1, 0.3, "lgbm")]  # below band
        self.assertFalse(pr._has_contested_span(detections, found_high_conf=False))

    def test_has_contested_span_respects_custom_band(self):
        pr = self._make(privacy_filter_contested_band=(0.2, 0.4))
        detections = [("X", 0, 1, 0.3, "lgbm")]  # in custom band
        self.assertTrue(pr._has_contested_span(detections, found_high_conf=False))

    def test_apply_veto_keeps_high_confidence(self):
        pr = self._make(privacy_filter_mode="veto")
        # High-conf Hush span, PF silent. Should be kept regardless.
        dets = [("Alice", 0, 5, 0.88, "lgbm")]
        pf_spans = []  # PF found nothing
        kept = pr._apply_veto(dets, pf_spans, text="Alice went home.")
        self.assertEqual(kept, dets)

    def test_apply_veto_drops_low_conf_unconfirmed(self):
        pr = self._make(privacy_filter_mode="veto")
        dets = [("Alice", 0, 5, 0.55, "lgbm")]
        pf_spans = []
        kept = pr._apply_veto(dets, pf_spans, text="Alice went home.")
        self.assertEqual(kept, [])

    def test_apply_veto_keeps_low_conf_when_pf_confirms(self):
        pr = self._make(privacy_filter_mode="veto")
        dets = [("Alice", 0, 5, 0.55, "lgbm")]
        pf_spans = [(0, 5, 0.99)]  # PF detected exactly the same span
        kept = pr._apply_veto(dets, pf_spans, text="Alice went home.")
        self.assertEqual(kept, dets)

    def test_apply_veto_keeps_pf_own_spans(self):
        pr = self._make(privacy_filter_mode="veto")
        # A PF-only span below 0.75 still survives veto (it's the evidence).
        dets = [("Bob", 10, 13, 0.65, "privacy_filter")]
        pf_spans = [(10, 13, 0.65)]
        kept = pr._apply_veto(dets, pf_spans, text="... Bob ...")
        self.assertEqual(kept, dets)

    def test_apply_veto_arbiter_invoked_on_disagreement(self):
        calls = []

        def arbiter(text, span_text, start, end, hush_score, pf_score):
            calls.append((span_text, hush_score, pf_score))
            return 0.8  # promote to keep

        pr = self._make(
            privacy_filter_mode="veto",
            privacy_filter_arbiter=arbiter,
            min_confidence=0.55,
        )
        dets = [("Alice", 0, 5, 0.55, "lgbm")]  # low conf, PF silent
        kept = pr._apply_veto(dets, pf_spans=[], text="Alice went home.")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "Alice")
        self.assertEqual(calls[0][1], 0.55)
        self.assertIsNone(calls[0][2])
        # Arbiter promoted it to 0.8 - stays in the kept list
        self.assertEqual(len(kept), 1)
        self.assertAlmostEqual(kept[0][3], 0.8)

    def test_apply_veto_arbiter_can_drop(self):
        pr = self._make(
            privacy_filter_mode="veto",
            privacy_filter_arbiter=lambda *a: None,  # drop
        )
        dets = [("Alice", 0, 5, 0.55, "lgbm")]
        kept = pr._apply_veto(dets, pf_spans=[], text="Alice went home.")
        self.assertEqual(kept, [])


class PrivacyFilterRecognizerExcludeTests(unittest.TestCase):
    """`excluded_entities` drops matching hush types from PF output."""

    def test_empty_requested_returns_empty(self):
        from hush_engine.detectors.privacy_filter_recognizer import PrivacyFilterRecognizer
        rec = PrivacyFilterRecognizer(excluded_entities=[])
        self.assertEqual(rec.analyze("hello", entities=[], nlp_artifacts=None), [])

    def test_excluded_matching_all_returns_empty(self):
        from hush_engine.detectors.privacy_filter_recognizer import PrivacyFilterRecognizer
        rec = PrivacyFilterRecognizer(excluded_entities=PrivacyFilterRecognizer.ENTITIES)
        # Every supported entity is excluded → no call, returns [].
        self.assertEqual(
            rec.analyze("x", entities=list(PrivacyFilterRecognizer.ENTITIES), nlp_artifacts=None),
            [],
        )

    def test_exclude_filters_single_type(self):
        """When PHONE_NUMBER is excluded, PF's filtered target_labels omit private_phone.

        This test patches `detect_with_privacy_filter` to verify the call site.
        """
        from hush_engine.detectors import privacy_filter_recognizer as pfr

        captured = {}

        def fake_detect(text, target_labels=None):
            captured["target_labels"] = set(target_labels or [])
            return []  # no hits

        with mock.patch.object(pfr, "detect_with_privacy_filter", side_effect=fake_detect):
            rec = pfr.PrivacyFilterRecognizer(excluded_entities=["PHONE_NUMBER"])
            rec.analyze(
                "call me",
                entities=list(pfr.PrivacyFilterRecognizer.ENTITIES),
                nlp_artifacts=None,
            )

        # private_phone must have been dropped from the requested labels
        self.assertNotIn("private_phone", captured["target_labels"])
        # Other labels still present
        self.assertIn("private_email", captured["target_labels"])


if __name__ == "__main__":
    unittest.main()
