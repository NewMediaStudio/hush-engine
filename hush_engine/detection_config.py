#!/usr/bin/env python3
"""
Detection Config - Manages PII detection thresholds with auto-adjustment
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Engine version - single source of truth
VERSION = "1.11.2"

# Detection library/integration toggles
# These control which detection backends are enabled
DEFAULT_INTEGRATIONS = {
    # Lightweight NER (always available, fast, low memory)
    "lgbm_ner": True,        # LightGBM token classifiers (~10MB, 5-10x faster)
    "name_dataset": True,    # Dictionary lookup for names (~5MB)

    # macOS native NLP (v1.8.0 - replaces spaCy for Minimal tier)
    "nltagger": True,        # macOS NLTagger NER/lemmatization (zero-install)

    # Standard NER (spaCy - now optional, install with: pip install hush-engine[spacy])
    "spacy": False,          # spaCy NER (50-100MB, reliable baseline)

    # Heavy NER models (disabled by default - install with: pip install hush-engine[accurate])
    "gliner": False,         # GLiNER zero-shot PII model (~1GB)
    "flair": False,          # Flair NER (~400MB, high accuracy)
    "transformers": False,   # Transformers BERT NER (~600MB, high precision)

    # OpenAI Privacy Filter add-on (disabled by default - install with: pip install hush-engine[privacy-filter])
    # Released 2026-04-22, Apache-2.0, bidirectional token classifier (~3GB BF16).
    # Covers 8 span categories: private_person, private_email, private_phone,
    # private_address, private_url, private_date, account_number, secret.
    #
    # These two booleans predate `privacy_filter_mode` and are kept for
    # backward compatibility. When `privacy_filter_mode` is set to anything
    # other than the empty string, it takes precedence. See
    # `_derive_privacy_filter_mode` for the mapping.
    "openai_privacy_filter": False,
    # If True, a Privacy Filter PERSON hit short-circuits the cascade (authoritative).
    # If False (default), it feeds into the ensemble vote like other engines (candidate).
    "openai_privacy_filter_authoritative": False,

    # Vision detectors (v1.8.0 - macOS native, replaces OpenCV/pyzbar)
    "vision_face": True,     # Vision VNDetectFaceRectanglesRequest (zero-install)
    "vision_qr": True,       # Vision VNDetectBarcodesRequest (zero-install)
    "opencv_face": False,    # OpenCV Haar cascade fallback (optional)
    "opencv_qr": False,      # OpenCV QRCodeDetector fallback (optional)

    # Address detection
    "libpostal": True,       # libpostal address parsing (99.45% accuracy)

    # URL detection
    "urlextract": True,      # urlextract for comprehensive URL detection

    # Other integrations
    "phonenumbers": True,    # Google libphonenumber validation
}

# Precision improvement feature flags (v1.4.0)
# These control new precision enhancements and can be toggled for gradual rollout
PRECISION_FEATURES = {
    "spatial_filtering": True,       # Form label detection and zone penalties
    "negative_gazetteer": True,      # Common word false positive filtering
    "version_disambiguation": True,  # IP address vs version string filtering
    "ivw_calibration": False,        # Inverse-variance weighted calibration (requires feedback data)
}


# Default confidence thresholds per entity type
# Calibrated on 2026-02-07 using Yellowbrick-style threshold analysis
DEFAULT_THRESHOLDS = {
    "PERSON": 0.55,        # Base threshold - precision controlled by consensus logic
    "EMAIL_ADDRESS": 0.30, # Calibrated: 99.6% F1 at low threshold (high precision)
    "PHONE_NUMBER": 0.35,  # Calibrated: lower for recall (was 43.9%)
    "LOCATION": 0.5,
    "AWS_ACCESS_KEY": 0.5,
    "STRIPE_KEY": 0.5,
    "CREDIT_CARD": 0.30,   # Calibrated: 96.3% F1 at low threshold (high precision)
    "DATE_TIME": 0.40,     # Calibrated: 93.3% F1
    "AGE": 0.30,           # Calibrated: 89.5% F1 at low threshold
    "NRP": 0.5,            # Nationality, Religion, Political group
    "ORGANIZATION": 0.5,
    "URL": 0.5,
    "IP_ADDRESS": 0.6,     # Higher threshold to reduce FPs (49 in benchmark)
    "FINANCIAL": 0.5,
    "COMPANY": 0.5,
    "GENDER": 0.5,
    "FACE": 0.5,
    "MEDICAL": 0.6,        # Higher threshold to reduce FPs (34 in benchmark)
    "QR_CODE": 0.5,
    "BARCODE": 0.5,
    "COORDINATES": 0.6,    # Higher threshold to reduce FPs (12 in benchmark)
    # New entity types (v1.4.0)
    "BIOMETRIC": 0.6,      # Higher threshold to reduce FPs (12 in benchmark)
    "CREDENTIAL": 0.5,     # Passwords, PINs, API keys
    "ID": 0.5,             # Customer ID, Employee ID, generic IDs
    "NATIONAL_ID": 0.45,   # Calibrated: lower for recall (was 44.6%)
    "NETWORK": 0.6,        # Higher threshold to reduce FPs (34 in benchmark)
    "VEHICLE": 0.5,        # VIN, license plates
}

# Minimum threshold (don't go below this even with auto-adjustment)
MIN_THRESHOLD = 0.3

# Maximum threshold (don't go above this)
MAX_THRESHOLD = 0.95


# =============================================================================
# OpenAI Privacy Filter cascade-mode controls (1.12.0+)
# =============================================================================
# `privacy_filter_mode` supersedes the two booleans above when set to anything
# other than "". It controls WHERE Privacy Filter slots into the detection
# pipeline, not WHETHER to run it (the old booleans still drive that, via
# the backward-compat derivation in `_derive_privacy_filter_mode`).
PRIVACY_FILTER_MODES = ("", "off", "candidate", "authoritative", "tiebreaker", "veto")

# Default: "" (empty) means "use whatever the legacy booleans say." This keeps
# existing 1.11.x configs working without migration.
DEFAULT_PRIVACY_FILTER_MODE = ""

# Tiebreaker fires when any ensemble span lands in this score band AND no
# span reaches the early-exit confidence. Widening reduces PF calls; narrowing
# increases them.
DEFAULT_PRIVACY_FILTER_CONTESTED_BAND = [0.45, 0.75]

# Default entity-type exclude list for Privacy Filter's non-PERSON output.
# Rationale: the 2026-04-23 ablation on 1,000 Kaggle samples showed PF dropping
# PHONE_NUMBER F1 by 5.71 pp because its phone spans disagree with Hush's
# libphonenumber-validated ones. Hush's validators are authoritative for
# numeric IDs, so we default-exclude PHONE. Users can empty the list to let
# PF contribute phone spans if their document mix benefits.
DEFAULT_PRIVACY_FILTER_EXCLUDED_ENTITIES = ["PHONE_NUMBER"]


def _derive_privacy_filter_mode(
    explicit_mode: str,
    enabled: bool,
    authoritative: bool,
) -> str:
    """Resolve the effective Privacy Filter mode.

    Precedence:
      1. If `explicit_mode` is one of the non-empty modes, use it.
      2. Otherwise, fall back to the legacy bool pair:
           - enabled=False             -> "off"
           - enabled=True, auth=True   -> "authoritative"
           - enabled=True, auth=False  -> "candidate"
    """
    if explicit_mode and explicit_mode in PRIVACY_FILTER_MODES and explicit_mode != "":
        return explicit_mode
    if not enabled:
        return "off"
    return "authoritative" if authoritative else "candidate"


class DetectionConfig:
    """
    Manages detection confidence thresholds with persistence and auto-adjustment
    """

    def __init__(self, config_path: str = None):
        """
        Initialize config manager

        Args:
            config_path: Path to config file (default: ~/.hush/detection_config.json)
        """
        if config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = Path.home() / ".hush" / "detection_config.json"

        self.config: Dict[str, Any] = {
            "thresholds": DEFAULT_THRESHOLDS.copy(),
            "enabled_entities": {k: True for k in DEFAULT_THRESHOLDS.keys()},  # All enabled by default
            "enabled_integrations": DEFAULT_INTEGRATIONS.copy(),  # Detection library toggles
            "calibrated_weights": {},  # IVW calibrated model weights
            "calibrated_thresholds": {},  # Per-entity calibrated thresholds
            "privacy_filter_mode": DEFAULT_PRIVACY_FILTER_MODE,
            "privacy_filter_excluded_entities": list(DEFAULT_PRIVACY_FILTER_EXCLUDED_ENTITIES),
            "privacy_filter_contested_band": list(DEFAULT_PRIVACY_FILTER_CONTESTED_BAND),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "adjustment_history": []
        }

        self._load_config()

    def _load_config(self):
        """Load config from file if it exists"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    saved = json.load(f)
                    # Merge with defaults (in case new entity types were added)
                    self.config["thresholds"] = {**DEFAULT_THRESHOLDS, **saved.get("thresholds", {})}
                    # Merge enabled_entities with defaults (all enabled by default)
                    default_enabled = {k: True for k in DEFAULT_THRESHOLDS.keys()}
                    self.config["enabled_entities"] = {**default_enabled, **saved.get("enabled_entities", {})}
                    # Merge enabled_integrations with defaults
                    self.config["enabled_integrations"] = {**DEFAULT_INTEGRATIONS, **saved.get("enabled_integrations", {})}
                    # Privacy Filter mode controls (new in 1.12.0; absent in older configs)
                    self.config["privacy_filter_mode"] = saved.get(
                        "privacy_filter_mode", DEFAULT_PRIVACY_FILTER_MODE
                    )
                    self.config["privacy_filter_excluded_entities"] = list(saved.get(
                        "privacy_filter_excluded_entities", DEFAULT_PRIVACY_FILTER_EXCLUDED_ENTITIES
                    ))
                    self.config["privacy_filter_contested_band"] = list(saved.get(
                        "privacy_filter_contested_band", DEFAULT_PRIVACY_FILTER_CONTESTED_BAND
                    ))
                    self.config["created_at"] = saved.get("created_at", self.config["created_at"])
                    self.config["updated_at"] = saved.get("updated_at", self.config["updated_at"])
                    self.config["adjustment_history"] = saved.get("adjustment_history", [])
            except (json.JSONDecodeError, IOError):
                pass  # Use defaults on error

    def save(self):
        """Save config to file"""
        self.config["updated_at"] = datetime.now().isoformat()
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

    def get_threshold(self, entity_type: str) -> float:
        """
        Get confidence threshold for an entity type

        Args:
            entity_type: Entity type (e.g., "PERSON", "EMAIL_ADDRESS")

        Returns:
            Confidence threshold (0.0 - 1.0)
        """
        return self.config["thresholds"].get(entity_type, 0.5)

    def set_threshold(self, entity_type: str, threshold: float, reason: str = None):
        """
        Set confidence threshold for an entity type

        Args:
            entity_type: Entity type
            threshold: New threshold (will be clamped to MIN/MAX)
            reason: Optional reason for the change
        """
        # Clamp to valid range
        threshold = max(MIN_THRESHOLD, min(MAX_THRESHOLD, threshold))

        old_value = self.config["thresholds"].get(entity_type, 0.5)
        self.config["thresholds"][entity_type] = threshold

        # Record adjustment
        self.config["adjustment_history"].append({
            "entity_type": entity_type,
            "old_value": old_value,
            "new_value": threshold,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        })

        # Keep only last 100 adjustments
        self.config["adjustment_history"] = self.config["adjustment_history"][-100:]

        self.save()

    def get_all_thresholds(self) -> Dict[str, float]:
        """Get all thresholds"""
        return self.config["thresholds"].copy()

    def get_enabled_entities(self) -> Dict[str, bool]:
        """Get all enabled entity settings"""
        return self.config["enabled_entities"].copy()

    def set_enabled_entity(self, entity_type: str, enabled: bool):
        """Set whether an entity type is enabled"""
        self.config["enabled_entities"][entity_type] = enabled
        self.save()

    def get_enabled_integrations(self) -> Dict[str, bool]:
        """Get all enabled integration/library settings"""
        return self.config.get("enabled_integrations", DEFAULT_INTEGRATIONS).copy()

    def set_enabled_integration(self, integration: str, enabled: bool):
        """Set whether a detection integration/library is enabled"""
        if "enabled_integrations" not in self.config:
            self.config["enabled_integrations"] = DEFAULT_INTEGRATIONS.copy()
        self.config["enabled_integrations"][integration] = enabled
        self.save()

    def is_integration_enabled(self, integration: str) -> bool:
        """Check if a specific integration is enabled"""
        integrations = self.config.get("enabled_integrations", DEFAULT_INTEGRATIONS)
        return integrations.get(integration, True)

    # -------------------------------------------------------------------------
    # Privacy Filter mode controls (1.12.0+)
    # -------------------------------------------------------------------------

    def get_privacy_filter_mode(self) -> str:
        """Resolved Privacy Filter cascade mode.

        Returns one of: "off", "candidate", "authoritative", "tiebreaker", "veto".
        If `privacy_filter_mode` is set explicitly, returns it. Otherwise
        derives from the legacy `openai_privacy_filter` + `openai_privacy_filter_authoritative`
        booleans so 1.11.x configs keep working.
        """
        integrations = self.config.get("enabled_integrations", DEFAULT_INTEGRATIONS)
        return _derive_privacy_filter_mode(
            explicit_mode=self.config.get("privacy_filter_mode", DEFAULT_PRIVACY_FILTER_MODE),
            enabled=integrations.get("openai_privacy_filter", False),
            authoritative=integrations.get("openai_privacy_filter_authoritative", False),
        )

    def set_privacy_filter_mode(self, mode: str):
        """Set the explicit Privacy Filter cascade mode.

        Pass "" (empty string) to clear the explicit setting and revert to the
        legacy-bool-derived behavior. Raises ValueError for an unknown mode.
        """
        if mode not in PRIVACY_FILTER_MODES:
            raise ValueError(
                f"Unknown privacy_filter_mode '{mode}'. "
                f"Expected one of: {PRIVACY_FILTER_MODES}"
            )
        self.config["privacy_filter_mode"] = mode
        self.save()

    def get_privacy_filter_excluded_entities(self) -> list:
        """List of entity types Privacy Filter must NOT output.

        Defaults to DEFAULT_PRIVACY_FILTER_EXCLUDED_ENTITIES (currently
        ["PHONE_NUMBER"]). Empty list allows every PF-mapped entity type.
        """
        return list(self.config.get(
            "privacy_filter_excluded_entities",
            DEFAULT_PRIVACY_FILTER_EXCLUDED_ENTITIES,
        ))

    def set_privacy_filter_excluded_entities(self, excluded: list):
        """Set the Privacy Filter per-entity exclude list."""
        self.config["privacy_filter_excluded_entities"] = list(excluded)
        self.save()

    def get_privacy_filter_contested_band(self) -> list:
        """The (low, high) confidence band that triggers tiebreaker mode.

        Returns a two-element list. A span with score strictly inside the
        band and no early-exit winner in the cascade qualifies the document
        for a PF tiebreaker call.
        """
        band = self.config.get(
            "privacy_filter_contested_band",
            DEFAULT_PRIVACY_FILTER_CONTESTED_BAND,
        )
        return list(band)

    def set_privacy_filter_contested_band(self, band):
        """Set the tiebreaker contested-band tuple.

        Accepts any 2-element iterable. Values are clamped to [0.0, 1.0] and
        enforced low < high.
        """
        if len(band) != 2:
            raise ValueError("privacy_filter_contested_band must have 2 elements")
        low, high = float(band[0]), float(band[1])
        low = max(0.0, min(1.0, low))
        high = max(0.0, min(1.0, high))
        if low >= high:
            raise ValueError(
                f"privacy_filter_contested_band low ({low}) must be < high ({high})"
            )
        self.config["privacy_filter_contested_band"] = [low, high]
        self.save()

    def update_all(
        self,
        thresholds: Dict[str, float] = None,
        enabled_entities: Dict[str, bool] = None,
        enabled_integrations: Dict[str, bool] = None,
        privacy_filter_mode: Optional[str] = None,
        privacy_filter_excluded_entities: Optional[list] = None,
        privacy_filter_contested_band: Optional[list] = None,
    ):
        """
        Update thresholds, enabled entities, integrations, and Privacy Filter
        controls in bulk. Any argument set to None is left unchanged.

        Args:
            thresholds: Dict of entity_type -> threshold value
            enabled_entities: Dict of entity_type -> enabled boolean
            enabled_integrations: Dict of integration -> enabled boolean
            privacy_filter_mode: One of "", "off", "candidate", "authoritative",
                "tiebreaker", "veto". Empty string reverts to legacy-bool behavior.
            privacy_filter_excluded_entities: List of entity types to exclude from
                Privacy Filter output.
            privacy_filter_contested_band: Two-element [low, high] for tiebreaker mode.
        """
        if thresholds:
            for entity_type, threshold in thresholds.items():
                # Clamp to valid range
                threshold = max(MIN_THRESHOLD, min(MAX_THRESHOLD, threshold))
                self.config["thresholds"][entity_type] = threshold

        if enabled_entities:
            for entity_type, enabled in enabled_entities.items():
                self.config["enabled_entities"][entity_type] = enabled

        if enabled_integrations:
            if "enabled_integrations" not in self.config:
                self.config["enabled_integrations"] = DEFAULT_INTEGRATIONS.copy()
            for integration, enabled in enabled_integrations.items():
                self.config["enabled_integrations"][integration] = enabled

        if privacy_filter_mode is not None:
            if privacy_filter_mode not in PRIVACY_FILTER_MODES:
                raise ValueError(
                    f"Unknown privacy_filter_mode '{privacy_filter_mode}'. "
                    f"Expected one of: {PRIVACY_FILTER_MODES}"
                )
            self.config["privacy_filter_mode"] = privacy_filter_mode

        if privacy_filter_excluded_entities is not None:
            self.config["privacy_filter_excluded_entities"] = list(privacy_filter_excluded_entities)

        if privacy_filter_contested_band is not None:
            if len(privacy_filter_contested_band) != 2:
                raise ValueError("privacy_filter_contested_band must have 2 elements")
            low, high = float(privacy_filter_contested_band[0]), float(privacy_filter_contested_band[1])
            low = max(0.0, min(1.0, low))
            high = max(0.0, min(1.0, high))
            if low >= high:
                raise ValueError(
                    f"privacy_filter_contested_band low ({low}) must be < high ({high})"
                )
            self.config["privacy_filter_contested_band"] = [low, high]

        self.save()

    def adjust_from_feedback(self, false_positive_rates: Dict[str, float], min_samples: int = 5):
        """
        Auto-adjust thresholds based on false positive rates

        Args:
            false_positive_rates: Dict mapping entity_type to false positive rate (0.0 - 1.0)
            min_samples: Minimum samples required to adjust
        """
        adjustments_made = []

        for entity_type, fp_rate in false_positive_rates.items():
            current = self.get_threshold(entity_type)

            # If false positive rate is high (> 30%), increase threshold
            if fp_rate > 0.3:
                # Increase threshold proportionally to false positive rate
                increase = fp_rate * 0.2  # Max 20% increase
                new_threshold = current + increase
                self.set_threshold(
                    entity_type,
                    new_threshold,
                    reason=f"Auto-adjusted: {fp_rate:.0%} false positive rate"
                )
                adjustments_made.append((entity_type, current, new_threshold, fp_rate))

            # If false positive rate is low (< 10%) and threshold is high, we can decrease
            elif fp_rate < 0.1 and current > 0.6:
                decrease = 0.05
                new_threshold = current - decrease
                self.set_threshold(
                    entity_type,
                    new_threshold,
                    reason=f"Auto-adjusted: low false positive rate ({fp_rate:.0%})"
                )
                adjustments_made.append((entity_type, current, new_threshold, fp_rate))

        return adjustments_made

    def get_calibrated_weights(self) -> Dict[str, float]:
        """
        Get calibrated model weights for NER ensemble.

        Returns IVW-calibrated weights if available, otherwise returns None.
        Callers should fall back to DEFAULT_MODEL_WEIGHTS if None.
        """
        return self.config.get("calibrated_weights", {}) or None

    def set_calibrated_weights(self, weights: Dict[str, float]):
        """
        Set calibrated model weights.

        Args:
            weights: Dict mapping model names to weights (0.0 - 1.0)
        """
        self.config["calibrated_weights"] = weights
        self.config["updated_at"] = datetime.now().isoformat()
        self.save()
        logger.info(f"Calibrated weights updated: {weights}")

    def get_calibrated_threshold(self, entity_type: str) -> Optional[float]:
        """
        Get calibrated threshold for a specific entity type.

        Returns the calibrated threshold if available, otherwise None.
        Callers should fall back to the standard threshold if None.
        """
        calibrated = self.config.get("calibrated_thresholds", {})
        return calibrated.get(entity_type)

    def set_calibrated_thresholds(self, thresholds: Dict[str, float]):
        """
        Set calibrated thresholds per entity type.

        Args:
            thresholds: Dict mapping entity types to calibrated thresholds
        """
        self.config["calibrated_thresholds"] = thresholds
        self.config["updated_at"] = datetime.now().isoformat()
        self.save()
        logger.info(f"Calibrated thresholds updated for {len(thresholds)} entity types")

    def recalibrate(self, feedback_path: str = None) -> bool:
        """
        Recalibrate weights and thresholds from feedback data.

        Uses the WeightCalibrator to compute IVW weights and optimal thresholds.

        Args:
            feedback_path: Path to feedback directory (default: training/feedback)

        Returns:
            True if calibration succeeded, False otherwise
        """
        try:
            from hush_engine.calibration import WeightCalibrator
        except ImportError:
            try:
                from .calibration import WeightCalibrator
            except ImportError:
                logger.warning("WeightCalibrator not available")
                return False

        if feedback_path is None:
            # Default to training/feedback in the repo
            feedback_path = Path(__file__).parent.parent / "training" / "feedback"
        else:
            feedback_path = Path(feedback_path)

        if not feedback_path.exists():
            logger.warning(f"Feedback path does not exist: {feedback_path}")
            return False

        calibrator = WeightCalibrator()
        weights, thresholds = calibrator.calibrate(feedback_path)

        # Store calibrated values
        self.set_calibrated_weights(weights)

        # Convert EntityThreshold objects to simple threshold dict
        threshold_dict = {k: v.threshold for k, v in thresholds.items()}
        self.set_calibrated_thresholds(threshold_dict)

        # Save calibration to a separate file for inspection
        calibration_file = self.config_path.parent / "calibration.json"
        calibrator.save_calibration(calibration_file)

        return True

    def reset(self):
        """Reset all thresholds to defaults"""
        self.config = {
            "thresholds": DEFAULT_THRESHOLDS.copy(),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "adjustment_history": [{
                "entity_type": "ALL",
                "old_value": "custom",
                "new_value": "defaults",
                "reason": "Manual reset by user",
                "timestamp": datetime.now().isoformat()
            }]
        }
        self.save()

    def is_modified(self) -> bool:
        """Check if config has been modified from defaults"""
        for entity_type, default_val in DEFAULT_THRESHOLDS.items():
            if abs(self.config["thresholds"].get(entity_type, default_val) - default_val) > 0.01:
                return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get config statistics"""
        feedback_path = Path.home() / ".hush" / "training_feedback.jsonl"
        total_feedback_entries = 0
        total_added_areas = 0
        total_removed_bars = 0

        if feedback_path.exists():
            try:
                with open(feedback_path, 'r') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            total_feedback_entries += 1
                            user_edits = data.get("user_edits", {})
                            total_added_areas += len(user_edits.get("added_areas", []))
                            total_removed_bars += len(user_edits.get("removed_bars", []))
                        except json.JSONDecodeError:
                            continue
            except IOError:
                pass

        return {
            "is_modified": self.is_modified() or total_feedback_entries > 0,
            "total_adjustments": len(self.config["adjustment_history"]),
            "total_feedback_sessions": total_feedback_entries,
            "total_added_areas": total_added_areas,
            "total_removed_bars": total_removed_bars,
            "created_at": self.config["created_at"],
            "updated_at": self.config["updated_at"],
            "thresholds": self.get_all_thresholds(),
            "enabled_entities": self.get_enabled_entities(),
            "enabled_integrations": self.get_enabled_integrations()
        }


# Global instance for convenience
_config_instance: Optional[DetectionConfig] = None


def get_config() -> DetectionConfig:
    """Get the global config instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = DetectionConfig()
    return _config_instance


def reset_config():
    """Reset to shipped defaults and clear training data (e.g. ~/.hush/training_feedback.jsonl)."""
    cfg = get_config()
    cfg.reset()
    feedback_path = Path.home() / ".hush" / "training_feedback.jsonl"
    if feedback_path.exists():
        try:
            feedback_path.unlink()
        except OSError:
            pass
