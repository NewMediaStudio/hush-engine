#!/usr/bin/env python3
"""
Detection Config - Manages PII detection thresholds with auto-adjustment
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Engine version - single source of truth
VERSION = "1.4.0"

# Detection library/integration toggles
# These control which detection backends are enabled
DEFAULT_INTEGRATIONS = {
    # Lightweight NER (always available, fast, low memory)
    "lgbm_ner": True,        # LightGBM token classifiers (~10MB, 5-10x faster)
    "name_dataset": True,    # Dictionary lookup for names (~5MB)

    # Standard NER (spaCy - moderate memory, good accuracy)
    "spacy": True,           # spaCy NER (50-100MB, reliable baseline)

    # Heavy NER models (disabled by default - install with: pip install hush-engine[accurate])
    "gliner": False,         # GLiNER zero-shot PII model (~1GB)
    "flair": False,          # Flair NER (~400MB, high accuracy)
    "transformers": False,   # Transformers BERT NER (~600MB, high precision)

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

    def update_all(self, thresholds: Dict[str, float] = None, enabled_entities: Dict[str, bool] = None, enabled_integrations: Dict[str, bool] = None):
        """
        Update thresholds, enabled entities, and/or integrations in bulk.

        Args:
            thresholds: Dict of entity_type -> threshold value
            enabled_entities: Dict of entity_type -> enabled boolean
            enabled_integrations: Dict of integration -> enabled boolean
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
