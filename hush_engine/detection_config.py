#!/usr/bin/env python3
"""
Detection Config - Manages PII detection thresholds with auto-adjustment
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


# Default confidence thresholds per entity type
DEFAULT_THRESHOLDS = {
    "PERSON": 0.5,
    "EMAIL_ADDRESS": 0.5,
    "PHONE_NUMBER": 0.5,
    "LOCATION": 0.5,
    "AWS_ACCESS_KEY": 0.5,
    "STRIPE_KEY": 0.5,
    "CREDIT_CARD": 0.5,
    "SSN": 0.5,
    "DATE_TIME": 0.5,
    "NRP": 0.5,  # Nationality, Religion, Political group
    "ORGANIZATION": 0.5,
    "URL": 0.5,
    "IP_ADDRESS": 0.5,
    "CURRENCY": 0.5,
    "COMPANY": 0.5,
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
            "thresholds": self.get_all_thresholds()
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
