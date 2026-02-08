#!/usr/bin/env python3
"""
Weight Calibrator for Hush Engine NER Models

Provides automated calibration of model weights using Inverse-Variance
Weighting (IVW) and precision-recall curve analysis from feedback data.

This module replaces static MODEL_WEIGHTS with data-driven calibration
for improved precision-recall trade-offs.

Usage:
    from hush_engine.calibration import WeightCalibrator, calibrate_from_feedback

    # Calibrate from feedback directory
    weights = calibrate_from_feedback("/path/to/feedback")

    # Or use the calibrator directly
    calibrator = WeightCalibrator()
    weights = calibrator.compute_ivw_weights(model_metrics)
    thresholds = calibrator.compute_optimal_thresholds(feedback_data)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
from collections import defaultdict
import json
import logging
import math

logger = logging.getLogger(__name__)


# Default static weights (used as fallback when calibration data unavailable)
DEFAULT_MODEL_WEIGHTS = {
    "patterns": 1.0,       # Highest precision - regex patterns
    "flair": 0.93,         # ~93% F1 on CoNLL-03
    "spacy": 0.90,         # ~90% F1
    "transformers": 0.88,  # BERT NER (dslim/bert-base-NER)
    "gliner": 0.82,        # Zero-shot ~82% F1
    "name_dataset": 0.70   # Dictionary lookup (high recall, lower precision)
}


@dataclass
class ModelMetrics:
    """Metrics for a single NER model."""
    model_name: str
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    total_predictions: int = 0

    @property
    def variance(self) -> float:
        """
        Compute variance approximation from precision and recall.

        Uses the formula: variance = (1 - precision) * (1 - recall)
        This penalizes models that are weak in either dimension.
        """
        return (1 - self.precision) * (1 - self.recall)

    def update_metrics(self):
        """Recompute precision, recall, F1 from counts."""
        if self.true_positives + self.false_positives > 0:
            self.precision = self.true_positives / (self.true_positives + self.false_positives)
        else:
            self.precision = 0.0

        if self.true_positives + self.false_negatives > 0:
            self.recall = self.true_positives / (self.true_positives + self.false_negatives)
        else:
            self.recall = 0.0

        if self.precision + self.recall > 0:
            self.f1 = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        else:
            self.f1 = 0.0


@dataclass
class EntityThreshold:
    """Calibrated threshold for a specific entity type."""
    entity_type: str
    threshold: float
    precision_at_threshold: float
    recall_at_threshold: float
    samples: int = 0


class WeightCalibrator:
    """
    Calibrates NER model weights using Inverse-Variance Weighting (IVW).

    IVW gives more weight to models with lower variance (more stable predictions),
    which typically correlates with higher precision.
    """

    def __init__(self, min_samples: int = 10):
        """
        Initialize the calibrator.

        Args:
            min_samples: Minimum samples required per model for calibration
        """
        self.min_samples = min_samples
        self._model_metrics: Dict[str, ModelMetrics] = {}
        self._entity_metrics: Dict[str, Dict[str, ModelMetrics]] = defaultdict(dict)
        self._calibrated_weights: Optional[Dict[str, float]] = None
        self._calibrated_thresholds: Optional[Dict[str, float]] = None

    def compute_ivw_weights(
        self,
        model_metrics: Dict[str, ModelMetrics]
    ) -> Dict[str, float]:
        """
        Compute Inverse-Variance Weighted scores for NER models.

        Models with lower variance (more consistent predictions) get higher weights.

        Args:
            model_metrics: Dict mapping model names to their metrics

        Returns:
            Dict mapping model names to normalized weights (sum to 1.0)
        """
        weights = {}

        for model_name, metrics in model_metrics.items():
            # Skip models with insufficient data
            if metrics.total_predictions < self.min_samples:
                logger.debug(f"Skipping {model_name}: insufficient samples ({metrics.total_predictions})")
                continue

            # Compute inverse variance weight
            # Add small epsilon to avoid division by zero
            variance = metrics.variance
            weights[model_name] = 1.0 / (variance + 1e-6)

        if not weights:
            logger.warning("No models had sufficient data for calibration")
            return DEFAULT_MODEL_WEIGHTS.copy()

        # Normalize weights to sum to 1.0
        total = sum(weights.values())
        normalized = {k: v / total for k, v in weights.items()}

        # Scale to match original weight range (0.7 - 1.0)
        # This keeps compatibility with existing consensus logic
        max_weight = max(normalized.values())
        min_weight = min(normalized.values())
        range_factor = 0.3  # Target range: 1.0 - 0.7 = 0.3

        if max_weight > min_weight:
            scaled = {}
            for k, v in normalized.items():
                # Scale to [0.7, 1.0] range
                scaled[k] = 0.7 + (range_factor * (v - min_weight) / (max_weight - min_weight))
            return scaled

        return {k: 0.85 for k in normalized}  # All equal if no variance

    def compute_ivw_weights_per_entity(
        self,
        entity_type: str
    ) -> Dict[str, float]:
        """
        Compute IVW weights for a specific entity type.

        Different entity types may have different optimal weightings
        (e.g., gliner might be better for non-Western names).

        Args:
            entity_type: The entity type (PERSON, LOCATION, etc.)

        Returns:
            Dict mapping model names to weights for this entity type
        """
        if entity_type not in self._entity_metrics:
            return DEFAULT_MODEL_WEIGHTS.copy()

        return self.compute_ivw_weights(self._entity_metrics[entity_type])

    def compute_optimal_thresholds(
        self,
        feedback_data: List[Dict],
        target_precision: float = 0.90
    ) -> Dict[str, EntityThreshold]:
        """
        Compute optimal confidence thresholds per entity type.

        Uses precision-recall curve analysis to find the threshold
        that achieves the target precision while maximizing recall.

        Args:
            feedback_data: List of feedback entries with detections and corrections
            target_precision: Target precision to achieve (default 0.90)

        Returns:
            Dict mapping entity types to calibrated thresholds
        """
        # Group feedback by entity type
        entity_feedback: Dict[str, List[Dict]] = defaultdict(list)
        for entry in feedback_data:
            entity_type = entry.get("detectedEntityType", "")
            if entity_type:
                entity_feedback[entity_type].append(entry)

        thresholds = {}

        for entity_type, entries in entity_feedback.items():
            if len(entries) < self.min_samples:
                continue

            threshold = self._find_optimal_threshold(entries, target_precision)
            if threshold:
                thresholds[entity_type] = threshold

        return thresholds

    def _find_optimal_threshold(
        self,
        entries: List[Dict],
        target_precision: float
    ) -> Optional[EntityThreshold]:
        """
        Find the optimal threshold for a set of feedback entries.

        Searches for the lowest threshold that achieves target precision.
        """
        # Collect (confidence, is_correct) pairs
        samples = []
        for entry in entries:
            confidence = entry.get("confidence", 0.5)
            detected_type = entry.get("detectedEntityType", "")
            suggested_types = entry.get("suggestedEntityTypes", [])

            # Detection is correct if detected type is in suggested types
            # (or if suggested is empty and user didn't provide correction)
            is_correct = detected_type in suggested_types if suggested_types else True
            samples.append((confidence, is_correct))

        if not samples:
            return None

        entity_type = entries[0].get("detectedEntityType", "UNKNOWN")

        # Sort by confidence descending
        samples.sort(key=lambda x: x[0], reverse=True)

        # Find threshold that achieves target precision
        best_threshold = 0.5  # Default
        best_recall = 0.0
        total_correct = sum(1 for _, correct in samples if correct)

        for i, (conf, _) in enumerate(samples):
            # Compute precision and recall at this threshold
            predictions_above = samples[:i+1]
            correct_above = sum(1 for _, c in predictions_above if c)
            precision = correct_above / len(predictions_above) if predictions_above else 0

            if total_correct > 0:
                recall = correct_above / total_correct
            else:
                recall = 0

            # Check if this threshold achieves target precision
            if precision >= target_precision:
                if recall > best_recall:
                    best_threshold = conf
                    best_recall = recall

        # Compute final metrics at best threshold
        predictions_at_threshold = [s for s in samples if s[0] >= best_threshold]
        if predictions_at_threshold:
            correct = sum(1 for _, c in predictions_at_threshold if c)
            precision_at_threshold = correct / len(predictions_at_threshold)
            recall_at_threshold = correct / total_correct if total_correct > 0 else 0
        else:
            precision_at_threshold = 0
            recall_at_threshold = 0

        return EntityThreshold(
            entity_type=entity_type,
            threshold=best_threshold,
            precision_at_threshold=precision_at_threshold,
            recall_at_threshold=recall_at_threshold,
            samples=len(samples)
        )

    def load_feedback(self, feedback_path: Path) -> List[Dict]:
        """
        Load feedback data from a directory of JSON files.

        Args:
            feedback_path: Path to feedback directory

        Returns:
            List of feedback entries
        """
        feedback_data = []

        if not feedback_path.exists():
            logger.warning(f"Feedback path does not exist: {feedback_path}")
            return feedback_data

        for f in feedback_path.glob("*.json"):
            try:
                with open(f, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                    # Handle both single entries (dict) and batch entries (list)
                    if isinstance(data, list):
                        feedback_data.extend(data)
                    else:
                        feedback_data.append(data)
            except (json.JSONDecodeError, IOError) as e:
                logger.debug(f"Skipping invalid feedback file {f}: {e}")
                continue

        logger.info(f"Loaded {len(feedback_data)} feedback entries from {feedback_path}")
        return feedback_data

    def calibrate(self, feedback_path: Path) -> Tuple[Dict[str, float], Dict[str, EntityThreshold]]:
        """
        Perform full calibration from feedback data.

        Args:
            feedback_path: Path to feedback directory

        Returns:
            Tuple of (model_weights, entity_thresholds)
        """
        feedback_data = self.load_feedback(feedback_path)

        if not feedback_data:
            logger.warning("No feedback data available for calibration")
            return DEFAULT_MODEL_WEIGHTS.copy(), {}

        # Compute model metrics from feedback
        model_metrics = self._compute_model_metrics_from_feedback(feedback_data)

        # Compute IVW weights
        weights = self.compute_ivw_weights(model_metrics)

        # Compute optimal thresholds
        thresholds = self.compute_optimal_thresholds(feedback_data)

        self._calibrated_weights = weights
        self._calibrated_thresholds = {k: v.threshold for k, v in thresholds.items()}

        return weights, thresholds

    def _compute_model_metrics_from_feedback(
        self,
        feedback_data: List[Dict]
    ) -> Dict[str, ModelMetrics]:
        """
        Compute model metrics from feedback data.

        Note: This requires feedback to include model-specific information.
        If not available, uses aggregate metrics.
        """
        # For now, compute aggregate metrics
        # Future: Extract per-model metrics if feedback includes source model
        metrics = ModelMetrics(model_name="aggregate")

        for entry in feedback_data:
            detected_type = entry.get("detectedEntityType", "")
            suggested_types = entry.get("suggestedEntityTypes", [])

            metrics.total_predictions += 1

            if suggested_types:
                if detected_type in suggested_types:
                    metrics.true_positives += 1
                else:
                    metrics.false_positives += 1

        metrics.update_metrics()

        # Return as aggregate for all models
        # In practice, you'd want per-model metrics from detailed feedback
        return {"aggregate": metrics}

    def get_calibrated_weights(self) -> Dict[str, float]:
        """Get the calibrated weights (or defaults if not calibrated)."""
        return self._calibrated_weights or DEFAULT_MODEL_WEIGHTS.copy()

    def get_calibrated_thresholds(self) -> Dict[str, float]:
        """Get the calibrated thresholds (or empty if not calibrated)."""
        return self._calibrated_thresholds or {}

    def save_calibration(self, output_path: Path):
        """Save calibration results to a JSON file."""
        data = {
            "weights": self._calibrated_weights or DEFAULT_MODEL_WEIGHTS,
            "thresholds": self._calibrated_thresholds or {},
            "min_samples": self.min_samples,
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved calibration to {output_path}")

    def load_calibration(self, input_path: Path) -> bool:
        """Load calibration results from a JSON file."""
        if not input_path.exists():
            return False

        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self._calibrated_weights = data.get("weights")
            self._calibrated_thresholds = data.get("thresholds")
            return True
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load calibration: {e}")
            return False


# Global calibrator instance
_calibrator_instance: Optional[WeightCalibrator] = None


def get_calibrator() -> WeightCalibrator:
    """Get the global calibrator instance."""
    global _calibrator_instance
    if _calibrator_instance is None:
        _calibrator_instance = WeightCalibrator()
    return _calibrator_instance


def calibrate_from_feedback(feedback_path: str) -> Dict[str, float]:
    """
    Convenience function to calibrate weights from a feedback directory.

    Args:
        feedback_path: Path to feedback directory

    Returns:
        Dict of calibrated model weights
    """
    calibrator = get_calibrator()
    weights, _ = calibrator.calibrate(Path(feedback_path))
    return weights
