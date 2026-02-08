#!/usr/bin/env python3
"""
LightGBM-based Named Entity Recognition

Lightweight NER using gradient boosting classifiers instead of heavy
transformer models. Each entity type has its own binary classifier.

Memory footprint: ~5-10MB per entity type (vs 500MB-1GB for transformers)
Inference speed: 5-10x faster than transformer models

Privacy: Models are trained ONLY on synthetic/public data.
No user data is ever used for training.
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Eager import for LightGBM (required for model loading)
# Note: Lazy import was causing segfaults due to import order issues
_lgbm = None
_LGBM_AVAILABLE = False

try:
    import lightgbm as lgbm
    _lgbm = lgbm
    _LGBM_AVAILABLE = True
    logger.debug("LightGBM available")
except ImportError:
    _LGBM_AVAILABLE = False
    logger.info("LightGBM not installed. Install with: pip install lightgbm")


def _check_lgbm():
    """Check if LightGBM is available."""
    return _LGBM_AVAILABLE


# Default model directory
DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "models" / "lgbm"

# Supported entity types for lightweight NER
SUPPORTED_ENTITIES = {
    "PERSON",
    "LOCATION",
    "ORGANIZATION",
    "DATE_TIME",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
}

# Default confidence thresholds per entity type (tuned during training)
# Updated 2026-02-07 for ai4privacy sentence-level models
DEFAULT_THRESHOLDS = {
    "PERSON": 0.6,        # F1=0.672 at this threshold (ai4privacy)
    "LOCATION": 0.5,      # F1=0.765 at this threshold (ai4privacy)
    "ORGANIZATION": 0.7,  # F1=0.959 at this threshold (synthetic)
    "DATE_TIME": 0.6,     # F1=0.857 at this threshold (ai4privacy)
    "ADDRESS": 0.5,       # F1=0.762 at this threshold (ai4privacy)
    "EMAIL_ADDRESS": 0.7,
    "PHONE_NUMBER": 0.6,
}


@dataclass
class LightweightEntity:
    """Detected entity from lightweight NER."""
    entity_type: str
    text: str
    start: int
    end: int
    confidence: float
    source: str = "lgbm_ner"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entity_type": self.entity_type,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "source": self.source,
        }


class LightGBMNERClassifier:
    """
    Token-level NER classifier using LightGBM.

    Uses gradient boosting to classify tokens as entity/non-entity
    based on extracted features.
    """

    def __init__(
        self,
        entity_type: str,
        model_path: Optional[Path] = None,
        threshold: float = None,
    ):
        """
        Initialize classifier for a specific entity type.

        Args:
            entity_type: The entity type this classifier detects (e.g., "PERSON")
            model_path: Path to trained model file (default: auto-detect)
            threshold: Confidence threshold for predictions (default: type-specific)
        """
        self.entity_type = entity_type
        self.threshold = threshold or DEFAULT_THRESHOLDS.get(entity_type, 0.5)
        self._model = None
        self._model_path = model_path

        if not _check_lgbm():
            logger.warning(f"LightGBM not available for {entity_type} classifier")

    @property
    def model_path(self) -> Path:
        """Get the model file path."""
        if self._model_path:
            return self._model_path
        return DEFAULT_MODEL_DIR / f"{self.entity_type.lower()}_classifier.txt"

    @property
    def is_available(self) -> bool:
        """Check if this classifier is available (model exists and LightGBM installed)."""
        return _check_lgbm() and self.model_path.exists()

    def load(self) -> bool:
        """
        Load the trained model from disk.

        Returns:
            True if model loaded successfully, False otherwise
        """
        if not _check_lgbm():
            return False

        if not self.model_path.exists():
            logger.debug(f"Model not found: {self.model_path}")
            return False

        try:
            self._model = _lgbm.Booster(model_file=str(self.model_path))
            logger.info(f"Loaded LightGBM model for {self.entity_type}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model {self.model_path}: {e}")
            return False

    def _ensure_loaded(self) -> bool:
        """Ensure model is loaded."""
        if self._model is None:
            return self.load()
        return True

    def predict_proba(self, features: List[Dict[str, Any]]) -> List[float]:
        """
        Predict probability of each token being this entity type.

        Args:
            features: List of feature dictionaries (from feature_extractor)

        Returns:
            List of probabilities (one per token)
        """
        if not self._ensure_loaded():
            return [0.0] * len(features)

        if not features:
            return []

        try:
            # Convert to format LightGBM expects
            import numpy as np
            from .feature_extractor import FEATURE_NAMES

            # Build feature matrix
            X = np.zeros((len(features), len(FEATURE_NAMES)))
            for i, feat_dict in enumerate(features):
                for j, name in enumerate(FEATURE_NAMES):
                    X[i, j] = feat_dict.get(name, 0)

            # Predict
            probs = self._model.predict(X)
            return probs.tolist()

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return [0.0] * len(features)

    def predict(
        self,
        features: List[Dict[str, Any]],
        threshold: float = None
    ) -> List[bool]:
        """
        Predict whether each token is this entity type.

        Args:
            features: List of feature dictionaries
            threshold: Custom threshold (default: self.threshold)

        Returns:
            List of boolean predictions
        """
        threshold = threshold or self.threshold
        probs = self.predict_proba(features)
        return [p >= threshold for p in probs]


class LightweightNER:
    """
    Lightweight NER system using LightGBM classifiers.

    Provides a unified interface for detecting multiple entity types
    with minimal memory footprint.
    """

    def __init__(
        self,
        entity_types: Set[str] = None,
        model_dir: Path = None,
        thresholds: Dict[str, float] = None,
    ):
        """
        Initialize lightweight NER system.

        Args:
            entity_types: Set of entity types to detect (default: all supported)
            model_dir: Directory containing trained models
            thresholds: Custom thresholds per entity type
        """
        self.entity_types = entity_types or SUPPORTED_ENTITIES
        self.model_dir = model_dir or DEFAULT_MODEL_DIR
        self.thresholds = thresholds or {}

        self._classifiers: Dict[str, LightGBMNERClassifier] = {}
        self._loaded = False

    def load(self) -> Dict[str, bool]:
        """
        Load all available classifiers.

        Returns:
            Dictionary mapping entity types to load success status
        """
        results = {}
        for entity_type in self.entity_types:
            if entity_type not in SUPPORTED_ENTITIES:
                logger.warning(f"Unsupported entity type: {entity_type}")
                results[entity_type] = False
                continue

            threshold = self.thresholds.get(entity_type)
            model_path = self.model_dir / f"{entity_type.lower()}_classifier.txt"

            classifier = LightGBMNERClassifier(
                entity_type=entity_type,
                model_path=model_path,
                threshold=threshold,
            )

            if classifier.load():
                self._classifiers[entity_type] = classifier
                results[entity_type] = True
            else:
                results[entity_type] = False

        self._loaded = True
        loaded_count = sum(results.values())
        logger.info(f"Loaded {loaded_count}/{len(self.entity_types)} LightGBM classifiers")
        return results

    @property
    def available_entities(self) -> Set[str]:
        """Get set of entity types with loaded classifiers."""
        return set(self._classifiers.keys())

    def detect(self, text: str) -> List[LightweightEntity]:
        """
        Detect entities in text using lightweight classifiers.

        Args:
            text: Input text to analyze

        Returns:
            List of detected entities
        """
        if not self._loaded:
            self.load()

        if not self._classifiers:
            logger.debug("No classifiers loaded, skipping lightweight NER")
            return []

        # Extract features
        from .feature_extractor import extract_features_with_context, tokenize, features_to_matrix

        tokens = tokenize(text)
        if not tokens:
            return []

        features_list = extract_features_with_context(text)
        feature_dicts = features_to_matrix(features_list)

        # Run each classifier
        entities = []
        for entity_type, classifier in self._classifiers.items():
            probs = classifier.predict_proba(feature_dicts)

            # Find sequences of high-probability tokens
            entity_spans = self._extract_entity_spans(
                tokens, probs, classifier.threshold
            )

            for start_idx, end_idx, confidence in entity_spans:
                start_char = tokens[start_idx][1]
                end_char = tokens[end_idx][2]
                entity_text = text[start_char:end_char]

                entities.append(LightweightEntity(
                    entity_type=entity_type,
                    text=entity_text,
                    start=start_char,
                    end=end_char,
                    confidence=confidence,
                ))

        # Sort by position
        entities.sort(key=lambda e: (e.start, -e.confidence))

        # Remove duplicates (keep highest confidence)
        return self._deduplicate_entities(entities)

    def _extract_entity_spans(
        self,
        tokens: List[Tuple[str, int, int]],
        probs: List[float],
        threshold: float,
    ) -> List[Tuple[int, int, float]]:
        """
        Extract contiguous entity spans from token probabilities.

        Uses BIO-style logic: adjacent high-probability tokens form a single entity.

        Args:
            tokens: List of (token, start, end) tuples
            probs: Probability for each token
            threshold: Minimum probability threshold

        Returns:
            List of (start_idx, end_idx, avg_confidence) tuples
        """
        spans = []
        current_start = None
        current_probs = []

        for i, prob in enumerate(probs):
            if prob >= threshold:
                if current_start is None:
                    current_start = i
                current_probs.append(prob)
            else:
                if current_start is not None:
                    # End of entity span
                    avg_conf = sum(current_probs) / len(current_probs)
                    spans.append((current_start, i - 1, avg_conf))
                    current_start = None
                    current_probs = []

        # Handle entity at end of text
        if current_start is not None:
            avg_conf = sum(current_probs) / len(current_probs)
            spans.append((current_start, len(probs) - 1, avg_conf))

        return spans

    def _deduplicate_entities(
        self,
        entities: List[LightweightEntity]
    ) -> List[LightweightEntity]:
        """
        Remove overlapping entities, keeping highest confidence.

        Args:
            entities: List of entities (sorted by position)

        Returns:
            Deduplicated list
        """
        if not entities:
            return []

        result = []
        for entity in entities:
            # Check for overlap with existing entities
            overlaps = False
            for existing in result:
                if (entity.start < existing.end and entity.end > existing.start):
                    # Overlapping - keep higher confidence
                    if entity.confidence > existing.confidence:
                        result.remove(existing)
                    else:
                        overlaps = True
                    break

            if not overlaps:
                result.append(entity)

        return result


# Global singleton instance
_lightweight_ner: Optional[LightweightNER] = None


def get_lightweight_ner() -> LightweightNER:
    """Get or create the global LightweightNER instance."""
    global _lightweight_ner
    if _lightweight_ner is None:
        _lightweight_ner = LightweightNER()
    return _lightweight_ner


def detect_entities_lightweight(text: str) -> List[Dict[str, Any]]:
    """
    Convenience function to detect entities using lightweight NER.

    Args:
        text: Input text

    Returns:
        List of entity dictionaries
    """
    ner = get_lightweight_ner()
    entities = ner.detect(text)
    return [e.to_dict() for e in entities]


def is_lightweight_available() -> bool:
    """Check if lightweight NER is available (LightGBM installed + models exist)."""
    if not _check_lgbm():
        return False

    # Check if any models exist
    if DEFAULT_MODEL_DIR.exists():
        model_files = list(DEFAULT_MODEL_DIR.glob("*_classifier.txt"))
        return len(model_files) > 0

    return False
