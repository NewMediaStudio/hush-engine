#!/usr/bin/env python3
"""
Threshold Calibration Tool for Hush Engine NER Models

Uses precision-recall curve analysis (inspired by Yellowbrick's DiscriminationThreshold)
to find optimal confidence thresholds for each entity type and source model.

Methodology:
-----------
This tool implements a Yellowbrick-style threshold calibration workflow:

1. **Data Collection**: Collects (confidence, is_correct) pairs from:
   - Fresh detections on datasets with ground truth (recommended)
   - User feedback files (training/feedback/*.json)
   - Benchmark results with ground truth comparisons

2. **Precision-Recall Curve Analysis**: For each threshold t in [0, 1]:
   - Predictions above t are treated as positive
   - Computes precision = TP / (TP + FP)
   - Computes recall = TP / (TP + FN)
   - Computes F1 = 2 * (precision * recall) / (precision + recall)

3. **Optimal Threshold Selection**: Supports multiple strategies:
   - F1 Maximization: Find threshold that maximizes F1 score (default)
   - Target Precision: Find lowest threshold achieving target precision
   - Target Recall: Find highest threshold achieving target recall

4. **Visualization**: Generates precision-recall tradeoff plots similar to
   Yellowbrick's DiscriminationThreshold visualizer, showing:
   - Precision, Recall, F1 curves vs threshold
   - Optimal threshold marker
   - Queue rate (fraction of items above threshold)

Yellowbrick Background:
----------------------
Yellowbrick's DiscriminationThreshold visualizer is a model selection tool that
helps find the optimal threshold for binary classifiers. Instead of using the
default 0.5 threshold, it evaluates precision, recall, and F1 across all possible
thresholds and identifies the optimal point based on the selected metric.

Key concepts from Yellowbrick:
- **Queue Rate**: The fraction of predictions above the threshold. Higher thresholds
  mean fewer predictions need review but may miss true positives.
- **F-beta Score**: Generalizes F1 to weight precision vs recall. F1 gives equal weight.
- **Threshold Sweep**: Evaluates metrics at each unique confidence score in the data.

This tool adapts Yellowbrick's methodology for NER confidence calibration,
computing separate optimal thresholds per entity type and per source model.

Usage:
------
    # Generate calibration data from synthetic golden set (RECOMMENDED)
    python calibrate_thresholds.py --dataset tests/data/synthetic_golden.json --samples 500

    # Calibrate from existing feedback data
    python calibrate_thresholds.py --feedback training/feedback

    # Generate visualizations
    python calibrate_thresholds.py --dataset tests/data/synthetic_golden.json --visualize

    # Target 90% precision (for high-precision applications)
    python calibrate_thresholds.py --dataset tests/data/synthetic_golden.json --target-precision 0.9

    # Target 95% recall (for high-recall applications)
    python calibrate_thresholds.py --dataset tests/data/synthetic_golden.json --target-recall 0.95

    # Output recommended config changes as Python code
    python calibrate_thresholds.py --dataset tests/data/synthetic_golden.json --output-python

    # Save full results to JSON
    python calibrate_thresholds.py --dataset tests/data/synthetic_golden.json \
        --output training/analysis/calibration_results.json

Output:
-------
- **Console**: Per-entity and per-model threshold recommendations with expected metrics
- **JSON** (--output): Full calibration data including precision-recall curves
- **Python** (--output-python): Ready-to-use CALIBRATED_THRESHOLDS dict for config
- **Plots** (--visualize): PNG files showing precision-recall-F1 curves per entity/model

References:
-----------
- Yellowbrick DiscriminationThreshold: https://www.scikit-yb.org/en/latest/api/classifier/threshold.html
- sklearn.metrics.precision_recall_curve
- "The Relationship Between Precision-Recall and ROC Curves" (Davis & Goadrich, 2006)
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CalibrationSample:
    """A single (confidence, is_correct) pair for threshold calibration."""
    confidence: float
    is_correct: bool
    entity_type: str
    source_model: str = "unknown"
    text: str = ""

    def __post_init__(self):
        # Clamp confidence to [0, 1]
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class ThresholdResult:
    """Result of threshold optimization for one entity type or model."""
    entity_type: str
    optimal_threshold: float
    precision_at_threshold: float
    recall_at_threshold: float
    f1_at_threshold: float
    samples_count: int
    strategy: str  # "f1_max", "target_precision", "target_recall"

    # Precision-recall curve data for visualization
    thresholds: List[float] = field(default_factory=list)
    precisions: List[float] = field(default_factory=list)
    recalls: List[float] = field(default_factory=list)
    f1_scores: List[float] = field(default_factory=list)
    queue_rates: List[float] = field(default_factory=list)


@dataclass
class CalibrationReport:
    """Complete calibration report with results per entity type."""
    timestamp: str
    strategy: str
    target_value: Optional[float]
    entity_results: Dict[str, ThresholdResult]
    model_results: Dict[str, ThresholdResult]
    overall_samples: int
    data_sources: List[str]


# =============================================================================
# DATA LOADERS
# =============================================================================

class FeedbackLoader:
    """Load calibration samples from user feedback files."""

    @staticmethod
    def load(feedback_dir: Path) -> List[CalibrationSample]:
        """Load feedback and convert to calibration samples.

        Feedback schema (from training/README.md):
        {
            "detectedText": str,
            "detectedEntityType": str,
            "suggestedEntityTypes": [str],
            "confidence": float,
            "notes": str
        }

        A detection is "correct" if detectedEntityType is in suggestedEntityTypes.
        """
        samples = []

        if not feedback_dir.exists():
            logger.warning(f"Feedback directory not found: {feedback_dir}")
            return samples

        for fb_file in feedback_dir.glob("*.json"):
            try:
                with open(fb_file, 'r') as f:
                    data = json.load(f)

                # Handle both single entries and batch entries
                entries = data if isinstance(data, list) else [data]

                for entry in entries:
                    sample = FeedbackLoader._entry_to_sample(entry)
                    if sample:
                        samples.append(sample)

            except (json.JSONDecodeError, IOError) as e:
                logger.debug(f"Skipping {fb_file.name}: {e}")

        logger.info(f"Loaded {len(samples)} samples from feedback")
        return samples

    @staticmethod
    def _entry_to_sample(entry: Dict) -> Optional[CalibrationSample]:
        """Convert a feedback entry to a calibration sample."""
        detected_type = entry.get("detectedEntityType", "")
        suggested_types = entry.get("suggestedEntityTypes", [])
        confidence = entry.get("confidence", 0.5)
        text = entry.get("detectedText", "")

        # Skip if no detected type or it's CUSTOM (missed detection)
        if not detected_type or detected_type == "CUSTOM":
            return None

        # Determine correctness:
        # - Correct if detected type is in suggested types
        # - Incorrect if suggested_types is empty (false positive)
        # - Incorrect if detected type not in suggested_types
        if suggested_types:
            is_correct = detected_type in suggested_types
        else:
            # Empty suggested = user marked as false positive
            is_correct = False

        # Try to extract source model from entry
        source_model = entry.get("source_model", "unknown")

        return CalibrationSample(
            confidence=confidence,
            is_correct=is_correct,
            entity_type=detected_type,
            source_model=source_model,
            text=text
        )


class BenchmarkLoader:
    """Load calibration samples from benchmark results."""

    @staticmethod
    def load(benchmark_dir: Path) -> List[CalibrationSample]:
        """Load benchmark cache files and extract calibration samples.

        Note: Benchmark feedback files typically only contain errors (missed
        detections and false positives), not true positives. For proper
        calibration with both correct and incorrect samples, use DatasetLoader
        to run fresh detections against a dataset with ground truth.
        """
        samples = []

        if not benchmark_dir.exists():
            logger.warning(f"Benchmark directory not found: {benchmark_dir}")
            return samples

        # Look for benchmark feedback files (generated by benchmark_accuracy.py)
        for fb_file in benchmark_dir.glob("benchmark_*.json"):
            try:
                with open(fb_file, 'r') as f:
                    data = json.load(f)

                entries = data if isinstance(data, list) else [data]

                for entry in entries:
                    sample = FeedbackLoader._entry_to_sample(entry)
                    if sample:
                        samples.append(sample)

            except (json.JSONDecodeError, IOError) as e:
                logger.debug(f"Skipping {fb_file.name}: {e}")

        logger.info(f"Loaded {len(samples)} samples from benchmark data")
        return samples


class DatasetLoader:
    """Run fresh detections on a dataset to generate calibration samples."""

    # Entity type mapping (from benchmark_accuracy.py)
    ENTITY_MAP = {
        'PHONE_NUMBER': 'PHONE',
        'EMAIL_ADDRESS': 'EMAIL',
        'LOCATION': 'ADDRESS',
        'PERSON': 'PERSON',
        'URL': 'URL',
        'US_SSN': 'NATIONAL_ID',
        'SSN': 'NATIONAL_ID',
        'CREDIT_CARD': 'CREDIT_CARD',
        'NRP': 'PERSON',
        'IP_ADDRESS': 'IP_ADDRESS',
        'COORDINATES': 'COORDINATES',
        'DATE_TIME': 'DATE_TIME',
        'AGE': 'AGE',
        'GENDER': 'GENDER',
        'FINANCIAL': 'FINANCIAL',
        'COMPANY': 'COMPANY',
        'MEDICAL': 'MEDICAL',
        'ORGANIZATION': 'COMPANY',
        'VEHICLE_ID': 'VEHICLE',
        'NATIONAL_ID': 'NATIONAL_ID',
        'ADDRESS': 'ADDRESS',
    }

    @staticmethod
    def load(
        dataset_path: Path,
        max_samples: int = None,
        verbose: bool = False
    ) -> List[CalibrationSample]:
        """
        Run PII detection on dataset and compare against ground truth.

        This generates proper (confidence, is_correct) pairs for calibration
        by running fresh detections and matching against ground truth labels.

        Args:
            dataset_path: Path to dataset (CSV, Parquet, or Arrow)
            max_samples: Maximum samples to process
            verbose: Print progress

        Returns:
            List of CalibrationSamples with both correct and incorrect detections
        """
        import csv
        import re

        samples = []

        if not dataset_path.exists():
            logger.error(f"Dataset not found: {dataset_path}")
            return samples

        # Load PII detector
        try:
            from hush_engine.detectors.pii_detector import PIIDetector
            detector = PIIDetector()
            logger.info("Loaded PII detector")
        except ImportError as e:
            logger.error(f"Could not import PIIDetector: {e}")
            return samples

        # Load dataset
        logger.info(f"Loading dataset: {dataset_path}")
        rows = DatasetLoader._load_rows(dataset_path, max_samples)
        logger.info(f"Loaded {len(rows)} rows")

        # Process each row
        total_detections = 0
        total_correct = 0
        total_incorrect = 0

        try:
            from tqdm import tqdm
            iterator = tqdm(rows, desc="Generating calibration data")
        except ImportError:
            iterator = rows
            if verbose:
                logger.info("Processing rows (install tqdm for progress bar)")

        for row in iterator:
            text = row.get('text', '')
            if not text:
                continue

            # Get ground truth
            ground_truth = DatasetLoader._extract_ground_truth(row)

            # Run detection
            try:
                entities = detector.analyze_text(text)
            except Exception as e:
                logger.debug(f"Detection error: {e}")
                continue

            # Match each detection against ground truth
            for entity in entities:
                entity_type = DatasetLoader.ENTITY_MAP.get(
                    entity.entity_type, entity.entity_type
                )
                entity_text = text[entity.start:entity.end]
                confidence = getattr(entity, 'confidence', 0.5)

                # Extract source model
                source_model = "unknown"
                if hasattr(entity, 'recognition_metadata') and entity.recognition_metadata:
                    source_model = entity.recognition_metadata.get(
                        'detection_source',
                        entity.recognition_metadata.get('recognizer_name', 'unknown')
                    )

                # Check if detection matches ground truth
                is_correct = DatasetLoader._check_match(
                    entity_text, entity_type, ground_truth
                )

                samples.append(CalibrationSample(
                    confidence=confidence,
                    is_correct=is_correct,
                    entity_type=entity_type,
                    source_model=source_model,
                    text=entity_text
                ))

                total_detections += 1
                if is_correct:
                    total_correct += 1
                else:
                    total_incorrect += 1

        logger.info(
            f"Generated {len(samples)} calibration samples: "
            f"{total_correct} correct, {total_incorrect} incorrect"
        )

        return samples

    @staticmethod
    def _load_rows(path: Path, max_rows: int = None) -> List[Dict]:
        """Load rows from various dataset formats."""
        import csv

        suffix = path.suffix.lower()
        rows = []

        if suffix == '.csv':
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if max_rows and i >= max_rows:
                        break
                    # Handle both 'text' and 'Text' columns
                    if 'text' not in row and 'Text' in row:
                        row['text'] = row['Text']
                    rows.append(row)

        elif suffix == '.parquet':
            try:
                import pyarrow.parquet as pq
                import ast
                df = pq.read_table(path).to_pandas()
                if max_rows:
                    df = df.head(max_rows)
                for _, row in df.iterrows():
                    rows.append(dict(row))
            except ImportError:
                logger.error("pyarrow required for Parquet files")

        elif suffix == '.arrow':
            try:
                import pyarrow as pa
                with pa.memory_map(str(path), 'r') as source:
                    try:
                        reader = pa.ipc.open_stream(source)
                    except Exception:
                        reader = pa.ipc.open_file(source)
                    df = reader.read_all().to_pandas()
                if max_rows:
                    df = df.head(max_rows)
                for _, row in df.iterrows():
                    rows.append(dict(row))
            except ImportError:
                logger.error("pyarrow required for Arrow files")

        return rows

    @staticmethod
    def _extract_ground_truth(row: Dict) -> Dict[str, List[str]]:
        """Extract ground truth labels from a row."""
        import ast

        ground_truth = defaultdict(list)

        # Check for pre-computed ground_truth dict
        if 'ground_truth' in row:
            gt = row['ground_truth']
            if isinstance(gt, str):
                try:
                    gt = ast.literal_eval(gt)
                except (ValueError, SyntaxError):
                    gt = {}
            if isinstance(gt, dict):
                for k, v in gt.items():
                    if isinstance(v, list):
                        ground_truth[k].extend(v)
                    else:
                        ground_truth[k].append(str(v))

        # Check for spans annotation
        elif 'spans' in row:
            text = row.get('text', '')
            spans_raw = row['spans']
            if isinstance(spans_raw, str):
                try:
                    spans = ast.literal_eval(spans_raw)
                except (ValueError, SyntaxError):
                    spans = []
            elif spans_raw is None:
                spans = []
            else:
                # Handle numpy arrays, pandas series, or lists
                try:
                    spans = list(spans_raw) if len(spans_raw) > 0 else []
                except (TypeError, ValueError):
                    spans = []

            for span in spans:
                if not isinstance(span, dict):
                    continue
                label = span.get('label', '')
                entity_type = DatasetLoader.ENTITY_MAP.get(label, label)
                if entity_type:
                    span_text = span.get('text', '')
                    if not span_text and 'start' in span and 'end' in span:
                        span_text = text[span['start']:span['end']]
                    if span_text:
                        ground_truth[entity_type].append(span_text)

        # Legacy CSV format with individual columns
        else:
            field_map = {
                'name': 'PERSON',
                'email': 'EMAIL',
                'phone': 'PHONE',
                'address': 'ADDRESS',
                'url': 'URL',
            }
            for field, entity_type in field_map.items():
                value = row.get(field, '')
                if value and str(value).strip():
                    ground_truth[entity_type].append(str(value).strip())

        return dict(ground_truth)

    @staticmethod
    def _check_match(
        detection_text: str,
        entity_type: str,
        ground_truth: Dict[str, List[str]]
    ) -> bool:
        """Check if a detection matches any ground truth entry."""
        import re

        def normalize(text):
            if not text:
                return ""
            return re.sub(r'\s+', ' ', str(text).lower().strip())

        gt_values = ground_truth.get(entity_type, [])
        if not gt_values:
            return False

        det_norm = normalize(detection_text)
        if not det_norm or len(det_norm) < 2:
            return False

        for gt in gt_values:
            gt_norm = normalize(gt)
            if not gt_norm:
                continue

            # Check substring match
            if det_norm in gt_norm or gt_norm in det_norm:
                return True

            # Check word overlap (50% threshold)
            det_words = set(det_norm.split())
            gt_words = set(gt_norm.split())
            if gt_words and len(gt_words & det_words) >= len(gt_words) * 0.5:
                return True

        return False


# =============================================================================
# THRESHOLD CALIBRATION ENGINE
# =============================================================================

class ThresholdCalibrator:
    """
    Computes optimal thresholds using precision-recall curve analysis.

    Inspired by Yellowbrick's DiscriminationThreshold visualizer which:
    1. Iterates through candidate thresholds
    2. Computes precision/recall at each threshold
    3. Finds optimal threshold based on selected metric
    """

    def __init__(
        self,
        strategy: str = "f1_max",
        target_value: float = None,
        min_samples: int = 10,
        threshold_steps: int = 100
    ):
        """
        Initialize the calibrator.

        Args:
            strategy: Optimization strategy
                - "f1_max": Maximize F1 score
                - "target_precision": Find threshold achieving target precision
                - "target_recall": Find threshold achieving target recall
            target_value: Target for "target_precision" or "target_recall" strategy
            min_samples: Minimum samples required for calibration
            threshold_steps: Number of threshold values to evaluate
        """
        self.strategy = strategy
        self.target_value = target_value or 0.90
        self.min_samples = min_samples
        self.threshold_steps = threshold_steps

    def calibrate(
        self,
        samples: List[CalibrationSample]
    ) -> CalibrationReport:
        """
        Run calibration on all samples.

        Returns calibration results grouped by entity type and source model.
        """
        # Group samples by entity type
        by_entity = defaultdict(list)
        by_model = defaultdict(list)

        for sample in samples:
            by_entity[sample.entity_type].append(sample)
            if sample.source_model != "unknown":
                by_model[sample.source_model].append(sample)

        # Calibrate each entity type
        entity_results = {}
        for entity_type, entity_samples in by_entity.items():
            if len(entity_samples) >= self.min_samples:
                result = self._calibrate_samples(entity_samples, entity_type)
                if result:
                    entity_results[entity_type] = result
            else:
                logger.debug(f"Skipping {entity_type}: only {len(entity_samples)} samples")

        # Calibrate each source model
        model_results = {}
        for model_name, model_samples in by_model.items():
            if len(model_samples) >= self.min_samples:
                result = self._calibrate_samples(model_samples, model_name)
                if result:
                    model_results[model_name] = result

        return CalibrationReport(
            timestamp=datetime.now().isoformat(),
            strategy=self.strategy,
            target_value=self.target_value if self.strategy != "f1_max" else None,
            entity_results=entity_results,
            model_results=model_results,
            overall_samples=len(samples),
            data_sources=["feedback", "benchmark"]
        )

    def _calibrate_samples(
        self,
        samples: List[CalibrationSample],
        name: str
    ) -> Optional[ThresholdResult]:
        """
        Compute optimal threshold for a set of samples.

        Uses the Yellowbrick methodology:
        1. Generate candidate thresholds from 0 to 1
        2. For each threshold, compute precision/recall/F1
        3. Select optimal based on strategy
        """
        if len(samples) < self.min_samples:
            return None

        # Sort samples by confidence (descending)
        sorted_samples = sorted(samples, key=lambda x: x.confidence, reverse=True)

        # Precompute total positives (correct detections)
        total_positives = sum(1 for s in samples if s.is_correct)
        total_samples = len(samples)

        if total_positives == 0:
            logger.warning(f"{name}: No correct detections in samples")
            return None

        # Generate threshold candidates
        thresholds = []
        precisions = []
        recalls = []
        f1_scores = []
        queue_rates = []

        # Use confidence values from samples as threshold candidates
        # Plus add evenly spaced values for smooth curves
        threshold_candidates = set()
        for s in samples:
            threshold_candidates.add(s.confidence)

        # Add evenly spaced thresholds
        for i in range(self.threshold_steps + 1):
            threshold_candidates.add(i / self.threshold_steps)

        threshold_candidates = sorted(threshold_candidates)

        # Compute metrics at each threshold
        for threshold in threshold_candidates:
            # Predictions above threshold
            above_threshold = [s for s in samples if s.confidence >= threshold]

            if not above_threshold:
                # No predictions at this threshold
                thresholds.append(threshold)
                precisions.append(1.0)  # No false positives
                recalls.append(0.0)     # No true positives
                f1_scores.append(0.0)
                queue_rates.append(0.0)
                continue

            # True positives: correct predictions above threshold
            tp = sum(1 for s in above_threshold if s.is_correct)
            # False positives: incorrect predictions above threshold
            fp = len(above_threshold) - tp
            # False negatives: correct samples below threshold
            fn = total_positives - tp

            # Precision: TP / (TP + FP)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            # Recall: TP / (TP + FN)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            # F1: harmonic mean
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            # Queue rate: fraction of samples above threshold (for cost analysis)
            queue_rate = len(above_threshold) / total_samples

            thresholds.append(threshold)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            queue_rates.append(queue_rate)

        # Find optimal threshold based on strategy
        optimal_idx = self._find_optimal_index(
            thresholds, precisions, recalls, f1_scores
        )

        return ThresholdResult(
            entity_type=name,
            optimal_threshold=thresholds[optimal_idx],
            precision_at_threshold=precisions[optimal_idx],
            recall_at_threshold=recalls[optimal_idx],
            f1_at_threshold=f1_scores[optimal_idx],
            samples_count=len(samples),
            strategy=self.strategy,
            thresholds=thresholds,
            precisions=precisions,
            recalls=recalls,
            f1_scores=f1_scores,
            queue_rates=queue_rates
        )

    def _find_optimal_index(
        self,
        thresholds: List[float],
        precisions: List[float],
        recalls: List[float],
        f1_scores: List[float]
    ) -> int:
        """Find index of optimal threshold based on strategy."""

        if self.strategy == "f1_max":
            # Find threshold that maximizes F1
            max_f1 = max(f1_scores)
            # If multiple thresholds have same F1, prefer lower threshold (more recall)
            for i, f1 in enumerate(f1_scores):
                if f1 == max_f1:
                    return i
            return f1_scores.index(max_f1)

        elif self.strategy == "target_precision":
            # Find lowest threshold that achieves target precision
            # (maximizes recall while meeting precision target)
            target = self.target_value
            best_idx = 0
            best_recall = 0.0

            for i, (prec, rec) in enumerate(zip(precisions, recalls)):
                if prec >= target and rec > best_recall:
                    best_idx = i
                    best_recall = rec

            return best_idx

        elif self.strategy == "target_recall":
            # Find highest threshold that achieves target recall
            # (maximizes precision while meeting recall target)
            target = self.target_value
            best_idx = len(thresholds) - 1
            best_precision = 0.0

            for i, (prec, rec) in enumerate(zip(precisions, recalls)):
                if rec >= target and prec > best_precision:
                    best_idx = i
                    best_precision = prec

            return best_idx

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")


# =============================================================================
# VISUALIZATION
# =============================================================================

class YellowbrickVisualizer:
    """
    Use Yellowbrick's DiscriminationThreshold visualizer for threshold calibration.

    Yellowbrick provides publication-quality visualizations that show:
    - Precision, Recall, F-beta curves vs threshold
    - Queue rate (fraction of items flagged at each threshold)
    - Optimal threshold marker

    This class adapts our calibration data to work with Yellowbrick's API.
    """

    @staticmethod
    def is_available() -> bool:
        """Check if Yellowbrick is installed."""
        try:
            import yellowbrick
            return True
        except ImportError:
            return False

    @staticmethod
    def plot_discrimination_threshold(
        samples: List[CalibrationSample],
        entity_type: str,
        output_path: Path,
        fbeta: float = 1.0
    ) -> Optional[ThresholdResult]:
        """
        Use Yellowbrick-style precision-recall threshold analysis.

        Since Yellowbrick's DiscriminationThreshold requires a proper classifier,
        we implement an equivalent visualization using sklearn's precision_recall_curve
        which is what Yellowbrick uses internally.

        Args:
            samples: List of CalibrationSample for this entity type
            entity_type: Name of the entity type
            output_path: Where to save the plot
            fbeta: Beta for F-beta score (1.0 = F1)

        Returns:
            ThresholdResult with optimal threshold and metrics
        """
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from sklearn.metrics import precision_recall_curve, f1_score
        except ImportError as e:
            logger.warning(f"sklearn not available: {e}")
            return None

        if len(samples) < 20:
            logger.warning(f"Not enough samples for {entity_type}: {len(samples)}")
            return None

        # Create arrays
        # y_true: 1 if correct detection, 0 if false positive
        # y_scores: confidence scores
        y_true = np.array([1 if s.is_correct else 0 for s in samples])
        y_scores = np.array([s.confidence for s in samples])

        # Check we have both classes
        n_positive = y_true.sum()
        n_negative = len(y_true) - n_positive
        if n_positive == 0 or n_negative == 0:
            logger.warning(f"{entity_type}: Need both correct and incorrect samples "
                          f"(pos={n_positive}, neg={n_negative})")
            return None

        # Compute precision-recall curve using sklearn
        # This is what Yellowbrick uses internally
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

        # precision_recall_curve returns one extra precision/recall value
        # Trim to match thresholds
        precisions = precisions[:-1]
        recalls = recalls[:-1]

        # Compute F-score at each threshold
        # F_beta = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)
        beta_sq = fbeta ** 2
        with np.errstate(divide='ignore', invalid='ignore'):
            f_scores = (1 + beta_sq) * (precisions * recalls) / (beta_sq * precisions + recalls)
            f_scores = np.nan_to_num(f_scores, nan=0.0)

        # Find optimal threshold (max F-score)
        optimal_idx = np.argmax(f_scores)
        optimal_thresh = thresholds[optimal_idx]
        precision_at_opt = precisions[optimal_idx]
        recall_at_opt = recalls[optimal_idx]
        f1_at_opt = f_scores[optimal_idx]

        # Create Yellowbrick-style plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot precision, recall, F-score curves (Yellowbrick style)
        ax.plot(thresholds, precisions, 'b-', label='Precision', linewidth=2, alpha=0.8)
        ax.plot(thresholds, recalls, 'g-', label='Recall', linewidth=2, alpha=0.8)
        ax.plot(thresholds, f_scores, 'r-', label=f'F{fbeta:.0f} Score', linewidth=2, alpha=0.8)

        # Mark optimal threshold with vertical line
        ax.axvline(x=optimal_thresh, color='#333333', linestyle='--', linewidth=2,
                   label=f'Optimal: {optimal_thresh:.3f}')

        # Add optimal point marker
        ax.scatter([optimal_thresh], [f1_at_opt], color='red', s=100, zorder=5,
                   edgecolors='black', linewidth=1.5)

        # Style the plot (Yellowbrick aesthetic)
        ax.set_xlabel('Discrimination Threshold', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.legend(loc='lower left', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

        # Add title with metrics
        ax.set_title(
            f"Discrimination Threshold: {entity_type}\n"
            f"Max F1 at {optimal_thresh:.3f} "
            f"(P={precision_at_opt:.1%}, R={recall_at_opt:.1%}, F1={f1_at_opt:.1%})\n"
            f"Samples: {len(samples)} ({n_positive} correct, {n_negative} false positives)",
            fontsize=11
        )

        # Add shaded region for queue rate
        queue_rates = []
        for t in thresholds:
            above = (y_scores >= t).sum()
            queue_rates.append(above / len(y_scores))
        queue_rates = np.array(queue_rates)

        ax2 = ax.twinx()
        ax2.fill_between(thresholds, queue_rates, alpha=0.1, color='gray')
        ax2.set_ylabel('Queue Rate', color='gray', fontsize=10)
        ax2.tick_params(axis='y', labelcolor='gray')
        ax2.set_ylim(0, 1.05)

        # Save the plot
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved Yellowbrick-style plot: {output_path}")

        return ThresholdResult(
            entity_type=entity_type,
            optimal_threshold=float(optimal_thresh),
            precision_at_threshold=float(precision_at_opt),
            recall_at_threshold=float(recall_at_opt),
            f1_at_threshold=float(f1_at_opt),
            samples_count=len(samples),
            strategy="yellowbrick_f1_max",
            thresholds=list(thresholds),
            precisions=list(precisions),
            recalls=list(recalls),
            f1_scores=list(f_scores),
            queue_rates=list(queue_rates)
        )

    @staticmethod
    def plot_all_entities(
        samples: List[CalibrationSample],
        output_dir: Path,
        entity_types: List[str] = None
    ) -> Dict[str, ThresholdResult]:
        """
        Generate Yellowbrick plots for all entity types.

        Args:
            samples: All calibration samples
            output_dir: Directory for output plots
            entity_types: Optional list of entity types to process

        Returns:
            Dict of entity_type -> ThresholdResult
        """
        # Group samples by entity type
        by_entity = defaultdict(list)
        for sample in samples:
            by_entity[sample.entity_type].append(sample)

        # Filter to requested entity types
        if entity_types:
            by_entity = {k: v for k, v in by_entity.items() if k in entity_types}

        results = {}
        output_dir.mkdir(parents=True, exist_ok=True)

        for entity_type, entity_samples in sorted(by_entity.items()):
            output_path = output_dir / f"yellowbrick_{entity_type.lower()}.png"
            result = YellowbrickVisualizer.plot_discrimination_threshold(
                entity_samples,
                entity_type,
                output_path
            )
            if result:
                results[entity_type] = result

        return results


class CalibrationVisualizer:
    """
    Generate Yellowbrick-style precision-recall threshold visualizations.

    Creates plots showing:
    - Precision vs threshold
    - Recall vs threshold
    - F1 vs threshold
    - Optimal threshold marker
    """

    @staticmethod
    def plot_threshold_curve(
        result: ThresholdResult,
        output_path: Path,
        title: str = None
    ):
        """Generate precision-recall threshold plot."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            logger.warning("matplotlib not available. Skipping visualization.")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot precision, recall, F1 curves
        ax.plot(result.thresholds, result.precisions, 'b-', label='Precision', linewidth=2)
        ax.plot(result.thresholds, result.recalls, 'g-', label='Recall', linewidth=2)
        ax.plot(result.thresholds, result.f1_scores, 'r-', label='F1 Score', linewidth=2)

        # Mark optimal threshold
        ax.axvline(
            x=result.optimal_threshold,
            color='k',
            linestyle='--',
            linewidth=1.5,
            label=f'Optimal: {result.optimal_threshold:.3f}'
        )

        # Add shaded region showing queue rate
        if result.queue_rates:
            ax2 = ax.twinx()
            ax2.fill_between(
                result.thresholds,
                result.queue_rates,
                alpha=0.1,
                color='gray',
                label='Queue Rate'
            )
            ax2.set_ylabel('Queue Rate (fraction above threshold)', color='gray')
            ax2.tick_params(axis='y', labelcolor='gray')
            ax2.set_ylim(0, 1.05)

        ax.set_xlabel('Confidence Threshold')
        ax.set_ylabel('Score')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        title = title or f"Threshold Calibration: {result.entity_type}"
        ax.set_title(f"{title}\n"
                     f"Optimal: {result.optimal_threshold:.3f} "
                     f"(P={result.precision_at_threshold:.2%}, "
                     f"R={result.recall_at_threshold:.2%}, "
                     f"F1={result.f1_at_threshold:.2%})")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved plot: {output_path}")

    @staticmethod
    def plot_all_entities(
        report: CalibrationReport,
        output_dir: Path
    ):
        """Generate plots for all entity types."""
        output_dir.mkdir(parents=True, exist_ok=True)

        for entity_type, result in report.entity_results.items():
            output_path = output_dir / f"threshold_{entity_type.lower()}.png"
            CalibrationVisualizer.plot_threshold_curve(
                result,
                output_path,
                title=f"Entity: {entity_type}"
            )

        for model_name, result in report.model_results.items():
            safe_name = model_name.replace("/", "_").replace("\\", "_")
            output_path = output_dir / f"threshold_model_{safe_name}.png"
            CalibrationVisualizer.plot_threshold_curve(
                result,
                output_path,
                title=f"Model: {model_name}"
            )


# =============================================================================
# OUTPUT FORMATTERS
# =============================================================================

class ConfigRecommender:
    """Generate recommended configuration changes."""

    @staticmethod
    def generate_recommendations(
        report: CalibrationReport,
        current_thresholds: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """Generate recommended threshold changes."""

        # Default thresholds from detection_config.py
        if current_thresholds is None:
            try:
                from hush_engine.detection_config import DEFAULT_THRESHOLDS
                current_thresholds = DEFAULT_THRESHOLDS
            except ImportError:
                current_thresholds = {}

        recommendations = {
            "generated_at": report.timestamp,
            "strategy": report.strategy,
            "samples_analyzed": report.overall_samples,
            "threshold_changes": [],
            "summary": {},
        }

        total_precision_improvement = 0.0
        total_recall_change = 0.0
        count = 0

        for entity_type, result in report.entity_results.items():
            current = current_thresholds.get(entity_type, 0.5)
            recommended = result.optimal_threshold
            delta = recommended - current

            if abs(delta) > 0.02:  # Only recommend if change > 2%
                change = {
                    "entity_type": entity_type,
                    "current_threshold": current,
                    "recommended_threshold": round(recommended, 3),
                    "delta": round(delta, 3),
                    "expected_precision": round(result.precision_at_threshold, 3),
                    "expected_recall": round(result.recall_at_threshold, 3),
                    "expected_f1": round(result.f1_at_threshold, 3),
                    "samples": result.samples_count,
                }
                recommendations["threshold_changes"].append(change)

                # Track improvements
                total_precision_improvement += (result.precision_at_threshold - 0.5) * 100
                count += 1

        # Sort by delta magnitude
        recommendations["threshold_changes"].sort(
            key=lambda x: abs(x["delta"]),
            reverse=True
        )

        recommendations["summary"] = {
            "entities_to_update": len(recommendations["threshold_changes"]),
            "average_f1_at_optimal": sum(
                r.f1_at_threshold for r in report.entity_results.values()
            ) / max(len(report.entity_results), 1),
        }

        return recommendations

    @staticmethod
    def format_as_python(recommendations: Dict) -> str:
        """Format recommendations as Python code for detection_config.py."""
        lines = [
            "# Calibrated thresholds (generated by calibrate_thresholds.py)",
            f"# Generated: {recommendations['generated_at']}",
            f"# Strategy: {recommendations['strategy']}",
            f"# Samples: {recommendations['samples_analyzed']}",
            "",
            "CALIBRATED_THRESHOLDS = {",
        ]

        for change in recommendations["threshold_changes"]:
            lines.append(
                f"    \"{change['entity_type']}\": {change['recommended_threshold']:.3f},  "
                f"# was {change['current_threshold']:.3f}, "
                f"F1={change['expected_f1']:.1%}"
            )

        lines.append("}")

        return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Calibrate NER confidence thresholds using precision-recall analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run fresh detections on dataset to generate calibration data (RECOMMENDED)
    python calibrate_thresholds.py --dataset tests/data/training/Training_Set_cache.csv --samples 500

    # Calibrate from existing feedback
    python calibrate_thresholds.py --feedback training/feedback

    # Calibrate to achieve 90% precision
    python calibrate_thresholds.py --dataset tests/data/training/Training_Set_cache.csv --target-precision 0.9

    # Generate visualizations
    python calibrate_thresholds.py --dataset tests/data/training/Training_Set_cache.csv --visualize

    # Output Python config code
    python calibrate_thresholds.py --dataset tests/data/training/Training_Set_cache.csv --output-python
        """
    )

    # Data sources
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Path to dataset with ground truth (CSV, Parquet, Arrow). "
             "Runs fresh detections to generate calibration samples."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Maximum samples to process from dataset"
    )
    parser.add_argument(
        "--feedback",
        type=Path,
        help="Path to feedback directory (alternative to --dataset)"
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        help="Path to benchmark feedback files"
    )

    # Calibration strategy
    parser.add_argument(
        "--strategy",
        choices=["f1_max", "target_precision", "target_recall"],
        default="f1_max",
        help="Threshold optimization strategy (default: f1_max)"
    )
    parser.add_argument(
        "--target-precision",
        type=float,
        help="Target precision (sets strategy to target_precision)"
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        help="Target recall (sets strategy to target_recall)"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Minimum samples required per entity type (default: 10)"
    )

    # Output options
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON file for calibration results"
    )
    parser.add_argument(
        "--output-python",
        action="store_true",
        help="Output recommended thresholds as Python code"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate precision-recall threshold plots"
    )
    parser.add_argument(
        "--yellowbrick",
        action="store_true",
        help="Use Yellowbrick's DiscriminationThreshold visualizer (requires: pip install yellowbrick)"
    )
    parser.add_argument(
        "--viz-dir",
        type=Path,
        default=REPO_ROOT / "training" / "analysis" / "calibration_plots",
        help="Directory for visualization outputs"
    )
    parser.add_argument(
        "--focus-entities",
        type=str,
        nargs="+",
        default=["PERSON", "ADDRESS", "COMPANY", "PHONE"],
        help="Entity types to focus calibration on (default: PERSON ADDRESS COMPANY PHONE)"
    )

    # Verbosity
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine strategy
    strategy = args.strategy
    target_value = None

    if args.target_precision:
        strategy = "target_precision"
        target_value = args.target_precision
    elif args.target_recall:
        strategy = "target_recall"
        target_value = args.target_recall

    # Load samples
    samples = []

    # Dataset mode: run fresh detections (recommended for proper calibration)
    if args.dataset:
        if args.dataset.exists():
            samples.extend(DatasetLoader.load(
                args.dataset,
                max_samples=args.samples,
                verbose=args.verbose
            ))
        else:
            logger.error(f"Dataset not found: {args.dataset}")
            sys.exit(1)

    # Feedback mode: load from feedback files
    if args.feedback and args.feedback.exists():
        samples.extend(FeedbackLoader.load(args.feedback))

    # Benchmark mode: load from benchmark feedback files
    if args.benchmark and args.benchmark.exists():
        samples.extend(BenchmarkLoader.load(args.benchmark))

    if not samples:
        logger.error("No calibration samples found. Use --dataset, --feedback, or --benchmark.")
        sys.exit(1)

    logger.info(f"Total samples: {len(samples)}")

    # Run calibration
    calibrator = ThresholdCalibrator(
        strategy=strategy,
        target_value=target_value,
        min_samples=args.min_samples
    )

    report = calibrator.calibrate(samples)

    # Output results
    print("\n" + "=" * 70)
    print("CALIBRATION RESULTS")
    print("=" * 70)
    print(f"Strategy: {report.strategy}")
    if report.target_value:
        print(f"Target: {report.target_value:.1%}")
    print(f"Total samples: {report.overall_samples}")
    print()

    # Entity type results
    if report.entity_results:
        print("Per-Entity Thresholds:")
        print("-" * 70)
        print(f"{'Entity Type':<20} {'Optimal':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Samples':>8}")
        print("-" * 70)

        for entity_type, result in sorted(report.entity_results.items()):
            print(
                f"{entity_type:<20} "
                f"{result.optimal_threshold:>10.3f} "
                f"{result.precision_at_threshold:>10.1%} "
                f"{result.recall_at_threshold:>10.1%} "
                f"{result.f1_at_threshold:>10.1%} "
                f"{result.samples_count:>8}"
            )
        print()

    # Model results
    if report.model_results:
        print("Per-Model Thresholds:")
        print("-" * 70)
        print(f"{'Model'::<30} {'Optimal':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print("-" * 70)

        for model_name, result in sorted(report.model_results.items()):
            print(
                f"{model_name:<30} "
                f"{result.optimal_threshold:>10.3f} "
                f"{result.precision_at_threshold:>10.1%} "
                f"{result.recall_at_threshold:>10.1%} "
                f"{result.f1_at_threshold:>10.1%}"
            )
        print()

    # Generate recommendations
    recommendations = ConfigRecommender.generate_recommendations(report)

    if recommendations["threshold_changes"]:
        print("\nRecommended Changes:")
        print("-" * 70)
        for change in recommendations["threshold_changes"]:
            direction = "+" if change["delta"] > 0 else ""
            print(
                f"  {change['entity_type']}: "
                f"{change['current_threshold']:.3f} -> {change['recommended_threshold']:.3f} "
                f"({direction}{change['delta']:.3f})"
            )

    # Save JSON output
    if args.output:
        output_data = {
            "calibration": {
                "timestamp": report.timestamp,
                "strategy": report.strategy,
                "target_value": report.target_value,
                "overall_samples": report.overall_samples,
            },
            "entity_thresholds": {
                et: {
                    "threshold": r.optimal_threshold,
                    "precision": r.precision_at_threshold,
                    "recall": r.recall_at_threshold,
                    "f1": r.f1_at_threshold,
                    "samples": r.samples_count,
                }
                for et, r in report.entity_results.items()
            },
            "model_thresholds": {
                mn: {
                    "threshold": r.optimal_threshold,
                    "precision": r.precision_at_threshold,
                    "recall": r.recall_at_threshold,
                    "f1": r.f1_at_threshold,
                    "samples": r.samples_count,
                }
                for mn, r in report.model_results.items()
            },
            "recommendations": recommendations,
        }

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved JSON output: {args.output}")

    # Output Python code
    if args.output_python:
        print("\n" + "=" * 70)
        print("PYTHON CONFIG CODE")
        print("=" * 70)
        print(ConfigRecommender.format_as_python(recommendations))

    # Generate visualizations
    if args.visualize:
        CalibrationVisualizer.plot_all_entities(report, args.viz_dir)
        print(f"\nSaved visualizations to: {args.viz_dir}")

    # Use Yellowbrick if requested
    if args.yellowbrick:
        if YellowbrickVisualizer.is_available():
            print("\n" + "=" * 70)
            print("YELLOWBRICK DISCRIMINATION THRESHOLD ANALYSIS")
            print("=" * 70)

            yb_results = YellowbrickVisualizer.plot_all_entities(
                samples,
                args.viz_dir / "yellowbrick",
                entity_types=args.focus_entities
            )

            if yb_results:
                print(f"\n{'Entity Type':<20} {'YB Threshold':>12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
                print("-" * 70)

                for entity_type, result in sorted(yb_results.items()):
                    print(
                        f"{entity_type:<20} "
                        f"{result.optimal_threshold:>12.3f} "
                        f"{result.precision_at_threshold:>10.1%} "
                        f"{result.recall_at_threshold:>10.1%} "
                        f"{result.f1_at_threshold:>10.1%}"
                    )

                # Comparison with current thresholds
                print("\n" + "-" * 70)
                print("COMPARISON WITH CURRENT THRESHOLDS:")
                print("-" * 70)

                try:
                    from hush_engine.detection_config import DEFAULT_THRESHOLDS
                    current_thresholds = DEFAULT_THRESHOLDS
                except ImportError:
                    current_thresholds = {
                        "PERSON": 0.55,
                        "ADDRESS": 0.50,
                        "COMPANY": 0.50,
                        "PHONE": 0.50,
                    }

                for entity_type in args.focus_entities:
                    if entity_type in yb_results:
                        current = current_thresholds.get(entity_type, 0.50)
                        optimal = yb_results[entity_type].optimal_threshold
                        delta = optimal - current
                        direction = "+" if delta > 0 else ""

                        # Assess impact
                        if delta > 0.1:
                            impact = "RAISE SIGNIFICANTLY - reduce false positives"
                        elif delta > 0.03:
                            impact = "Raise slightly - improve precision"
                        elif delta < -0.1:
                            impact = "LOWER SIGNIFICANTLY - improve recall"
                        elif delta < -0.03:
                            impact = "Lower slightly - improve recall"
                        else:
                            impact = "Keep current threshold"

                        print(
                            f"  {entity_type}: {current:.3f} -> {optimal:.3f} "
                            f"({direction}{delta:.3f}) | {impact}"
                        )

                print(f"\nSaved Yellowbrick plots to: {args.viz_dir / 'yellowbrick'}")
        else:
            print("\nYellowbrick not available. Install with: pip install yellowbrick")

    # Focus entity summary
    print("\n" + "=" * 70)
    print("FOCUS ENTITY THRESHOLD RECOMMENDATIONS")
    print("=" * 70)

    try:
        from hush_engine.detection_config import DEFAULT_THRESHOLDS
        current_thresholds = DEFAULT_THRESHOLDS
    except ImportError:
        current_thresholds = {}

    for entity_type in args.focus_entities:
        if entity_type in report.entity_results:
            result = report.entity_results[entity_type]
            current = current_thresholds.get(entity_type, 0.50)

            print(f"\n{entity_type}:")
            print(f"  Current threshold:     {current:.3f}")
            print(f"  Optimal (Max F1):      {result.optimal_threshold:.3f}")
            print(f"  Expected precision:    {result.precision_at_threshold:.1%}")
            print(f"  Expected recall:       {result.recall_at_threshold:.1%}")
            print(f"  Expected F1:           {result.f1_at_threshold:.1%}")
            print(f"  Samples analyzed:      {result.samples_count}")

    print("\nCalibration complete.")


if __name__ == "__main__":
    main()
