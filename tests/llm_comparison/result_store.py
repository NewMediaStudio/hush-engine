"""Incremental result persistence with resume support."""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path


class ResultStore:
    """Persists benchmark results to JSON, supports resuming interrupted runs."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load_or_create()

    def _load_or_create(self) -> dict:
        if self.path.exists():
            with open(self.path, "r") as f:
                return json.load(f)
        return {
            "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "started": datetime.now().isoformat(),
            "config": {},
            "models": {},
        }

    def set_config(self, config: dict):
        self.data["config"] = config
        self._save()

    def get_completed_indices(self, model_id: str, dataset: str) -> set:
        """Return set of sample indices already completed for this model+dataset."""
        model_data = self.data.get("models", {}).get(model_id, {}).get(dataset, {})
        return set(model_data.get("completed_indices", []))

    def save_sample_result(self, model_id: str, dataset: str, index: int,
                           detections: dict, latency_ms: float,
                           input_tokens: int = 0, output_tokens: int = 0,
                           parse_failed: bool = False):
        """Save a single sample result incrementally."""
        models = self.data.setdefault("models", {})
        model = models.setdefault(model_id, {})
        ds = model.setdefault(dataset, {
            "completed_indices": [],
            "per_sample_detections": {},
            "latencies_ms": [],
            "input_tokens": [],
            "output_tokens": [],
            "parse_failures": 0,
        })
        ds["completed_indices"].append(index)
        ds["per_sample_detections"][str(index)] = detections
        ds["latencies_ms"].append(latency_ms)
        ds["input_tokens"].append(input_tokens)
        ds["output_tokens"].append(output_tokens)
        if parse_failed:
            ds["parse_failures"] = ds.get("parse_failures", 0) + 1

    def save_batch(self):
        """Flush current state to disk."""
        self._save()

    def save_model_summary(self, model_id: str, dataset: str, summary: dict):
        """Save final computed metrics and summary for a model+dataset."""
        models = self.data.setdefault("models", {})
        model = models.setdefault(model_id, {})
        ds = model.setdefault(dataset, {})
        ds["summary"] = summary
        self._save()

    def get_all_detections(self, model_id: str, dataset: str) -> dict:
        """Aggregate all per-sample detections into a single dict for metrics calculation."""
        ds = self.data.get("models", {}).get(model_id, {}).get(dataset, {})
        all_dets = {}
        for _idx, dets in ds.get("per_sample_detections", {}).items():
            for entity_type, entities in dets.items():
                all_dets.setdefault(entity_type, []).extend(entities)
        return all_dets

    def get_model_data(self, model_id: str) -> dict:
        return self.data.get("models", {}).get(model_id, {})

    def get_all_model_ids(self) -> list:
        return list(self.data.get("models", {}).keys())

    def get_results(self) -> dict:
        return self.data

    def _save(self):
        """Atomic write: write to temp file then rename."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.path.parent), suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.data, f, indent=2, default=str)
            os.replace(tmp_path, self.path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
