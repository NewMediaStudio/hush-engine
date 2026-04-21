#!/usr/bin/env python3
"""
Bootstrap Confidence Interval Calculator for Hush Engine.

Runs Hush Engine on each sample individually, scores using the exact same
calculate_metrics() logic as benchmark_accuracy.py, and computes bootstrap
95% CIs for F1, precision, and recall.

Usage:
    python tools/bootstrap_ci.py --dataset tests/data/synthetic_golden.json
    python tools/bootstrap_ci.py --dataset tests/data/kaggle_pii.json
    python tools/bootstrap_ci.py --dataset tests/data/holdout_test_set.json
    python tools/bootstrap_ci.py --dataset tests/data/synthetic_golden.json --samples 200 --latex
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from benchmark_accuracy import calculate_metrics, detect_pii


def load_dataset(path: str) -> list:
    """Load dataset rows with ground truth."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "samples" in data:
        return data["samples"]
    elif isinstance(data, list):
        return data
    raise ValueError(f"Unknown format: {path}")


def score_single_sample(detections: dict, ground_truth: dict, allowed_types: set = None) -> dict:
    """Score a single sample using the same calculate_metrics logic.

    Returns per-type TP (recall), TP (precision), FP, and total GT counts.
    """
    # Filter detections to allowed types if specified
    if allowed_types:
        detections = {k: v for k, v in detections.items() if k in allowed_types}
        ground_truth = {k: v for k, v in ground_truth.items() if k in allowed_types}

    metrics = calculate_metrics(detections, ground_truth)

    tp_recall = 0
    tp_precision = 0
    fp = 0
    total_gt = 0

    for pii_type, m in metrics.items():
        tp_recall += m.get("tp", 0)
        tp_precision += m.get("tp_precision", 0)
        fp += m.get("fp", 0)
        total_gt += m.get("total", 0)

    return {
        "tp_recall": tp_recall,
        "tp_precision": tp_precision,
        "fp": fp,
        "total_gt": total_gt,
        "per_type": metrics,
    }


def bootstrap_metric(samples: list, metric_fn, n_iter: int = 5000, seed: int = 42) -> dict:
    """Bootstrap a metric from per-sample scores."""
    n = len(samples)
    rng = np.random.default_rng(seed=seed)
    boot = []

    for _ in range(n_iter):
        idx = rng.choice(n, size=n, replace=True)
        val = metric_fn([samples[i] for i in idx])
        boot.append(val)

    boot = np.array(boot)
    return {
        "mean": float(np.mean(boot)),
        "ci_lower": float(np.percentile(boot, 2.5)),
        "ci_upper": float(np.percentile(boot, 97.5)),
        "std": float(np.std(boot)),
    }


def agg_f1(samples):
    tp_p = sum(s["tp_precision"] for s in samples)
    fp = sum(s["fp"] for s in samples)
    tp_r = sum(s["tp_recall"] for s in samples)
    total = sum(s["total_gt"] for s in samples)
    p = tp_p / (tp_p + fp) if (tp_p + fp) > 0 else 1.0
    r = tp_r / total if total > 0 else 1.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def agg_precision(samples):
    tp_p = sum(s["tp_precision"] for s in samples)
    fp = sum(s["fp"] for s in samples)
    return tp_p / (tp_p + fp) if (tp_p + fp) > 0 else 1.0

def agg_recall(samples):
    tp_r = sum(s["tp_recall"] for s in samples)
    total = sum(s["total_gt"] for s in samples)
    return tp_r / total if total > 0 else 1.0


def main():
    parser = argparse.ArgumentParser(description="Bootstrap CIs for Hush Engine")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--latex", action="store_true")
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    rows = load_dataset(args.dataset)
    if args.samples:
        rows = rows[:args.samples]

    # Determine allowed entity types from ground truth
    all_gt_types = set()
    for row in rows:
        all_gt_types.update(row.get("ground_truth", {}).keys())

    print(f"Dataset: {args.dataset}")
    print(f"Samples: {len(rows)}")
    print(f"GT entity types: {sorted(all_gt_types)}")
    print(f"Bootstrap iterations: {args.iterations}")
    print()

    # Process each sample
    print("Running Hush Engine on each sample...")
    per_sample = []
    per_entity_agg = defaultdict(lambda: {"tp": 0, "tp_precision": 0, "fp": 0, "total": 0})
    latencies = []

    for i, row in enumerate(rows):
        text = row.get("text", "")
        gt = row.get("ground_truth", {})
        if not text:
            continue

        t0 = time.time()
        detections = detect_pii(text)
        lat = (time.time() - t0) * 1000
        latencies.append(lat)

        result = score_single_sample(detections, gt, allowed_types=all_gt_types)
        per_sample.append(result)

        for pii_type, m in result["per_type"].items():
            per_entity_agg[pii_type]["tp"] += m.get("tp", 0)
            per_entity_agg[pii_type]["tp_precision"] += m.get("tp_precision", 0)
            per_entity_agg[pii_type]["fp"] += m.get("fp", 0)
            per_entity_agg[pii_type]["total"] += m.get("total", 0)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(rows)}...")

    # Overall metrics
    overall_f1 = agg_f1(per_sample)
    overall_prec = agg_precision(per_sample)
    overall_rec = agg_recall(per_sample)

    print(f"\n{'='*70}")
    print(f"Results ({len(per_sample)} samples)")
    print(f"{'='*70}")
    print(f"  Precision: {overall_prec:.1%}")
    print(f"  Recall:    {overall_rec:.1%}")
    print(f"  F1:        {overall_f1:.1%}")
    print(f"  Latency:   {np.median(latencies):.0f} ms (median), {np.mean(latencies):.0f} ms (mean)")

    # Per-entity
    print(f"\n{'='*70}")
    print("Per-Entity")
    print(f"{'='*70}")
    print(f"  {'Type':<20} {'Prec':>7} {'Recall':>7} {'F1':>7} {'TP':>5} {'FP':>5} {'GT':>5}")
    for etype in sorted(per_entity_agg.keys()):
        t = per_entity_agg[etype]
        p = t["tp_precision"] / (t["tp_precision"] + t["fp"]) if (t["tp_precision"] + t["fp"]) > 0 else 0
        r = t["tp"] / t["total"] if t["total"] > 0 else 0
        f = 2*p*r/(p+r) if (p+r) > 0 else 0
        print(f"  {etype:<20} {p:>6.1%} {r:>6.1%} {f:>6.1%} {t['tp']:>5} {t['fp']:>5} {t['total']:>5}")

    # Bootstrap
    print(f"\n{'='*70}")
    print(f"Bootstrap 95% CIs ({args.iterations} iterations)")
    print(f"{'='*70}")

    f1_ci = bootstrap_metric(per_sample, agg_f1, args.iterations)
    p_ci = bootstrap_metric(per_sample, agg_precision, args.iterations)
    r_ci = bootstrap_metric(per_sample, agg_recall, args.iterations)

    print(f"  F1:        {f1_ci['mean']:.1%}  (95% CI: {f1_ci['ci_lower']:.1%} – {f1_ci['ci_upper']:.1%})")
    print(f"  Precision: {p_ci['mean']:.1%}  (95% CI: {p_ci['ci_lower']:.1%} – {p_ci['ci_upper']:.1%})")
    print(f"  Recall:    {r_ci['mean']:.1%}  (95% CI: {r_ci['ci_lower']:.1%} – {r_ci['ci_upper']:.1%})")

    if args.latex:
        print("\n% LaTeX:")
        print(f"Hush Engine & Local & "
              f"{p_ci['mean']:.1%} & {r_ci['mean']:.1%} & "
              f"{f1_ci['mean']:.1%} ({{95\\% CI: {f1_ci['ci_lower']:.1%}--{f1_ci['ci_upper']:.1%}}}) & "
              f"{np.median(latencies):.0f}\\,ms & 0.0\\% & ${{\\sim}}$15\\,MB \\\\")

    if args.save:
        out = {
            "dataset": args.dataset,
            "n_samples": len(per_sample),
            "overall": {
                "f1": overall_f1, "precision": overall_prec, "recall": overall_rec,
                "f1_ci": f1_ci, "precision_ci": p_ci, "recall_ci": r_ci,
            },
            "per_entity": dict(per_entity_agg),
            "per_sample": [{"tp_r": s["tp_recall"], "tp_p": s["tp_precision"],
                           "fp": s["fp"], "gt": s["total_gt"]} for s in per_sample],
            "latencies_ms": latencies,
        }
        with open(args.save, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved to {args.save}")


if __name__ == "__main__":
    main()
