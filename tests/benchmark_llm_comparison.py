#!/usr/bin/env python3
"""
LLM Comparison Benchmark for Hush Engine

Benchmarks Hush Engine PII detection against LLM models via Ollama,
measuring accuracy (F1/precision/recall), latency, memory, and cost.

Usage:
    python benchmark_llm_comparison.py --samples 500
    python benchmark_llm_comparison.py --samples 100 --models llama3.2:1b,mistral:7b
    python benchmark_llm_comparison.py --resume
    python benchmark_llm_comparison.py --hush-only
    python benchmark_llm_comparison.py --report
    python benchmark_llm_comparison.py --list-models
"""

import argparse
import sys
import time
import random
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Add parent directory to path for hush_engine imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# Import lightgbm early to avoid segfault
try:
    import lightgbm  # noqa: F401
except ImportError:
    pass

# Progress bar support
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs):
        return iterable

# Import reusable components from existing benchmark
from benchmark_accuracy import (
    DatasetLoader,
    extract_ground_truth,
    calculate_metrics,
    normalize,
    digits_only,
    get_detection_text,
    sample_rows,
    detect_pii,
    _get_detector,
    get_engine_version,
)

# Import LLM comparison modules
from llm_comparison.ollama_client import OllamaClient
from llm_comparison.prompt_templates import build_prompt
from llm_comparison.output_parser import parse_llm_pii_output
from llm_comparison.model_registry import (
    MODELS, get_llm_model_ids, get_ollama_model_ids, get_claude_model_ids,
    is_claude_model, is_gemini_model, estimate_cost_per_1k_docs,
)
from llm_comparison.memory_profiler import HushMemoryProfiler, get_ollama_model_memory_mb
from llm_comparison.result_store import ResultStore

# Claude client (optional)
try:
    from llm_comparison.claude_client import ClaudeClient
    HAS_CLAUDE = True
except ImportError:
    HAS_CLAUDE = False

# Gemini client (optional)
try:
    from llm_comparison.gemini_client import GeminiClient
    HAS_GEMINI = True
except (ImportError, Exception):
    HAS_GEMINI = False

# Default paths
TESTS_DIR = Path(__file__).parent
DATA_DIR = TESTS_DIR / "data"
TRAINING_DIR = DATA_DIR / "training"
RESULTS_DIR = TESTS_DIR / "benchmark_history"
DEFAULT_RESULTS_PATH = RESULTS_DIR / "llm_comparison_results.json"

# Dataset search locations (same as benchmark_accuracy.py)
DATASET_SEARCH_DIRS = [TRAINING_DIR, DATA_DIR]


def find_datasets() -> dict:
    """Find available datasets on disk (searches same dirs as benchmark_accuracy.py)."""
    known_names = [
        "sample_3000.json",
        "synthetic_golden.json",
        "golden_test_set.json",
        "pii_dataset_2.parquet",
    ]
    available = {}
    for name in known_names:
        for search_dir in DATASET_SEARCH_DIRS:
            path = search_dir / name
            if path.exists():
                available[name] = path
                break
    # Also scan training dir for any .json/.parquet files
    if TRAINING_DIR.exists():
        for p in TRAINING_DIR.iterdir():
            if p.suffix in (".json", ".parquet", ".arrow", ".jsonl") and p.name not in available:
                available[p.name] = p
    return available


def run_hush_engine(rows: list, dataset_name: str, store: ResultStore):
    """Run Hush Engine baseline on all samples."""
    model_id = "hush_engine"
    completed = store.get_completed_indices(model_id, dataset_name)
    remaining = [(i, row) for i, row in enumerate(rows) if i not in completed]

    if not remaining:
        print(f"  Hush Engine: already completed on {dataset_name}")
        return

    print(f"\n  Running Hush Engine on {len(remaining)} samples...")

    # Memory profiling
    mem_profiler = HushMemoryProfiler()
    mem_profiler.start()

    # Warmup
    _get_detector()

    batch_count = 0
    for idx, row in tqdm(remaining, desc="Hush Engine", disable=not TQDM_AVAILABLE):
        text = row.get("text", "")
        if not text:
            store.save_sample_result(model_id, dataset_name, idx, {}, 0.0)
            batch_count += 1
            continue

        start = time.perf_counter()
        detections = detect_pii(text)
        latency_ms = (time.perf_counter() - start) * 1000

        store.save_sample_result(model_id, dataset_name, idx, detections, latency_ms)
        batch_count += 1

        if batch_count % 50 == 0:
            store.save_batch()

    peak_mb = mem_profiler.get_peak_mb()
    mem_profiler.stop()

    # Calculate final metrics
    all_dets = store.get_all_detections(model_id, dataset_name)
    ground_truth = extract_ground_truth(rows)
    metrics = calculate_metrics(all_dets, ground_truth)

    ds_data = store.data["models"][model_id][dataset_name]
    latencies = ds_data.get("latencies_ms", [])

    summary = {
        "metrics": metrics,
        "memory_peak_mb": peak_mb,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
        "median_latency_ms": sorted(latencies)[len(latencies) // 2] if latencies else 0,
        "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
        "samples_per_sec": 1000 / (sum(latencies) / len(latencies)) if latencies else 0,
        "total_samples": len(rows),
        "engine_version": get_engine_version(),
    }

    # Compute overall metrics
    total_tp = sum(m.get("tp", 0) for m in metrics.values())
    total_fp = sum(m.get("fp", 0) for m in metrics.values())
    total_gt = sum(m.get("total", 0) for m in metrics.values())
    overall_recall = total_tp / total_gt if total_gt > 0 else 0
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    summary["overall_recall"] = overall_recall
    summary["overall_precision"] = overall_precision
    summary["overall_f1"] = overall_f1

    store.save_model_summary(model_id, dataset_name, summary)
    _print_model_result("Hush Engine", summary)


def run_llm_model(client: OllamaClient, model_id: str, rows: list,
                   dataset_name: str, store: ResultStore, few_shot: bool = False):
    """Run a single LLM model on all samples."""
    model_info = MODELS.get(model_id, {})
    display_name = model_info.get("display_name", model_id)
    ollama_tag = model_info.get("ollama_tag", model_id)

    completed = store.get_completed_indices(model_id, dataset_name)
    remaining = [(i, row) for i, row in enumerate(rows) if i not in completed]

    if not remaining:
        print(f"  {display_name}: already completed on {dataset_name}")
        return

    # Check model is available
    installed = [m.get("name", "").split(":")[0] for m in client.list_models()]
    tag_base = ollama_tag.split(":")[0] if ollama_tag else ""
    if tag_base and tag_base not in installed and ollama_tag not in [m.get("name", "") for m in client.list_models()]:
        print(f"  {display_name}: not installed in Ollama. Run: ollama pull {ollama_tag}")
        return

    print(f"\n  Running {display_name} on {len(remaining)} samples...")

    # Warmup
    try:
        client.generate(ollama_tag, "Say hello.", timeout=60)
    except Exception as e:
        print(f"  {display_name}: warmup failed: {e}")
        return

    batch_count = 0
    for idx, row in tqdm(remaining, desc=display_name, disable=not TQDM_AVAILABLE):
        text = row.get("text", "")
        if not text:
            store.save_sample_result(model_id, dataset_name, idx, {}, 0.0)
            batch_count += 1
            continue

        prompt = build_prompt(text, few_shot=few_shot)

        try:
            start = time.perf_counter()
            response = client.generate(ollama_tag, prompt, timeout=300)
            latency_ms = (time.perf_counter() - start) * 1000

            raw_output = response.get("response", "")
            detections = parse_llm_pii_output(raw_output)
            parse_failed = len(detections) == 0 and len(text.strip()) > 20

            input_tokens = response.get("prompt_eval_count", 0)
            output_tokens = response.get("eval_count", 0)

        except Exception as e:
            latency_ms = 0
            detections = {}
            parse_failed = True
            input_tokens = 0
            output_tokens = 0
            if batch_count < 3:
                print(f"\n    Error on sample {idx}: {e}")

        store.save_sample_result(
            model_id, dataset_name, idx, detections, latency_ms,
            input_tokens=input_tokens, output_tokens=output_tokens,
            parse_failed=parse_failed,
        )
        batch_count += 1

        if batch_count % 50 == 0:
            store.save_batch()

    store.save_batch()

    # Memory measurement
    mem_mb = get_ollama_model_memory_mb(ollama_tag)

    # Calculate final metrics
    all_dets = store.get_all_detections(model_id, dataset_name)
    ground_truth = extract_ground_truth(rows)
    metrics = calculate_metrics(all_dets, ground_truth)

    ds_data = store.data["models"][model_id][dataset_name]
    latencies = [l for l in ds_data.get("latencies_ms", []) if l > 0]
    input_tokens_list = ds_data.get("input_tokens", [])
    output_tokens_list = ds_data.get("output_tokens", [])
    parse_failures = ds_data.get("parse_failures", 0)

    avg_input_tokens = sum(input_tokens_list) / len(input_tokens_list) if input_tokens_list else 0
    avg_output_tokens = sum(output_tokens_list) / len(output_tokens_list) if output_tokens_list else 0

    summary = {
        "metrics": metrics,
        "memory_mb": mem_mb or model_info.get("ram_estimate_mb"),
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
        "median_latency_ms": sorted(latencies)[len(latencies) // 2] if latencies else 0,
        "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
        "samples_per_sec": 1000 / (sum(latencies) / len(latencies)) if latencies else 0,
        "total_samples": len(rows),
        "parse_failures": parse_failures,
        "parse_failure_rate": parse_failures / len(rows) if rows else 0,
        "avg_input_tokens": avg_input_tokens,
        "avg_output_tokens": avg_output_tokens,
        "cost_per_1k_docs": estimate_cost_per_1k_docs(model_id, avg_input_tokens, avg_output_tokens),
    }

    # Overall metrics
    total_tp = sum(m.get("tp", 0) for m in metrics.values())
    total_fp = sum(m.get("fp", 0) for m in metrics.values())
    total_gt = sum(m.get("total", 0) for m in metrics.values())
    overall_recall = total_tp / total_gt if total_gt > 0 else 0
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    summary["overall_recall"] = overall_recall
    summary["overall_precision"] = overall_precision
    summary["overall_f1"] = overall_f1

    store.save_model_summary(model_id, dataset_name, summary)
    _print_model_result(display_name, summary)


def run_api_model(api_client, model_id: str, rows: list,
                  dataset_name: str, store: ResultStore, few_shot: bool = False):
    """Run an API-based model (Claude, Gemini) on all samples."""
    model_info = MODELS.get(model_id, {})
    display_name = model_info.get("display_name", model_id)
    api_model_id = (model_info.get("claude_model_id")
                    or model_info.get("gemini_model_id")
                    or model_id)

    completed = store.get_completed_indices(model_id, dataset_name)
    remaining = [(i, row) for i, row in enumerate(rows) if i not in completed]

    if not remaining:
        print(f"  {display_name}: already completed on {dataset_name}")
        return

    print(f"\n  Running {display_name} on {len(remaining)} samples...")

    # Warmup
    try:
        api_client.generate(api_model_id, "Say hello.", timeout=30)
    except Exception as e:
        print(f"  {display_name}: warmup failed: {e}")
        return

    batch_count = 0
    for idx, row in tqdm(remaining, desc=display_name, disable=not TQDM_AVAILABLE):
        text = row.get("text", "")
        if not text:
            store.save_sample_result(model_id, dataset_name, idx, {}, 0.0)
            batch_count += 1
            continue

        prompt = build_prompt(text, few_shot=few_shot)

        try:
            start = time.perf_counter()
            response = api_client.generate(api_model_id, prompt, timeout=120)
            latency_ms = (time.perf_counter() - start) * 1000

            raw_output = response.get("response", "")
            detections = parse_llm_pii_output(raw_output)
            parse_failed = len(detections) == 0 and len(text.strip()) > 20

            input_tokens = response.get("prompt_eval_count", 0)
            output_tokens = response.get("eval_count", 0)

        except Exception as e:
            latency_ms = 0
            detections = {}
            parse_failed = True
            input_tokens = 0
            output_tokens = 0
            if batch_count < 3:
                print(f"\n    Error on sample {idx}: {e}")

        store.save_sample_result(
            model_id, dataset_name, idx, detections, latency_ms,
            input_tokens=input_tokens, output_tokens=output_tokens,
            parse_failed=parse_failed,
        )
        batch_count += 1

        if batch_count % 50 == 0:
            store.save_batch()

    store.save_batch()

    # Calculate final metrics
    all_dets = store.get_all_detections(model_id, dataset_name)
    ground_truth = extract_ground_truth(rows)
    metrics = calculate_metrics(all_dets, ground_truth)

    ds_data = store.data["models"][model_id][dataset_name]
    latencies = [l for l in ds_data.get("latencies_ms", []) if l > 0]
    input_tokens_list = ds_data.get("input_tokens", [])
    output_tokens_list = ds_data.get("output_tokens", [])
    parse_failures = ds_data.get("parse_failures", 0)

    avg_input_tokens = sum(input_tokens_list) / len(input_tokens_list) if input_tokens_list else 0
    avg_output_tokens = sum(output_tokens_list) / len(output_tokens_list) if output_tokens_list else 0

    summary = {
        "metrics": metrics,
        "memory_mb": None,  # Cloud API, N/A
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
        "median_latency_ms": sorted(latencies)[len(latencies) // 2] if latencies else 0,
        "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
        "samples_per_sec": 1000 / (sum(latencies) / len(latencies)) if latencies else 0,
        "total_samples": len(rows),
        "parse_failures": parse_failures,
        "parse_failure_rate": parse_failures / len(rows) if rows else 0,
        "avg_input_tokens": avg_input_tokens,
        "avg_output_tokens": avg_output_tokens,
        "cost_per_1k_docs": estimate_cost_per_1k_docs(model_id, avg_input_tokens, avg_output_tokens),
    }

    # Overall metrics
    total_tp = sum(m.get("tp", 0) for m in metrics.values())
    total_fp = sum(m.get("fp", 0) for m in metrics.values())
    total_gt = sum(m.get("total", 0) for m in metrics.values())
    overall_recall = total_tp / total_gt if total_gt > 0 else 0
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    summary["overall_recall"] = overall_recall
    summary["overall_precision"] = overall_precision
    summary["overall_f1"] = overall_f1

    store.save_model_summary(model_id, dataset_name, summary)
    _print_model_result(display_name, summary)


def _print_model_result(name: str, summary: dict):
    """Print a single model's results."""
    f1 = summary.get("overall_f1", 0) * 100
    prec = summary.get("overall_precision", 0) * 100
    rec = summary.get("overall_recall", 0) * 100
    lat = summary.get("avg_latency_ms", 0)
    sps = summary.get("samples_per_sec", 0)
    pf = summary.get("parse_failure_rate", 0) * 100

    print(f"\n  {name}:")
    print(f"    F1: {f1:.1f}%  Precision: {prec:.1f}%  Recall: {rec:.1f}%")
    print(f"    Avg latency: {lat:.0f}ms  Throughput: {sps:.1f} samples/sec")
    if "parse_failure_rate" in summary:
        print(f"    Parse failures: {pf:.1f}%")
    if summary.get("cost_per_1k_docs"):
        print(f"    Est. cost: ${summary['cost_per_1k_docs']:.4f}/1K docs")


def print_comparison_table(store: ResultStore):
    """Print a comparison table across all models."""
    results = store.get_results()
    models = results.get("models", {})
    if not models:
        print("No results to display.")
        return

    print("\n" + "=" * 100)
    print("COMPARISON RESULTS")
    print("=" * 100)

    # Find all datasets tested
    all_datasets = set()
    for model_data in models.values():
        all_datasets.update(k for k in model_data.keys() if k != "summary")

    for dataset in sorted(all_datasets):
        print(f"\n--- Dataset: {dataset} ---\n")
        header = f"{'Model':<25} {'F1':>6} {'Prec':>6} {'Rec':>6} {'ms/doc':>8} {'samp/s':>8} {'RAM(MB)':>9} {'Parse%':>7} {'$/1K':>8}"
        print(header)
        print("-" * len(header))

        # Collect and sort by F1
        rows = []
        for model_id, model_data in models.items():
            ds = model_data.get(dataset, {})
            summary = ds.get("summary", {})
            if not summary:
                continue
            display = MODELS.get(model_id, {}).get("display_name", model_id)
            rows.append((summary.get("overall_f1", 0), display, summary, model_id))

        rows.sort(reverse=True)

        for f1_val, display, summary, model_id in rows:
            f1 = summary.get("overall_f1", 0) * 100
            prec = summary.get("overall_precision", 0) * 100
            rec = summary.get("overall_recall", 0) * 100
            lat = summary.get("avg_latency_ms", 0)
            sps = summary.get("samples_per_sec", 0)
            mem = summary.get("memory_peak_mb") or summary.get("memory_mb") or 0
            pf = summary.get("parse_failure_rate", 0) * 100
            cost = summary.get("cost_per_1k_docs", 0)

            mem_str = f"{mem:.0f}" if mem else "N/A"
            pf_str = f"{pf:.1f}%" if "parse_failure_rate" in summary else "-"
            cost_str = f"${cost:.4f}" if cost else "$0"

            print(f"{display:<25} {f1:>5.1f}% {prec:>5.1f}% {rec:>5.1f}% {lat:>7.0f} {sps:>7.1f} {mem_str:>9} {pf_str:>7} {cost_str:>8}")

    # Per-entity type breakdown for best dataset
    if all_datasets:
        primary_dataset = "sample_3000.json" if "sample_3000.json" in all_datasets else sorted(all_datasets)[0]
        print(f"\n--- Per-Entity F1 Breakdown ({primary_dataset}) ---\n")
        _print_entity_breakdown(models, primary_dataset)


def _print_entity_breakdown(models: dict, dataset: str):
    """Print per-entity type F1 scores across models."""
    # Collect all entity types
    all_types = set()
    model_metrics = {}
    for model_id, model_data in models.items():
        ds = model_data.get(dataset, {})
        summary = ds.get("summary", {})
        metrics = summary.get("metrics", {})
        model_metrics[model_id] = metrics
        all_types.update(metrics.keys())

    if not all_types:
        return

    # Sort models by overall F1
    model_order = sorted(
        model_metrics.keys(),
        key=lambda m: models[m].get(dataset, {}).get("summary", {}).get("overall_f1", 0),
        reverse=True,
    )

    # Header
    model_names = [MODELS.get(m, {}).get("display_name", m)[:12] for m in model_order]
    header = f"{'Entity':<16} " + " ".join(f"{n:>12}" for n in model_names)
    print(header)
    print("-" * len(header))

    # Rows (sorted by entity type name)
    for entity_type in sorted(all_types):
        row_parts = [f"{entity_type:<16}"]
        for model_id in model_order:
            m = model_metrics.get(model_id, {}).get(entity_type, {})
            f1 = m.get("f1", 0) * 100
            if m.get("total", 0) > 0:
                row_parts.append(f"{f1:>11.1f}%")
            else:
                row_parts.append(f"{'N/A':>12}")
        print(" ".join(row_parts))


def list_available_models(client: OllamaClient):
    """List models from registry and their Ollama install status."""
    print("\nModel Registry:")
    print(f"{'Model':<25} {'Params':>8} {'Disk (MB)':>10} {'Ollama Tag':<25} {'Installed':>9}")
    print("-" * 85)

    installed = set()
    if client.is_available():
        for m in client.list_models():
            installed.add(m.get("name", ""))
            installed.add(m.get("name", "").split(":")[0])

    for model_id, info in MODELS.items():
        tag = info.get("ollama_tag") or "-"
        tag_base = tag.split(":")[0] if tag != "-" else ""
        is_installed = "Yes" if (tag in installed or tag_base in installed) else "No"
        if model_id == "hush_engine":
            is_installed = "Built-in"
        params = f"{info['params_b']}B" if info['params_b'] else "-"
        print(f"{info['display_name']:<25} {params:>8} {info.get('disk_size_mb', 0):>10} {tag:<25} {is_installed:>9}")


def run_comparison(args):
    """Main benchmark runner."""
    # Initialize
    results_path = Path(args.output) if args.output else DEFAULT_RESULTS_PATH
    store = ResultStore(str(results_path))

    # Find datasets
    available_ds = find_datasets()
    if not available_ds:
        print(f"No datasets found in {DATA_DIR}")
        sys.exit(1)

    if args.datasets:
        selected = {}
        for name in args.datasets.split(","):
            name = name.strip()
            if name in available_ds:
                selected[name] = available_ds[name]
            else:
                print(f"Dataset not found: {name}. Available: {', '.join(available_ds.keys())}")
        if not selected:
            sys.exit(1)
    else:
        # Default to sample_3000.json if available, else all
        if "sample_3000.json" in available_ds:
            selected = {"sample_3000.json": available_ds["sample_3000.json"]}
        else:
            selected = available_ds

    # Determine models to run
    if args.hush_only:
        model_ids = []
    elif args.models:
        model_ids = [m.strip() for m in args.models.split(",")]
    else:
        model_ids = get_llm_model_ids()

    # Save config
    store.set_config({
        "samples": args.samples,
        "datasets": list(selected.keys()),
        "models": ["hush_engine"] + model_ids,
        "few_shot": args.few_shot,
        "timestamp": datetime.now().isoformat(),
    })

    # Split models by provider
    ollama_models = [m for m in model_ids if not is_claude_model(m) and not is_gemini_model(m)]
    claude_models = [m for m in model_ids if is_claude_model(m)]
    gemini_models = [m for m in model_ids if is_gemini_model(m)]

    # Initialize Ollama client
    client = OllamaClient()
    if ollama_models and not client.is_available():
        print("Ollama is not running. Skipping Ollama models.")
        ollama_models = []

    # Initialize Claude client
    claude_client = None
    if claude_models:
        if HAS_CLAUDE:
            try:
                claude_client = ClaudeClient()
                print("  Claude API: connected")
            except Exception as e:
                print(f"  Claude API: failed to initialize ({e}). Skipping Claude models.")
                claude_models = []
        else:
            print("  anthropic SDK not installed. pip install anthropic")
            claude_models = []

    # Initialize Gemini client
    gemini_client = None
    if gemini_models:
        if HAS_GEMINI:
            try:
                gemini_client = GeminiClient()
                print("  Gemini API: connected")
            except Exception as e:
                print(f"  Gemini API: failed to initialize ({e}). Skipping Gemini models.")
                gemini_models = []
        else:
            print("  google-generativeai SDK not installed. pip install google-generativeai")
            gemini_models = []

    total_models = len(ollama_models) + len(claude_models) + len(gemini_models)
    print(f"\nBenchmark Configuration:")
    print(f"  Datasets: {', '.join(selected.keys())}")
    print(f"  Samples per dataset: {args.samples}")
    providers = []
    if ollama_models: providers.append(f"{len(ollama_models)} Ollama")
    if claude_models: providers.append(f"{len(claude_models)} Claude")
    if gemini_models: providers.append(f"{len(gemini_models)} Gemini")
    print(f"  Models: Hush Engine + {total_models} LLMs ({', '.join(providers)})")
    print(f"  Prompt: {'few-shot' if args.few_shot else 'zero-shot'}")
    print(f"  Results: {results_path}")

    # Run benchmarks per dataset
    for ds_name, ds_path in selected.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        # Load and sample data
        print(f"  Loading {ds_name}...")
        all_rows = DatasetLoader.load(str(ds_path))
        rows = sample_rows(all_rows, args.samples)
        print(f"  Sampled {len(rows)} rows from {len(all_rows)} total")

        gt = extract_ground_truth(rows)
        gt_count = sum(len(v) for v in gt.values())
        print(f"  Ground truth: {gt_count} entities across {len(gt)} types")

        # Run Hush Engine
        run_hush_engine(rows, ds_name, store)

        # Run Ollama models
        for model_id in ollama_models:
            run_llm_model(client, model_id, rows, ds_name, store, few_shot=args.few_shot)

        # Run Claude models
        for model_id in claude_models:
            run_api_model(claude_client, model_id, rows, ds_name, store, few_shot=args.few_shot)

        # Run Gemini models
        for model_id in gemini_models:
            run_api_model(gemini_client, model_id, rows, ds_name, store, few_shot=args.few_shot)

    # Print comparison
    print_comparison_table(store)
    print(f"\nResults saved to: {results_path}")
    print(f"Generate report: python benchmark_llm_report.py --input {results_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Hush Engine vs LLM models for PII detection"
    )
    parser.add_argument("--samples", type=int, default=500,
                        help="Number of samples per dataset (default: 500)")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated dataset filenames (default: sample_3000.json)")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated Ollama model tags (default: all registered)")
    parser.add_argument("--hush-only", action="store_true",
                        help="Only run Hush Engine baseline")
    parser.add_argument("--few-shot", action="store_true",
                        help="Use few-shot prompt instead of zero-shot")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing results file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output results JSON path")
    parser.add_argument("--report", action="store_true",
                        help="Generate report from existing results (use with --input)")
    parser.add_argument("--input", type=str, default=None,
                        help="Input results JSON for report generation")
    parser.add_argument("--list-models", action="store_true",
                        help="List registered models and Ollama install status")

    args = parser.parse_args()

    if args.list_models:
        client = OllamaClient()
        list_available_models(client)
        return

    if args.report:
        input_path = args.input or str(DEFAULT_RESULTS_PATH)
        print(f"Generating report from {input_path}...")
        try:
            from benchmark_llm_report import generate_report
            generate_report(input_path)
        except ImportError:
            print("benchmark_llm_report.py not found. Run from tests/ directory.")
        return

    if not args.resume:
        # Start fresh (unless resuming)
        results_path = Path(args.output) if args.output else DEFAULT_RESULTS_PATH
        if results_path.exists() and not args.resume:
            # Archive old results
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive = results_path.with_name(f"llm_comparison_{ts}.json")
            results_path.rename(archive)
            print(f"Archived previous results to {archive.name}")

    run_comparison(args)


if __name__ == "__main__":
    main()
