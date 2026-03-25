#!/usr/bin/env python3
"""
Research Paper Report Generator for LLM Comparison Benchmark

Generates LaTeX tables and matplotlib figures from benchmark results.

Usage:
    python benchmark_llm_report.py
    python benchmark_llm_report.py --input path/to/results.json
    python benchmark_llm_report.py --output-dir paper_figures/
    python benchmark_llm_report.py --format png  # or pdf, svg
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent))
from llm_comparison.model_registry import MODELS

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

DEFAULT_RESULTS = Path(__file__).parent / "benchmark_history" / "llm_comparison_results.json"
HUSH_COLOR = "#F5A623"  # Gold/amber for Hush Engine
LLM_COLORS = ["#4A90D9", "#7B68EE", "#E74C3C", "#2ECC71", "#9B59B6",
              "#E67E22", "#1ABC9C", "#34495E", "#D35400", "#8E44AD"]


def load_results(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def get_model_summaries(results: dict, dataset: str) -> list:
    """Extract sorted model summaries for a dataset."""
    summaries = []
    for model_id, model_data in results.get("models", {}).items():
        ds = model_data.get(dataset, {})
        summary = ds.get("summary", {})
        if not summary:
            continue
        info = MODELS.get(model_id, {})
        summaries.append({
            "model_id": model_id,
            "display_name": info.get("display_name", model_id),
            "params_b": info.get("params_b", 0),
            "disk_size_mb": info.get("disk_size_mb", 0),
            "is_hush": model_id == "hush_engine",
            **summary,
        })
    summaries.sort(key=lambda x: x.get("overall_f1", 0), reverse=True)
    return summaries


def find_primary_dataset(results: dict) -> str | None:
    """Find the primary dataset in results."""
    models = results.get("models", {})
    datasets = set()
    for model_data in models.values():
        datasets.update(k for k in model_data.keys())
    if "sample_3000.json" in datasets:
        return "sample_3000.json"
    return sorted(datasets)[0] if datasets else None


# ============================================================================
# LaTeX TABLE GENERATORS
# ============================================================================

def generate_overall_table(summaries: list, output_path: Path):
    """Generate Table 1: Overall comparison LaTeX table."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{PII Detection: Hush Engine vs. LLM Models}",
        r"\label{tab:overall}",
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        r"Model & Params & Size (MB) & F1 (\%) & Prec (\%) & Rec (\%) & ms/doc & Samp/s & RAM (MB) & \$/1K docs \\",
        r"\midrule",
    ]

    for s in summaries:
        params = f"{s['params_b']:.0f}B" if s["params_b"] else "--"
        disk = f"{s['disk_size_mb']:,}" if s["disk_size_mb"] else "15"
        f1 = f"{s.get('overall_f1', 0) * 100:.1f}"
        prec = f"{s.get('overall_precision', 0) * 100:.1f}"
        rec = f"{s.get('overall_recall', 0) * 100:.1f}"
        lat = f"{s.get('avg_latency_ms', 0):.0f}"
        sps = f"{s.get('samples_per_sec', 0):.1f}"
        mem = s.get("memory_peak_mb") or s.get("memory_mb") or 0
        mem_str = f"{mem:,.0f}" if mem else "N/A"
        cost = s.get("cost_per_1k_docs", 0)
        cost_str = f"\\${cost:.4f}" if cost else "\\$0"

        name = s["display_name"].replace("_", r"\_")
        if s["is_hush"]:
            lines.append(f"\\textbf{{{name}}} & {params} & {disk} & \\textbf{{{f1}}} & {prec} & {rec} & \\textbf{{{lat}}} & {sps} & {mem_str} & {cost_str} \\\\")
        else:
            lines.append(f"{name} & {params} & {disk} & {f1} & {prec} & {rec} & {lat} & {sps} & {mem_str} & {cost_str} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    output_path.write_text("\n".join(lines))
    print(f"  Table 1 (overall): {output_path}")


def generate_entity_table(summaries: list, output_path: Path):
    """Generate Table 2: Per-entity-type F1 LaTeX table."""
    # Collect all entity types
    all_types = set()
    for s in summaries:
        all_types.update(s.get("metrics", {}).keys())

    # Filter to types with ground truth
    scored_types = sorted(t for t in all_types
                          if any(s.get("metrics", {}).get(t, {}).get("total", 0) > 0 for s in summaries))

    # Limit columns to top 6 models by F1
    top_models = summaries[:6]
    ncols = len(top_models)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Per-Entity F1 Scores (\%) Across Models}",
        r"\label{tab:entity}",
        f"\\begin{{tabular}}{{l{'r' * ncols}}}",
        r"\toprule",
    ]

    # Header
    model_headers = " & ".join(
        f"\\textbf{{{s['display_name'][:12]}}}" if s["is_hush"]
        else s["display_name"][:12]
        for s in top_models
    )
    lines.append(f"Entity Type & {model_headers} \\\\")
    lines.append(r"\midrule")

    for entity_type in scored_types:
        row_parts = [entity_type.replace("_", r"\_")]
        for s in top_models:
            m = s.get("metrics", {}).get(entity_type, {})
            if m.get("total", 0) > 0:
                f1 = m["f1"] * 100
                # Bold if best in row
                row_parts.append(f"{f1:.1f}")
            else:
                row_parts.append("--")
        lines.append(" & ".join(row_parts) + " \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    output_path.write_text("\n".join(lines))
    print(f"  Table 2 (per-entity): {output_path}")


# ============================================================================
# MATPLOTLIB FIGURE GENERATORS
# ============================================================================

def _setup_style():
    """Set consistent matplotlib style for paper figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    })


def figure_f1_vs_size(summaries: list, output_path: Path, fmt: str = "pdf"):
    """Figure 1: F1 vs Model Size scatter plot."""
    if not HAS_MATPLOTLIB:
        print("  Skipping figures (matplotlib not installed)")
        return

    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, s in enumerate(summaries):
        f1 = s.get("overall_f1", 0) * 100
        size = s.get("disk_size_mb", 15) or 15

        if s["is_hush"]:
            ax.scatter(size, f1, c=HUSH_COLOR, s=200, zorder=5, marker="*",
                       edgecolors="black", linewidth=0.5, label=s["display_name"])
        else:
            color = LLM_COLORS[i % len(LLM_COLORS)]
            ax.scatter(size, f1, c=color, s=80, zorder=3,
                       edgecolors="white", linewidth=0.5, label=s["display_name"])

    ax.set_xscale("log")
    ax.set_xlabel("Model Size (MB)")
    ax.set_ylabel("F1 Score (%)")
    ax.set_title("PII Detection F1 vs. Model Size")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=max(0, ax.get_ylim()[0] - 5))

    fig.tight_layout()
    fig.savefig(output_path.with_suffix(f".{fmt}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure 1 (F1 vs size): {output_path.with_suffix(f'.{fmt}')}")


def figure_f1_vs_latency(summaries: list, output_path: Path, fmt: str = "pdf"):
    """Figure 2: F1 vs Latency Pareto frontier."""
    if not HAS_MATPLOTLIB:
        return

    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, s in enumerate(summaries):
        f1 = s.get("overall_f1", 0) * 100
        lat = s.get("avg_latency_ms", 0)
        if lat == 0:
            continue

        if s["is_hush"]:
            ax.scatter(lat, f1, c=HUSH_COLOR, s=200, zorder=5, marker="*",
                       edgecolors="black", linewidth=0.5, label=s["display_name"])
        else:
            color = LLM_COLORS[i % len(LLM_COLORS)]
            ax.scatter(lat, f1, c=color, s=80, zorder=3,
                       edgecolors="white", linewidth=0.5, label=s["display_name"])

    ax.set_xscale("log")
    ax.set_xlabel("Average Latency per Document (ms)")
    ax.set_ylabel("F1 Score (%)")
    ax.set_title("PII Detection: Accuracy vs. Speed")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path.with_suffix(f".{fmt}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure 2 (F1 vs latency): {output_path.with_suffix(f'.{fmt}')}")


def figure_radar_chart(summaries: list, output_path: Path, fmt: str = "pdf"):
    """Figure 3: Per-entity radar chart for top models + Hush."""
    if not HAS_MATPLOTLIB:
        return

    # Select models: Hush + top 3 LLMs
    hush = [s for s in summaries if s["is_hush"]]
    llms = [s for s in summaries if not s["is_hush"]][:3]
    selected = hush + llms
    if len(selected) < 2:
        return

    # Find common entity types with enough data
    all_metrics = {}
    for s in selected:
        for etype, m in s.get("metrics", {}).items():
            if m.get("total", 0) >= 5:
                all_metrics.setdefault(etype, []).append(m["f1"])

    # Keep types present in all selected models
    entity_types = sorted(t for t, scores in all_metrics.items() if len(scores) == len(selected))
    if len(entity_types) < 3:
        return

    _setup_style()
    n = len(entity_types)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # Close polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    colors = [HUSH_COLOR] + LLM_COLORS[:len(llms)]
    for i, s in enumerate(selected):
        values = [s.get("metrics", {}).get(t, {}).get("f1", 0) * 100 for t in entity_types]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=s["display_name"],
                color=colors[i], markersize=4)
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([t.replace("_", "\n") for t in entity_types], size=8)
    ax.set_ylim(0, 105)
    ax.set_title("Per-Entity F1 Comparison", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    fig.tight_layout()
    fig.savefig(output_path.with_suffix(f".{fmt}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure 3 (radar): {output_path.with_suffix(f'.{fmt}')}")


def figure_latency_boxplot(summaries: list, results: dict, dataset: str,
                           output_path: Path, fmt: str = "pdf"):
    """Figure 4: Latency distribution box plots per model."""
    if not HAS_MATPLOTLIB:
        return

    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    model_latencies = []
    model_names = []
    model_colors = []

    for i, s in enumerate(summaries):
        model_id = s["model_id"]
        ds_data = results.get("models", {}).get(model_id, {}).get(dataset, {})
        latencies = [l for l in ds_data.get("latencies_ms", []) if l > 0]
        if not latencies:
            continue
        model_latencies.append(latencies)
        model_names.append(s["display_name"][:15])
        model_colors.append(HUSH_COLOR if s["is_hush"] else LLM_COLORS[i % len(LLM_COLORS)])

    if not model_latencies:
        return

    bp = ax.boxplot(model_latencies, labels=model_names, patch_artist=True,
                    showfliers=False, widths=0.6)

    for patch, color in zip(bp["boxes"], model_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Latency per Document (ms)")
    ax.set_title("Inference Latency Distribution")
    ax.set_yscale("log")
    plt.xticks(rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_path.with_suffix(f".{fmt}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure 4 (latency boxplot): {output_path.with_suffix(f'.{fmt}')}")


def figure_cost_efficiency(summaries: list, output_path: Path, fmt: str = "pdf"):
    """Figure 5: F1 vs estimated cost per 1K documents."""
    if not HAS_MATPLOTLIB:
        return

    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, s in enumerate(summaries):
        f1 = s.get("overall_f1", 0) * 100
        cost = s.get("cost_per_1k_docs", 0)

        if s["is_hush"]:
            ax.scatter(cost, f1, c=HUSH_COLOR, s=200, zorder=5, marker="*",
                       edgecolors="black", linewidth=0.5, label=s["display_name"])
            ax.annotate(s["display_name"], (cost, f1),
                        textcoords="offset points", xytext=(10, 5), fontsize=8)
        else:
            color = LLM_COLORS[i % len(LLM_COLORS)]
            ax.scatter(cost, f1, c=color, s=80, zorder=3,
                       edgecolors="white", linewidth=0.5, label=s["display_name"])

    ax.set_xlabel("Estimated Cost per 1,000 Documents (USD)")
    ax.set_ylabel("F1 Score (%)")
    ax.set_title("PII Detection: Accuracy vs. Cost")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path.with_suffix(f".{fmt}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure 5 (cost efficiency): {output_path.with_suffix(f'.{fmt}')}")


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(input_path: str, output_dir: str = None, fmt: str = "pdf"):
    """Generate all paper artifacts from benchmark results."""
    results = load_results(input_path)
    dataset = find_primary_dataset(results)
    if not dataset:
        print("No datasets found in results.")
        return

    summaries = get_model_summaries(results, dataset)
    if not summaries:
        print("No model summaries found.")
        return

    out_dir = Path(output_dir) if output_dir else Path(input_path).parent / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating research paper artifacts ({dataset})...")
    print(f"Output directory: {out_dir}\n")

    # LaTeX tables
    generate_overall_table(summaries, out_dir / "table_overall.tex")
    generate_entity_table(summaries, out_dir / "table_entity.tex")

    # Matplotlib figures
    if HAS_MATPLOTLIB:
        figure_f1_vs_size(summaries, out_dir / "fig_f1_vs_size", fmt)
        figure_f1_vs_latency(summaries, out_dir / "fig_f1_vs_latency", fmt)
        figure_radar_chart(summaries, out_dir / "fig_radar", fmt)
        figure_latency_boxplot(summaries, results, dataset, out_dir / "fig_latency_boxplot", fmt)
        figure_cost_efficiency(summaries, out_dir / "fig_cost_efficiency", fmt)
    else:
        print("  matplotlib not installed - skipping figures")
        print("  Install with: pip install matplotlib numpy")

    # Summary statistics for paper text
    _write_summary_stats(summaries, dataset, out_dir / "summary_stats.json")

    print(f"\nDone. Include tables with \\input{{table_overall.tex}} in your LaTeX paper.")


def _write_summary_stats(summaries: list, dataset: str, output_path: Path):
    """Write summary statistics JSON for easy reference in paper writing."""
    hush = next((s for s in summaries if s["is_hush"]), None)
    best_llm = next((s for s in summaries if not s["is_hush"]), None)

    stats = {
        "dataset": dataset,
        "num_models": len(summaries),
        "num_llm_models": len([s for s in summaries if not s["is_hush"]]),
    }

    if hush:
        stats["hush"] = {
            "f1": round(hush.get("overall_f1", 0) * 100, 1),
            "precision": round(hush.get("overall_precision", 0) * 100, 1),
            "recall": round(hush.get("overall_recall", 0) * 100, 1),
            "avg_latency_ms": round(hush.get("avg_latency_ms", 0), 1),
            "samples_per_sec": round(hush.get("samples_per_sec", 0), 1),
        }

    if best_llm:
        stats["best_llm"] = {
            "name": best_llm["display_name"],
            "f1": round(best_llm.get("overall_f1", 0) * 100, 1),
            "avg_latency_ms": round(best_llm.get("avg_latency_ms", 0), 1),
            "disk_size_mb": best_llm.get("disk_size_mb", 0),
        }

    if hush and best_llm:
        hush_lat = hush.get("avg_latency_ms", 1)
        llm_lat = best_llm.get("avg_latency_ms", 1)
        stats["speedup_vs_best_llm"] = round(llm_lat / hush_lat, 1) if hush_lat > 0 else 0
        stats["size_ratio_vs_best_llm"] = round(
            (best_llm.get("disk_size_mb", 0) or 1) / 15, 0
        )

    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Summary stats: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate research paper artifacts from benchmark results")
    parser.add_argument("--input", type=str, default=str(DEFAULT_RESULTS),
                        help="Input results JSON path")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for tables and figures")
    parser.add_argument("--format", type=str, default="pdf", choices=["pdf", "png", "svg"],
                        help="Figure output format (default: pdf)")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Results file not found: {args.input}")
        print("Run the benchmark first: python benchmark_llm_comparison.py --samples 100")
        sys.exit(1)

    generate_report(args.input, args.output_dir, args.format)


if __name__ == "__main__":
    main()
