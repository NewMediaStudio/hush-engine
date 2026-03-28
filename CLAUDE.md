# Claude Instructions for Hush Engine

## Important: Repository Exclusions

The `training/` folder should NOT be committed to the repository:
- It contains large datasets and generated files
- The `tests/` folder IS committed (benchmark code, dashboard, results)
- Test datasets (`.json`, `.parquet`) in `tests/data/` may need to be downloaded separately

## Training Feedback Integration

The Hush macOS app writes user feedback directly to `training/feedback/` in this repo.

**Feedback Location:** `training/feedback/*.json`
**Analysis Output:** `training/analysis/`
**Tracking:** `training/analysis/processed_feedback.json`

Each JSON file contains:

- `detectedText` - The text that was detected
- `detectedEntityType` - What the engine classified it as
- `suggestedEntityTypes` - User's corrected entity types
- `confidence` - Engine's confidence score
- `engineVersion` - Version that made the detection
- `notes` - User's notes

### Key Tasks

1. Read `training/README.md` for full schema documentation
2. Analyze patterns in feedback to identify common misclassifications
3. Use feedback to improve recognizers (especially PersonRecognizer)
4. Track improvements by filtering feedback by `engineVersion`

### Example Analysis

```python
from pathlib import Path
import json

for f in Path("training/feedback").glob("*.json"):
    entry = json.load(open(f))
    if entry["detectedEntityType"] not in entry["suggestedEntityTypes"]:
        print(f"Misclassified: {entry['detectedText']} as {entry['detectedEntityType']}")
```

## Key Files

| File | Purpose |
|------|---------|
| `hush_engine/detectors/pii_detector.py` | Main PII detection logic |
| `hush_engine/detectors/person_recognizer.py` | Person name detection (multi-NER cascade) |
| `hush_engine/data/names_database.py` | Lightweight names lookup database |
| `hush_engine/detection_config.py` | Detection thresholds and entity config |
| `tools/feedback_analyzer.py` | Feedback analysis tool |
| `tools/bootstrap_ci.py` | Bootstrap 95% confidence intervals for F1/precision/recall |
| `tools/kaggle_pii_adapter.py` | Kaggle PII Detection 2024 dataset → Hush format converter |
| `tools/create_kaggle_golden.py` | Golden 1,000-sample Kaggle test set creator |
| `tools/generate_holdout_set.py` | Held-out test set generator with overlap verification |
| `tests/benchmark_accuracy.py` | Accuracy benchmarking |
| `tests/benchmark_llm_comparison.py` | LLM comparison benchmark (Hush vs Ollama/Claude models) |
| `tests/benchmark_llm_report.py` | Research paper artifact generator (LaTeX tables + figures) |
| `tests/llm_comparison/` | LLM benchmark modules (clients, parsers, model registry) |

## Feedback Analysis Tool

Run the feedback analyzer to get actionable recommendations:

```bash
python3 tools/feedback_analyzer.py
```

This generates:
- Categorized feedback (false positives, missed detections, misclassifications)
- Pattern analysis by entity type
- Claude-actionable JSON at `~/Library/Application Support/Hush/analysis/claude_actions.json`

## Common Improvement Patterns

### False Positives
Add filters to `_filter_false_positives()` in `pii_detector.py`:
- Document header phrases detected as PERSON
- UI/navigation text detected as COMPANY
- Phone numbers detected as LOCATION

### Missed Detections
Add patterns to appropriate `_add_*_recognizers()` methods:
- International address formats (LOCATION)
- Currency with spaces after symbol (FINANCIAL)
- DD/MM/YYYY date format (DATE_TIME)
- Names with middle initials (PERSON)

## Optional Dependencies

The following are **not** shipped in the Hush macOS app bundle and must be installed separately:

| Extra | Install | What it adds |
|-------|---------|-------------|
| `medical` | `pip install hush-engine[medical]` | Fast Data Science disease/drug NER (supplements built-in pattern-based MEDICAL detection) |
| `address` | `pip install hush-engine[address]` | libpostal address parsing (99.45% accuracy). Also requires `brew install libpostal` |
| `accurate` | `pip install hush-engine[accurate]` | Flair, Transformers (BERT), GLiNER heavyweight NER models (~2GB) |
| `full` | `pip install hush-engine[full]` | All of the above |

The app ships with pattern-based MEDICAL detection (blood types, ICD codes, conditions, medications via regex) and heuristic + LightGBM address verification — the optional extras improve recall.

## Benchmarking

Run accuracy benchmark after changes:

```bash
python3 tests/benchmark_accuracy.py --samples 100  # Quick test
python3 tests/benchmark_accuracy.py --samples 1000  # Full test
```

### LLM Comparison Benchmark

Benchmark Hush Engine against LLM models for research paper:

```bash
# List available models
python3 tests/benchmark_llm_comparison.py --list-models

# Run comparison (Hush + selected LLMs via Ollama)
python3 tests/benchmark_llm_comparison.py --samples 1000 --models llama3.2:1b,mistral:7b

# Run with Claude API models
ANTHROPIC_API_KEY=sk-... python3 tests/benchmark_llm_comparison.py --models claude-haiku-4-5,claude-sonnet-4-6

# Resume interrupted run
python3 tests/benchmark_llm_comparison.py --resume

# Generate research paper figures (LaTeX + matplotlib)
python3 tests/benchmark_llm_report.py --format png
```

**Latest results (1,000 samples, ai4privacy):**

| Model | F1 | Precision | Recall | Latency | Parse Failures |
|-------|-----|-----------|--------|---------|---------------|
| Hush Engine v1.8.0 | 89.9% | 89.5% | 90.4% | 166ms | 0% |
| Llama 3.2 (1B) | 49.9% | 36.3% | 80.1% | 21,239ms | 49.6% |

**Latest results (1,000 samples, synthetic golden set):**

| Model | F1 | Precision | Recall | Latency | Parse Failures |
|-------|-----|-----------|--------|---------|---------------|
| Hush Engine v1.8.0 | 91.4% | 84.4% | 99.7% | 93ms | 0% |
| Llama 3.2 (1B) | 86.2% | 76.0% | 99.6% | 2,058ms | 3.5% |

### Benchmark Dashboard

```bash
python3 tests/benchmark_server.py --port 8000
# Open http://localhost:8000
```

The dashboard shows historical benchmark runs with LLM comparison runs marked as red triangles. Use the "Run Test" button to configure and launch benchmarks (supports Hush Engine, LLM Comparison, or both).

Ground truth data is cached in `tests/data/training/Training_Set_cache.csv`.

### Bootstrap Confidence Intervals

Compute 95% CIs for Hush Engine metrics on any dataset:

```bash
python3 tools/bootstrap_ci.py --dataset tests/data/synthetic_golden.json
python3 tools/bootstrap_ci.py --dataset tests/data/kaggle_pii.json --latex
python3 tools/bootstrap_ci.py --dataset tests/data/holdout_test_set.json --save results.json
```

Runs Hush Engine per-sample, resamples 5,000× (configurable via `--iterations`), and reports micro-averaged 95% CIs for F1, precision, and recall. Uses the same `calculate_metrics()` logic as `benchmark_accuracy.py`.

### Kaggle PII Dataset

Convert the Kaggle "PII Detection 2024" competition dataset for benchmarking:

```bash
# Download train.json from Kaggle (requires account + competition rules acceptance)
# Convert to Hush Engine format
python3 tools/kaggle_pii_adapter.py --input tests/data/kaggle_train.json --output tests/data/kaggle_pii.json
python3 tools/kaggle_pii_adapter.py --input tests/data/kaggle_train.json --only-pii --stats  # preview
```

Maps 7 Kaggle BIO-tagged entity types (NAME_STUDENT, EMAIL, PHONE_NUM, URL_PERSONAL, STREET_ADDRESS, ID_NUM, USERNAME) → Hush Engine types.

### Kaggle Golden 1,000

Create a fixed, deterministic 1,000-sample golden set from the Kaggle competition data:

```bash
python3 tools/create_kaggle_golden.py                    # Create (945 PII + 55 non-PII)
python3 tools/create_kaggle_golden.py --stats            # Preview without writing
python3 tools/create_kaggle_golden.py --validate         # Validate existing set
```

Output: `tests/data/kaggle_golden_1000.json` (4.3 MB, 1,606 entities across 7 types). Compatible with `benchmark_accuracy.py`, `benchmark_llm_comparison.py`, and `bootstrap_ci.py`.

### Held-Out Test Set

Generate a deterministic, non-overlapping held-out set for fair evaluation:

```bash
# Slice from existing sample_3000.json
python3 tools/generate_holdout_set.py --slice 1 --samples 500

# Download full ai4privacy (300K samples, completely independent)
python3 tools/generate_holdout_set.py --download --samples 1000

# Verify zero overlap with development data
python3 tools/generate_holdout_set.py --verify tests/data/holdout_test_set.json
```

Requires `pip install datasets` for the HuggingFace download option.
