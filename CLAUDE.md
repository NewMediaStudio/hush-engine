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
