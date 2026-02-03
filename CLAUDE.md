# Claude Instructions for Hush Engine

## Important: Repository Exclusions

The `tests/` and `training/` folders should NOT be committed to the repository:
- They contain large datasets and generated files
- Add them to `.gitignore` if not already excluded
- These folders are for local development and benchmarking only

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

## Benchmarking

Run accuracy benchmark after changes:

```bash
python3 tests/benchmark_accuracy.py --samples 100  # Quick test
python3 tests/benchmark_accuracy.py --samples 1000  # Full test
```

Ground truth data is cached in `tests/data/training/Training_Set_cache.csv`.
