# Debugging Detection Decisions

This guide explains how to use Hush Engine's debugging tools to understand why PII detections are made and troubleshoot false positives.

## Overview

Hush Engine uses Presidio's `return_decision_process=True` feature to expose detailed information about each detection decision, including:

- **Which recognizer fired** (SpacyRecognizer, PatternRecognizer, PersonRecognizer, etc.)
- **Original score vs context-boosted score**
- **What context words triggered score boosts**
- **Recognition metadata** (detection sources, engine counts for ensemble models)

## Quick Start

### Using the Debug Tool

The debug tool at `tools/debug_presidio_decisions.py` provides a command-line interface for analyzing detection decisions.

```bash
# Analyze specific text
python3 tools/debug_presidio_decisions.py --text "John Smith works at Apple Inc."

# Compare raw Presidio vs Hush Engine filtering
python3 tools/debug_presidio_decisions.py --compare "Dr. Jane Doe"

# Filter by entity type
python3 tools/debug_presidio_decisions.py --entity PERSON --text "Bob Jones"

# Run all false positive test scenarios
python3 tools/debug_presidio_decisions.py --all-scenarios

# List all registered recognizers
python3 tools/debug_presidio_decisions.py --list-recognizers

# Use Hush Engine (with custom recognizers) instead of raw Presidio
python3 tools/debug_presidio_decisions.py --hush --text "123 Main Street"

# JSON output for programmatic use
python3 tools/debug_presidio_decisions.py --json --text "email@example.com"

# Batch analysis from file
python3 tools/debug_presidio_decisions.py --file samples.txt
```

### Using the PIIDetector API

The `PIIDetector` class includes a debug method for programmatic access:

```python
from hush_engine.detectors.pii_detector import PIIDetector

detector = PIIDetector()

# Get detailed debug information
result = detector.analyze_text_with_debug("Dr. John Smith works at Apple Inc.")

# Access the results
print(f"Raw detections: {result['raw_count']}")
print(f"Final entities: {result['final_count']}")
print(f"Filtered: {result['filtered_count']}")

# See which recognizers fired
for recognizer, detections in result['by_recognizer'].items():
    print(f"\n{recognizer}:")
    for d in detections:
        print(f"  {d['entity_type']}: \"{d['text']}\" (score: {d['final_score']:.2f})")

# See context-boosted detections
for d in result['context_boosted']:
    print(f"Boosted: {d['entity_type']} \"{d['text']}\" (+{d['context_boost']:.2f})")

# List all registered recognizers
print(detector.get_registered_recognizers())

# Find recognizers for a specific entity type
print(detector.get_recognizers_for_entity("PERSON"))
```

## Understanding Detection Output

### Decision Trace Fields

Each detection includes the following information:

| Field | Description |
|-------|-------------|
| `entity_type` | Type of PII detected (PERSON, LOCATION, etc.) |
| `text` | The detected text span |
| `start`, `end` | Character positions in the input |
| `final_score` | Final confidence score after context enhancement |
| `original_score` | Base score from the recognizer |
| `context_boost` | Score improvement from context words |
| `recognizer` | Name of the recognizer that fired |
| `pattern_name` | Name of the pattern (for PatternRecognizer) |
| `supportive_context` | The context word that triggered the boost |
| `textual_explanation` | Human-readable explanation |
| `recognition_metadata` | Additional metadata (e.g., detection sources) |

### Example Output

```
============================================================
Entity Type: PERSON
Text: "Dr. John Smith"
Position: 0:14
Final Score: 0.990

Decision Details:
  Recognizer: PersonRecognizer
  Pattern Name: person_ensemble_pattern+spacy+lgbm
  Original Score: 0.990
  Context Boost: +0.000
  Explanation: Multi-engine ensemble detection

Recognition Metadata:
  recognizer_name: PersonRecognizer
  detection_source: pattern+spacy+lgbm
  engine_count: 3
```

## Troubleshooting False Positives

### Step 1: Identify the Recognizer

Run the debug tool to see which recognizer is causing the false positive:

```bash
python3 tools/debug_presidio_decisions.py --text "Your false positive text here"
```

### Step 2: Compare Raw vs Filtered

Use the compare mode to see if Hush Engine's post-processing is filtering the detection:

```bash
python3 tools/debug_presidio_decisions.py --compare "Your false positive text"
```

This shows:
- **Raw Presidio**: All detections before filtering
- **Hush Engine**: Final filtered results
- **Filtered by Hush**: Detections removed by post-processing
- **Added by Hush**: Detections added by custom recognizers

### Step 3: Check Context Boost

If a false positive has a high score due to context boost, you may need to:

1. Add the text to the denylist in `PIIDetector.__init__`
2. Add a filter rule in `_filter_false_positives()`
3. Update the negative gazetteer at `hush_engine/data/negative_gazetteer.py`

### Step 4: Check Pattern Matches

For PatternRecognizer false positives:

1. Identify the pattern name in the debug output
2. Find the pattern in `_add_*_recognizers()` methods
3. Adjust the regex or add exclusion rules

## Common False Positive Scenarios

The debug tool includes built-in test scenarios for common issues:

```bash
python3 tools/debug_presidio_decisions.py --all-scenarios
```

Scenarios include:

| Category | Example | Issue |
|----------|---------|-------|
| Form Labels | "Name: Address: Phone:" | Labels detected as PII |
| UI Text | "Customer Hub \| Dashboard" | Navigation items detected |
| Business Terms | "customer", "portal" | Generic words detected |
| Company Names | "Apple headquarters" | Company vs Person confusion |
| Addresses | Various international formats | Missing or wrong detection |

## Fixing False Positives

### 1. Add to Denylist

For common words that should never be detected:

```python
# In PIIDetector.__init__
self.denylist = {
    "email", "phone", "name", "address",
    # Add your terms here
    "customer", "portal", "dashboard",
}
```

### 2. Add Filter Rule

For pattern-based filtering in `_filter_false_positives()`:

```python
# Skip form labels detected as PERSON
if entity.entity_type == "PERSON":
    if entity.text.lower() in {"name", "first name", "last name"}:
        continue
```

### 3. Add to Negative Gazetteer

For larger-scale exclusions:

```python
# In hush_engine/data/negative_gazetteer.py
NEGATIVE_PATTERNS = {
    "PERSON": [
        "customer", "patient", "client",
        # Add more terms
    ],
}
```

### 4. Adjust Confidence Thresholds

Thresholds are configured per entity type in `analyze_text()`:

```python
ENTITY_THRESHOLDS = {
    'PERSON': 0.90,
    'COMPANY': 0.75,
    # Adjust as needed
}
```

## Interactive Mode

The debug tool includes an interactive mode for exploratory analysis:

```bash
python3 tools/debug_presidio_decisions.py

# Then at the prompt:
> John Smith at 123 Main St
> compare Dr. Jane Doe
> recognizers
> recognizers PERSON
> quit
```

## JSON Output for Automation

For integration with other tools:

```bash
# Single text
python3 tools/debug_presidio_decisions.py --json --text "test@example.com" > result.json

# All scenarios
python3 tools/debug_presidio_decisions.py --json --all-scenarios > scenarios.json
```

The JSON output includes full decision traces suitable for automated analysis.

## Batch Processing

For analyzing multiple samples:

```bash
# Create a file with one sample per line
echo "John Smith
Dr. Jane Doe
123 Main Street" > samples.txt

# Process all samples
python3 tools/debug_presidio_decisions.py --file samples.txt

# With JSON output
python3 tools/debug_presidio_decisions.py --file samples.txt --json > results.json
```

## Related Documentation

- [PII Reference](PII_REFERENCE.md) - Supported entity types and patterns
- [Precision Improvements](PRECISION_IMPROVEMENTS.md) - How filtering works
- [CLAUDE.md](../CLAUDE.md) - Development instructions
