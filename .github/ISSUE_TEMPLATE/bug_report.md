---
name: Bug report
about: Report a problem with PII detection or engine behavior
title: "[Bug] "
labels: bug
---

## Description
A clear and concise description of the bug.

## Reproduction

Minimal code sample that reproduces the issue:

```python
from hush_engine import PIIDetector
detector = PIIDetector()
result = detector.analyze_text("...")
# Expected: ...
# Actual: ...
```

Or, if the issue is with image/PDF processing, describe the file type and content (do **not** attach files containing real PII — use synthetic/redacted samples).

## Expected behavior
What should happen.

## Actual behavior
What actually happens. Include the full output including confidence scores.

## Environment
- Hush Engine version: (`python -c "import hush_engine; print(hush_engine.__version__)"`)
- Python version: (`python --version`)
- OS: (macOS 14.x, etc.)
- Optional extras installed: (none / medical / address / accurate / full)

## Additional context
Any other details — screenshots (with PII redacted), related issues, etc.
