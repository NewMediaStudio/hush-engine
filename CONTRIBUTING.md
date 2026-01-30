# Contributing to Hush Engine

Thank you for your interest in contributing to Hush Engine! This document provides guidelines for contributing to this open-source PII detection engine.

## Ways to Contribute

- **Bug Reports**: Found a bug? Open an issue with a clear description and reproduction steps
- **Feature Requests**: Have an idea? Open an issue to discuss it
- **Code Contributions**: Submit pull requests with bug fixes or new features
- **Documentation**: Improve docs, add examples, or fix typos
- **Custom Recognizers**: Share your custom PII detection patterns

## Development Setup

### Prerequisites

- Python 3.10 or higher
- macOS 10.15+ (for Apple Vision OCR)
- Poppler (for PDF processing): `brew install poppler`

### Installation

```bash
# Clone repository
git clone https://github.com/NewMediaStudio/hush-engine.git
cd hush-engine

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e .

# Download spaCy model
python -m spacy download en_core_web_lg

# Run tests
python -m pytest tests/
```

## Code Guidelines

### Python Style

- Follow PEP 8 style guide
- Use type hints where appropriate
- Write docstrings for public functions and classes
- Keep functions focused and single-purpose

### Testing

- Add tests for new features
- Ensure all tests pass before submitting PR
- Aim for high test coverage on core detection logic

### Commits

- Write clear, descriptive commit messages
- Use present tense ("Add feature" not "Added feature")
- Reference issue numbers when applicable

## Adding Custom PII Recognizers

To add a new PII detection pattern:

1. **Create recognizer in `hush_engine/detectors/pii_detector.py`:**

```python
from presidio_analyzer import Pattern, PatternRecognizer

custom_recognizer = PatternRecognizer(
    supported_entity="YOUR_ENTITY_TYPE",
    patterns=[
        Pattern(
            name="your_pattern_name",
            regex=r"your-regex-pattern",
            score=0.8
        )
    ]
)

# Add to PIIDetector.analyzer.registry in __init__
```

2. **Add to detection config in `hush_engine/detection_config.py`:**

```python
CONFIDENCE_THRESHOLDS = {
    # ... existing thresholds
    "YOUR_ENTITY_TYPE": 0.75,
}
```

3. **Add tests in `tests/`:**

```python
def test_your_entity_detection():
    detector = PIIDetector()
    text = "Sample text with your entity"
    results = detector.analyze_text(text)
    assert any(r.entity_type == "YOUR_ENTITY_TYPE" for r in results)
```

4. **Update README with new entity type**

## Pull Request Process

1. **Fork** the repository
2. **Create a branch** from `main`: `git checkout -b feature/your-feature-name`
3. **Make your changes** following the guidelines above
4. **Write or update tests** as needed
5. **Run tests** locally: `pytest tests/`
6. **Commit your changes** with clear messages
7. **Push** to your fork: `git push origin feature/your-feature-name`
8. **Open a Pull Request** against the `main` branch

### PR Checklist

- [ ] Tests pass locally
- [ ] New code has tests
- [ ] Documentation updated (if needed)
- [ ] No unnecessary dependencies added
- [ ] Code follows style guidelines
- [ ] Commit messages are clear

## Recognition Patterns We're Looking For

We especially welcome contributions for:

- **International PII**: Non-US government IDs, tax numbers, etc.
- **Domain-specific patterns**: Healthcare, legal, financial industry identifiers
- **API keys and tokens**: New services and authentication patterns
- **Blockchain identifiers**: Additional crypto wallet formats
- **Performance improvements**: Faster detection or reduced false positives

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Assume good intentions

### Unacceptable Behavior

- Harassment or discriminatory language
- Trolling or personal attacks
- Publishing others' private information
- Any conduct inappropriate in a professional setting

## Questions?

- **Issues**: Open a GitHub issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Security**: Email security@newmediastudio.com for security issues

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for helping make Hush Engine better! 🎉
