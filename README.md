<p align="center">
  <img src="https://raw.githubusercontent.com/NewMediaStudio/hush-engine/main/assets/hush-engine-logo.png" alt="Hush Engine" width="400" />
</p>

# Hush Engine

[![PyPI version](https://img.shields.io/pypi/v/hush-engine.svg)](https://pypi.org/project/hush-engine/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/NewMediaStudio/hush-engine/actions/workflows/tests.yml/badge.svg)](https://github.com/NewMediaStudio/hush-engine/actions/workflows/tests.yml)

Local-first PII detection for images, PDFs, and spreadsheets. Uses Microsoft Presidio for text detection and Apple Vision for OCR. Runs on your machine; nothing is uploaded.

> **Prefer a GUI?** [hushbee.app](https://hushbee.app) ships a free macOS app built on this engine. Drop files in, get redacted versions out.

## Features

**Formats**
Images (PNG, JPEG, HEIC), PDFs, Excel and CSV. Apple Vision OCR runs at 400 DPI.

**Detection**
27 PII types out of the box: names, emails, phone numbers, SSN, credit cards, IBAN, API keys, crypto wallets, passports, medical identifiers, and more. The full table is below.

**NER stack**
LightGBM classifiers handle token-level PERSON, LOCATION, ORGANIZATION, DATE_TIME, and ADDRESS at ~10MB total. A 7,500-name curated database across 54 locales provides fallback coverage. Optional heavyweight models (Flair, Transformers/BERT, GLiNER, OpenAI Privacy Filter) slot into the cascade for workloads where recall matters more than throughput.

**International**
116 IBAN countries via `python-stdnum`. 249 phone country codes via `phonenumbers`. 35+ national ID formats. 800+ cities for LOCATION disambiguation.

**Validation**
Luhn for credit cards. Verhoeff for Aadhaar. Mod-11 and Mod-97 for other IDs. Context-aware thresholds boost confidence when headers or labels disambiguate.

**Extras**
Face detection (OpenCV Haar). QR and barcode decoding (Apple Vision). Runtime toggle for each NER backend.

## Install

```bash
pip install hush-engine
python -m spacy download en_core_web_lg
brew install poppler  # for PDFs
```

Optional extras:

```bash
pip install hush-engine[accurate]         # Flair + Transformers + GLiNER (~2GB)
pip install hush-engine[medical]          # Disease + drug NER
pip install hush-engine[address]          # libpostal bindings (99.45% accuracy, requires brew install libpostal)
pip install hush-engine[names]            # names-dataset (GPL-3.0, opt-in)
pip install hush-engine[privacy-filter]   # OpenAI Privacy Filter add-on backend (~3GB, Apache-2.0)
pip install hush-engine[full]             # medical + address + accurate + privacy-filter
```

## Quick start

```python
from hush_engine import FileRouter

router = FileRouter()

# Image
result = router.detect_pii_image("screenshot.png")
for d in result["detections"]:
    print(f"{d['entity_type']}: {d['text']} ({d['confidence']:.2f})")

# PDF
result = router.detect_pii_pdf("document.pdf")
print(f"{result['total_pages']} pages, {len(result['detections'])} detections")
```

Direct use of the detector:

```python
from hush_engine import PIIDetector
detector = PIIDetector()
for e in detector.analyze_text("John Doe's email is john@example.com"):
    print(f"{e.entity_type}: {e.text}")
```

## Entity types

| Category | Type | Notes |
|---|---|---|
| Personal | `PERSON` | Multi-NER cascade with 7,500-name database |
| | `EMAIL_ADDRESS` | Regex with validation |
| | `PHONE_NUMBER` | 249 countries via libphonenumber |
| | `DATE_TIME` | Multiple formats including DD/MM/YYYY and card expiry (MM/YY) |
| | `AGE` | "25 years old", "Age: 45" |
| | `GENDER`, `NRP` | Demographic references |
| Financial | `CREDIT_CARD` | Luhn-validated, reconstructs fragmented OCR blocks |
| | `FINANCIAL` | SWIFT/BIC, IBAN (116 countries), crypto wallets, salaries (`$128k/yr`), labeled balances, masked accounts (`****7823`) |
| | `AWS_ACCESS_KEY`, `STRIPE_KEY` | Pattern-matched API keys |
| Government | `NATIONAL_ID` | SSN, passport, driver's license across 35+ countries |
| Medical | `MEDICAL` | ICD-10, conditions, medications (pattern-based by default) |
| Technical | `CREDENTIAL` | Passwords, tokens, keys (Shannon entropy) |
| | `IP_ADDRESS` | IPv4/IPv6 with version-string disambiguation |
| | `URL` | via `urlextract` |
| | `NETWORK` | MAC, IMEI, UUID, cookies, device IDs |
| Location | `LOCATION` | Addresses, cities, countries (libpostal optional) |
| | `COORDINATES` | Lat/long |
| Visual | `FACE`, `QR_CODE`, `BARCODE` | Apple Vision framework |
| Organization | `COMPANY`, `ORGANIZATION` | S&P 500 + international database |
| Vehicle | `VEHICLE` | VIN, license plates |
| Biometric | `BIOMETRIC` | Fingerprint IDs |
| Generic | `ID` | Employee ID, customer ID, generic identifiers |

See [docs/PII_REFERENCE.md](docs/PII_REFERENCE.md) for regulatory mapping (HIPAA, GDPR, CCPA).

## Architecture

| Component | Role |
|---|---|
| `FileRouter` | Entry point for file-level processing |
| `PIIDetector` | Presidio analyzer with 50+ custom recognizers |
| `PersonRecognizer` | NER cascade: NLTagger → LightGBM → name database → (optional) spaCy/Flair/Transformers/GLiNER |
| `VisionOCR` | Apple Vision wrapper at 400 DPI |
| `PDFProcessor` | PDF-to-image with parallel page processing |
| `TableDetector` | Context-aware detection for spreadsheets and tables |
| `ImageAnonymizer`, `SpreadsheetAnonymizer` | Redaction output |
| `FaceDetector` | OpenCV Haar cascade |
| `AddressVerifier`, `CompanyVerifier`, `CredentialEntropy`, `HeuristicVerifier` | Precision verifiers |
| `DetectionConfig` | Runtime thresholds and toggles |

### PERSON cascade

```
pattern match → LightGBM NER → NLTagger → names database → [spaCy] → [Flair] → [Transformers] → [GLiNER] → [Privacy Filter]
```

Lightweight engines run first. The cascade exits early when a high-confidence match is found. Heavy engines are skipped unless installed and explicitly enabled.

When `openai_privacy_filter_authoritative=True`, Privacy Filter runs before anything else and its verdict replaces the rest of the cascade for PERSON.

## Custom recognizers

```python
from hush_engine import PIIDetector
from presidio_analyzer import Pattern, PatternRecognizer

detector = PIIDetector()
detector.analyzer.registry.add_recognizer(
    PatternRecognizer(
        supported_entity="CUSTOM_ID",
        patterns=[Pattern("custom", r"[A-Z]{3}-\d{6}", 0.8)],
    )
)
```

## Configuration

```python
from hush_engine import DetectionConfig

config = DetectionConfig()
config.set_threshold("PERSON", 0.60)
config.set_enabled_entity("FACE", False)
config.set_enabled_integration("flair", False)
```

Thresholds persist to `~/.hush/detection_config.json`. Integrations: `lgbm_ner`, `spacy`, `flair`, `transformers`, `gliner`, `name_dataset`, `libpostal`, `urlextract`, `phonenumbers`, `openai_privacy_filter`, `openai_privacy_filter_authoritative`.

## Add-on backend: OpenAI Privacy Filter

OpenAI released [Privacy Filter](https://openai.com/index/introducing-openai-privacy-filter/) on 2026-04-22 as an open-weight PII-redaction model: Apache-2.0, 1.5B parameters total with 50M active (mixture-of-experts), 128K context, bidirectional token classifier with constrained Viterbi span decoding. Weights sit on [HuggingFace](https://huggingface.co/openai/privacy-filter); source is at [github.com/openai/privacy-filter](https://github.com/openai/privacy-filter); the full methodology is in the [model card PDF](https://cdn.openai.com/pdf/c66281ed-b638-456a-8ce1-97e9f5264a90/OpenAI-Privacy-Filter-Model-Card.pdf).

Hush 1.11.0 ships an opt-in integration. Install the extra, then enable it through the config:

```bash
pip install hush-engine[privacy-filter]
```

```python
from hush_engine import DetectionConfig
cfg = DetectionConfig()
cfg.set_enabled_integration("openai_privacy_filter", True)
# Optional: let Privacy Filter's PERSON verdict short-circuit the cascade.
cfg.set_enabled_integration("openai_privacy_filter_authoritative", False)
```

Two gating modes:

- **candidate** (default when enabled): Privacy Filter votes in the ensemble alongside LightGBM, spaCy, Flair, Transformers. The cascade's early-exit threshold still applies, so it runs only when lighter engines haven't produced a high-confidence hit.
- **authoritative**: Privacy Filter's PERSON decision replaces the cascade output. Verifiers skip.

Privacy Filter covers 8 span categories: `private_person`, `private_email`, `private_phone`, `private_address`, `private_url`, `private_date`, `account_number`, `secret`. The 6 non-PERSON categories register as a Presidio recognizer that feeds into Hush's standard entity-type pipeline. To load weights from disk instead of HuggingFace Hub, set `HUSH_PRIVACY_FILTER_MODEL=/path/to/dir`.

License compatibility: Privacy Filter ships under Apache-2.0, which the AGPL-3.0 engine can link against. See the [LICENSE](LICENSE) for Hush and [COMMERCIAL-LICENSING.md](COMMERCIAL-LICENSING.md) for proprietary-deployment terms. The add-on does not change either.

## Agent integrations

Hush ships wrappers for the three current agent transports so any Claude Code, Cursor, Zed, Cline, Windsurf, or Continue user — plus any OpenAI Agents or Claude Agent SDK builder — can drop PII detection into their tool chain without writing glue code.

### MCP server (any stdio MCP client)

```bash
pip install hush-engine[mcp]
```

Add to your client config. Example for Claude Code:

```bash
claude mcp add hush -- hush-mcp
```

Or drop this into the `mcpServers` block of your client's settings:

```json
{
  "mcpServers": {
    "hush": { "command": "hush-mcp" }
  }
}
```

Exposes two tools the model can call: `detect_pii(text, entity_types=None)` and `redact_text(text, entity_types=None, mask="[{entity_type}]")`.

### Claude Agent SDK (in-process)

```bash
pip install hush-engine[agent-claude]
```

```python
from claude_agent_sdk import ClaudeAgentOptions, query
from hush_engine.agents.claude_sdk import hush_server

options = ClaudeAgentOptions(
    mcp_servers={"hush": hush_server},
    allowed_tools=["mcp__hush__detect_pii", "mcp__hush__redact_text"],
)
async for msg in query(prompt="Redact PII then summarize: ...", options=options):
    print(msg)
```

### OpenAI Agents SDK

```bash
pip install hush-engine[agent-openai]
```

```python
from agents import Agent, Runner
from hush_engine.agents.openai_agent import detect_pii_tool, redact_text_tool

agent = Agent(
    name="SafeAssistant",
    instructions="Redact PII before answering questions about user data.",
    tools=[detect_pii_tool, redact_text_tool],
)
result = await Runner.run(agent, "Summarize: My name is John Doe, email john@x.com.")
print(result.final_output)
```

### Bundle

```bash
pip install hush-engine[agents]  # mcp + agent-claude + agent-openai
```

All three wrappers share the same `hush_engine.agents._core` helpers so detector behavior stays consistent across transports.

## Release privacy gates

Set the `HUSH_AUDIT=1` environment variable to opt into internal audit logging (dev + calibration use). Release builds should leave it unset, which:

- Attaches a `NullHandler` to `hush.audit`, so `~/.hush/audit.log` never gets created.
- Removes `ingestTrainingFeedback` from the RPC allow-list, so the Swift UI has no path to read `~/.hush/training_feedback.jsonl` on end-user machines.
- Hashes filenames in any audit line that does emit (defense-in-depth), so a 10-char SHA-256 prefix takes the place of the filename.

`~/.hush/config.json` and `~/.hush/detection_config.json` stay unchanged. Those are user settings (locale, thresholds, enabled libraries), not telemetry.

`FileRouter` also sweeps stragglers out of `~/.hush/tmp` on startup and wraps every temp-file caller in `try/finally` unlink, so preview JPEGs don't accumulate between runs.

## Performance

Synthetic golden set (1,000 samples generated with Faker):

| Metric | Score |
|---|---|
| F1 | 97.2% |
| Precision | 98.3% |
| Recall | 96.2% |

Kaggle PII Detection 2024 (1,000 student essays, 1,606 GT entities):

| Metric | Score |
|---|---|
| F1 | 93.2% |
| Precision | 94.4% |
| Recall | 91.9% |

Per-entity on the Kaggle set: PERSON 93.7% F1, EMAIL 98.7%, ID 88.6%, URL 88.8%, PHONE 85.7%. Latency: 289 ms/doc with libpostal enabled.

## Hush vs LLMs

Same Kaggle set, 1,000 samples. The Privacy Filter rows come from the same benchmark harness, run with `[privacy-filter]` installed and `openai_privacy_filter` enabled.

| Model | F1 | Precision | Recall | Latency | RAM |
|---|---|---|---|---|---|
| **Hush Engine v1.11.0** | **93.2%** | **94.4%** | 91.9% | **289ms** | **~15MB** |
| Hush + OpenAI Privacy Filter | 93.0% | 94.2% | 91.9% | 5,017ms | ~3GB |
| OpenAI Privacy Filter (standalone) | 86.9% | 77.2% | **99.4%** | 5,386ms | ~3GB |
| Mistral 7B | 77.8% | 64.6% | 97.9% | 3,486ms | 10.2GB |
| Phi-4 (14B) | 75.3% | 65.0% | 89.5% | 6,046ms | 14.3GB |
| Qwen 2.5 (7B) | 65.7% | 49.8% | 96.5% | 3,105ms | 8.4GB |
| Gemma 2 (9B) | 63.7% | 47.2% | 97.9% | 4,250ms | 9.0GB |
| Llama 3.2 (1B) | 21.2% | 11.9% | 95.3% | 4,208ms | 4.7GB |

Two results stand out.

OpenAI Privacy Filter alone catches almost every PII span (99.4% recall) and flags 23% false positives. In a redaction pipeline, each false positive deletes text the user wants kept. The 17-point precision gap translates into real content loss.

Adding Privacy Filter to Hush in candidate mode does not lift F1 (93.0% vs 93.2% baseline) and costs 17x the runtime. Hush sits at the ceiling its validators produce on this set. A learned model cannot push past it for entities that already pass Luhn, mod-97, or similar arithmetic.

Reproduce:

```bash
# LLM comparison: Hush vs LLMs (includes openai-privacy-filter as a row)
python tests/benchmark_llm_comparison.py --samples 1000 \
  --models mistral:7b,phi4:latest,openai-privacy-filter

# Ablation: baseline vs Hush + Privacy Filter
python tests/benchmark_accuracy.py --samples 1000 \
  --datasets kaggle_golden_1000.json --privacy-filter-ablation --no-pdf
```

## Development

```bash
git clone https://github.com/NewMediaStudio/hush-engine.git
cd hush-engine
pip install -e ".[dev]"
python -m spacy download en_core_web_lg
pytest tests/
```

### Benchmarks

```bash
python tests/benchmark_accuracy.py --samples 100
python tests/benchmark_accuracy.py --samples 1000
python tests/benchmark_server.py  # dashboard at http://localhost:8000
```

Bootstrap 95% confidence intervals:

```bash
python tools/bootstrap_ci.py --dataset tests/data/synthetic_golden.json
```

### Training LightGBM classifiers

```bash
python tools/train_lgbm_ner.py --entity-type PERSON --samples 5000
python tools/train_lgbm_ner.py --all --ai4privacy --augment --samples 10000
python tools/train_lgbm_ner.py --entity-type PERSON --custom-dataset path/to/data.json
```

### Kaggle dataset (optional)

The Kaggle PII Detection 2024 set requires a Kaggle account. After downloading `train.json`:

```bash
python tools/create_kaggle_golden.py  # 1,000-sample golden set for benchmarks
python tools/kaggle_pii_adapter.py --input tests/data/kaggle_train.json
```

## Requirements

- macOS 10.15+ (Apple Vision OCR)
- Python 3.10+

Windows and Linux support is on the roadmap but not yet available.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Report security issues per [SECURITY.md](SECURITY.md) instead of the public tracker.

## Maintainers

Built and maintained by [Valentine Makhouleen](https://new-media.ca) at [New Media Studio](https://wearenewmedia.com/).

## License

Hush Engine is dual-licensed.

**Open source:** [AGPL-3.0](LICENSE). Free to use, modify, and distribute under AGPL terms. If you run Hush over a network (for example, inside a SaaS), AGPL § 13 requires you to open-source the service that uses it.

**Commercial:** a paid commercial license is available for proprietary products, closed-source SaaS, or any use where AGPL obligations don't fit. See [COMMERCIAL-LICENSING.md](COMMERCIAL-LICENSING.md) or email **studio@newmediastudio.com**.

## Related

- **[Hushbee](https://hushbee.app)** — free macOS app built on this engine. Download there for a drag-and-drop GUI over the same detection pipeline.
- **[Microsoft Presidio](https://github.com/microsoft/presidio)** — the detection framework Hush builds on.
- **[OpenAI Privacy Filter](https://openai.com/index/introducing-openai-privacy-filter/)** — add-on backend for contextual PII redaction ([HuggingFace](https://huggingface.co/openai/privacy-filter), [source](https://github.com/openai/privacy-filter), [model card PDF](https://cdn.openai.com/pdf/c66281ed-b638-456a-8ce1-97e9f5264a90/OpenAI-Privacy-Filter-Model-Card.pdf)).

## Acknowledgments

Built on [Presidio](https://github.com/microsoft/presidio), [Apple Vision](https://developer.apple.com/documentation/vision), [spaCy](https://spacy.io/), [Flair](https://github.com/flairNLP/flair), [GLiNER](https://github.com/urchade/GLiNER), [libpostal](https://github.com/openvenues/libpostal), and [python-stdnum](https://github.com/arthurdejong/python-stdnum). Optional add-on: [OpenAI Privacy Filter](https://huggingface.co/openai/privacy-filter).
