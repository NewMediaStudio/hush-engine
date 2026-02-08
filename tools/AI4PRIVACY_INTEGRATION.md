# AI4Privacy Dataset Integration

This document describes how to use the ai4privacy/pii-masking-300k dataset with the Hush Engine's LightGBM NER training pipeline.

## Overview

The [ai4privacy/pii-masking-300k](https://huggingface.co/datasets/ai4privacy/pii-masking-300k) dataset provides 225,000+ annotated samples of PII data across 27+ entity types and 6 languages. This dataset can significantly improve our NER models by providing:

1. **Real-world patterns** - More diverse examples than synthetic Faker data
2. **Multilingual coverage** - English, French, German, Italian, Spanish, Dutch
3. **Fine-grained labels** - Multiple variants for names, addresses, IDs

## Quick Start

### 1. Install Dependencies

```bash
pip install datasets
```

### 2. Convert the Dataset

```bash
# Convert all entity types (recommended for first run)
python tools/convert_ai4privacy.py --all --output tests/data/ai4privacy

# Or convert a specific entity type
python tools/convert_ai4privacy.py --entity-type PERSON --output tests/data/ai4privacy

# Quick test with limited samples
python tools/convert_ai4privacy.py --all --max-samples 1000 --output tests/data/ai4privacy
```

### 3. View Conversion Statistics

The conversion script prints statistics about:
- Records processed and skipped
- Entity counts by Hush Engine type
- Original ai4privacy labels and their mappings
- Language distribution

### 4. Use the Data for Training

After conversion, use `external_data_loader.py` to load the data:

```python
from tools.external_data_loader import load_external_training_data, combine_training_data

# Load converted ai4privacy data
external_samples = load_external_training_data(
    entity_type="PERSON",
    data_dir=Path("tests/data/ai4privacy")
)

# Combine with synthetic data (50/50 split)
all_samples = combine_training_data(
    synthetic_samples=synthetic_samples,
    external_samples=external_samples,
    synthetic_ratio=0.5
)
```

## Entity Type Mapping

The ai4privacy dataset uses fine-grained labels. We consolidate them into Hush Engine's broader categories:

| AI4Privacy Label | Hush Engine Type | Notes |
|-----------------|------------------|-------|
| `GIVENNAME1/2`, `LASTNAME1/2/3`, `TITLE`, `MIDDLENAME`, `PREFIX`, `SUFFIX`, `FULLNAME`, `FIRSTNAME`, `LASTNAME` | `PERSON` | Name components |
| `USERNAME`, `DISPLAYNAME` | `PERSON` | User identifiers |
| `CITY`, `COUNTRY`, `STATE`, `POSTCODE`, `COUNTY`, `ZIPCODE` | `LOCATION` | Geographic entities |
| `STREET`, `BUILDING`, `SECADDRESS`, `BUILDINGNUMBER`, `SECONDARYADDRESS`, `STREETADDRESS` | `ADDRESS` | Street addresses |
| `COMPANYNAME`, `COMPANY` | `ORGANIZATION` | Companies |
| `DATE`, `TIME`, `BOD`, `DOB`, `BIRTHDATE`, `DATE_TIME` | `DATE_TIME` | Temporal entities |
| `IBAN`, `BIC`, `CREDITCARD`, `ACCOUNTNUMBER`, `AMOUNT`, `BITCOINADDRESS`, etc. | `FINANCIAL` | Financial data |
| `EMAIL` | `EMAIL_ADDRESS` | Email addresses |
| `TEL`, `PHONENUMBER`, `MOBILE`, `LANDLINE`, `FAX` | `PHONE_NUMBER` | Phone numbers |
| `IP`, `IPV4`, `IPV6` | `IP_ADDRESS` | IP addresses |
| `SOCIALNUMBER`, `IDCARD`, `PASSPORT`, `DRIVERLICENSE`, `SSN` | `SSN` | Government IDs |
| `GEOCOORD`, `NEARBYGPSCOORDINATE` | `COORDINATES` | GPS coordinates |
| `URL` | `URL` | URLs |

### Excluded Labels

These labels are not mapped (too granular or not relevant for OCR):
- `SEX`, `GENDER`, `AGE` - Demographics
- `PASS`, `PASSWORD`, `PIN` - Credentials (not visible in OCR)
- `CURRENCYCODE`, `CURRENCYSYMBOL` - Currency indicators
- `JOBDESCRIPTOR`, `JOBTITLE`, `JOBTYPE`, `JOBAREA` - Job metadata
- `MAC`, `USERAGENT` - Technical identifiers

## Output Formats

### JSON Format (Default)

Each entity type produces a JSON file:

```json
{
  "entity_type": "PERSON",
  "source": "ai4privacy/pii-masking-300k",
  "sample_count": 45000,
  "samples": [
    {
      "text": "Subject: Meeting Tomorrow\n\nHi John Smith, please confirm attendance...",
      "entities": [[26, 36, "PERSON"]],
      "source_id": "40767A",
      "language": "English"
    }
  ]
}
```

### BIO Format (Optional)

Use `--format bio` to export in CoNLL-style BIO format:

```
token	start	end	tag	sample_id
Subject	0	7	O	40767A
:	7	8	O	40767A
Meeting	9	16	O	40767A
John	26	30	B-PERSON	40767A
Smith	31	36	I-PERSON	40767A
```

## Advanced Usage

### Language Filtering

```bash
# English only
python tools/convert_ai4privacy.py --all --language English --output tests/data/ai4privacy_en

# French only
python tools/convert_ai4privacy.py --all --language French --output tests/data/ai4privacy_fr
```

### Validation Split

```bash
# Use validation data for testing
python tools/convert_ai4privacy.py --all --split validation --output tests/data/ai4privacy_val
```

### Statistics Only

```bash
# Preview without exporting
python tools/convert_ai4privacy.py --all --stats-only --output /dev/null
```

### Both Formats

```bash
# Export both JSON and BIO
python tools/convert_ai4privacy.py --all --format both --output tests/data/ai4privacy
```

## Integration with train_lgbm_ner.py

To use the converted data with the existing training script, modify `train_lgbm_ner.py`:

```python
# Add import
from tools.external_data_loader import load_external_training_data, combine_training_data

# In create_training_data() function, add:
def create_training_data(entity_type, n_positive, n_negative, generator, use_external=False):
    # ... existing synthetic data generation ...

    if use_external:
        external_dir = Path("tests/data/ai4privacy")
        external_samples = load_external_training_data(entity_type, external_dir)

        if external_samples:
            # Combine synthetic and external data
            all_samples = combine_training_data(
                synthetic_samples=positive_samples + negative_samples,
                external_samples=external_samples,
                synthetic_ratio=0.3  # 30% synthetic, 70% ai4privacy
            )
            # Continue with feature extraction...
```

## Expected Entity Counts

Based on the dataset structure, approximate entity counts after conversion:

| Entity Type | Estimated Count | Notes |
|-------------|-----------------|-------|
| PERSON | 80,000+ | Most common type (names, usernames) |
| DATE_TIME | 50,000+ | Dates, times, DOB |
| LOCATION | 40,000+ | Cities, countries, states, postcodes |
| EMAIL_ADDRESS | 30,000+ | Email addresses |
| ADDRESS | 25,000+ | Street addresses |
| PHONE_NUMBER | 20,000+ | Phone numbers |
| FINANCIAL | 15,000+ | IBANs, credit cards, amounts |
| SSN | 10,000+ | Government IDs |
| IP_ADDRESS | 5,000+ | IP addresses |
| ORGANIZATION | 5,000+ | Company names |

## Best Practices

1. **Start with English** - Filter by language for initial testing
2. **Balance synthetic + external** - Use 30-50% synthetic data to maintain diversity
3. **Validate entity spans** - Check that entities map correctly to text positions
4. **Monitor training** - Compare validation F1 with/without external data
5. **Iterate on mapping** - Adjust `ENTITY_TYPE_MAPPING` based on training results

## File Structure

After conversion:

```
tests/data/
└── ai4privacy/
    ├── person_training_data.json
    ├── location_training_data.json
    ├── address_training_data.json
    ├── organization_training_data.json
    ├── date_time_training_data.json
    ├── financial_training_data.json
    ├── email_address_training_data.json
    ├── phone_number_training_data.json
    ├── ip_address_training_data.json
    ├── ssn_training_data.json
    ├── coordinates_training_data.json
    └── url_training_data.json
```

## Troubleshooting

### "datasets" Library Not Found

```bash
pip install datasets
```

### Network/Download Issues

The dataset is ~100MB and requires internet access for first download. Hugging Face caches it locally after the first run.

### Memory Issues

For very large conversions, use `--max-samples` to limit:

```bash
python tools/convert_ai4privacy.py --all --max-samples 50000 --output tests/data/ai4privacy
```

### Entity Mapping Issues

Check unmapped labels in the conversion stats output. Add new mappings to `ENTITY_TYPE_MAPPING` in `convert_ai4privacy.py` if needed.

## References

- Dataset: https://huggingface.co/datasets/ai4privacy/pii-masking-300k
- Training priority plan: `/training/priority_plan.md`
- LightGBM training script: `/tools/train_lgbm_ner.py`
- Feature extractor: `/hush_engine/detectors/feature_extractor.py`

---

*Created: 2026-02-07*
*Hush Engine v1.4.0*
