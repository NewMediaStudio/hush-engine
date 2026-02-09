"""
Person Name Recognition using Multi-NER Cascade

Implements a "smart cascade" approach for high-accuracy person name detection:
1. Pattern matching: Fast lookup using title patterns (Dr., Mr., etc.)
2. LightGBM NER: Lightweight gradient boosting classifier (~10MB, 5-10x faster)
3. Dictionary lookup: name-dataset for contextless names (spreadsheets)
4. Standard NER (spaCy): Industrial-speed processing for bulk text
5. Advanced NER (GLiNER/Flair): High-accuracy for ambiguous names (optional)

Supports:
- LightGBM: Lightweight, fast NER (~85% F1 estimated, ~10MB)
- spaCy: Fast, industrial-grade NER (~89-91% F1)
- GLiNER: Zero-shot flexibility for PII (~80-83% F1) [optional]
- Flair: State-of-the-art accuracy (~93% F1 CoNLL-03) [optional]
- name-dataset: Dictionary lookup for contextless names (spreadsheets)

License: Apache 2.0
"""

import sys
import re
from typing import Dict, List, Optional, Set, Tuple
from presidio_analyzer import EntityRecognizer, RecognizerResult, AnalysisExplanation

# ============================================================================
# Lazy-loaded NER engines
# ============================================================================

# spaCy NER (standard, fast)
_spacy_nlp = None
SPACY_AVAILABLE = False

# GLiNER (zero-shot PII detection)
_gliner_model = None
GLINER_AVAILABLE = False

# Flair NER (high accuracy)
_flair_tagger = None
FLAIR_AVAILABLE = False

# name-dataset (dictionary lookup)
_name_dataset = None
NAME_DATASET_AVAILABLE = False

# LightGBM NER (lightweight, fast)
_lgbm_ner = None
LGBM_AVAILABLE = False

# LightGBM ADDRESS model (pre-loaded to avoid OpenMP conflict with spaCy)
_address_lgbm_model = None
ADDRESS_LGBM_AVAILABLE = False


def _load_lgbm_ner():
    """Lazy-load LightGBM NER classifier.

    IMPORTANT: This must be called BEFORE spaCy/presidio are loaded to avoid
    OpenMP library conflicts (libomp vs libiomp5 on macOS).
    """
    global _lgbm_ner, LGBM_AVAILABLE
    global _address_lgbm_model, ADDRESS_LGBM_AVAILABLE

    if _lgbm_ner is not None:
        return

    try:
        from hush_engine.detectors.lgbm_ner import LightweightNER, is_lightweight_available

        if is_lightweight_available():
            _lgbm_ner = LightweightNER(entity_types={"PERSON"})
            _lgbm_ner.load()
            LGBM_AVAILABLE = True
            print("Loaded LightGBM NER for PERSON detection", file=sys.stderr)

            # Also load ADDRESS model early to avoid spaCy/OpenMP conflict
            _load_address_lgbm_model()
        else:
            LGBM_AVAILABLE = False
    except ImportError:
        LGBM_AVAILABLE = False
    except Exception as e:
        print(f"LightGBM NER initialization failed: {e}", file=sys.stderr)
        LGBM_AVAILABLE = False


def _load_address_lgbm_model():
    """Load ADDRESS LightGBM model early to avoid OpenMP conflict.

    This is called from _load_lgbm_ner() to ensure the ADDRESS model is loaded
    BEFORE spaCy/presidio, which prevents the OpenMP runtime library conflict
    that causes segfaults on macOS.
    """
    global _address_lgbm_model, ADDRESS_LGBM_AVAILABLE

    if _address_lgbm_model is not None:
        return

    try:
        import lightgbm as lgbm
        from pathlib import Path

        model_path = Path(__file__).parent.parent / "models" / "lgbm" / "address_classifier.txt"
        if model_path.exists():
            _address_lgbm_model = lgbm.Booster(model_file=str(model_path))
            ADDRESS_LGBM_AVAILABLE = True
            print("Loaded LightGBM ADDRESS classifier (early load)", file=sys.stderr)
        else:
            ADDRESS_LGBM_AVAILABLE = False
    except Exception as e:
        print(f"ADDRESS LightGBM model failed to load: {e}", file=sys.stderr)
        ADDRESS_LGBM_AVAILABLE = False


def get_address_lgbm_model():
    """Get the pre-loaded ADDRESS LightGBM model.

    Returns:
        The LightGBM Booster model, or None if not available.
    """
    return _address_lgbm_model


def is_address_lgbm_available() -> bool:
    """Check if ADDRESS LightGBM model is available."""
    return ADDRESS_LGBM_AVAILABLE


def _load_spacy():
    """Lazy-load spaCy NER model."""
    global _spacy_nlp, SPACY_AVAILABLE

    if _spacy_nlp is not None:
        return

    try:
        import spacy

        # Try transformer model first (most accurate), fall back to medium/small
        for model in ["en_core_web_trf", "en_core_web_lg", "en_core_web_md", "en_core_web_sm"]:
            try:
                _spacy_nlp = spacy.load(model, disable=["parser", "lemmatizer"])
                sys.stderr.write(f"[PersonRecognizer] Loaded spaCy model: {model}\n")
                SPACY_AVAILABLE = True
                return
            except OSError:
                continue

        sys.stderr.write("[PersonRecognizer] No spaCy model found. Run: python -m spacy download en_core_web_md\n")
    except ImportError:
        sys.stderr.write("[PersonRecognizer] spaCy not installed.\n")


def _load_gliner():
    """Lazy-load GLiNER zero-shot PII model."""
    global _gliner_model, GLINER_AVAILABLE

    if _gliner_model is not None:
        return

    try:
        from gliner import GLiNER

        # Use the PII-specific model for best results
        try:
            _gliner_model = GLiNER.from_pretrained("knowledgator/gliner-pii-large-v1.0")
            sys.stderr.write("[PersonRecognizer] Loaded GLiNER PII model\n")
            GLINER_AVAILABLE = True
        except Exception as e:
            # Fall back to base model
            try:
                _gliner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
                sys.stderr.write("[PersonRecognizer] Loaded GLiNER base model\n")
                GLINER_AVAILABLE = True
            except Exception:
                sys.stderr.write(f"[PersonRecognizer] GLiNER model not available: {e}\n")
    except ImportError:
        sys.stderr.write("[PersonRecognizer] GLiNER not installed. Run: pip install gliner\n")


def _load_flair():
    """Lazy-load Flair NER tagger."""
    global _flair_tagger, FLAIR_AVAILABLE

    if _flair_tagger is not None:
        return

    try:
        from flair.data import Sentence
        from flair.models import SequenceTagger

        # Use the fast NER model (good balance of speed and accuracy)
        try:
            _flair_tagger = SequenceTagger.load("flair/ner-english-fast")
            sys.stderr.write("[PersonRecognizer] Loaded Flair NER model\n")
            FLAIR_AVAILABLE = True
        except Exception as e:
            sys.stderr.write(f"[PersonRecognizer] Flair model not available: {e}\n")
    except ImportError:
        sys.stderr.write("[PersonRecognizer] Flair not installed. Run: pip install flair\n")


def _load_name_dataset():
    """Lazy-load name-dataset for dictionary lookup."""
    global _name_dataset, NAME_DATASET_AVAILABLE

    if _name_dataset is not None:
        return

    try:
        from names_dataset import NameDataset

        _name_dataset = NameDataset()
        sys.stderr.write("[PersonRecognizer] Loaded name-dataset\n")
        NAME_DATASET_AVAILABLE = True
    except ImportError:
        sys.stderr.write("[PersonRecognizer] names-dataset not installed. Run: pip install names-dataset\n")
    except Exception as e:
        sys.stderr.write(f"[PersonRecognizer] name-dataset error: {e}\n")


# Transformers NER (BERT-based, high precision)
_transformers_ner = None
TRANSFORMERS_NER_AVAILABLE = False


def _load_transformers_ner():
    """Lazy-load Hugging Face Transformers NER pipeline."""
    global _transformers_ner, TRANSFORMERS_NER_AVAILABLE

    if _transformers_ner is not None:
        return

    try:
        from transformers import pipeline

        # Use BERT-base NER - good balance of speed and accuracy
        _transformers_ner = pipeline(
            'ner',
            model='dslim/bert-base-NER',
            aggregation_strategy='simple',
            device=-1  # CPU by default, change to 0 for GPU
        )
        sys.stderr.write("[PersonRecognizer] Loaded Transformers BERT NER\n")
        TRANSFORMERS_NER_AVAILABLE = True
    except ImportError:
        sys.stderr.write("[PersonRecognizer] transformers not installed. Run: pip install transformers\n")
    except Exception as e:
        sys.stderr.write(f"[PersonRecognizer] Transformers NER error: {e}\n")


# ============================================================================
# NER Backend Functions
# ============================================================================

def detect_with_spacy(text: str) -> List[Tuple[str, int, int, float]]:
    """
    Detect PERSON entities using spaCy.

    Returns: List of (text, start, end, confidence) tuples
    """
    _load_spacy()

    if not SPACY_AVAILABLE or _spacy_nlp is None:
        return []

    results = []
    doc = _spacy_nlp(text)

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            # spaCy doesn't provide confidence; estimate based on entity length
            # Lower base scores to stay below threshold even with consensus boost
            base_score = 0.68  # Boosted from 0.65 for better single-word recall
            if len(ent.text.split()) >= 2:
                base_score = 0.72  # Multi-word names - still needs consensus
            results.append((ent.text, ent.start_char, ent.end_char, base_score))

    return results


def detect_with_gliner(text: str) -> List[Tuple[str, int, int, float]]:
    """
    Detect PERSON entities using GLiNER zero-shot model.

    Returns: List of (text, start, end, confidence) tuples
    """
    _load_gliner()

    if not GLINER_AVAILABLE or _gliner_model is None:
        return []

    results = []

    # GLiNER labels for person detection
    labels = ["person", "name", "person name", "full name"]

    try:
        entities = _gliner_model.predict_entities(text, labels, threshold=0.3)

        for ent in entities:
            # GLiNER provides confidence scores
            score = ent.get("score", 0.75)
            start = ent.get("start", 0)
            end = ent.get("end", 0)
            entity_text = ent.get("text", text[start:end])

            results.append((entity_text, start, end, score))
    except Exception as e:
        sys.stderr.write(f"[PersonRecognizer] GLiNER error: {e}\n")

    return results


def detect_with_flair(text: str) -> List[Tuple[str, int, int, float]]:
    """
    Detect PERSON entities using Flair NER.

    Returns: List of (text, start, end, confidence) tuples
    """
    _load_flair()

    if not FLAIR_AVAILABLE or _flair_tagger is None:
        return []

    results = []

    try:
        from flair.data import Sentence

        sentence = Sentence(text)
        _flair_tagger.predict(sentence)

        for entity in sentence.get_spans("ner"):
            if entity.tag == "PER":
                # Flair provides confidence scores
                score = entity.score
                results.append((entity.text, entity.start_position, entity.end_position, score))
    except Exception as e:
        sys.stderr.write(f"[PersonRecognizer] Flair error: {e}\n")

    return results


def detect_with_transformers(text: str) -> List[Tuple[str, int, int, float]]:
    """
    Detect PERSON entities using Hugging Face Transformers BERT NER.

    This model has excellent false positive rejection (won't flag UI phrases as names)
    and high precision. Uses subword tokenization, so results may need post-processing.

    Returns: List of (text, start, end, confidence) tuples
    """
    _load_transformers_ner()

    if not TRANSFORMERS_NER_AVAILABLE or _transformers_ner is None:
        return []

    raw_results = []

    try:
        entities = _transformers_ner(text)

        for ent in entities:
            if ent.get('entity_group') == 'PER':
                score = ent.get('score', 0.0)
                start = ent.get('start', 0)
                end = ent.get('end', 0)

                # Use the positions from the model to get clean text from source
                entity_text = text[start:end].strip()
                if entity_text:
                    raw_results.append((entity_text, start, end, score))

        # Merge adjacent person entities (handles subword splitting like "Alex" + "a Samuels")
        # BERT sometimes splits names into multiple entities
        merged_results = []
        i = 0
        while i < len(raw_results):
            current = raw_results[i]
            curr_text, curr_start, curr_end, curr_score = current

            # Look ahead to merge adjacent entities
            while i + 1 < len(raw_results):
                next_ent = raw_results[i + 1]
                next_text, next_start, next_end, next_score = next_ent

                # Check if adjacent (separated by whitespace only, max 2 chars gap)
                gap = next_start - curr_end
                if 0 <= gap <= 2:
                    # Merge: extend to include next entity
                    curr_end = next_end
                    curr_text = text[curr_start:curr_end].strip()
                    curr_score = min(curr_score, next_score)  # Conservative score
                    i += 1
                else:
                    break

            merged_results.append((curr_text, curr_start, curr_end, curr_score))
            i += 1

        return merged_results

    except Exception as e:
        sys.stderr.write(f"[PersonRecognizer] Transformers error: {e}\n")

    return []


def lookup_with_name_dataset(text: str) -> List[Tuple[str, int, int, float]]:
    """
    Look up potential names using name-dataset dictionary.
    Best for contextless text like spreadsheet cells.

    Returns: List of (text, start, end, confidence) tuples
    """
    _load_name_dataset()

    if not NAME_DATASET_AVAILABLE or _name_dataset is None:
        return []

    results = []

    # Split text into potential name tokens
    # Match capitalized words that could be names (supports apostrophes and hyphens)
    # Handles: O'Brien, D'Arcy, Jean-Claude, Mary-Jane
    name_pattern = re.compile(r"\b([A-Z][a-z'-]+(?:\s+[A-Z][a-z'-]+)?)\b")

    # Also match ALL CAPS names (common in forms, PDFs)
    # Requires 2+ words to avoid matching random acronyms
    # Handles: JOHN SMITH, MARY JANE DOE
    allcaps_pattern = re.compile(r"\b([A-Z][A-Z'-]+(?:\s+[A-Z][A-Z'-]+){1,2})\b")

    for match in allcaps_pattern.finditer(text):
        potential_name = match.group(1)
        # Convert to Title Case for name-dataset lookup
        title_case_name = potential_name.title()
        parts = title_case_name.split()

        # Check each part against name-dataset
        is_likely_name = False
        matches_found = 0

        for part in parts:
            if len(part) < 2:
                continue
            result = _name_dataset.search(part)
            if result:
                first_name_data = result.get("first_name", {})
                last_name_data = result.get("last_name", {})
                first_rank = first_name_data.get("rank", {}).get("*", 0) if first_name_data else 0
                last_rank = last_name_data.get("rank", {}).get("*", 0) if last_name_data else 0
                if first_rank > 0 or last_rank > 0:
                    matches_found += 1

        # Require at least 2 parts match the name database for ALL CAPS
        if matches_found >= 2:
            confidence = 0.75 + min(0.15, matches_found * 0.05)
            results.append((potential_name, match.start(), match.end(), confidence))

    for match in name_pattern.finditer(text):
        potential_name = match.group(1)
        parts = potential_name.split()

        # Check each part against name-dataset
        is_likely_name = False
        confidence = 0.0

        for part in parts:
            result = _name_dataset.search(part)

            # Check if it's a known first name or last name
            if result:
                first_name_data = result.get("first_name", {})
                last_name_data = result.get("last_name", {})

                # Calculate confidence based on name frequency
                first_rank = first_name_data.get("rank", {}).get("*", 0) if first_name_data else 0
                last_rank = last_name_data.get("rank", {}).get("*", 0) if last_name_data else 0

                if first_rank > 0 or last_rank > 0:
                    is_likely_name = True
                    # Higher rank = more common = higher confidence (boosted base)
                    part_confidence = 0.75 + min(0.20, (first_rank + last_rank) / 10000)
                    confidence = max(confidence, part_confidence)

        if is_likely_name:
            # Boost confidence for multi-part names (increased from 0.15 to 0.20)
            if len(parts) >= 2:
                confidence = min(0.95, confidence + 0.20)
            elif len(parts) == 1:
                # Smaller boost for single-word names (helps unusual names pass filters)
                confidence = min(0.90, confidence + 0.10)

            results.append((potential_name, match.start(), match.end(), confidence))

    return results


# ============================================================================
# Pattern-based Detection (Title + Name)
# ============================================================================

# High-confidence title patterns
# Matches titles (case-insensitive) followed by proper names (Title Case)
# Does NOT use IGNORECASE to preserve name capitalization rules
# Supports apostrophes and hyphens: O'Brien, Jean-Claude
TITLE_PATTERN = re.compile(
    r"\b(?:[Dd][Rr]|[Mm][Rr]|[Mm][Rr][Ss]|[Mm][Ss]|[Mm][Ii][Ss][Ss]|"
    r"[Pp][Rr][Oo][Ff]|[Pp]rofessor|[Rr][Ee][Vv]|[Rr]everend|"
    r"[Ss][Rr]|[Jj][Rr]|[Ee][Ss][Qq]|[Hh][Oo][Nn]|"
    r"[Cc][Aa][Pp][Tt]|[Cc]aptain|[Cc][Oo][Ll]|[Cc]olonel|"
    r"[Gg][Ee][Nn]|[Gg]eneral|[Ll][Tt]|[Ll]ieutenant|"
    r"[Mm][Aa][Jj]|[Mm]ajor|[Ss][Gg][Tt]|[Ss]ergeant|"
    r"[Rr]abbi|[Ii]mam|[Ff]ather|[Ss]ister|[Bb]rother|"
    r"[Dd]ame|[Ss]ir|[Ll]ady|[Ll]ord)"
    r"\.?\s+([A-Z][a-z'-]+(?:\s+[A-Z][a-z'-]+)?)\b"
)

# Labeled name patterns (require context)
# Handles: "Name: John Smith", "Name: A S Eusuf", "Name: J. R. Smith", "Name: O'Brien"
LABELED_NAME_PATTERN = re.compile(
    r"(?:[Ff]irst|[Ll]ast|[Ff]ull|[Mm]iddle|[Gg]iven|[Ff]amily|[Ss]ur|"
    r"[Pp]atient|[Cc]lient|[Cc]ustomer|[Aa]pplicant|[Cc]ardholder|"
    r"[Aa]ccount\s*[Hh]older|[Bb]eneficiary|[Ee]mployee|[Cc]ontact|"
    r"[Aa]uthorized|[Pp]rimary|[Ss]econdary|[Ee]mergency|"
    r"[Nn]ext\s*[Oo]f\s*[Kk]in|[Ss]pouse|[Pp]arent|"
    r"[Gg]uardian|[Ww]itness|[Nn]otary|[Aa]gent|[Rr]epresentative|"
    r"[Rr]ecipient|[Ss]ender|[Oo]wner|[Hh]older)?"
    r"\s*[Nn]ame\s*[:]\s*"
    r"([A-Z][a-z'-]*(?:\.?\s+[A-Z][a-z'-]*){0,3})"
)

# ALL CAPS labeled name pattern for forms: "NAME: JOHN DOE", "APPLICANT: ABDUL RAHMAN"
LABELED_NAME_CAPS_PATTERN = re.compile(
    r"(?:NAME|CUSTOMER|CLIENT|APPLICANT|BENEFICIARY|CARDHOLDER|ACCOUNT\s*HOLDER|"
    r"PATIENT|EMPLOYEE|CONTACT|OWNER|HOLDER|RECIPIENT|SENDER)"
    r"\s*[:]\s*"
    r"([A-Z][A-Z'-]+(?:\s+[A-Z][A-Z'-]+){0,3})"
)

# Initials pattern: "A S EUSUF", "J. R. Smith", "M. L. King Jr"
INITIALS_NAME_PATTERN = re.compile(
    r"\b([A-Z]\.?\s+[A-Z]\.?\s+[A-Z][a-zA-Z'-]+(?:\s+(?:Jr|Sr|II|III|IV)\.?)?)\b"
)

# Document signature context patterns for PDFs
# Handles: "Signed by: John Smith", "Authorized by John Doe", "Prepared by: Jane"
# NOTE: Capture group requires uppercase start to avoid matching sentences like "Reviewed all pending"
SIGNATURE_CONTEXT_PATTERN = re.compile(
    r"(?:[Ss]igned|[Aa]uthorized|[Pp]repared|[Ss]ubmitted|[Aa]pproved|[Rr]eviewed|[Cc]ertified|"
    r"[Ww]itnessed|[Aa]ttested|[Vv]erified|[Cc]onfirmed|[Aa]cknowledged|[Ee]xecuted|[Nn]otarized)"
    r"(?:\s+[Bb]y)?\s*[:.]?\s*"
    r"([A-Z][a-z'-]+(?:\s+[A-Z][a-z'-]+){0,3})"
)

# ALL CAPS signature context: "SIGNED BY: JOHN DOE", "AUTHORIZED: ABDUL RAHMAN"
SIGNATURE_CONTEXT_CAPS_PATTERN = re.compile(
    r"(?:SIGNED|AUTHORIZED|PREPARED|SUBMITTED|APPROVED|REVIEWED|CERTIFIED|"
    r"WITNESSED|ATTESTED|VERIFIED|CONFIRMED|ACKNOWLEDGED|EXECUTED)"
    r"(?:\s+BY)?\s*[:.]?\s*"
    r"([A-Z][A-Z'-]+(?:\s+[A-Z][A-Z'-]+){0,3})"
)

# Role-based context patterns for PDFs
# Handles: "Customer: John Smith", "Patient: Jane Doe", etc.
# NOTE: Requires colon/period to avoid matching sentences like "Customer was notified"
ROLE_CONTEXT_PATTERN = re.compile(
    r"(?:[Cc]ustomer|[Cc]lient|[Pp]atient|[Ee]mployee|[Aa]pplicant|[Bb]eneficiary|[Cc]ardholder|"
    r"[Aa]ccount\s*[Hh]older|[Pp]olicyholder|[Ii]nsured|[Cc]laimant|[Mm]ember|[Ss]ubscriber|"
    r"[Tt]enant|[Ll]andlord|[Bb]uyer|[Ss]eller|[Bb]orrower|[Ll]ender|[Gg]uarantor|"
    r"[Dd]ebtor|[Cc]reditor|[Pp]ayee|[Pp]ayer|[Ss]ender|[Rr]ecipient|"
    r"[Dd]river|[Oo]wner|[Rr]egistrant|[Rr]espondent|[Pp]etitioner)"
    r"\s*[:.]\s*"  # REQUIRE colon or period (no longer optional)
    r"([A-Z][a-z'-]+(?:\s+[A-Z][a-z'-]+){0,3})"
)

# Salutation patterns (Dear Name, Hi Name, Hello Name)
# Handles: "Dear Allison Hill,", "Dear Margaret Hawkins DDS,", "Hi John,"
# Requires trailing comma/newline/exclamation to avoid matching "Dear Customer Service"
SALUTATION_PATTERN = re.compile(
    r"\b(?:Dear|Hi|Hello|Greetings|Attention|Attn)\s+"
    r"([A-Z][a-z'-]+(?:\s+[A-Z][a-z'-]+){0,3}"
    r"(?:\s+(?:DDS|MD|PhD|Jr\.?|Sr\.?|II|III|IV|Esq|CPA|RN|DO)\.?)?)"
    r"\s*[,\n!]"
)

# Common international name prefixes that indicate full names follow
# Handles: "Mohammed Ali", "Abdul Rahman", "Md. Hassan"
INTL_NAME_PREFIX_PATTERN = re.compile(
    r"\b((?:Mohammed|Mohammad|Muhammad|Mohamed|Md\.?|Abdul|Abu|"
    r"Sri|Shri|Smt\.?|Kumar|Devi|Bai|"
    r"Van|Von|De|Del|La|Le|El|Al-?|Ibn|Bin|Binti|"
    r"Mac|Mc|O'|D'|St\.?)"
    r"\s+[A-Z][a-zA-Z'-]+(?:\s+[A-Z][a-zA-Z'-]+){0,2})\b"
)

# Username patterns that may contain person names
# Handles: "john_smith", "jane.doe", "michael_johnson"
USERNAME_UNDERSCORE_PATTERN = re.compile(
    r"\b([a-z]{2,15})_([a-z]{2,15})\b"
)

# Employee ID patterns (user identifiers that may link to persons)
# Handles: "u123456", "U00012345", "emp123456"
EMPLOYEE_ID_PATTERN = re.compile(
    r"\b(?:u|U|emp|EMP)\d{5,8}\b"
)

# Username context words that boost confidence
USERNAME_CONTEXT = frozenset({
    "user", "username", "userid", "login", "account", "created",
    "author", "owner", "assigned", "by", "from", "updated", "modified"
})

# Cache for NamesDatabase to avoid repeated imports
_internal_names_db = None
_internal_names_db_checked = False


def _get_internal_names_db():
    """Lazy-load internal NamesDatabase for username validation."""
    global _internal_names_db, _internal_names_db_checked

    if _internal_names_db_checked:
        return _internal_names_db

    _internal_names_db_checked = True
    try:
        from hush_engine.data.names_database import NamesDatabase
        _internal_names_db = NamesDatabase()
    except ImportError:
        pass
    except Exception as e:
        sys.stderr.write(f"[PersonRecognizer] NamesDatabase error: {e}\n")

    return _internal_names_db


def validate_username_as_person(username: str) -> Tuple[bool, float, str]:
    """
    Check if username contains known first/last name components.

    Validates usernames like "john_smith" by checking if parts match
    known first names or last names.

    Args:
        username: Username string (e.g., "john_smith")

    Returns:
        Tuple of (is_valid_person, confidence, extracted_name)
        - is_valid_person: True if username likely represents a person
        - confidence: Detection confidence (0.0 - 1.0)
        - extracted_name: Reconstructed name (e.g., "John Smith")
    """
    # Try underscore-separated format first
    if '_' in username:
        parts = username.lower().split('_')
        if len(parts) == 2:
            first, last = parts
            # Check using internal NamesDatabase
            names_db = _get_internal_names_db()
            if names_db:
                is_first = names_db.is_first_name(first)
                is_last = names_db.is_last_name(last)

                if is_first and is_last:
                    # Both parts are known names - high confidence
                    return True, 0.80, f"{first.title()} {last.title()}"
                elif is_first or is_last:
                    # One part matches - moderate confidence
                    return True, 0.70, f"{first.title()} {last.title()}"

            # Fallback: try names-dataset package
            if NAME_DATASET_AVAILABLE and _name_dataset is not None:
                first_result = _name_dataset.search(first.title())
                last_result = _name_dataset.search(last.title())

                first_match = False
                last_match = False

                if first_result:
                    first_data = first_result.get("first_name")
                    if first_data and first_data.get("rank", {}).get("*", 0) > 0:
                        first_match = True

                if last_result:
                    last_data = last_result.get("last_name")
                    if last_data and last_data.get("rank", {}).get("*", 0) > 0:
                        last_match = True

                if first_match and last_match:
                    return True, 0.78, f"{first.title()} {last.title()}"
                elif first_match or last_match:
                    return True, 0.68, f"{first.title()} {last.title()}"

    # Try dot-separated format
    elif '.' in username and '@' not in username:  # Avoid emails
        parts = username.lower().split('.')
        if len(parts) == 2:
            first, last = parts
            if len(first) >= 2 and len(last) >= 2:
                names_db = _get_internal_names_db()
                if names_db:
                    if names_db.is_first_name(first) or names_db.is_last_name(last):
                        return True, 0.65, f"{first.title()} {last.title()}"

    return False, 0.0, ""


def detect_with_lgbm(text: str) -> List[Tuple[str, int, int, float]]:
    """
    Detect PERSON entities using LightGBM classifier.

    This is a lightweight alternative to transformer models, using
    gradient boosting for token classification with ~5-10x faster inference.

    Returns: List of (text, start, end, confidence) tuples
    """
    if not LGBM_AVAILABLE or _lgbm_ner is None:
        return []

    try:
        entities = _lgbm_ner.detect(text)
        results = []
        for entity in entities:
            if entity.entity_type == "PERSON":
                results.append((
                    entity.text,
                    entity.start,
                    entity.end,
                    entity.confidence
                ))
        return results
    except Exception as e:
        print(f"LightGBM PERSON detection failed: {e}", file=sys.stderr)
        return []


def detect_with_patterns(text: str) -> List[Tuple[str, int, int, float]]:
    """
    Detect PERSON names using high-confidence regex patterns.

    Returns: List of (text, start, end, confidence) tuples
    """
    results = []

    # Title + Name patterns (Dr. Smith, Mr. Jones)
    # Require 2+ word names for highest score
    for match in TITLE_PATTERN.finditer(text):
        name = match.group(1)
        full_match_start = match.start()
        full_match_end = match.end()
        # Multi-word names with title get higher score
        score = 0.92 if len(name.split()) >= 2 else 0.88
        results.append((match.group(0), full_match_start, full_match_end, score))

    # Labeled name patterns (Name: John Smith)
    for match in LABELED_NAME_PATTERN.finditer(text):
        name = match.group(1)
        # Skip very short names (1-2 chars) - avoids "Name: L", "Name: AB"
        if len(name.strip()) < 3:
            continue
        # Multi-word labeled names are more reliable
        score = 0.85 if len(name.split()) >= 2 else 0.78  # Lowered scores
        results.append((match.group(0), match.start(), match.end(), score))

    # ALL CAPS labeled patterns (NAME: JOHN DOE)
    for match in LABELED_NAME_CAPS_PATTERN.finditer(text):
        name = match.group(1)
        # Only accept if name has 2+ parts (to avoid single word matches)
        if len(name.split()) >= 2:
            results.append((match.group(0), match.start(), match.end(), 0.88))

    # Initials patterns (A S Eusuf, J. R. Smith)
    for match in INITIALS_NAME_PATTERN.finditer(text):
        name = match.group(1)
        # Initials are less reliable, lower score
        results.append((name, match.start(), match.end(), 0.80))

    # Signature context patterns (Signed by: John Smith)
    for match in SIGNATURE_CONTEXT_PATTERN.finditer(text):
        name = match.group(1)
        if len(name.split()) >= 2 and len(name) >= 5:
            # Multi-word signature names are reliable
            results.append((match.group(0), match.start(), match.end(), 0.90))
        elif len(name.split()) >= 1 and len(name) >= 3:
            # Single word signature names less reliable
            results.append((match.group(0), match.start(), match.end(), 0.82))

    # ALL CAPS signature context (SIGNED BY: JOHN DOE)
    for match in SIGNATURE_CONTEXT_CAPS_PATTERN.finditer(text):
        name = match.group(1)
        if len(name.split()) >= 2:
            results.append((match.group(0), match.start(), match.end(), 0.88))

    # Role context patterns (Customer: John Smith)
    for match in ROLE_CONTEXT_PATTERN.finditer(text):
        name = match.group(1)
        if len(name.split()) >= 2 and len(name) >= 5:
            # Multi-word role context names
            results.append((match.group(0), match.start(), match.end(), 0.88))
        elif len(name.split()) >= 1 and len(name) >= 3:
            # Single word role context names less reliable
            results.append((match.group(0), match.start(), match.end(), 0.80))

    # Salutation patterns (Dear John Smith, Hi Jane Doe)
    for match in SALUTATION_PATTERN.finditer(text):
        name = match.group(1)
        if len(name.split()) >= 2 and len(name) >= 5:
            # Multi-word salutation names are very reliable
            results.append((name, match.start(1), match.end(1), 0.92))
        elif len(name.split()) >= 1 and len(name) >= 3:
            # Single word salutation names
            results.append((name, match.start(1), match.end(1), 0.82))

    # International name prefix patterns (Mohammed Ali, Abdul Rahman)
    for match in INTL_NAME_PREFIX_PATTERN.finditer(text):
        name = match.group(1)
        if len(name.split()) >= 2:
            results.append((name, match.start(), match.end(), 0.85))

    # Username patterns (john_smith, jane.doe)
    # Check for context words nearby to boost confidence
    text_lower = text.lower()
    has_username_context = any(ctx in text_lower for ctx in USERNAME_CONTEXT)

    for match in USERNAME_UNDERSCORE_PATTERN.finditer(text):
        username = match.group(0)
        is_person, confidence, extracted_name = validate_username_as_person(username)

        if is_person:
            # Boost confidence if username context words are nearby
            if has_username_context:
                confidence = min(0.90, confidence + 0.10)
            results.append((username, match.start(), match.end(), confidence))

    return results


# ============================================================================
# Smart Cascade Recognizer
# ============================================================================

class PersonRecognizer(EntityRecognizer):
    """
    Presidio recognizer implementing smart cascade for PERSON detection.

    Cascade order:
    1. Pattern matching (titles, labels) - fastest, highest precision
    2. LightGBM NER - lightweight, fast (~5-10x faster than transformers)
    3. Dictionary lookup (name-dataset) - for spreadsheet contexts
    4. Standard NER (spaCy) - fast, good accuracy
    5. Advanced NER (GLiNER/Flair/Transformers) - for low-confidence or ambiguous spans

    The cascade can be configured to balance speed vs accuracy.
    By default, only lightweight models (LightGBM, patterns, dictionary) are enabled.
    Heavy models (GLiNER, Flair, Transformers) require: pip install hush-engine[accurate]

    Voting Modes:
    - Default: Tiered scoring with ensemble bonuses (balanced precision/recall)
    - Strict (STRICT_PERSON_MODE=True): Higher precision via refined voting:
        * Tier 1: Accept if LightGBM > 0.92 (high-precision model)
        * Tier 2: Accept if 2+ models agree with cumulative score > 1.5
        * Tier 3: Reject single-model detections below 0.70
      Expected tradeoff: +10-15% precision, -5-10% recall
    """

    ENTITIES = ["PERSON"]

    # Feature flag for strict precision mode
    # Set to True to enable refined voting strategy
    # Expected: +10-15% precision, -5-10% recall
    STRICT_PERSON_MODE = True

    def __init__(
        self,
        supported_language: str = "en",
        supported_entities: Optional[List[str]] = None,
        use_lgbm_ner: bool = True,
        use_spacy: bool = True,
        use_gliner: bool = False,
        use_flair: bool = False,
        use_transformers: bool = False,
        use_name_dataset: bool = True,
        use_patterns: bool = True,
        min_confidence: float = 0.55,  # Lowered from 0.70 to allow Tier 3 single-model detections through
        early_exit_confidence: float = 0.85,
        spreadsheet_mode: bool = False,
    ):
        """
        Initialize the PersonRecognizer.

        Args:
            supported_language: Language code
            supported_entities: Entity types to detect
            use_lgbm_ner: Enable LightGBM NER (lightweight, fast)
            use_spacy: Enable spaCy NER
            use_gliner: Enable GLiNER zero-shot NER
            use_flair: Enable Flair NER (high accuracy)
            use_transformers: Enable Transformers BERT NER (high precision)
            use_name_dataset: Enable dictionary lookup
            use_patterns: Enable regex pattern matching
            min_confidence: Minimum confidence threshold
            early_exit_confidence: Skip expensive models when lighter models find
                results above this threshold (default 0.85). Set to 1.0 to disable.
            spreadsheet_mode: Optimize for contextless spreadsheet cells
        """
        # Set instance variables BEFORE super().__init__() because the parent
        # class calls load() which invokes _preload_models() that uses these
        self.use_lgbm_ner = use_lgbm_ner
        self.use_spacy = use_spacy
        self.use_gliner = use_gliner
        self.use_flair = use_flair
        self.use_transformers = use_transformers
        self.use_name_dataset = use_name_dataset
        self.use_patterns = use_patterns
        self.min_confidence = min_confidence
        self.early_exit_confidence = early_exit_confidence
        self.spreadsheet_mode = spreadsheet_mode

        supported_entities = supported_entities or self.ENTITIES
        super().__init__(
            supported_entities=supported_entities,
            supported_language=supported_language,
            name="PersonRecognizer",
        )

    def _preload_models(self):
        """Preload configured NER models."""
        if self.use_lgbm_ner:
            _load_lgbm_ner()
        if self.use_spacy:
            _load_spacy()
        if self.use_gliner:
            _load_gliner()
        if self.use_flair:
            _load_flair()
        if self.use_transformers:
            _load_transformers_ner()
        if self.use_name_dataset:
            _load_name_dataset()

    def load(self) -> None:
        """Load all configured NER models."""
        self._preload_models()

    def _preprocess_text(self, text: str) -> Tuple[str, Dict[int, int]]:
        """
        Preprocess text to handle multi-line names and OCR artifacts.

        Handles:
        - Hyphenated line breaks: "MOHAM-\\nMED" → "MOHAMMED"
        - Line breaks splitting names: "John\\nSmith" → "John Smith"

        Returns:
            Tuple of (processed_text, position_map) where position_map maps
            new positions to original positions for accurate span reporting.
        """
        # Join hyphenated words split across lines
        # Pattern: word ending with hyphen followed by newline and continuation
        processed = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)

        # If no changes, return original with identity mapping
        if processed == text:
            return text, {}

        # Build position mapping (simplified - works for most cases)
        # For now, return processed text without detailed mapping
        # This is acceptable as the positions will be approximately correct
        return processed, {}

    def analyze(
        self,
        text: str,
        entities: List[str],
        nlp_artifacts=None,
    ) -> List[RecognizerResult]:
        """
        Analyze text for PERSON entities using smart cascade with ensemble scoring.

        Uses ensemble scoring: when multiple engines detect overlapping spans,
        their scores are aggregated for higher confidence. This improves both
        precision (multiple engines agreeing) and recall (lower threshold works).

        Args:
            text: Text to analyze
            entities: List of entity types to detect
            nlp_artifacts: Not used (we use our own NER engines)

        Returns:
            List of RecognizerResult objects
        """
        if "PERSON" not in entities:
            return []

        # Preprocess text to handle hyphenated line breaks
        processed_text, _ = self._preprocess_text(text)

        # Collect ALL detections from all engines for ensemble scoring
        all_detections: List[Tuple[str, int, int, float, str]] = []

        # Step 1: Pattern matching (fastest, highest precision)
        if self.use_patterns:
            for text_match, start, end, score in detect_with_patterns(processed_text):
                all_detections.append((text_match, start, end, score, "pattern"))

        # Step 2: LightGBM NER (lightweight, fast - ~5-10x faster than transformers)
        if self.use_lgbm_ner:
            for text_match, start, end, score in detect_with_lgbm(processed_text):
                all_detections.append((text_match, start, end, score, "lgbm"))

        # Step 3: Dictionary lookup (good for spreadsheets and moderate texts)
        # Raised from 500 to 1000 for better recall on paragraph-length texts
        if self.use_name_dataset and (self.spreadsheet_mode or len(processed_text) < 1000):
            for text_match, start, end, score in lookup_with_name_dataset(processed_text):
                all_detections.append((text_match, start, end, score, "name_dataset"))

        # Step 4: Standard NER (spaCy - fast, reliable)
        if self.use_spacy:
            for text_match, start, end, score in detect_with_spacy(processed_text):
                all_detections.append((text_match, start, end, score, "spacy"))

        # Step 5: Advanced NER for low-confidence or missed spans
        # Use configurable early-exit threshold to skip expensive models
        # when lighter models find high-confidence results
        found_high_conf = any(
            d[3] >= self.early_exit_confidence for d in all_detections
        )

        if not found_high_conf:
            # Try GLiNER
            if self.use_gliner:
                for text_match, start, end, score in detect_with_gliner(processed_text):
                    all_detections.append((text_match, start, end, score, "gliner"))

                # Check again after GLiNER - skip Flair/Transformers if confident
                found_high_conf = any(
                    d[3] >= self.early_exit_confidence for d in all_detections
                )

            # Try Flair for highest accuracy (only if still not confident)
            if self.use_flair and not found_high_conf:
                for text_match, start, end, score in detect_with_flair(processed_text):
                    all_detections.append((text_match, start, end, score, "flair"))

                # Check again after Flair
                found_high_conf = any(
                    d[3] >= self.early_exit_confidence for d in all_detections
                )

            # Try Transformers BERT NER (high precision, only if still not confident)
            if self.use_transformers and not found_high_conf:
                for text_match, start, end, score in detect_with_transformers(processed_text):
                    all_detections.append((text_match, start, end, score, "transformers"))

        # Ensemble scoring: aggregate overlapping detections
        merged_detections = self._merge_overlapping_detections(all_detections)

        # Convert to RecognizerResults
        results = []
        for text_match, start, end, score, sources in merged_detections:
            if score < self.min_confidence:
                continue

            # Apply false positive filters
            if self._is_false_positive(text_match):
                continue

            explanation = AnalysisExplanation(
                recognizer=self.name,
                original_score=score,
                pattern_name=f"person_ensemble_{'+'.join(sources)}",
                pattern=None,
                validation_result=None,
            )

            results.append(
                RecognizerResult(
                    entity_type="PERSON",
                    start=start,
                    end=end,
                    score=score,
                    analysis_explanation=explanation,
                    recognition_metadata={
                        "recognizer_name": self.name,
                        "detection_source": "+".join(sources),
                        "engine_count": len(sources),
                    },
                )
            )

        return results

    # Default model accuracy weights for SOFT weighted voting
    # Weights calibrated for recall optimization (2026-02-07 tuning)
    # Formula: combined_score = weighted_sum / sum_of_weights_used
    DEFAULT_MODEL_WEIGHTS = {
        "patterns": 0.82,      # UP from 0.70 - high precision deserves higher weight
        "flair": 0.93,         # Flair NER - highest accuracy when available
        "spacy": 0.90,         # spaCy NER - industrial grade, good accuracy
        "lgbm": 0.88,          # UP from 0.85 - trained on real data
        "transformers": 0.88,  # BERT NER - high precision
        "gliner": 0.82,        # Zero-shot NER
        "name_dataset": 0.68,  # UP from 0.60 - better for context detection
    }

    # Soft voting acceptance threshold (lowered for better recall)
    SOFT_VOTING_THRESHOLD = 0.50  # DOWN from 0.55

    # Single-model minimum score threshold (lowered for borderline detections)
    SINGLE_MODEL_THRESHOLD = 0.58  # DOWN from 0.60

    # Alias for backward compatibility
    MODEL_WEIGHTS = DEFAULT_MODEL_WEIGHTS

    @classmethod
    def get_model_weights(cls) -> Dict[str, float]:
        """
        Get model weights, preferring IVW-calibrated weights if available.

        Returns calibrated weights from detection_config if set,
        otherwise falls back to DEFAULT_MODEL_WEIGHTS.
        """
        try:
            from hush_engine.detection_config import get_config
        except ImportError:
            try:
                from ..detection_config import get_config
            except ImportError:
                return cls.DEFAULT_MODEL_WEIGHTS.copy()

        config = get_config()
        calibrated = config.get_calibrated_weights()

        if calibrated:
            # Merge calibrated with defaults (calibrated takes precedence)
            weights = cls.DEFAULT_MODEL_WEIGHTS.copy()
            weights.update(calibrated)
            return weights

        return cls.DEFAULT_MODEL_WEIGHTS.copy()

    def _merge_overlapping_detections(
        self, detections: List[Tuple[str, int, int, float, str]],
        return_decision_process: bool = False
    ) -> List[Tuple[str, int, int, float, List[str]]]:
        """
        Merge overlapping detections using SOFT weighted voting.

        SOFT Weighted Voting Algorithm:
        1. Groups overlapping detections together
        2. Uses the span with the best coverage (longest match)
        3. Computes combined_score = weighted_sum / sum_of_weights_used
        4. Accept if combined_score >= SOFT_VOTING_THRESHOLD (0.55)
        5. Single-model detections accepted if score >= SINGLE_MODEL_THRESHOLD (0.60)

        Model Weights (calibrated for recall optimization):
        - spacy: 0.90 (industrial grade, good accuracy)
        - lgbm: 0.85 (lightweight, fast)
        - patterns: 0.70 (high precision but narrow)
        - name_dataset: 0.60 (broad coverage, lower precision)

        Args:
            detections: List of (text, start, end, score, source)
            return_decision_process: If True, return decision log for debugging

        Returns:
            List of (text, start, end, aggregated_score, list_of_sources)
            If return_decision_process=True, also returns decision_log list
        """
        if not detections:
            return [] if not return_decision_process else ([], [])

        # Sort by start position, then by length (longest first)
        sorted_dets = sorted(detections, key=lambda x: (x[1], -(x[2] - x[1])))

        merged = []
        decision_log = [] if return_decision_process else None
        i = 0

        while i < len(sorted_dets):
            text_match, start, end, score, source = sorted_dets[i]
            sources = [source]
            scores = [score]

            # Find all overlapping detections
            j = i + 1
            while j < len(sorted_dets):
                _, other_start, other_end, other_score, other_source = sorted_dets[j]

                # Check for overlap (not disjoint)
                if not (other_end <= start or other_start >= end):
                    # Overlapping - extend the span to cover both if needed
                    if other_start < start:
                        start = other_start
                    if other_end > end:
                        end = other_end
                        text_match = sorted_dets[j][0]  # Use longer match's text
                    sources.append(other_source)
                    scores.append(other_score)
                    j += 1
                elif other_start < end:
                    # Continue checking nearby spans
                    j += 1
                else:
                    # No more overlaps possible
                    break

            # Compute common values
            max_score = max(scores)
            unique_sources = list(dict.fromkeys(sources))  # Preserve order, remove dups
            num_engines = len(unique_sources)

            # Get model weights (IVW-calibrated if available, otherwise defaults)
            model_weights = self.get_model_weights()

            # ================================================================
            # SOFT WEIGHTED VOTING
            # ================================================================
            # Formula: combined_score = weighted_sum / sum_of_weights_used
            # This gives higher scores when high-weight models agree
            # ================================================================

            # Calculate weighted sum and sum of weights
            weighted_sum = 0.0
            weight_sum = 0.0
            weighted_scores = []

            # Group scores by unique source (take max score per source)
            source_best_scores = {}
            for s, src in zip(scores, sources):
                if src not in source_best_scores or s > source_best_scores[src]:
                    source_best_scores[src] = s

            for src, s in source_best_scores.items():
                weight = model_weights.get(src, 0.70)  # Default weight for unknown sources
                weighted_sum += s * weight
                weight_sum += weight
                weighted_scores.append(s * weight)

            # Compute combined score via soft weighted voting
            if weight_sum > 0:
                combined_score = weighted_sum / weight_sum
            else:
                combined_score = max_score

            # Initialize for decision logging
            tier_used = "soft_voting"
            agreement_bonus = 0.0
            accepted = False

            # ================================================================
            # STRICT_PERSON_MODE: Refined voting for higher precision
            # ================================================================
            if self.STRICT_PERSON_MODE:
                # Get LightGBM-specific scores
                lgbm_scores = [s for s, src in zip(scores, sources) if src == "lgbm"]
                max_lgbm_score = max(lgbm_scores) if lgbm_scores else 0.0

                # Strict Tier 1: LightGBM high confidence (> 0.92)
                if max_lgbm_score > 0.92:
                    final_score = max_lgbm_score
                    tier_used = "strict_tier1_lgbm_high"
                    accepted = True

                # Strict Tier 2: Multi-model consensus with soft voting
                elif num_engines >= 2:
                    if combined_score >= self.SOFT_VOTING_THRESHOLD:
                        # Apply small agreement bonus for multi-model consensus
                        agreement_bonus = min(0.08, (num_engines - 1) * 0.03)
                        final_score = min(0.88, combined_score + agreement_bonus)
                        tier_used = "strict_tier2_soft_vote"
                        accepted = True
                    else:
                        final_score = combined_score
                        tier_used = "strict_tier2_weak_consensus"
                        accepted = False

                # Strict Tier 3: Single-model detection
                # Lowered from 0.62 to 0.58 - aligned with DEFAULT mode for better recall
                else:
                    if max_score >= 0.58:
                        final_score = max_score
                        tier_used = "strict_tier3_single_acceptable"
                        accepted = True
                    else:
                        final_score = max_score
                        tier_used = "strict_tier3_single_rejected"
                        accepted = False

            # ================================================================
            # DEFAULT MODE: SOFT Weighted Voting (recall-optimized)
            # ================================================================
            else:
                # Tier 1: Very high confidence from any single model (>= 0.90)
                if max_score >= 0.90:
                    final_score = max_score
                    tier_used = "tier1_high_confidence"
                    accepted = True

                # Tier 2: Multi-model soft voting
                elif num_engines >= 2:
                    # Use combined score from weighted voting
                    if combined_score >= self.SOFT_VOTING_THRESHOLD:
                        # Apply agreement bonus for consensus
                        agreement_bonus = min(0.12, (num_engines - 1) * 0.05)  # Tuned up from 0.04
                        final_score = min(0.92, combined_score + agreement_bonus)
                        tier_used = "tier2_soft_vote_accept"
                        accepted = True
                    else:
                        # Below threshold but still multi-model - check max score
                        if max_score >= self.SINGLE_MODEL_THRESHOLD:
                            final_score = max_score
                            tier_used = "tier2_fallback_max"
                            accepted = True
                        else:
                            final_score = combined_score
                            tier_used = "tier2_soft_vote_reject"
                            accepted = False

                # Tier 3: Single-model detection
                elif num_engines == 1:
                    # Accept single-model if above reduced threshold (0.60)
                    if max_score >= self.SINGLE_MODEL_THRESHOLD:
                        final_score = max_score
                        tier_used = "tier3_single_accept"
                        accepted = True
                    else:
                        final_score = max_score
                        tier_used = "tier3_single_reject"
                        accepted = False

                # Tier 4: Very low confidence
                else:
                    final_score = max_score
                    tier_used = "tier4_very_low"
                    accepted = False

            # Log decision process if requested
            if decision_log is not None:
                log_entry = {
                    "text": text_match,
                    "sources": unique_sources,
                    "raw_scores": list(scores),
                    "source_best_scores": source_best_scores,
                    "weighted_scores": weighted_scores,
                    "weighted_sum": weighted_sum,
                    "weight_sum": weight_sum,
                    "combined_score": combined_score,
                    "max_score": max_score,
                    "num_engines": num_engines,
                    "tier": tier_used,
                    "agreement_bonus": agreement_bonus,
                    "final_score": final_score,
                    "accepted": accepted,
                    "strict_mode": self.STRICT_PERSON_MODE,
                    "soft_voting_threshold": self.SOFT_VOTING_THRESHOLD,
                    "single_model_threshold": self.SINGLE_MODEL_THRESHOLD,
                }
                # Add LightGBM-specific info in strict mode
                if self.STRICT_PERSON_MODE:
                    lgbm_scores = [s for s, src in zip(scores, sources) if src == "lgbm"]
                    log_entry["lgbm_scores"] = lgbm_scores
                    log_entry["max_lgbm_score"] = max(lgbm_scores) if lgbm_scores else 0.0
                decision_log.append(log_entry)

            # Only add detection if it was accepted by the voting logic
            if accepted:
                merged.append((text_match, start, end, final_score, unique_sources))

            # Move past all processed detections
            i = j if j > i + 1 else i + 1

        if return_decision_process:
            return merged, decision_log
        return merged

    def _overlaps_existing(
        self, start: int, end: int, detections: List[Tuple]
    ) -> bool:
        """Check if a span overlaps with existing detections."""
        for _, det_start, det_end, _, _ in detections:
            if not (end <= det_start or start >= det_end):
                return True
        return False

    def _is_false_positive(self, text: str) -> bool:
        """Filter out common false positives."""
        # Clean span-merge artifacts: truncate at first newline and strip
        # trailing punctuation. LightGBM often bleeds spans past line breaks
        # (e.g., "Allison Hill,\n\nYour" → "Allison Hill")
        if '\n' in text or '\r' in text:
            text = text.split('\n')[0].split('\r')[0].rstrip(' \t,:;')

        text_lower = text.lower().strip()
        text_stripped = text.strip()

        # Skip if too short (likely abbreviation)
        if len(text_stripped) <= 2:
            return True

        # Skip if contains numbers
        if any(c.isdigit() for c in text):
            return True

        # Skip if excessive internal whitespace (OCR artifact)
        if '  ' in text:
            return True

        # Skip if too long (likely a phrase, not a name)
        if len(text_stripped) > 40:
            return True

        # OCR artifact patterns - common from PDF extraction
        # Ends with hyphen (line break artifact): "MOHAMMED ALDAK-"
        if text_stripped.endswith('-'):
            return True

        # Starts with punctuation (likely fragment): ": John", "- Smith"
        if text_stripped and text_stripped[0] in ':;,.-!?':
            return True

        # Contains unusual punctuation sequences
        if any(seq in text for seq in ['--', '..', '__', '//', '\\\\', '||']):
            return True

        # All uppercase single word less than 4 chars (likely acronym, not name)
        if text_stripped.isupper() and ' ' not in text_stripped and len(text_stripped) < 4:
            return True

        # Common false positives from UI/navigation/documents
        false_positives = {
            # UI/Navigation
            "home", "back", "next", "previous", "submit", "cancel", "ok", "yes", "no",
            "save", "edit", "delete", "view", "search", "filter", "sort", "settings",
            "menu", "file", "help", "tools", "window", "close", "open", "print",
            "copy", "paste", "cut", "undo", "redo", "select", "all", "none",
            "click", "tap", "press", "enter", "exit", "logout", "login", "signin",
            "signup", "register", "subscribe", "unsubscribe", "download", "upload",
            "attach", "attachment", "browse", "refresh", "reload", "update",
            # Additional verbs (common FPs)
            "read", "write", "create", "change", "execute", "process", "run", "build",
            "start", "stop", "pause", "resume", "continue", "skip", "retry",
            # Additional UI elements
            "button", "dialog", "panel", "tab", "toolbar", "sidebar", "header", "footer",
            "modal", "popup", "dropdown", "checkbox", "radio", "slider", "toggle",

            # Days and months
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
            "january", "february", "march", "april", "may", "june", "july",
            "august", "september", "october", "november", "december",

            # Document/form labels
            "label", "field", "form", "input", "output", "value", "data", "info",
            "name", "address", "phone", "email", "date", "time", "number", "amount",
            "total", "subtotal", "balance", "payment", "invoice", "invoices", "receipt",
            "document", "attachment", "signature", "required", "optional", "mandatory",
            "photo", "image", "picture", "scan", "copy", "original", "duplicate",

            # Common phrases from OCR
            "adhere", "affix", "apply", "attach", "here", "this", "that", "each",
            "every", "unique", "please", "thank", "thanks", "regards", "sincerely",

            # Action phrases
            "do not", "must not", "cannot", "should not", "will not",

            # Software/tech terms
            "error", "warning", "success", "failed", "loading", "processing",
            "pending", "completed", "approved", "rejected", "denied", "confirmed",
            "version", "update", "upgrade", "install", "uninstall", "configure",

            # Form role labels (common FPs from document headers)
            "applicant", "beneficiary", "holder", "owner", "recipient", "sender",
            "service", "system", "admin", "root", "user", "users",
            "account", "profile", "administrator", "supervisor", "representative",
        }
        if text_lower in false_positives:
            return True

        # Multi-word phrase patterns that are commonly flagged
        phrase_patterns = [
            "do not", "must not", "please note", "click here", "tap here",
            "each click", "label is", "adhere your", "affix your", "apply here",
            "open invoices", "see attached", "per attached", "as attached",
        ]
        for pattern in phrase_patterns:
            if pattern in text_lower:
                return True

        # Single words that look like names but aren't (common FPs)
        # These are words that NER models often misclassify as person names
        name_like_words = {
            # Food items often detected as names
            'caesar', 'napoleon', 'benedict', 'wellington',
            # Places commonly detected as names
            'america', 'american', 'asia', 'asian', 'africa', 'african',
            'europe', 'european', 'austin', 'boston', 'denver', 'phoenix',
            'dallas', 'houston', 'charlotte', 'orlando', 'brooklyn', 'jersey',
            # Common words misdetected as names
            'customer', 'client', 'patient', 'member', 'agent',
            'manager', 'director', 'president', 'chairman', 'secretary',
            'admin', 'administrator', 'moderator', 'owner', 'creator',
            # Status words
            'active', 'inactive', 'pending', 'approved', 'denied',
            'available', 'unavailable', 'online', 'offline',
            # Document sections
            'summary', 'details', 'notes', 'comments', 'history',
            'overview', 'introduction', 'conclusion', 'appendix',
            # Generic terms
            'unknown', 'anonymous', 'default', 'sample', 'test', 'demo',
            'example', 'template', 'placeholder', 'generic', 'standard',
            # Occupations (not names)
            'doctor', 'nurse', 'lawyer', 'engineer', 'teacher', 'professor',
            'accountant', 'developer', 'designer', 'analyst', 'consultant',
        }
        if text_lower.strip() in name_like_words:
            return True

        # Single word starting with lowercase is not a name
        # (Unless it's a known name from name_dataset)
        if ' ' not in text_stripped and text_stripped[0].islower():
            return True

        return False


# ============================================================================
# Factory Functions
# ============================================================================

def get_person_recognizer(
    mode: str = "balanced",
    spreadsheet_mode: bool = False,
) -> PersonRecognizer:
    """
    Get a PersonRecognizer configured for specific use case.

    Args:
        mode: Configuration mode
            - "fast": Only patterns + spaCy (fastest)
            - "balanced": Patterns + name-dataset + spaCy (default)
            - "accurate": All engines including GLiNER/Flair/Transformers
        spreadsheet_mode: Optimize for contextless spreadsheet cells

    Returns:
        Configured PersonRecognizer instance
    """
    if mode == "fast":
        return PersonRecognizer(
            use_spacy=True,
            use_gliner=False,
            use_flair=False,
            use_transformers=False,
            use_name_dataset=False,
            use_patterns=True,
            spreadsheet_mode=spreadsheet_mode,
        )
    elif mode == "accurate":
        return PersonRecognizer(
            use_spacy=True,
            use_gliner=True,
            use_flair=True,
            use_transformers=True,
            use_name_dataset=True,
            use_patterns=True,
            early_exit_confidence=0.80,  # Lower threshold to allow more cascade
            spreadsheet_mode=spreadsheet_mode,
        )
    else:  # balanced
        return PersonRecognizer(
            use_spacy=True,
            use_gliner=False,
            use_flair=False,
            use_transformers=False,
            use_name_dataset=True,
            use_patterns=True,
            spreadsheet_mode=spreadsheet_mode,
        )


def is_person_ner_available() -> bool:
    """Check if any NER engine is available for person detection."""
    _load_spacy()
    _load_gliner()
    _load_flair()
    _load_transformers_ner()
    return SPACY_AVAILABLE or GLINER_AVAILABLE or FLAIR_AVAILABLE or TRANSFORMERS_NER_AVAILABLE


def get_available_engines() -> List[str]:
    """Get list of available NER engines."""
    _load_spacy()
    _load_gliner()
    _load_flair()
    _load_transformers_ner()
    _load_name_dataset()

    engines = ["patterns"]  # Always available
    if SPACY_AVAILABLE:
        engines.append("spacy")
    if GLINER_AVAILABLE:
        engines.append("gliner")
    if FLAIR_AVAILABLE:
        engines.append("flair")
    if TRANSFORMERS_NER_AVAILABLE:
        engines.append("transformers")
    if NAME_DATASET_AVAILABLE:
        engines.append("name_dataset")

    return engines
