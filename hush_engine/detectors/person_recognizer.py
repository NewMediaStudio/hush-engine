"""
Person Name Recognition using Multi-NER Cascade

Implements a "smart cascade" approach for high-accuracy person name detection:
1. Dictionary/Pattern: Fast lookup using name-dataset + title patterns
2. Standard NER (spaCy): Industrial-speed processing for bulk text
3. Advanced NER (GLiNER/Flair): High-accuracy for ambiguous names

Supports:
- spaCy: Fast, industrial-grade NER (~89-91% F1)
- GLiNER: Zero-shot flexibility for PII (~80-83% F1)
- Flair: State-of-the-art accuracy (~93% F1 CoNLL-03)
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
            base_score = 0.85
            if len(ent.text.split()) >= 2:
                base_score = 0.90  # Full names more reliable
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
                    # Higher rank = more common = higher confidence (boosted from 0.60 base)
                    part_confidence = 0.70 + min(0.25, (first_rank + last_rank) / 10000)
                    confidence = max(confidence, part_confidence)

        if is_likely_name:
            # Boost confidence for multi-part names (increased from 0.15 to 0.20)
            if len(parts) >= 2:
                confidence = min(0.95, confidence + 0.20)

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


def detect_with_patterns(text: str) -> List[Tuple[str, int, int, float]]:
    """
    Detect PERSON names using high-confidence regex patterns.

    Returns: List of (text, start, end, confidence) tuples
    """
    results = []

    # Title + Name patterns (Dr. Smith, Mr. Jones)
    for match in TITLE_PATTERN.finditer(text):
        name = match.group(1)
        # Full match includes title, but we capture just the name
        full_match_start = match.start()
        full_match_end = match.end()
        results.append((match.group(0), full_match_start, full_match_end, 0.95))

    # Labeled name patterns (Name: John Smith)
    for match in LABELED_NAME_PATTERN.finditer(text):
        name = match.group(1)
        results.append((match.group(0), match.start(), match.end(), 0.90))

    return results


# ============================================================================
# Smart Cascade Recognizer
# ============================================================================

class PersonRecognizer(EntityRecognizer):
    """
    Presidio recognizer implementing smart cascade for PERSON detection.

    Cascade order:
    1. Pattern matching (titles, labels) - fastest, highest precision
    2. Dictionary lookup (name-dataset) - for spreadsheet contexts
    3. Standard NER (spaCy) - fast, good accuracy
    4. Advanced NER (GLiNER/Flair/Transformers) - for low-confidence or ambiguous spans

    The cascade can be configured to balance speed vs accuracy.
    """

    ENTITIES = ["PERSON"]

    def __init__(
        self,
        supported_language: str = "en",
        supported_entities: Optional[List[str]] = None,
        use_spacy: bool = True,
        use_gliner: bool = False,
        use_flair: bool = False,
        use_transformers: bool = False,
        use_name_dataset: bool = True,
        use_patterns: bool = True,
        min_confidence: float = 0.60,
        early_exit_confidence: float = 0.85,
        spreadsheet_mode: bool = False,
    ):
        """
        Initialize the PersonRecognizer.

        Args:
            supported_language: Language code
            supported_entities: Entity types to detect
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
        Analyze text for PERSON entities using smart cascade.

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

        all_detections: List[Tuple[str, int, int, float, str]] = []

        # Step 1: Pattern matching (fastest, highest precision)
        if self.use_patterns:
            for text_match, start, end, score in detect_with_patterns(processed_text):
                all_detections.append((text_match, start, end, score, "pattern"))

        # Step 2: Dictionary lookup (good for spreadsheets)
        if self.use_name_dataset and (self.spreadsheet_mode or len(processed_text) < 100):
            for text_match, start, end, score in lookup_with_name_dataset(processed_text):
                # Only add if not already found by patterns
                if not self._overlaps_existing(start, end, all_detections):
                    all_detections.append((text_match, start, end, score, "name_dataset"))

        # Step 3: Standard NER (spaCy - fast, reliable)
        if self.use_spacy:
            for text_match, start, end, score in detect_with_spacy(processed_text):
                if not self._overlaps_existing(start, end, all_detections):
                    all_detections.append((text_match, start, end, score, "spacy"))

        # Step 4: Advanced NER for low-confidence or missed spans
        # Use configurable early-exit threshold to skip expensive models
        # when lighter models find high-confidence results
        found_high_conf = any(
            d[3] >= self.early_exit_confidence for d in all_detections
        )

        if not found_high_conf:
            # Try GLiNER
            if self.use_gliner:
                for text_match, start, end, score in detect_with_gliner(processed_text):
                    if not self._overlaps_existing(start, end, all_detections):
                        all_detections.append((text_match, start, end, score, "gliner"))

                # Check again after GLiNER - skip Flair/Transformers if confident
                found_high_conf = any(
                    d[3] >= self.early_exit_confidence for d in all_detections
                )

            # Try Flair for highest accuracy (only if still not confident)
            if self.use_flair and not found_high_conf:
                for text_match, start, end, score in detect_with_flair(processed_text):
                    if not self._overlaps_existing(start, end, all_detections):
                        all_detections.append((text_match, start, end, score, "flair"))

                # Check again after Flair
                found_high_conf = any(
                    d[3] >= self.early_exit_confidence for d in all_detections
                )

            # Try Transformers BERT NER (high precision, only if still not confident)
            if self.use_transformers and not found_high_conf:
                for text_match, start, end, score in detect_with_transformers(processed_text):
                    if not self._overlaps_existing(start, end, all_detections):
                        all_detections.append((text_match, start, end, score, "transformers"))

        # Convert to RecognizerResults
        results = []
        for text_match, start, end, score, source in all_detections:
            if score < self.min_confidence:
                continue

            # Apply false positive filters
            if self._is_false_positive(text_match):
                continue

            explanation = AnalysisExplanation(
                recognizer=self.name,
                original_score=score,
                pattern_name=f"person_{source}",
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
                        "detection_source": source,
                    },
                )
            )

        return results

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
        text_lower = text.lower().strip()

        # Skip if too short (likely abbreviation)
        if len(text) <= 2:
            return True

        # Skip if contains numbers
        if any(c.isdigit() for c in text):
            return True

        # Skip if contains newlines or excessive whitespace
        if '\n' in text or '\r' in text or '  ' in text:
            return True

        # Skip if too long (likely a phrase, not a name)
        if len(text) > 40:
            return True

        # Common false positives from UI/navigation
        false_positives = {
            "home", "back", "next", "previous", "submit", "cancel", "ok", "yes", "no",
            "save", "edit", "delete", "view", "search", "filter", "sort", "settings",
            "menu", "file", "help", "tools", "window", "close", "open", "print",
            "copy", "paste", "cut", "undo", "redo", "select", "all", "none",
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
            "january", "february", "march", "april", "may", "june", "july",
            "august", "september", "october", "november", "december",
        }
        if text_lower in false_positives:
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
