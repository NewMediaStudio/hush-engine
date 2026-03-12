"""
macOS NLTagger-based NLP engine for Presidio.

Replaces spaCy as the default NLP backend for the Minimal tier.
Uses Apple's NaturalLanguage framework (available macOS 10.14+) for:
- Tokenization (NLTokenUnit.word)
- Lemmatization (NLTagScheme.lemma)
- Named Entity Recognition (NLTagScheme.nameType: PersonalName, PlaceName, OrganizationName)
- Stopword/punctuation detection

License: MIT
"""

import string
import sys
from typing import List, Optional, Iterator, Iterable, Tuple

# Defer Presidio imports to avoid triggering the spaCy import chain at module load.
# Presidio's NlpArtifacts does `from spacy.tokens import Doc, Span` at import time,
# which fails on Python 3.14+ due to spaCy/Pydantic v1 incompatibility.
# We import these lazily in the class methods that need them.
_NlpEngine = None
_NlpArtifacts = None


def _get_presidio_types():
    """Lazy-import Presidio NLP types."""
    global _NlpEngine, _NlpArtifacts
    if _NlpEngine is None:
        from presidio_analyzer.nlp_engine import NlpEngine, NlpArtifacts
        _NlpEngine = NlpEngine
        _NlpArtifacts = NlpArtifacts
    return _NlpEngine, _NlpArtifacts

# Try to import NaturalLanguage framework
NLTAGGER_AVAILABLE = False
try:
    import NaturalLanguage
    NLTAGGER_AVAILABLE = True
except ImportError:
    pass

# English stopwords (subset matching spaCy's defaults for context filtering)
_ENGLISH_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "need",
    "not", "no", "nor", "so", "if", "then", "than", "too", "very",
    "just", "about", "above", "after", "again", "all", "also", "am",
    "any", "because", "before", "below", "between", "both", "each",
    "few", "further", "get", "got", "here", "her", "hers", "herself",
    "him", "himself", "his", "how", "i", "into", "it", "its", "itself",
    "me", "more", "most", "my", "myself", "now", "only", "other", "our",
    "ours", "ourselves", "out", "over", "own", "re", "s", "same", "she",
    "some", "such", "t", "that", "their", "theirs", "them", "themselves",
    "there", "these", "they", "this", "those", "through", "under", "until",
    "up", "we", "what", "when", "where", "which", "while", "who", "whom",
    "why", "you", "your", "yours", "yourself", "yourselves",
})

_PUNCTUATION = frozenset(string.punctuation)

# NLTagger nameType → Presidio entity type mapping
_NER_TAG_MAP = {
    "PersonalName": "PERSON",
    "PlaceName": "LOCATION",
    "OrganizationName": "ORGANIZATION",
}


class NLToken:
    """Lightweight token object duck-typed to match spaCy Token interface.

    Provides the attributes that Presidio's NlpArtifacts and recognizers access:
    .text, .lemma_, .idx, .is_stop, .is_punct, .pos_
    """

    __slots__ = ("text", "lemma_", "idx", "is_stop", "is_punct", "pos_", "i")

    def __init__(self, text: str, lemma: str, idx: int, pos: str = "", index: int = 0):
        self.text = text
        self.lemma_ = lemma
        self.idx = idx
        self.pos_ = pos
        self.i = index
        self.is_stop = text.lower() in _ENGLISH_STOPWORDS
        self.is_punct = all(c in _PUNCTUATION for c in text) if text else False

    def __str__(self):
        return self.text

    def __repr__(self):
        return f"NLToken({self.text!r})"

    def __len__(self):
        return len(self.text)


class NLSpan:
    """Lightweight span object duck-typed to match spaCy Span interface.

    Provides: .text, .label_, .start_char, .end_char, .start, .end
    """

    __slots__ = ("text", "label_", "start_char", "end_char", "start", "end", "score")

    def __init__(self, text: str, label: str, start_char: int, end_char: int,
                 start: int = 0, end: int = 0, score: float = 0.85):
        self.text = text
        self.label_ = label
        self.start_char = start_char
        self.end_char = end_char
        self.start = start
        self.end = end
        self.score = score

    def __str__(self):
        return self.text

    def __repr__(self):
        return f"NLSpan({self.text!r}, {self.label_!r})"


class NLDoc:
    """Lightweight document object duck-typed to match spaCy Doc interface.

    Iterable over NLToken objects. Has .ents, .text, and len() support.
    """

    def __init__(self, text: str, tokens: List[NLToken], ents: List[NLSpan]):
        self.text = text
        self._tokens = tokens
        self.ents = ents

    def __iter__(self):
        return iter(self._tokens)

    def __len__(self):
        return len(self._tokens)

    def __getitem__(self, index):
        return self._tokens[index]

    def __str__(self):
        return self.text


class NLTaggerNlpEngine:
    """
    Presidio NLP engine backed by macOS NLTagger.

    Zero-dependency alternative to SpacyNlpEngine for the Minimal tier.
    Uses Apple's NaturalLanguage framework for tokenization, lemmatization, and NER.

    Note: Does not inherit from NlpEngine at class-definition time to avoid
    triggering the spaCy import chain. Presidio's AnalyzerEngine checks for
    duck-type compatibility (process_text, is_loaded, etc.) not isinstance.
    """

    engine_name = "nltagger"

    def __init__(self):
        self._loaded = False
        self._tagger = None

    def load(self) -> None:
        """Initialize the NLTagger."""
        if not NLTAGGER_AVAILABLE:
            raise ImportError(
                "pyobjc-framework-NaturalLanguage is required. "
                "Install with: pip install pyobjc-framework-NaturalLanguage"
            )

        self._tagger = NaturalLanguage.NLTagger.alloc().initWithTagSchemes_([
            NaturalLanguage.NLTagSchemeTokenType,
            NaturalLanguage.NLTagSchemeNameType,
            NaturalLanguage.NLTagSchemeLemma,
        ])
        self._loaded = True
        sys.stderr.write("[NLTaggerNlpEngine] Loaded macOS NLTagger\n")

    def is_loaded(self) -> bool:
        return self._loaded

    def process_text(self, text: str, language: str):
        """
        Process text through NLTagger pipeline.

        Produces NlpArtifacts with duck-typed Doc/Token/Span objects
        compatible with Presidio's recognizers and context enhancers.
        """
        if not self._loaded:
            self.load()

        self._tagger.setString_(text)

        tokens = []
        entities = []
        lemmas = []
        tokens_indices = []
        entity_scores = []

        # Current NER entity tracking (for multi-token entities)
        current_entity_text = ""
        current_entity_label = ""
        current_entity_start = -1
        current_entity_token_start = -1
        token_index = 0

        # Enumerate tokens with their tags
        # NLTagger enumerates over the string range
        tag_schemes = [
            NaturalLanguage.NLTagSchemeTokenType,
            NaturalLanguage.NLTagSchemeNameType,
            NaturalLanguage.NLTagSchemeLemma,
        ]

        # We need to enumerate once for each tag scheme, or use a combined approach.
        # NLTagger supports enumerating one scheme at a time.
        # Strategy: enumerate with nameType (NER) and collect token info simultaneously.

        # First pass: tokenize and get NER tags
        token_data = []  # [(text, range_start, range_len, ner_tag)]

        def _collect_tokens(tag, token_range, stop):
            """Callback for NLTagger enumeration."""
            start = token_range.location
            length = token_range.length
            token_text = text[start:start + length]
            ner_tag = str(tag) if tag else ""
            token_data.append((token_text, start, length, ner_tag))

        self._tagger.enumerateTagsInRange_unit_scheme_options_usingBlock_(
            (0, len(text)),
            NaturalLanguage.NLTokenUnitWord,
            NaturalLanguage.NLTagSchemeNameType,
            NaturalLanguage.NLTaggerOmitWhitespace | NaturalLanguage.NLTaggerOmitPunctuation,
            _collect_tokens,
        )

        # Second pass: get lemmas for each token range
        # We need a separate tagger call since we can't get multiple schemes in one enumeration
        lemma_data = []

        def _collect_lemmas(tag, token_range, stop):
            """Callback for lemma enumeration."""
            lemma = str(tag) if tag else ""
            lemma_data.append(lemma)

        self._tagger.enumerateTagsInRange_unit_scheme_options_usingBlock_(
            (0, len(text)),
            NaturalLanguage.NLTokenUnitWord,
            NaturalLanguage.NLTagSchemeLemma,
            NaturalLanguage.NLTaggerOmitWhitespace | NaturalLanguage.NLTaggerOmitPunctuation,
            _collect_lemmas,
        )

        # Also enumerate punctuation tokens (Presidio expects all tokens including punct)
        punct_data = []

        def _collect_all_tokens(tag, token_range, stop):
            start = token_range.location
            length = token_range.length
            token_text = text[start:start + length]
            punct_data.append((token_text, start, length))

        self._tagger.enumerateTagsInRange_unit_scheme_options_usingBlock_(
            (0, len(text)),
            NaturalLanguage.NLTokenUnitWord,
            NaturalLanguage.NLTagSchemeTokenType,
            0,  # No omit flags - include everything
            _collect_all_tokens,
        )

        # Build token list from all tokens (including punctuation)
        # Use punct_data for the full token list, and match NER/lemma data by position
        word_idx = 0
        for i, (tok_text, tok_start, tok_len) in enumerate(punct_data):
            # Check if this token matches the next word token (non-punct)
            lemma = tok_text.lower()
            if word_idx < len(token_data):
                wd_text, wd_start, wd_len, wd_ner = token_data[word_idx]
                if tok_start == wd_start and tok_len == wd_len:
                    # This is a word token - use its lemma and NER tag
                    if word_idx < len(lemma_data) and lemma_data[word_idx]:
                        lemma = lemma_data[word_idx].lower()
                    ner_tag = wd_ner
                    word_idx += 1
                else:
                    ner_tag = ""
            else:
                ner_tag = ""

            token = NLToken(
                text=tok_text,
                lemma=lemma,
                idx=tok_start,
                index=i,
            )
            tokens.append(token)
            lemmas.append(lemma)
            tokens_indices.append(tok_start)

            # Determine if this is a word token (has NER data) or whitespace/punct
            is_word_token = (ner_tag != "" or not tok_text.strip() == "" and
                             word_idx > 0 and not token.is_punct)
            is_whitespace = tok_text.strip() == ""

            # Accumulate NER entities (merge consecutive tokens with same tag)
            presidio_label = _NER_TAG_MAP.get(ner_tag, "")
            if presidio_label:
                if presidio_label == current_entity_label and current_entity_start >= 0:
                    # Continue current entity
                    current_entity_text = text[current_entity_start:tok_start + tok_len]
                else:
                    # Flush previous entity
                    if current_entity_label and current_entity_text.strip():
                        entities.append(NLSpan(
                            text=current_entity_text.strip(),
                            label=current_entity_label,
                            start_char=current_entity_start,
                            end_char=current_entity_start + len(current_entity_text),
                            start=current_entity_token_start,
                            end=i,
                        ))
                        entity_scores.append(0.85)
                    # Start new entity
                    current_entity_text = tok_text
                    current_entity_label = presidio_label
                    current_entity_start = tok_start
                    current_entity_token_start = i
            elif is_whitespace and current_entity_label:
                # Whitespace between entity tokens — extend entity span but don't flush.
                # This allows "Maria Garcia" to merge instead of splitting at the space.
                current_entity_text = text[current_entity_start:tok_start + tok_len]
            else:
                # Non-entity word token — flush any accumulated entity
                if current_entity_label and current_entity_text.strip():
                    entities.append(NLSpan(
                        text=current_entity_text.strip(),
                        label=current_entity_label,
                        start_char=current_entity_start,
                        end_char=current_entity_start + len(current_entity_text),
                        start=current_entity_token_start,
                        end=i,
                    ))
                    entity_scores.append(0.85)
                current_entity_label = ""
                current_entity_text = ""
                current_entity_start = -1

        # Flush final entity
        if current_entity_label and current_entity_text.strip():
            entities.append(NLSpan(
                text=current_entity_text.strip(),
                label=current_entity_label,
                start_char=current_entity_start,
                end_char=current_entity_start + len(current_entity_text),
                start=current_entity_token_start,
                end=len(tokens),
            ))
            entity_scores.append(0.85)

        doc = NLDoc(text=text, tokens=tokens, ents=entities)

        _, NlpArtifactsCls = _get_presidio_types()
        return NlpArtifactsCls(
            entities=entities,
            tokens=doc,
            tokens_indices=tokens_indices,
            lemmas=lemmas,
            nlp_engine=self,
            language=language,
            scores=entity_scores if entity_scores else None,
        )

    def process_batch(
        self,
        texts: Iterable[str],
        language: str,
        batch_size: int = 1,
        n_process: int = 1,
        **kwargs,
    ) -> Iterator[Tuple[str, NlpArtifacts]]:
        """Process multiple texts sequentially (NLTagger doesn't support batch mode)."""
        for text in texts:
            yield text, self.process_text(text, language)

    def is_stopword(self, word: str, language: str) -> bool:
        """Check if word is a stopword."""
        return word.lower() in _ENGLISH_STOPWORDS

    def is_punct(self, word: str, language: str) -> bool:
        """Check if word is punctuation."""
        return all(c in _PUNCTUATION for c in word) if word else False

    def get_supported_entities(self) -> List[str]:
        """Return entity types recognized by NLTagger."""
        return ["PERSON", "LOCATION", "ORGANIZATION"]

    def get_supported_languages(self) -> List[str]:
        """Return supported languages."""
        return ["en"]
