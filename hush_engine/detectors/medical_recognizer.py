"""
Medical Entity Recognition using Fast Data Science NER libraries.

Uses lightweight, zero-dependency libraries to detect:
- Diseases and medical conditions (medical-named-entity-recognition)
- Drugs and medications (drug-named-entity-recognition)

License: MIT (both libraries)
"""

import sys
from typing import List, Optional
from presidio_analyzer import EntityRecognizer, RecognizerResult, AnalysisExplanation

# Try to import medical NER libraries
_medical_ner = None
_drug_ner = None
MEDICAL_NER_AVAILABLE = False
DRUG_NER_AVAILABLE = False


def _load_medical_ner():
    """Lazy-load medical NER on first use."""
    global _medical_ner, MEDICAL_NER_AVAILABLE

    if _medical_ner is not None:
        return

    try:
        from medical_named_entity_recognition import find_diseases
        _medical_ner = find_diseases
        MEDICAL_NER_AVAILABLE = True
    except ImportError:
        pass


def _load_drug_ner():
    """Lazy-load drug NER on first use."""
    global _drug_ner, DRUG_NER_AVAILABLE

    if _drug_ner is not None:
        return

    try:
        from drug_named_entity_recognition import find_drugs
        _drug_ner = find_drugs
        DRUG_NER_AVAILABLE = True
    except ImportError:
        pass


class FastMedicalRecognizer(EntityRecognizer):
    """
    Presidio recognizer using Fast Data Science medical NER libraries.

    Detects:
    - MEDICAL: Diseases, conditions, symptoms, drugs, medications
    """

    ENTITIES = ["MEDICAL"]

    def __init__(
        self,
        supported_language: str = "en",
        supported_entities: Optional[List[str]] = None,
    ):
        supported_entities = supported_entities or self.ENTITIES
        super().__init__(
            supported_entities=supported_entities,
            supported_language=supported_language,
            name="FastMedicalRecognizer",
        )
        # Lazy load on first use
        _load_medical_ner()
        _load_drug_ner()

    def load(self) -> None:
        """Load the NER models."""
        _load_medical_ner()
        _load_drug_ner()

    def analyze(
        self,
        text: str,
        entities: List[str],
        nlp_artifacts=None,
    ) -> List[RecognizerResult]:
        """
        Analyze text for medical entities.

        Args:
            text: Text to analyze
            entities: List of entity types to detect
            nlp_artifacts: Not used

        Returns:
            List of RecognizerResult objects
        """
        results = []

        if not MEDICAL_NER_AVAILABLE and not DRUG_NER_AVAILABLE:
            return results

        # Tokenize text (simple whitespace tokenization)
        tokens = text.split()

        # Find diseases/conditions
        if MEDICAL_NER_AVAILABLE and _medical_ner:
            try:
                disease_matches = _medical_ner(tokens)
                for match in disease_matches:
                    # match is typically (mesh_code, term, start_idx, end_idx)
                    if len(match) >= 4:
                        mesh_code, term, start_idx, end_idx = match[:4]
                        # Find character positions
                        char_start = self._token_to_char_pos(text, tokens, start_idx)
                        char_end = self._token_to_char_pos(text, tokens, end_idx - 1, end=True)

                        if char_start is not None and char_end is not None:
                            score = 0.85  # Base confidence
                            # Higher confidence for longer terms
                            if len(term.split()) > 1:
                                score = min(0.95, score + 0.05)

                            results.append(
                                RecognizerResult(
                                    entity_type="MEDICAL",
                                    start=char_start,
                                    end=char_end,
                                    score=score,
                                    analysis_explanation=AnalysisExplanation(
                                        recognizer=self.name,
                                        original_score=score,
                                        pattern_name="medical_disease",
                                        pattern=None,
                                        validation_result=None,
                                    ),
                                    recognition_metadata={
                                        "recognizer_name": self.name,
                                        "mesh_code": mesh_code,
                                        "medical_type": "disease",
                                    },
                                )
                            )
            except Exception as e:
                sys.stderr.write(f"[FastMedicalRecognizer] Disease NER error: {e}\n")

        # Find drugs/medications
        if DRUG_NER_AVAILABLE and _drug_ner:
            try:
                drug_matches = _drug_ner(tokens)
                for match in drug_matches:
                    if len(match) >= 4:
                        mesh_code, term, start_idx, end_idx = match[:4]
                        char_start = self._token_to_char_pos(text, tokens, start_idx)
                        char_end = self._token_to_char_pos(text, tokens, end_idx - 1, end=True)

                        if char_start is not None and char_end is not None:
                            # Check if we already found this span
                            already_found = any(
                                r.start == char_start and r.end == char_end
                                for r in results
                            )
                            if not already_found:
                                score = 0.85

                                results.append(
                                    RecognizerResult(
                                        entity_type="MEDICAL",
                                        start=char_start,
                                        end=char_end,
                                        score=score,
                                        analysis_explanation=AnalysisExplanation(
                                            recognizer=self.name,
                                            original_score=score,
                                            pattern_name="medical_drug",
                                            pattern=None,
                                            validation_result=None,
                                        ),
                                        recognition_metadata={
                                            "recognizer_name": self.name,
                                            "mesh_code": mesh_code,
                                            "medical_type": "drug",
                                        },
                                    )
                                )
            except Exception as e:
                sys.stderr.write(f"[FastMedicalRecognizer] Drug NER error: {e}\n")

        return results

    def _token_to_char_pos(
        self, text: str, tokens: List[str], token_idx: int, end: bool = False
    ) -> Optional[int]:
        """Convert token index to character position in text."""
        if token_idx < 0 or token_idx >= len(tokens):
            return None

        pos = 0
        for i, token in enumerate(tokens):
            # Find next occurrence of token
            token_start = text.find(token, pos)
            if token_start == -1:
                return None

            if i == token_idx:
                if end:
                    return token_start + len(token)
                return token_start

            pos = token_start + len(token)

        return None


def get_medical_recognizer() -> Optional[FastMedicalRecognizer]:
    """Get a medical recognizer instance if libraries are available."""
    _load_medical_ner()
    _load_drug_ner()
    if MEDICAL_NER_AVAILABLE or DRUG_NER_AVAILABLE:
        return FastMedicalRecognizer()
    return None


def is_medical_ner_available() -> bool:
    """Check if medical NER libraries are available."""
    _load_medical_ner()
    _load_drug_ner()
    return MEDICAL_NER_AVAILABLE or DRUG_NER_AVAILABLE
