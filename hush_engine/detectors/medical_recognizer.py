"""
Medical Entity Recognition using scispaCy

Uses biomedical NER models trained on PubMed/clinical text to detect:
- Diseases and medical conditions
- Drugs and chemicals
- Medical procedures

License: Apache 2.0 (scispaCy)
"""

import sys
from typing import List, Optional
from presidio_analyzer import EntityRecognizer, RecognizerResult, AnalysisExplanation

# Try to import scispacy and load models
SCISPACY_AVAILABLE = False
_nlp_disease_chem = None
_nlp_anatomy = None


def _load_scispacy_models():
    """Lazy-load scispaCy models on first use."""
    global SCISPACY_AVAILABLE, _nlp_disease_chem, _nlp_anatomy

    if _nlp_disease_chem is not None:
        return  # Already loaded

    try:
        import scispacy
        import spacy

        # Try to load the BC5CDR model (diseases + chemicals)
        # This model is trained on the BC5CDR corpus with ~5800 disease
        # and ~4400 chemical annotations
        try:
            _nlp_disease_chem = spacy.load("en_ner_bc5cdr_md")
            sys.stderr.write("[MedicalRecognizer] Loaded en_ner_bc5cdr_md model\n")
            SCISPACY_AVAILABLE = True
        except OSError:
            sys.stderr.write("[MedicalRecognizer] en_ner_bc5cdr_md not installed. "
                           "Run: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz\n")

        # Optionally load anatomy model for anatomical terms
        try:
            _nlp_anatomy = spacy.load("en_ner_bionlp13cg_md")
            sys.stderr.write("[MedicalRecognizer] Loaded en_ner_bionlp13cg_md model\n")
        except OSError:
            # This model is optional
            pass

    except ImportError:
        sys.stderr.write("[MedicalRecognizer] scispacy not installed. Medical NER disabled.\n")


class ScispaCyMedicalRecognizer(EntityRecognizer):
    """
    Presidio recognizer that uses scispaCy biomedical NER models.

    Detects:
    - DISEASE: Medical conditions, diseases, symptoms
    - CHEMICAL: Drugs, medications, chemical compounds
    """

    ENTITIES = ["MEDICAL"]

    # Mapping from scispaCy entity types to our unified MEDICAL type
    SCISPACY_TO_MEDICAL = {
        "DISEASE": "MEDICAL",
        "CHEMICAL": "MEDICAL",
        # From bionlp13cg model
        "CANCER": "MEDICAL",
        "ORGAN": "MEDICAL",
        "TISSUE": "MEDICAL",
        "CELL": "MEDICAL",
        "AMINO_ACID": "MEDICAL",
        "GENE_OR_GENE_PRODUCT": "MEDICAL",
        "SIMPLE_CHEMICAL": "MEDICAL",
        "ANATOMICAL_SYSTEM": "MEDICAL",
        "IMMATERIAL_ANATOMICAL_ENTITY": "MEDICAL",
        "MULTI-TISSUE_STRUCTURE": "MEDICAL",
        "DEVELOPING_ANATOMICAL_STRUCTURE": "MEDICAL",
        "ORGANISM_SUBDIVISION": "MEDICAL",
        "CELLULAR_COMPONENT": "MEDICAL",
        "PATHOLOGICAL_FORMATION": "MEDICAL",
    }

    def __init__(
        self,
        supported_language: str = "en",
        supported_entities: Optional[List[str]] = None,
    ):
        supported_entities = supported_entities or self.ENTITIES
        super().__init__(
            supported_entities=supported_entities,
            supported_language=supported_language,
            name="ScispaCyMedicalRecognizer",
        )
        # Lazy load models on first use
        _load_scispacy_models()

    def load(self) -> None:
        """Load the scispaCy models."""
        _load_scispacy_models()

    def analyze(
        self,
        text: str,
        entities: List[str],
        nlp_artifacts=None,
    ) -> List[RecognizerResult]:
        """
        Analyze text for medical entities using scispaCy.

        Args:
            text: Text to analyze
            entities: List of entity types to detect
            nlp_artifacts: Not used (we use scispaCy's own pipeline)

        Returns:
            List of RecognizerResult objects
        """
        results = []

        if not SCISPACY_AVAILABLE:
            return results

        # Process with disease/chemical model
        if _nlp_disease_chem is not None:
            doc = _nlp_disease_chem(text)
            for ent in doc.ents:
                if ent.label_ in self.SCISPACY_TO_MEDICAL:
                    # Higher confidence for longer entities (more specific)
                    base_score = 0.85
                    length_bonus = min(0.1, len(ent.text) * 0.01)
                    score = min(0.95, base_score + length_bonus)

                    explanation = AnalysisExplanation(
                        recognizer=self.name,
                        original_score=score,
                        pattern_name=f"scispacy_{ent.label_.lower()}",
                        pattern=None,
                        validation_result=None,
                    )

                    results.append(
                        RecognizerResult(
                            entity_type="MEDICAL",
                            start=ent.start_char,
                            end=ent.end_char,
                            score=score,
                            analysis_explanation=explanation,
                            recognition_metadata={
                                "recognizer_name": self.name,
                                "scispacy_label": ent.label_,
                            },
                        )
                    )

        # Process with anatomy model if available
        if _nlp_anatomy is not None:
            doc = _nlp_anatomy(text)
            for ent in doc.ents:
                if ent.label_ in self.SCISPACY_TO_MEDICAL:
                    # Check if we already have this span from the other model
                    already_found = any(
                        r.start == ent.start_char and r.end == ent.end_char
                        for r in results
                    )
                    if not already_found:
                        score = 0.80  # Slightly lower for anatomy model

                        explanation = AnalysisExplanation(
                            recognizer=self.name,
                            original_score=score,
                            pattern_name=f"scispacy_anatomy_{ent.label_.lower()}",
                            pattern=None,
                            validation_result=None,
                        )

                        results.append(
                            RecognizerResult(
                                entity_type="MEDICAL",
                                start=ent.start_char,
                                end=ent.end_char,
                                score=score,
                                analysis_explanation=explanation,
                                recognition_metadata={
                                    "recognizer_name": self.name,
                                    "scispacy_label": ent.label_,
                                },
                            )
                        )

        return results


def get_medical_recognizer() -> Optional[ScispaCyMedicalRecognizer]:
    """Get a medical recognizer instance if scispaCy is available."""
    _load_scispacy_models()
    if SCISPACY_AVAILABLE:
        return ScispaCyMedicalRecognizer()
    return None


def is_scispacy_available() -> bool:
    """Check if scispaCy and required models are available."""
    _load_scispacy_models()
    return SCISPACY_AVAILABLE
