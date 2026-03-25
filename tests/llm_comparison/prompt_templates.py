"""PII detection prompt templates for LLM benchmarking."""

# Entity types that match the benchmark's ground truth taxonomy
ENTITY_TYPES = [
    "PERSON", "EMAIL", "PHONE", "ADDRESS", "DATE_TIME",
    "CREDIT_CARD", "NATIONAL_ID", "URL", "IP_ADDRESS",
    "COMPANY", "FINANCIAL", "MEDICAL", "AGE", "GENDER",
    "CREDENTIAL", "COORDINATES", "VEHICLE", "NETWORK",
    "BIOMETRIC", "ID",
]

ZERO_SHOT_PROMPT = """You are a PII (Personally Identifiable Information) detection system. Analyze the following text and identify all PII entities.

For each PII entity found, return a JSON object with:
- "text": the exact text span containing the PII
- "type": one of the following entity types: {entity_types}

Return ONLY a JSON array of objects. No explanation, no markdown formatting.

Text to analyze:
\"\"\"
{text}
\"\"\"

JSON output:"""

FEW_SHOT_PROMPT = """You are a PII (Personally Identifiable Information) detection system. Analyze the following text and identify all PII entities.

For each PII entity found, return a JSON object with:
- "text": the exact text span containing the PII
- "type": one of the following entity types: {entity_types}

Return ONLY a JSON array of objects. No explanation, no markdown formatting.

Example 1:
Text: "Contact John Smith at john.smith@email.com or call 555-123-4567."
Output: [{{"text": "John Smith", "type": "PERSON"}}, {{"text": "john.smith@email.com", "type": "EMAIL"}}, {{"text": "555-123-4567", "type": "PHONE"}}]

Example 2:
Text: "Patient DOB 03/15/1990, diagnosed with Type 2 diabetes. SSN: 123-45-6789."
Output: [{{"text": "03/15/1990", "type": "DATE_TIME"}}, {{"text": "Type 2 diabetes", "type": "MEDICAL"}}, {{"text": "123-45-6789", "type": "NATIONAL_ID"}}]

Text to analyze:
\"\"\"
{text}
\"\"\"

JSON output:"""


def build_prompt(text: str, few_shot: bool = False) -> str:
    """Build a PII detection prompt for an LLM.

    Args:
        text: The text to analyze for PII
        few_shot: If True, use few-shot prompt with examples

    Returns:
        Formatted prompt string
    """
    template = FEW_SHOT_PROMPT if few_shot else ZERO_SHOT_PROMPT
    entity_types_str = ", ".join(ENTITY_TYPES)
    return template.format(text=text, entity_types=entity_types_str)
