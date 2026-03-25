"""Google Gemini API client for PII detection benchmarking."""

import time


class GeminiClient:
    """Wrapper around Google GenAI SDK for PII detection benchmarking."""

    def __init__(self):
        from google import genai
        import os
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable")
        self._client = genai.Client(api_key=api_key)
        self._available = True

    def is_available(self) -> bool:
        return self._available

    def generate(self, model: str, prompt: str, timeout: int = 120) -> dict:
        """Run PII detection inference via Gemini API.

        Returns dict matching Ollama response format.
        """
        start = time.perf_counter()

        from google.genai import types
        result = self._client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=4096,
            ),
        )

        duration_ns = int((time.perf_counter() - start) * 1e9)

        response_text = result.text if result.text else ""

        input_tokens = 0
        output_tokens = 0
        if hasattr(result, "usage_metadata") and result.usage_metadata:
            input_tokens = getattr(result.usage_metadata, "prompt_token_count", 0) or 0
            output_tokens = getattr(result.usage_metadata, "candidates_token_count", 0) or 0

        return {
            "response": response_text,
            "total_duration": duration_ns,
            "prompt_eval_count": input_tokens,
            "eval_count": output_tokens,
        }
