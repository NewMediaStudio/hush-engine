"""Google Gemini API client for PII detection benchmarking."""

import time


class GeminiClient:
    """Wrapper around Google Generative AI SDK for PII detection benchmarking."""

    def __init__(self):
        import google.generativeai as genai
        import os
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable")
        genai.configure(api_key=api_key)
        self._genai = genai
        self._available = True

    def is_available(self) -> bool:
        return self._available

    def generate(self, model: str, prompt: str, timeout: int = 120) -> dict:
        """Run PII detection inference via Gemini API.

        Returns dict matching Ollama response format:
            response: str - generated text
            total_duration: int - nanoseconds
            prompt_eval_count: int - input tokens
            eval_count: int - output tokens
        """
        start = time.perf_counter()

        model_obj = self._genai.GenerativeModel(
            model,
            generation_config=self._genai.GenerationConfig(
                temperature=0.0,
                max_output_tokens=4096,
            ),
        )
        result = model_obj.generate_content(prompt)

        duration_ns = int((time.perf_counter() - start) * 1e9)

        response_text = result.text if result.text else ""

        # Extract token counts from usage metadata
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
