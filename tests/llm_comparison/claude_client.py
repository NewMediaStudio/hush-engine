"""Claude API client for PII detection benchmarking."""

import time


class ClaudeClient:
    """Wrapper around Anthropic SDK for PII detection benchmarking."""

    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var
        self._available = True

    def is_available(self) -> bool:
        return self._available

    def generate(self, model: str, prompt: str, timeout: int = 120) -> dict:
        """Run PII detection inference via Claude API.

        Returns dict matching Ollama response format:
            response: str - generated text
            total_duration: int - nanoseconds
            prompt_eval_count: int - input tokens
            eval_count: int - output tokens
        """
        start = time.perf_counter()

        message = self.client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )

        duration_ns = int((time.perf_counter() - start) * 1e9)

        response_text = ""
        for block in message.content:
            if block.type == "text":
                response_text += block.text

        return {
            "response": response_text,
            "total_duration": duration_ns,
            "prompt_eval_count": message.usage.input_tokens,
            "eval_count": message.usage.output_tokens,
        }
