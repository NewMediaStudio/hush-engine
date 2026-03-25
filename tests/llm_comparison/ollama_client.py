"""Thin wrapper around the Ollama REST API."""

import json
import urllib.request
import urllib.error


class OllamaClient:
    """Client for Ollama local LLM inference."""

    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url.rstrip("/")

    def is_available(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5):
                return True
        except (urllib.error.URLError, OSError):
            return False

    def list_models(self) -> list:
        """Return list of installed models with metadata."""
        data = self._get("/api/tags")
        return data.get("models", [])

    def model_info(self, model: str) -> dict:
        """Get model details (parameters, size, quantization)."""
        return self._post("/api/show", {"name": model})

    def pull_model(self, model: str) -> None:
        """Download a model if not present."""
        self._post("/api/pull", {"name": model, "stream": False}, timeout=600)

    def generate(self, model: str, prompt: str, timeout: int = 180) -> dict:
        """Run inference. Returns response text + timing/token metadata.

        Returns dict with keys:
            response: str - generated text
            total_duration: int - nanoseconds
            prompt_eval_count: int - input tokens
            eval_count: int - output tokens
            eval_duration: int - generation time in nanoseconds
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,  # Deterministic for reproducibility
                "num_predict": 4096,  # Enough for PII JSON output
            },
        }
        return self._post("/api/generate", payload, timeout=timeout)

    def running_models(self) -> list:
        """Return currently loaded models with memory usage."""
        data = self._get("/api/ps")
        return data.get("models", [])

    def _get(self, path: str, timeout: int = 10) -> dict:
        req = urllib.request.Request(f"{self.base_url}{path}")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    def _post(self, path: str, payload: dict, timeout: int = 30) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
