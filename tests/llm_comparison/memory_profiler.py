"""Memory profiling for Hush Engine and Ollama models."""

import subprocess
import tracemalloc


class HushMemoryProfiler:
    """Measure Hush Engine memory via tracemalloc."""

    def __init__(self):
        self._started = False

    def start(self):
        if not self._started:
            tracemalloc.start()
            self._started = True

    def get_peak_mb(self) -> float:
        if self._started:
            _, peak = tracemalloc.get_traced_memory()
            return round(peak / (1024 * 1024), 1)
        return 0.0

    def stop(self):
        if self._started:
            tracemalloc.stop()
            self._started = False


def get_ollama_model_memory_mb(model_name: str = None) -> float | None:
    """Get memory usage of currently loaded Ollama model(s).

    Parses `ollama ps` output for model memory info.
    Returns size in MB, or None if unavailable.
    """
    try:
        result = subprocess.run(
            ["ollama", "ps"], capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return None

        for line in result.stdout.strip().split("\n")[1:]:  # Skip header
            parts = line.split()
            if len(parts) >= 2:
                if model_name and not any(model_name in p for p in parts):
                    continue
                # Find the SIZE column (e.g., "6.7 GB", "1.3 GB")
                for i, part in enumerate(parts):
                    if part in ("GB", "MB") and i > 0:
                        try:
                            size = float(parts[i - 1])
                            if part == "GB":
                                return round(size * 1024, 1)
                            return round(size, 1)
                        except ValueError:
                            continue
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def get_ollama_model_disk_size_mb(client, model_id: str) -> float | None:
    """Get model disk size from Ollama API."""
    try:
        info = client.model_info(model_id)
        # Size in bytes from model info
        size_bytes = info.get("size", 0)
        if size_bytes:
            return round(size_bytes / (1024 * 1024), 1)
        # Try modelinfo.general.parameter_count as fallback
        details = info.get("details", {})
        if details.get("parameter_size"):
            param_str = details["parameter_size"]  # e.g., "7B", "70B"
            return None  # Can't reliably convert to disk size
        return None
    except Exception:
        return None
