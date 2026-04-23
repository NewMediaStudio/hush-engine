"""Model metadata registry for LLM comparison benchmark."""

# API pricing based on published rates (Together AI / Groq, as of March 2026)
# Prices are per 1M tokens in USD
MODELS = {
    "hush_engine": {
        "display_name": "Hush Engine v1.11.0",
        "family": "Rule-based + LightGBM",
        "params_b": 0,
        "disk_size_mb": 15,
        "ram_estimate_mb": 200,
        "ollama_tag": None,
        "api_pricing_input": 0,
        "api_pricing_output": 0,
    },
    # OpenAI Privacy Filter (standalone): the Apache-2.0 open-weight token
    # classifier released 2026-04-22. Not an Ollama model — driven through
    # transformers.pipeline() by run_privacy_filter(). Unlike the autoregressive
    # LLMs in this table, it outputs BIOES spans directly (no prompt parsing).
    "openai-privacy-filter": {
        "display_name": "OpenAI Privacy Filter",
        "family": "Token classifier (MoE)",
        "params_b": 1.5,
        "active_params_b": 0.05,
        "disk_size_mb": 3000,  # BF16 safetensors
        "ram_estimate_mb": 3000,
        "ollama_tag": None,
        "backend": "privacy_filter",
        "api_pricing_input": 0,
        "api_pricing_output": 0,
    },
    "llama3.2:1b": {
        "display_name": "Llama 3.2 (1B)",
        "family": "Llama",
        "params_b": 1.0,
        "disk_size_mb": 1300,
        "ram_estimate_mb": 3000,
        "ollama_tag": "llama3.2:1b",
        "api_pricing_input": 0.04,
        "api_pricing_output": 0.04,
    },
    "llama3.2:3b": {
        "display_name": "Llama 3.2 (3B)",
        "family": "Llama",
        "params_b": 3.0,
        "disk_size_mb": 3400,
        "ram_estimate_mb": 6000,
        "ollama_tag": "llama3.2:3b",
        "api_pricing_input": 0.06,
        "api_pricing_output": 0.06,
    },
    "mistral:7b": {
        "display_name": "Mistral 7B",
        "family": "Mistral",
        "params_b": 7.0,
        "disk_size_mb": 7200,
        "ram_estimate_mb": 10000,
        "ollama_tag": "mistral:7b",
        "api_pricing_input": 0.20,
        "api_pricing_output": 0.20,
    },
    "llama3.1:8b": {
        "display_name": "Llama 3.1 (8B)",
        "family": "Llama",
        "params_b": 8.0,
        "disk_size_mb": 8500,
        "ram_estimate_mb": 12000,
        "ollama_tag": "llama3.1:8b",
        "api_pricing_input": 0.18,
        "api_pricing_output": 0.18,
    },
    "phi3:medium": {
        "display_name": "Phi-3 Medium (14B)",
        "family": "Phi",
        "params_b": 14.0,
        "disk_size_mb": 14000,
        "ram_estimate_mb": 20000,
        "ollama_tag": "phi3:medium",
        "api_pricing_input": 0.30,
        "api_pricing_output": 0.30,
    },
    "llama3.1:70b": {
        "display_name": "Llama 3.1 (70B)",
        "family": "Llama",
        "params_b": 70.0,
        "disk_size_mb": 70000,
        "ram_estimate_mb": 48000,
        "ollama_tag": "llama3.1:70b",
        "api_pricing_input": 0.88,
        "api_pricing_output": 0.88,
    },
    "llama3.3:latest": {
        "display_name": "Llama 3.3 (70B)",
        "family": "Llama",
        "params_b": 70.0,
        "disk_size_mb": 43000,
        "ram_estimate_mb": 48000,
        "ollama_tag": "llama3.3:latest",
        "api_pricing_input": 0.88,
        "api_pricing_output": 0.88,
    },
    "gemma2:latest": {
        "display_name": "Gemma 2 (9B)",
        "family": "Gemma",
        "params_b": 9.0,
        "disk_size_mb": 5400,
        "ram_estimate_mb": 10000,
        "ollama_tag": "gemma2:latest",
        "api_pricing_input": 0.20,
        "api_pricing_output": 0.20,
    },
    "qwen2.5:latest": {
        "display_name": "Qwen 2.5 (7B)",
        "family": "Qwen",
        "params_b": 7.0,
        "disk_size_mb": 4700,
        "ram_estimate_mb": 8000,
        "ollama_tag": "qwen2.5:latest",
        "api_pricing_input": 0.20,
        "api_pricing_output": 0.20,
    },
    "phi4:latest": {
        "display_name": "Phi-4 (14B)",
        "family": "Phi",
        "params_b": 14.0,
        "disk_size_mb": 9100,
        "ram_estimate_mb": 16000,
        "ollama_tag": "phi4:latest",
        "api_pricing_input": 0.30,
        "api_pricing_output": 0.30,
    },
    "qwen3-coder:30b": {
        "display_name": "Qwen3 Coder (30B)",
        "family": "Qwen",
        "params_b": 30.0,
        "disk_size_mb": 18000,
        "ram_estimate_mb": 24000,
        "ollama_tag": "qwen3-coder:30b",
        "api_pricing_input": 0.50,
        "api_pricing_output": 0.50,
    },
    # Claude models (API-based)
    "claude-haiku-4-5": {
        "display_name": "Claude Haiku 4.5",
        "family": "Claude",
        "params_b": None,
        "disk_size_mb": None,
        "ram_estimate_mb": None,
        "ollama_tag": None,
        "claude_model_id": "claude-haiku-4-5-20251001",
        "api_pricing_input": 0.80,
        "api_pricing_output": 4.00,
    },
    "claude-sonnet-4-6": {
        "display_name": "Claude Sonnet 4.6",
        "family": "Claude",
        "params_b": None,
        "disk_size_mb": None,
        "ram_estimate_mb": None,
        "ollama_tag": None,
        "claude_model_id": "claude-sonnet-4-6",
        "api_pricing_input": 3.00,
        "api_pricing_output": 15.00,
    },
    # Gemini models (API-based, free tier available)
    "gemini-2.5-flash": {
        "display_name": "Gemini 2.5 Flash",
        "family": "Gemini",
        "params_b": None,
        "disk_size_mb": None,
        "ram_estimate_mb": None,
        "ollama_tag": None,
        "gemini_model_id": "gemini-2.5-flash",
        "api_pricing_input": 0.15,
        "api_pricing_output": 0.60,
    },
    "gemini-2.5-pro": {
        "display_name": "Gemini 2.5 Pro",
        "family": "Gemini",
        "params_b": None,
        "disk_size_mb": None,
        "ram_estimate_mb": None,
        "ollama_tag": None,
        "gemini_model_id": "gemini-2.5-pro",
        "api_pricing_input": 1.25,
        "api_pricing_output": 10.00,
    },
}


def get_model_ids(include_hush: bool = True) -> list:
    """Return list of model IDs."""
    ids = list(MODELS.keys())
    if not include_hush:
        ids = [m for m in ids if m != "hush_engine"]
    return ids


def get_llm_model_ids() -> list:
    """Return only LLM model IDs (exclude hush_engine)."""
    return get_model_ids(include_hush=False)


def get_ollama_model_ids() -> list:
    """Return only Ollama-based model IDs."""
    return [m for m in get_llm_model_ids() if MODELS[m].get("ollama_tag")]


def get_claude_model_ids() -> list:
    """Return only Claude API model IDs."""
    return [m for m in get_llm_model_ids() if MODELS[m].get("claude_model_id")]


def is_claude_model(model_id: str) -> bool:
    return bool(MODELS.get(model_id, {}).get("claude_model_id"))


def get_gemini_model_ids() -> list:
    """Return only Gemini API model IDs."""
    return [m for m in get_llm_model_ids() if MODELS[m].get("gemini_model_id")]


def is_gemini_model(model_id: str) -> bool:
    return bool(MODELS.get(model_id, {}).get("gemini_model_id"))


def estimate_cost_per_1k_docs(model_id: str, avg_input_tokens: float, avg_output_tokens: float) -> float:
    """Estimate cost per 1000 documents based on token usage and API pricing.

    Args:
        model_id: Model identifier
        avg_input_tokens: Average input tokens per document
        avg_output_tokens: Average output tokens per document

    Returns:
        Estimated cost in USD per 1000 documents
    """
    model = MODELS.get(model_id, {})
    input_price = model.get("api_pricing_input", 0)
    output_price = model.get("api_pricing_output", 0)
    cost = (avg_input_tokens * input_price + avg_output_tokens * output_price) / 1_000_000 * 1000
    return round(cost, 4)
