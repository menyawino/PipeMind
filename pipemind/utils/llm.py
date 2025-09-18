from __future__ import annotations
from typing import List, Dict, Optional
import os
import re

import httpx  # httpx already a dependency; used for Ollama/local OpenAI-compatible endpoints

try:  # pragma: no cover - import guarded for minimal envs
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


def chat(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.2,
) -> str:
    """Send a chat to an LLM provider (OpenAI-compatible or Ollama) and return assistant text.

    Provider selection rules (in order):
    1) Explicit env PIPEMIND_LLM_PROVIDER = 'ollama' or 'openai'
    2) If base_url contains '11434' or 'ollama' -> ollama
    3) Else default 'openai'

    OpenAI:
      - model: env PIPEMIND_LLM_MODEL / OPENAI_MODEL / default 'gpt-4o-mini'
      - api_key: OPENAI_API_KEY / OPENAI_API / OPENAI_KEY or key file

    Ollama:
      - model default: env PIPEMIND_LLM_MODEL / OLLAMA_MODEL / 'llama3'
      - base_url default: OLLAMA_HOST or http://localhost:11434
      - no API key required
    """
    provider = os.getenv("PIPEMIND_LLM_PROVIDER")
    # preliminary heuristics for provider detection if not explicitly set
    heuristic_base = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OLLAMA_HOST")
    if not provider:
        if heuristic_base and re.search(r"(11434|ollama)", heuristic_base, re.IGNORECASE):
            provider = "ollama"
        else:
            provider = "openai"

    # Normalize provider string
    provider = provider.lower().strip()

    # Gather model early (some defaults depend on provider)
    env_model = os.getenv("PIPEMIND_LLM_MODEL") or os.getenv("OPENAI_MODEL") or os.getenv("OLLAMA_MODEL")
    if model is None:
        if env_model:
            model = env_model
        else:
            model = "llama3" if provider == "ollama" else "gpt-4o-mini"

    if provider == "ollama":
        # Determine base URL (allow overriding via base_url arg)
        base_url = base_url or os.getenv("OLLAMA_HOST") or os.getenv("OPENAI_BASE_URL") or "http://localhost:11434"
        endpoint = base_url.rstrip("/") + "/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        try:
            resp = httpx.post(endpoint, json=payload, timeout=60)
            resp.raise_for_status()
        except Exception as e:  # pragma: no cover - network errors environment-specific
            raise RuntimeError(f"Ollama request failed: {e}") from e
        data = resp.json()
        # Ollama returns either message.content (chat) or 'response' (generate). Prefer message.content.
        return (
            (data.get("message") or {}).get("content")
            or data.get("response")
            or ""
        )

    # OpenAI / compatible path
    if OpenAI is None:
        raise RuntimeError("openai package not installed; please install optional dependency or set PIPEMIND_LLM_PROVIDER=ollama")

    api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API") or os.getenv("OPENAI_KEY")
    base_url = base_url or os.getenv("OPENAI_BASE_URL")

    # Fallback: if no API key in env/param, try reading from a key file.
    if not api_key:
        key_file = os.getenv("PIPEMIND_OPENAI_KEY_FILE")
        if key_file and os.path.exists(key_file):
            try:
                with open(key_file, "r", encoding="utf-8") as f:
                    api_key = f.read().strip()
            except Exception:  # pragma: no cover
                pass
        elif os.path.exists("openai.api"):
            try:  # pragma: no cover
                with open("openai.api", "r", encoding="utf-8") as f:
                    api_key = f.read().strip()
            except Exception:
                pass

    if not api_key:
        raise RuntimeError(
            "Missing API key: set OPENAI_API_KEY or provide a key file (PIPEMIND_OPENAI_KEY_FILE or ./openai.api)"
        )
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
    choice = resp.choices[0]
    return choice.message.content or ""
