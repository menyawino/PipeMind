from __future__ import annotations
from typing import List, Dict, Optional
import os

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


def chat(messages: List[Dict[str, str]], model: Optional[str] = None, api_key: Optional[str] = None, base_url: Optional[str] = None, temperature: float = 0.2) -> str:
    """Send a chat to OpenAI-compatible API and return the assistant text.

    - model: defaults to env PIPEMIND_LLM_MODEL or 'gpt-4o-mini'
    - api_key: from OPENAI_API_KEY
    - base_url: override with OPENAI_BASE_URL for compatible endpoints
    """
    if OpenAI is None:
        raise RuntimeError("openai package not installed; please install optional dependency.")
    model = model or os.getenv("PIPEMIND_LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
    api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API") or os.getenv("OPENAI_KEY")
    base_url = base_url or os.getenv("OPENAI_BASE_URL")

    # Fallback: if no API key in env/param, try reading from a key file.
    # 1) Respect PIPEMIND_OPENAI_KEY_FILE if provided
    # 2) Otherwise look for ./openai.api in the current working directory
    if not api_key:
        key_file = os.getenv("PIPEMIND_OPENAI_KEY_FILE")
        if key_file and os.path.exists(key_file):
            try:
                with open(key_file, "r", encoding="utf-8") as f:
                    api_key = f.read().strip()
            except Exception:
                pass
        elif os.path.exists("openai.api"):
            try:
                with open("openai.api", "r", encoding="utf-8") as f:
                    api_key = f.read().strip()
            except Exception:
                pass

    if not api_key:
        raise RuntimeError("Missing API key: set OPENAI_API_KEY or provide a key file (PIPEMIND_OPENAI_KEY_FILE or ./openai.api)")
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
    choice = resp.choices[0]
    return choice.message.content or ""
