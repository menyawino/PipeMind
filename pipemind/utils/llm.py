from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import os
import re

import httpx  # httpx already a dependency; used for Ollama/local OpenAI-compatible endpoints

try:  # pragma: no cover - import guarded for minimal envs
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


def _build_prompt_from_messages(messages: List[Dict[str, str]]) -> str:
    """Flatten chat style messages into a single prompt suitable for Ollama /api/generate fallback.

    We annotate roles minimally so the model retains conversational context.
    """
    parts: List[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if not content:
            continue
        if role == "system":
            parts.append(f"[SYSTEM] {content}")
        elif role == "assistant":
            parts.append(f"[ASSISTANT] {content}")
        else:
            parts.append(f"[USER] {content}")
    parts.append("[ASSISTANT] ")  # hint model to continue as assistant
    return "\n".join(parts)


def _ollama_request(
    messages: List[Dict[str, str]],
    model: str,
    base_url: str,
    temperature: float,
    json_mode: bool = False,
) -> Tuple[str, Dict[str, str]]:
    """Attempt Ollama /api/chat first; on 404/405 or unsupported shape, fallback to /api/generate.

    Returns (text, meta) where meta contains 'endpoint' used and 'fallback' flag.
    """
    chat_endpoint = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if json_mode:
        # Hint Ollama to return valid JSON
        payload["format"] = "json"
    fallback_used = False
    try:
        resp = httpx.post(chat_endpoint, json=payload, timeout=60)
    except Exception as e:  # pragma: no cover - network conditions
        raise RuntimeError(f"Ollama request failed (chat attempt): {e}") from e

    if resp.status_code in {404, 405}:
        # Old server or method mismatch; fallback
        fallback_used = True
    else:
        # If other 2xx we parse; if >=400 we may inspect JSON for missing model
        if 200 <= resp.status_code < 300:
            data = resp.json()
            content = (data.get("message") or {}).get("content") or data.get("response") or ""
            if content:
                return content, {"endpoint": chat_endpoint, "fallback": str(fallback_used)}
            # If empty, still allow fallback attempt to generate for safety
            fallback_used = True
        else:
            # Non-2xx; if model not found we raise a clearer error
            try:
                data = resp.json()
            except Exception:  # pragma: no cover
                data = {}
            err_text = data.get("error") or resp.text
            if "model" in err_text.lower() and "not found" in err_text.lower():
                raise RuntimeError(
                    f"Ollama model '{model}' not found. Pull it first: `ollama pull {model}`"
                )
            # For other errors, attempt fallback generate anyway
            fallback_used = True

    # Fallback path using /api/generate
    generate_endpoint = base_url.rstrip("/") + "/api/generate"
    gen_payload = {
        "model": model,
        "prompt": _build_prompt_from_messages(messages),
        "stream": False,
        "options": {"temperature": temperature},
    }
    if json_mode:
        gen_payload["format"] = "json"
    try:
        gresp = httpx.post(generate_endpoint, json=gen_payload, timeout=60)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Ollama fallback generate failed: {e}") from e
    if not (200 <= gresp.status_code < 300):
        try:
            data = gresp.json()
        except Exception:  # pragma: no cover
            data = {}
        err_text = data.get("error") or gresp.text
        if "model" in err_text.lower() and "not found" in err_text.lower():
            raise RuntimeError(
                f"Ollama model '{model}' not found. Pull it: `ollama pull {model}` (fallback generate)."
            )
        raise RuntimeError(f"Ollama fallback generate error ({gresp.status_code}): {err_text}")
    data = gresp.json()
    return data.get("response") or (data.get("message") or {}).get("content") or "", {"endpoint": generate_endpoint, "fallback": str(fallback_used)}


def chat(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.2,
    return_meta: bool = False,
    provider: Optional[str] = None,
    json_mode: bool = False,
) -> str | Tuple[str, Dict[str, str]]:
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
    # Provider resolution: explicit argument > env > heuristic
    provider = provider or os.getenv("PIPEMIND_LLM_PROVIDER")
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
        base_url = base_url or os.getenv("OLLAMA_HOST") or os.getenv("OPENAI_BASE_URL") or "http://localhost:11434"
        text, meta = _ollama_request(
            messages=messages,
            model=model,
            base_url=base_url,
            temperature=temperature,
            json_mode=json_mode,
        )  # type: ignore[arg-type]
        return (text, meta) if return_meta else text

    # OpenAI / compatible path
    if OpenAI is None:
        raise RuntimeError("openai package not installed; install 'openai' or set provider='ollama'.")

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
        # Helpful hint if user likely intended ollama
        hint = " (did you mean to use Ollama? pass provider='ollama' or export PIPEMIND_LLM_PROVIDER=ollama)"
        raise RuntimeError(
            "Missing API key: set OPENAI_API_KEY or provide a key file (PIPEMIND_OPENAI_KEY_FILE or ./openai.api)" + hint
        )
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    kwargs = {"model": model, "messages": messages, "temperature": temperature}
    # Prefer OpenAI JSON guidance if requested
    if json_mode:
        try:
            kwargs["response_format"] = {"type": "json_object"}  # type: ignore[assignment]
        except Exception:
            # If SDK doesn't support response_format, fall back silently
            pass
    resp = client.chat.completions.create(**kwargs)  # type: ignore[arg-type]
    choice = resp.choices[0]
    content = choice.message.content or ""
    return (content, {"endpoint": base_url or "https://api.openai.com", "fallback": "False"}) if return_meta else content
