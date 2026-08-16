from __future__ import annotations
import os
from pipemind.utils import llm
from typing import cast, Tuple, Dict

class DummyResp:
    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code
        self.text = ""
    def raise_for_status(self):
        return None
    def json(self):
        return self._json


def test_ollama_path(monkeypatch):  # type: ignore
    calls = {}
    def fake_post(url, json, timeout):  # noqa: A002 - shadow builtin acceptable in test
        calls['url'] = url
        calls['payload'] = json
        return DummyResp({"message": {"content": "hi there"}}, status_code=200)
    monkeypatch.setenv("PIPEMIND_LLM_PROVIDER", "ollama")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    monkeypatch.setenv("PIPEMIND_LLM_MODEL", "llama3")
    monkeypatch.setenv("OPENAI_MODEL", "")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3")
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    monkeypatch.setenv("PIPEMIND_OPENAI_KEY_FILE", "")
    monkeypatch.setenv("PIPEMIND_LLM_PROVIDER", "ollama")

    monkeypatch.setattr(llm, "httpx", type("H", (), {"post": staticmethod(fake_post)}))
    out = llm.chat([{"role": "user", "content": "hello"}], model="llama3")
    assert out == "hi there"
    assert calls['url'].endswith('/api/chat')
    assert calls['payload']['model'] == 'llama3'


def test_ollama_fallback_generate(monkeypatch):  # type: ignore
    calls = {"posts": []}

    class R:
        def __init__(self, status_code, data):
            self.status_code = status_code
            self._data = data
            self.text = data.get("error", "")
        def json(self):
            return self._data

    def fake_post(url, json, timeout):  # noqa: A002
        calls['posts'].append(url)
        # Simulate 405 for /api/chat then success for /api/generate
        if url.endswith('/api/chat'):
            return R(405, {"error": "method not allowed"})
        return R(200, {"response": "fallback ok"})

    monkeypatch.setenv("PIPEMIND_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    monkeypatch.setenv("PIPEMIND_LLM_MODEL", "phi")
    monkeypatch.setattr(llm, "httpx", type("H", (), {"post": staticmethod(fake_post)}))

    res = llm.chat([{"role": "user", "content": "hi"}], model="phi", return_meta=True)
    # chat returns Tuple[str, Dict[str,str]] when return_meta=True
    out, meta = cast(Tuple[str, Dict[str,str]], res)
    assert out == "fallback ok"
    assert any(u.endswith('/api/chat') for u in calls['posts'])
    assert any(u.endswith('/api/generate') for u in calls['posts'])
    assert meta['fallback'] == 'True'
