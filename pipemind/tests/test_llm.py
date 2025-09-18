from __future__ import annotations
import os
from pipemind.utils import llm

class DummyResp:
    def __init__(self, json_data):
        self._json = json_data
    def raise_for_status(self):
        return None
    def json(self):
        return self._json


def test_ollama_path(monkeypatch):  # type: ignore
    calls = {}
    def fake_post(url, json, timeout):  # noqa: A002 - shadow builtin acceptable in test
        calls['url'] = url
        calls['payload'] = json
        return DummyResp({"message": {"content": "hi there"}})
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
