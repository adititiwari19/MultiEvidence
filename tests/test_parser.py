import json
import sys
from types import SimpleNamespace

import pytest

from app.services.openrouter_client import OpenRouterOutputError, parse_model_json, verify_with_openrouter


VALID = {"relation": "SUPPORTS", "confidence": 0.82, "observable_facts": ["Visible water."], "evidence": [{"frame_id": "e_0", "observation": "Water visible."}], "explanation": "Visual evidence supports the claim.", "suggested_new_text": "Water is visible.", "limitations": []}


def test_strict_parser():
    assert parse_model_json(json.dumps(VALID)).relation.value == "SUPPORTS"
    with pytest.raises(ValueError):
        parse_model_json(json.dumps({**VALID, "relation": "IRRELEVANT"}))
    with pytest.raises(ValueError):
        parse_model_json(json.dumps({**VALID, "confidence": 1.4}))
    with pytest.raises(ValueError):
        parse_model_json("```json\n{}\n```")


def test_openrouter_repairs_once(monkeypatch):
    calls = []
    always_bad = False

    def create(**kwargs):
        calls.append(kwargs)
        raw = "invalid" if always_bad or len(calls) == 1 else json.dumps(VALID)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=raw))])

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    result = verify_with_openrouter("check", [("e_0", "data:image/jpeg;base64,AA==")])
    assert result.relation.value == "SUPPORTS"
    assert len(calls) == 2
    calls.clear()
    always_bad = True
    with pytest.raises(OpenRouterOutputError):
        verify_with_openrouter("check", [("e_0", "data:image/jpeg;base64,AA==")])
