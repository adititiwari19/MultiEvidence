import json
import sys
from types import SimpleNamespace

import pytest

from app.models.schemas import SearchRequest
from app.services.retrieval.query_builder import build_gdelt_text_query, build_gdelt_visual_query, build_queries, parse_query_plan


def request():
    return SearchRequest.model_validate({"report_id": "r", "topic": "North Kivu flooding", "claim": {"claim_id": "c", "text": "The road to Goma remains open."}})


def test_query_plan_parser():
    raw = json.dumps({"queries": ["Goma road flooding", "Goma road closure"], "entities": ["road"], "locations": ["Goma"], "event_terms": ["flooding"]})
    assert parse_query_plan(raw).locations == ["Goma"]
    five = json.dumps({"queries": ["one", "two", "three", "four", "five"], "entities": [], "locations": [], "event_terms": []})
    assert parse_query_plan(five).queries == ["one", "two", "three"]
    with pytest.raises(ValueError):
        parse_query_plan('{"queries": ["one"], "entities": [], "locations": [], "event_terms": []}')


def test_query_generation_and_cache(monkeypatch, tmp_path):
    monkeypatch.setattr("app.services.retrieval.cache.CACHE_DIR", tmp_path)
    monkeypatch.setattr("app.services.retrieval.query_builder.read_json", lambda group, key: None)
    saved = []
    monkeypatch.setattr("app.services.retrieval.query_builder.write_json", lambda group, key, value: saved.append(value))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        raw = json.dumps({"queries": ["Goma road flooding", "Goma road closure"], "entities": ["road"], "locations": ["Goma"], "event_terms": ["flooding"]})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=raw))])

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    plan = build_queries(request())
    assert plan.queries[0] == 'Goma (road OR access OR closure OR blocked OR transport)'
    assert plan.queries[1].startswith('Goma (imagetag:"road" OR imagewebtag:"road"')
    assert len(calls) == 1
    assert saved[0]["locations"] == ["Goma"]


def test_cached_query_plan_becomes_two_structured_queries(monkeypatch):
    monkeypatch.setattr("app.services.retrieval.query_builder.read_json", lambda group, key: {"queries": ["one", "two", "three", "four", "five"], "entities": [], "locations": [], "event_terms": []})
    queries = build_queries(request()).queries
    assert len(queries) == 2
    assert 'Goma' in queries[0]
    assert "imagewebtag:" in queries[1]


def test_structured_queries_quote_locations_and_group_synonyms():
    plan = parse_query_plan(json.dumps({"queries": ["a", "b"], "entities": ["Goma"], "locations": ["Goma", "North Kivu"], "event_terms": ["flood"]}))
    text = build_gdelt_text_query(request(), plan)
    visual = build_gdelt_visual_query(request(), plan)
    assert text.startswith('Goma "North Kivu"')
    assert '"Goma"' not in text and '"Goma"' not in visual
    assert "(road OR access OR closure OR blocked OR transport)" in text
    assert 'imagetag:"flood"' in visual and 'imagewebtag:"flood"' in visual
    assert '"Goma"' not in visual


def test_gdelt_operator_values_are_quoted_but_single_place_is_not():
    request = SearchRequest.model_validate({"report_id": "r", "topic": "Spokane fire", "claim": {"claim_id": "c", "text": "A fire burned near Spokane."}})
    plan = parse_query_plan(json.dumps({"queries": ["a", "b"], "entities": [], "locations": ["Spokane"], "event_terms": ["fire"]}))
    visual = build_gdelt_visual_query(request, plan)
    assert visual.startswith("Spokane (")
    assert 'imagetag:"fire"' in visual
    assert 'imagewebtag:"fire"' in visual
    assert '"Spokane"' not in visual
