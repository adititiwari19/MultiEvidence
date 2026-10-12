import logging
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlsplit

import pytest
import requests

from app.services.retrieval import cache, news_search
from app.services.retrieval.news_search import GDELT_URL, canonical_url, dedupe_articles, gdelt_request_scope, parse_gdelt_images, parse_gdelt_results, search_news, search_status


PAYLOAD = {"articles": [
    {"url": "https://example.org/news?utm_source=x", "title": "Road update", "domain": "example.org", "seendate": "20260317T120000Z"},
    {"url": "https://example.org/news", "title": "Duplicate", "domain": "example.org", "seendate": "bad"},
]}


class Response:
    def __init__(self, url, status_code=200, text="", payload=None):
        self.url = url
        self.status_code = status_code
        self.text = text
        self.payload = payload

    def json(self):
        if self.payload is None:
            raise ValueError("not JSON")
        return self.payload


@pytest.fixture(autouse=True)
def isolated_search(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(news_search, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(news_search, "GDELT_SPACING_SECONDS", 10.0)
    monkeypatch.setattr(news_search, "GDELT_BACKOFF_SECONDS", (15.0, 30.0, 60.0))
    monkeypatch.setattr(news_search, "GDELT_MAX_429_RETRIES", 3)
    clock = [100.0]
    waits = []
    monkeypatch.setattr(news_search, "_now", lambda: clock[0])

    def sleep(seconds):
        waits.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(news_search, "_sleep", sleep)
    monkeypatch.setattr(news_search, "_jitter", lambda: 0.0)
    return clock, waits


def response(url, params, **kwargs):
    prepared = requests.Request("GET", url, params=params).prepare().url
    return Response(prepared, payload=PAYLOAD)


def test_gdelt_only_and_request_parameters(monkeypatch, caplog):
    calls = []

    def get(url, params, **kwargs):
        calls.append(url)
        prepared = requests.Request("GET", url, params=params).prepare().url
        return Response(prepared, payload=PAYLOAD)

    monkeypatch.setattr(news_search.requests, "get", get)
    with caplog.at_level(logging.INFO):
        results = search_news("Goma flood road")
    assert [item.url for item in results] == ["https://example.org/news"]
    assert calls == [GDELT_URL]
    assert "source=GDELT" in caplog.text and "results=2" in caplog.text


def test_gdelt_query_encoding_and_deduplication(monkeypatch):
    parsed = parse_gdelt_results(PAYLOAD)
    assert len(parsed) == 2
    assert len(dedupe_articles(parsed)) == 1
    urls = []

    def get(url, params, **kwargs):
        prepared = requests.Request("GET", url, params=params).prepare().url
        urls.append(prepared)
        return Response(prepared, payload=PAYLOAD)

    monkeypatch.setattr(news_search.requests, "get", get)
    assert len(search_news("Goma flood road")) == 1
    params = parse_qs(urlsplit(urls[0]).query)
    assert params["query"] == ["Goma flood road"]
    assert params["maxrecords"] == ["100"]
    assert params["sort"] == ["datedesc"]
    assert params["timespan"] == ["1week"]
    assert canonical_url(parsed[0].url) == "https://example.org/news"


def test_no_broad_fallback_after_empty_result(monkeypatch, isolated_search):
    clock, waits = isolated_search
    starts = []

    def get(url, params, **kwargs):
        prepared = requests.Request("GET", url, params=params).prepare().url
        if url == GDELT_URL:
            starts.append(clock[0])
            return Response(prepared, payload={"articles": []})
        return Response(prepared, payload={"articles": []})

    monkeypatch.setattr(news_search.requests, "get", get)
    search_news("Goma flood", published_after=datetime(2026, 3, 1, tzinfo=timezone.utc))
    assert len(starts) == 1
    assert waits == []


def test_empty_gdelt_object_is_zero_results(monkeypatch):
    monkeypatch.setattr(news_search.requests, "get", lambda url, params, **kwargs: Response(requests.Request("GET", url, params=params).prepare().url, payload={}))
    with gdelt_request_scope():
        assert search_news("Dubai airport") == []
        assert search_status() == "OK"


def test_ten_second_spacing_across_separate_queries(monkeypatch, isolated_search):
    clock, waits = isolated_search
    starts = []

    def get(url, params, **kwargs):
        prepared = requests.Request("GET", url, params=params).prepare().url
        if url == GDELT_URL:
            starts.append((params["query"], clock[0]))
            return Response(prepared, payload=PAYLOAD)
        return Response(prepared, payload=PAYLOAD)

    monkeypatch.setattr(news_search.requests, "get", get)
    search_news("Goma flood")
    search_news("Goma road")
    assert [query for query, _ in starts] == ["Goma flood", "Goma road"]
    assert starts[1][1] - starts[0][1] >= 10.0
    assert 10.0 in waits


def test_429_backoff_three_retries_then_disables_gdelt(monkeypatch, isolated_search, caplog):
    clock, waits = isolated_search
    gdelt_calls = []

    def get(url, params, **kwargs):
        prepared = requests.Request("GET", url, params=params).prepare().url
        if url == GDELT_URL:
            gdelt_calls.append((params["query"], clock[0]))
            return Response(prepared, status_code=429, text="Please limit requests to one every 5 seconds")
        return Response(prepared, payload=PAYLOAD)

    monkeypatch.setattr(news_search.requests, "get", get)
    with caplog.at_level(logging.WARNING):
        assert search_news("Goma flood road", published_after=datetime(2026, 3, 1, tzinfo=timezone.utc)) == []
    assert len(gdelt_calls) == 4
    assert [round(gdelt_calls[i + 1][1] - gdelt_calls[i][1]) for i in range(3)] == [15, 30, 60]
    assert waits == [15.0, 30.0, 60.0]
    assert "source=GDELT" in caplog.text and "status=429" in caplog.text
    assert "body=Please limit requests to one every 5 seconds" in caplog.text
    retry_lines = [record.getMessage() for record in caplog.records if "reason=HTTP_429" in record.getMessage()]
    assert len(retry_lines) == 4
    for attempt, line in enumerate(retry_lines, 1):
        assert f"attempt={attempt}" in line
        assert "timestamp_utc=" in line and "source=GDELT" in line
        assert "status=429 results=0 rate_limit_wait=" in line
        assert "request_url=https://api.gdeltproject.org/api/v2/doc/doc?" in line


def test_search_cache_avoids_repeat_requests(monkeypatch):
    calls = []

    def get(url, params, **kwargs):
        calls.append(url)
        return response(url, params)

    monkeypatch.setattr(news_search.requests, "get", get)
    cutoff = datetime(2026, 3, 1, tzinfo=timezone.utc)
    first = search_news("Goma flood road", published_after=cutoff)
    second = search_news("Goma flood road", published_after=cutoff)
    assert first == second
    assert calls.count(GDELT_URL) == 1
    assert len(calls) == 1


def test_non_json_gdelt_logs_body_without_broadening(monkeypatch, caplog):
    queries = []

    def get(url, params, **kwargs):
        prepared = requests.Request("GET", url, params=params).prepare().url
        queries.append(params["query"])
        if params["query"] == "Goma flood":
            return Response(prepared, payload=PAYLOAD)
        return Response(prepared, text="<html>not JSON</html>" + "x" * 500)

    monkeypatch.setattr(news_search.requests, "get", get)
    with caplog.at_level(logging.WARNING), gdelt_request_scope():
        results = search_news("Goma flood road")
        assert search_status() == "QUERY_ERROR"
    assert results == []
    assert queries == ["Goma flood road"]
    assert "reason=QUERY_ERROR" in caplog.text
    assert "body=<html>not JSON</html>" + "x" * 500 in caplog.text
    search_news("Goma flood road")
    assert len(queries) == 2  # A malformed 200 response was not cached.


def test_maxrecords_cap_and_precise_cutoff(monkeypatch):
    params_seen = []

    def get(url, params, **kwargs):
        params_seen.append(params)
        return Response(requests.Request("GET", url, params=params).prepare().url, payload=PAYLOAD)

    monkeypatch.setattr(news_search.requests, "get", get)
    search_news('"Goma" (road OR access)', max_results=250, max_records=500, published_after=datetime(2026, 3, 1, tzinfo=timezone.utc))
    assert params_seen[0]["maxrecords"] == 250
    assert params_seen[0]["startdatetime"] == "20260301000000"
    assert "timespan" not in params_seen[0]


def test_429_disables_second_query_in_claim_scope(monkeypatch):
    calls = []

    def get(url, params, **kwargs):
        calls.append(params["query"])
        return Response(requests.Request("GET", url, params=params).prepare().url, status_code=429, text="rate limited")

    monkeypatch.setattr(news_search.requests, "get", get)
    with gdelt_request_scope():
        assert search_news("text query") == []
        assert search_news("visual query") == []
    assert calls == ["text query"] * 4


def test_socialimage_metadata_and_cache(monkeypatch):
    calls = []
    payload = {"articles": [{"url": "https://example.org/a", "title": "Goma road", "domain": "example.org", "socialimage": "https://cdn.example.org/a.jpg", "language": "English", "sourcecountry": "DR Congo"}]}

    def get(url, params, **kwargs):
        calls.append(params)
        return Response(requests.Request("GET", url, params=params).prepare().url, payload=payload)

    monkeypatch.setattr(news_search.requests, "get", get)
    first = search_news("socialimage query")
    second = search_news("socialimage query")
    assert first[0].socialimage == second[0].socialimage == "https://cdn.example.org/a.jpg"
    assert first[0].source_country == "DR Congo"
    assert len(calls) == 1


def test_image_collage_mode_parses_direct_media(monkeypatch):
    payload = {"images": [{"imageurl": "https://cdn.example/road.jpg", "articleurl": "https://news.example/road", "title": "Goma road", "seendate": "20260317T120000Z", "domain": "news.example"}]}
    seen = []

    def get(url, params, **kwargs):
        seen.append(params)
        return Response(requests.Request("GET", url, params=params).prepare().url, payload=payload)

    monkeypatch.setattr(news_search.requests, "get", get)
    items = search_news("Goma (imagetag:road OR imagewebtag:flood)")
    assert seen[0]["mode"] == "ImageCollageInfo"
    assert items[0].socialimage == "https://cdn.example/road.jpg"
    assert items[0].url == "https://news.example/road"
    assert items[0].published_at == datetime(2026, 3, 17, 12, tzinfo=timezone.utc)
    assert items[0].image_collage is True
    assert parse_gdelt_images(payload) == items


def test_429_cooldown_is_shared_between_request_scopes(monkeypatch, isolated_search):
    clock, waits = isolated_search
    monkeypatch.setattr(news_search, "GDELT_MAX_429_RETRIES", 0)
    calls = []

    def get(url, params, **kwargs):
        calls.append(clock[0])
        payload = None if len(calls) == 1 else {"articles": []}
        return Response(requests.Request("GET", url, params=params).prepare().url, status_code=429 if payload is None else 200, payload=payload, text="rate limited" if payload is None else "")

    monkeypatch.setattr(news_search.requests, "get", get)
    with gdelt_request_scope():
        search_news("first")
    state = json.loads((news_search.CACHE_DIR / "gdelt_rate.lock").read_text())
    assert state["last_429_at"] == 100.0
    assert state["blocked_until"] == 160.0
    with gdelt_request_scope():
        search_news("second")
    assert calls == [100.0, 160.0]
    assert 60.0 in waits


def test_429_backoff_adds_jitter_and_logs_actual_wait(monkeypatch, isolated_search, caplog):
    clock, waits = isolated_search
    monkeypatch.setattr(news_search, "_jitter", lambda: 0.75)
    calls = []

    def get(url, params, **kwargs):
        calls.append(clock[0])
        return Response(requests.Request("GET", url, params=params).prepare().url, status_code=429, text="limited") if len(calls) == 1 else Response(requests.Request("GET", url, params=params).prepare().url, payload=PAYLOAD)

    monkeypatch.setattr(news_search.requests, "get", get)
    with caplog.at_level(logging.INFO):
        assert search_news("jitter test")
    assert calls == [100.0, 115.75]
    assert waits == [15.75]
    assert "rate_limit_wait=15.750" in caplog.text


def test_configurable_interval_from_environment():
    env = dict(os.environ, GDELT_MIN_INTERVAL_SECONDS="12")
    result = subprocess.run([sys.executable, "-c", "from app.services.retrieval.news_search import GDELT_SPACING_SECONDS; print(GDELT_SPACING_SECONDS)"], env=env, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "12.0"


def test_timespan_changes_request_and_cache_key(monkeypatch):
    params_seen = []

    def get(url, params, **kwargs):
        params_seen.append(params)
        return Response(requests.Request("GET", url, params=params).prepare().url, payload=PAYLOAD)

    monkeypatch.setattr(news_search.requests, "get", get)
    search_news("Goma road", timespan="1week")
    search_news("Goma road", timespan="3months")
    assert [item["timespan"] for item in params_seen] == ["1week", "3months"]


def test_json_decode_failure_still_reserves_timestamp_and_waits(monkeypatch, isolated_search, caplog):
    clock, waits = isolated_search
    starts = []

    class BadJSON(Response):
        def json(self):
            raise json.JSONDecodeError("invalid JSON", self.text, 0)

    def get(url, params, **kwargs):
        starts.append(clock[0])
        state = json.loads((news_search.CACHE_DIR / "gdelt_rate.lock").read_text())
        assert state["last_request"] == clock[0]  # Written before requests.get runs.
        prepared = requests.Request("GET", url, params=params).prepare().url
        return BadJSON(prepared, text="not JSON\nsecond line") if len(starts) == 1 else Response(prepared, payload=PAYLOAD)

    monkeypatch.setattr(news_search.requests, "get", get)
    with caplog.at_level(logging.INFO):
        assert search_news("first malformed") == []
        assert search_news("second valid")
    assert starts == [100.0, 110.0]
    assert waits == [10.0]
    assert "reason=QUERY_ERROR" in caplog.text
    assert "body=not JSON\\nsecond line" in caplog.text
    gdelt_lines = [record.getMessage() for record in caplog.records if "source=GDELT" in record.getMessage()]
    assert gdelt_lines and all(re.match(r"timestamp_utc=\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}\+00:00 source=GDELT", line) for line in gdelt_lines)


@pytest.mark.parametrize("outcome", ["ok", "429", "timeout", "connection_error"])
def test_request_timestamp_precedes_http_and_failures_still_space_calls(monkeypatch, isolated_search, outcome):
    clock, _ = isolated_search
    monkeypatch.setattr(news_search, "GDELT_MAX_429_RETRIES", 0)
    starts = []

    def get(url, params, **kwargs):
        starts.append(clock[0])
        state = json.loads((news_search.CACHE_DIR / "gdelt_rate.lock").read_text())
        assert state["last_request"] == clock[0]
        prepared = requests.Request("GET", url, params=params).prepare().url
        if len(starts) == 1:
            if outcome == "429":
                return Response(prepared, status_code=429, text="rate limited")
            if outcome == "timeout":
                raise requests.Timeout("timed out")
            if outcome == "connection_error":
                raise requests.ConnectionError("unreachable")
        return Response(prepared, payload=PAYLOAD)

    monkeypatch.setattr(news_search.requests, "get", get)
    search_news("first " + outcome)
    search_news("second " + outcome)
    assert len(starts) == 2
    assert starts[1] - starts[0] >= 10.0
