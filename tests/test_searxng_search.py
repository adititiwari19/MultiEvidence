import tempfile
from pathlib import Path
from unittest.mock import patch

from app.models.schemas import SearchRequest
from app.services import search_and_check as search_module
from app.services.retrieval import cache, searxng_search


class FakeResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {"results": [{"url": "https://news.example/ship", "title": "Hormuz tanker",
                             "content": "Ship in the strait", "img_src": "https://cdn.example/ship.jpg"}]}


def test_searxng_image_search_and_cache():
    with tempfile.TemporaryDirectory() as folder, \
         patch.object(cache, "CACHE_DIR", Path(folder)), \
         patch.object(searxng_search, "SEARXNG_BASE_URL", "http://127.0.0.1:8888"), \
         patch.object(searxng_search.requests, "get", return_value=FakeResponse()) as get:
        first = searxng_search.search_searxng("Hormuz ship", image=True)
        second = searxng_search.search_searxng("Hormuz ship", image=True)
    assert first == second
    assert first[0].socialimage == "https://cdn.example/ship.jpg"
    assert first[0].published_at is None
    assert get.call_count == 1
    assert get.call_args.kwargs["params"]["categories"] == "images"


def test_auto_prefers_configured_searxng_without_google():
    request = SearchRequest.model_validate({"report_id": "live", "topic": "Strait of Hormuz",
                                            "claim": {"claim_id": "c", "text": "A tanker crossed the strait."}})
    with patch.object(search_module, "google_configured", return_value=False), \
         patch.object(search_module, "searxng_configured", return_value=True), \
         patch.object(search_module, "search_searxng", return_value=[]) as searxng, \
         patch.object(search_module, "search_duckduckgo", side_effect=AssertionError("DuckDuckGo must not run")):
        result = search_module.search_and_check(request)
    assert searxng.called
    assert all(query.startswith("SearXNG") for query in result.queries)
