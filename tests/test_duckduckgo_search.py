import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from app.models.schemas import SearchRequest
from app.services import search_and_check as search_module
from app.services.retrieval import cache, duckduckgo_search
from app.services.retrieval.news_search import RetrievedArticle


class FakeResponse:
    url = "https://duckduckgo.com/?q=Hormuz"

    def __init__(self, text="", payload=None):
        self.text = text
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class DuckDuckGoSearchTests(unittest.TestCase):
    def test_parses_web_source_and_direct_images(self):
        html = '<div class="result"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fnews.example%2Fstory">Hormuz tanker</a><span class="result__snippet">Ship near strait</span></div>'
        web = duckduckgo_search.parse_web_results(html)
        images = duckduckgo_search.parse_image_results({"results": [
            {"url": "https://news.example/story", "image": "https://cdn.example/ship.jpg", "title": "Ship"},
            {"url": "https://news.example/invalid", "image": "javascript:bad", "title": "Bad"},
        ]})
        self.assertEqual(web[0].url, "https://news.example/story")
        self.assertEqual(images[0].socialimage, "https://cdn.example/ship.jpg")
        self.assertTrue(images[0].image_collage)
        self.assertIsNone(images[0].published_at)
        self.assertEqual(len(images), 1)

    def test_image_search_uses_token_and_caches(self):
        calls = []

        def get(url, **kwargs):
            calls.append((url, kwargs.get("params")))
            if url == duckduckgo_search.TOKEN_URL:
                return FakeResponse("var vqd='4-123456789';")
            return FakeResponse(payload={"results": [{"url": "https://news.example/story",
                                                     "image": "https://cdn.example/ship.jpg"}]})

        with tempfile.TemporaryDirectory() as folder, \
             patch.object(cache, "CACHE_DIR", Path(folder)), \
             patch.object(duckduckgo_search.requests.Session, "get", side_effect=get):
            first = duckduckgo_search.search_duckduckgo("Hormuz ship", image=True)
            second = duckduckgo_search.search_duckduckgo("Hormuz ship", image=True)
        self.assertEqual(first, second)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1][1]["vqd"], "4-123456789")

    def test_empty_image_cache_is_retried(self):
        payload = {"results": [{"url": "https://news.example/story", "image": "https://cdn.example/ship.jpg"}]}
        with patch.object(duckduckgo_search, "read_json", return_value={"cached_at": duckduckgo_search.time.time(), "results": []}), \
             patch.object(duckduckgo_search, "write_json"), \
             patch.object(duckduckgo_search.requests.Session, "get", side_effect=[
                 FakeResponse("var vqd='4-123456789';"), FakeResponse(payload=payload)]):
            rows = duckduckgo_search.search_duckduckgo("Hormuz ship", image=True)
        self.assertEqual(len(rows), 1)

    def test_explicit_duckduckgo_uses_keyless_provider(self):
        request = SearchRequest.model_validate({"report_id": "live", "topic": "Strait of Hormuz",
                                                "claim": {"claim_id": "c", "text": "A tanker crossed the strait."},
                                                "search": {"provider": "duckduckgo"}})
        article = RetrievedArticle(url="https://news.example/ship", title="Hormuz tanker",
                                   source_name="News", socialimage="https://cdn.example/ship.jpg",
                                   image_collage=True)
        with patch.object(search_module, "google_configured", return_value=False), \
             patch.object(search_module, "search_duckduckgo", return_value=[article]) as keyless, \
             patch.object(search_module, "search_news", side_effect=AssertionError("GDELT must not run")), \
             patch.object(search_module, "candidate_score", return_value=0.8), \
             patch.object(search_module, "download_media", return_value=None), \
             patch("app.services.retrieval.candidate_ranker.text_similarity", return_value=0.8):
            result = search_module.search_and_check(request)
        self.assertTrue(keyless.called)
        self.assertEqual(result.media_candidates, 1)
        self.assertTrue(all(query.startswith("DuckDuckGo") for query in result.queries))

    def test_historical_result_needs_publisher_date(self):
        request = SearchRequest.model_validate({"report_id": "T0", "topic": "Strait of Hormuz",
                                                "claim": {"claim_id": "c", "text": "A tanker crossed the strait.",
                                                          "report_timestamp": "2026-03-05T23:59:59Z"},
                                                "search": {"provider": "duckduckgo", "timespan": "3months"}})
        page = "https://news.example/ship"
        rows = [RetrievedArticle(url=page, title="Hormuz tanker", source_name="News",
                                 socialimage="https://cdn.example/ship.jpg", image_collage=True),
                RetrievedArticle(url=page, title="Hormuz tanker", source_name="News")]
        parsed = type("Parsed", (), {"article": rows[1].model_copy(update={
            "published_at": datetime(2026, 3, 4, 12, tzinfo=timezone.utc)}), "candidates": []})()
        with patch.object(search_module, "search_duckduckgo", return_value=rows), \
             patch.object(search_module, "fetch_article", return_value=parsed), \
             patch.object(search_module, "candidate_score", return_value=0.8), \
             patch.object(search_module, "download_media", return_value=None), \
             patch("app.services.retrieval.candidate_ranker.text_similarity", return_value=0.8):
            result = search_module.search_and_check(request)
        self.assertEqual(result.media_candidates, 1)

        with patch.object(search_module, "search_duckduckgo", return_value=rows), \
             patch.object(search_module, "fetch_article", return_value=None), \
             patch.object(search_module, "download_media", return_value=None):
            undated = search_module.search_and_check(request)
        self.assertEqual(undated.media_candidates, 0)


if __name__ == "__main__":
    unittest.main()
