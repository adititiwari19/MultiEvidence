import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.models.schemas import QueryPlan, SearchRequest
from app.services.retrieval import cache, google_search
from app.services.retrieval.news_search import RetrievedArticle
from app.services import search_and_check as search_module


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class GoogleSearchTests(unittest.TestCase):
    def test_parse_web_and_image_results_with_source_dates(self):
        web = {"items": [{"link": "https://news.example/story", "title": "Hormuz tanker story",
                          "displayLink": "news.example", "snippet": "Tanker report",
                          "pagemap": {"metatags": [{"article:published_time": "2026-03-05T12:00:00Z",
                                                     "og:image": "https://cdn.example/photo.jpg"}]}}]}
        image = {"items": [{"link": "https://cdn.example/photo.jpg", "title": "Tanker",
                            "image": {"contextLink": "https://news.example/story"},
                            "pagemap": {"metatags": [{"article:published_time": "2026-03-05T12:00:00Z"}]}}]}
        web_result = google_search.parse_google_results(web, image=False)[0]
        image_result = google_search.parse_google_results(image, image=True)[0]
        self.assertEqual(web_result.socialimage, "https://cdn.example/photo.jpg")
        self.assertEqual(image_result.url, "https://news.example/story")
        self.assertTrue(image_result.image_collage)
        self.assertEqual(web_result.published_at, image_result.published_at)

    def test_google_api_uses_image_search_and_caches_result(self):
        seen = []
        payload = {"items": [{"link": "https://cdn.example/ship.jpg", "title": "Ship",
                             "image": {"contextLink": "https://news.example/ship"}}]}
        def get(url, *, params, **kwargs):
            seen.append((url, params))
            return FakeResponse(payload)
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(google_search, "GOOGLE_CSE_API_KEY", "test-key"), \
                 patch.object(google_search, "GOOGLE_CSE_CX", "test-cx"), \
                 patch.object(cache, "CACHE_DIR", Path(folder)), \
                 patch.object(google_search.requests, "get", side_effect=get):
                first = google_search.search_google('"Strait of Hormuz" tanker', image=True, timespan="1week")
                second = google_search.search_google('"Strait of Hormuz" tanker', image=True, timespan="1week")
        self.assertEqual(first, second)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][1]["searchType"], "image")
        self.assertEqual(seen[0][1]["dateRestrict"], "w1")

    def test_google_provider_does_not_call_gdelt(self):
        article = RetrievedArticle(url="https://news.example/ship", title="Hormuz tanker",
                                   source_name="News", socialimage="https://cdn.example/ship.jpg",
                                   image_collage=True)
        for provider in ("google", "auto"):
            request = SearchRequest.model_validate({"report_id": "live", "topic": "Strait of Hormuz",
                                                    "claim": {"claim_id": "c", "text": "A tanker crossed the strait."},
                                                    "search": {"provider": provider}})
            with patch.object(search_module, "google_configured", return_value=True), \
                 patch.object(search_module, "google_queries", return_value=["Hormuz tanker"]), \
                 patch.object(search_module, "search_google", return_value=[article]) as google_call, \
                 patch.object(search_module, "search_news", side_effect=AssertionError("GDELT must not run")), \
                 patch.object(search_module, "candidate_score", return_value=0.8), \
                 patch.object(search_module, "download_media", return_value=None), \
                 patch("app.services.retrieval.candidate_ranker.text_similarity", return_value=0.8):
                result = search_module.search_and_check(request)
            self.assertTrue(google_call.called)
            self.assertEqual(result.media_candidates, 1)
            self.assertTrue(all(query.startswith("Google") for query in result.queries))

    def test_explicit_google_without_credentials_fails_clearly(self):
        request = SearchRequest.model_validate({"report_id": "live", "topic": "Hormuz",
                                                "claim": {"claim_id": "c", "text": "Tanker"},
                                                "search": {"provider": "google"}})
        with patch.object(search_module, "google_configured", return_value=False), \
             patch.object(search_module, "build_queries", return_value=QueryPlan(queries=["one", "two"], entities=[], locations=[], event_terms=[])):
            with self.assertRaisesRegex(ValueError, "GOOGLE_CSE_API_KEY"):
                search_module.search_and_check(request)


if __name__ == "__main__":
    unittest.main()
