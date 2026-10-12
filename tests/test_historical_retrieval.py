import unittest
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from app.models.schemas import MediaCandidate, NewsArticle, QueryPlan, SearchRequest
from app.services.retrieval.candidate_ranker import rank_candidates
from app.services.retrieval import cache, news_search
from app.services.retrieval.query_builder import structured_queries
from app.services.relevance import entity_overlap


class HistoricalRetrievalTests(unittest.TestCase):
    def test_hormuz_queries_use_maritime_terms(self):
        request = SearchRequest.model_validate({
            "report_id": "T3", "topic": "Iran conflict and Strait of Hormuz",
            "claim": {"claim_id": "c", "text": "The Strait of Hormuz route remains open."},
        })
        plan = structured_queries(request, QueryPlan(queries=["seed one", "seed two"], entities=[], locations=[], event_terms=[]))
        self.assertIn('"Strait of Hormuz"', plan.queries[0])
        self.assertIn("shipping", plan.queries[0])
        self.assertNotIn("road", plan.queries[0])
        self.assertIn('imagetag:"ship"', plan.queries[1])

    def test_report_timestamp_excludes_later_and_undated_images(self):
        request = SearchRequest.model_validate({
            "report_id": "T0", "topic": "Strait of Hormuz",
            "claim": {"claim_id": "c", "text": "Shipping through the strait is disrupted.",
                      "report_timestamp": "2026-03-05T23:59:59Z"},
        })
        def candidate(name, published):
            return MediaCandidate(candidate_id=name, article_url=f"https://news.example/{name}",
                                  source_name="News", media_type="image",
                                  media_url=f"https://cdn.example/{name}.jpg", title="Hormuz shipping",
                                  published_at=published)
        before = candidate("before", datetime(2026, 3, 5, 12, tzinfo=timezone.utc))
        after = candidate("after", datetime(2026, 3, 6, 12, tzinfo=timezone.utc))
        undated = candidate("undated", None)
        with patch("app.services.retrieval.candidate_ranker.text_similarity", side_effect=entity_overlap):
            ranked = rank_candidates(request, [after, undated, before], ["Strait of Hormuz"])
        self.assertEqual([item.candidate_id for item in ranked], ["before"])

    def test_gdelt_results_respect_historical_end_date(self):
        end = datetime(2026, 3, 5, 23, 59, 59, tzinfo=timezone.utc)
        rows = [
            NewsArticle(url="https://news.example/old", title="Old", source_name="News",
                        published_at=datetime(2026, 3, 5, 12, tzinfo=timezone.utc)),
            NewsArticle(url="https://news.example/new", title="New", source_name="News",
                        published_at=datetime(2026, 3, 6, 12, tzinfo=timezone.utc)),
            NewsArticle(url="https://news.example/undated", title="Undated", source_name="News"),
        ]
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(news_search, "CACHE_DIR", Path(folder)), patch.object(cache, "CACHE_DIR", Path(folder)), patch.object(news_search, "_gdelt", return_value=rows):
                result = news_search.search_news("Hormuz shipping", enddatetime=end)
        self.assertEqual([item.url for item in result], ["https://news.example/old"])


if __name__ == "__main__":
    unittest.main()
