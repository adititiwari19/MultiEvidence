import asyncio
import json
from datetime import datetime, timezone

from app.main import app
from app.models.schemas import CheckResponse, MediaCandidate, NewsArticle, QueryPlan
from app.services.retrieval.article_parser import ParsedArticle
from app.services.retrieval.news_search import RetrievedArticle
from app.services.search_and_check import to_evidence
from app.services.search_and_check import search_and_check
from app.models.schemas import SearchRequest


def post(payload):
    payload = {**payload, "search": {**payload.get("search", {}), "provider": "gdelt"}}
    async def run():
        messages = []
        first = True

        async def receive():
            nonlocal first
            if first:
                first = False
                return {"type": "http.request", "body": json.dumps(payload).encode(), "more_body": False}
            await asyncio.sleep(3600)

        async def send(message):
            messages.append(message)

        path = "/api/v1/multimodal/search-and-check"
        scope = {"type": "http", "asgi": {"version": "3.0"}, "method": "POST", "path": path, "raw_path": path.encode(), "query_string": b"", "headers": [(b"content-type", b"application/json")], "client": ("test", 1), "server": ("test", 80), "scheme": "http", "root_path": ""}
        await asyncio.wait_for(app(scope, receive, send), timeout=5)
        return messages[0]["status"], json.loads(messages[1]["body"])

    return asyncio.run(run())


def test_auto_endpoint_with_mocked_sources(monkeypatch, tmp_path):
    article = NewsArticle(url="https://news.example/story", title="Goma road flooded", source_name="News", published_at=datetime(2026, 3, 17, tzinfo=timezone.utc))
    candidate = MediaCandidate(candidate_id="e1", article_url=article.url, source_name=article.source_name, published_at=article.published_at, media_type="image", media_url="https://cdn.example/road.jpg", title="Goma road flooded", article_text_excerpt="The Goma road flooded.", caption="Flooded road")
    media_path = tmp_path / "image.jpg"
    media_path.write_bytes(b"test media")
    monkeypatch.setattr("app.services.search_and_check.build_queries", lambda request: QueryPlan(queries=["Goma road", "Goma flooding"], entities=[], locations=["Goma"], event_terms=[]))
    monkeypatch.setattr("app.services.search_and_check.search_news", lambda *args, **kwargs: [article])
    monkeypatch.setattr("app.services.search_and_check.fetch_article", lambda item: ParsedArticle(item, "The Goma road flooded.", [candidate]))
    monkeypatch.setattr("app.services.search_and_check.download_media", lambda item, claim_id: item.model_copy(update={"local_path": str(media_path)}))
    monkeypatch.setattr("app.services.retrieval.candidate_ranker.text_similarity", lambda claim, text: 0.8)
    monkeypatch.setattr("app.services.retrieval.candidate_ranker.image_visual_score", lambda claim, path: None)

    def verify(request, media_url):
        assert request.evidence[0].context_text.startswith("Goma road flooded")
        return CheckResponse.model_validate({"claim_id": "c", "affected": True, "relation": "CONTRADICTS", "relevance": 0.9, "confidence": 0.91, "observable_facts": ["Flood water covers the road."], "evidence": [{"evidence_id": "e1", "frame_id": "e1_image", "frame_url": "/media/frames/e1_image.jpg", "observation": "Flood water covers the road."}], "suggested_update": {"action": "REPLACE", "old_text": "The Goma road is open.", "new_text": "The Goma road appears flooded.", "reason": "Visible flood water."}, "explanation": "The road is flooded.", "limitations": []})

    monkeypatch.setattr("app.services.search_and_check.cached_check", verify)
    status, body = post({"report_id": "r", "topic": "Goma flooding", "claim": {"claim_id": "c", "text": "The Goma road is open."}, "search": {"published_after": "2026-03-01T00:00:00Z"}})
    assert status == 200
    assert body["claim_status"] == "CONTRADICTED"
    assert body["verified_evidence_count"] == 1
    assert body["articles_considered"] == 1
    assert body["media_candidates"] == 1
    assert body["evidence"][0]["source_url"] == article.url
    assert body["suggested_update"]["action"] == "REPLACE"


def test_auto_endpoint_no_media_is_uncertain(monkeypatch):
    monkeypatch.setattr("app.services.search_and_check.build_queries", lambda request: QueryPlan(queries=["query one", "query two"], entities=[], locations=[], event_terms=[]))
    monkeypatch.setattr("app.services.search_and_check.search_news", lambda *args, **kwargs: [])
    status, body = post({"report_id": "r", "topic": "Goma", "claim": {"claim_id": "c", "text": "The road is open."}})
    assert status == 200
    assert body["claim_status"] == "UNCERTAIN"
    assert body["evidence"] == []
    assert body["message"] == "No sufficiently relevant multimodal evidence was found."


def test_rate_limited_retrieval_is_reported_and_stops_queries(monkeypatch):
    calls = []
    monkeypatch.setattr("app.services.search_and_check.build_queries", lambda request: QueryPlan(queries=["first", "second"], entities=[], locations=[], event_terms=[]))
    monkeypatch.setattr("app.services.search_and_check.search_news", lambda query, **kwargs: calls.append((query, kwargs["timespan"])) or [])
    monkeypatch.setattr("app.services.search_and_check.search_status", lambda: "RATE_LIMITED")
    monkeypatch.setattr("app.services.search_and_check.search_bing_news", lambda *args, **kwargs: [])
    status, body = post({"report_id": "r", "topic": "Dubai airport", "claim": {"claim_id": "c", "text": "Dubai airport closed"}, "search": {"timespan": "3months"}})
    assert status == 200
    assert calls == [("first", "3months")]
    assert "rate-limited" in body["message"]
    assert body["verified_evidence_count"] == 0


def test_rate_limited_gdelt_uses_rss_articles(monkeypatch):
    article = NewsArticle(url="https://publisher.example/story", title="Dubai airport flight update", source_name="Publisher")
    monkeypatch.setattr("app.services.search_and_check.build_queries", lambda request: QueryPlan(queries=["text", "visual"], entities=[], locations=["Dubai airport"], event_terms=[]))
    monkeypatch.setattr("app.services.search_and_check.search_news", lambda *args, **kwargs: [])
    monkeypatch.setattr("app.services.search_and_check.search_status", lambda: "RATE_LIMITED")
    monkeypatch.setattr("app.services.search_and_check.search_bing_news", lambda *args, **kwargs: [article])
    monkeypatch.setattr("app.services.search_and_check.fetch_article", lambda item: None)
    monkeypatch.setattr("app.services.search_and_check.candidate_score", lambda *args: 0.5)
    result = search_and_check(SearchRequest.model_validate({"report_id": "r", "topic": "Dubai airport", "claim": {"claim_id": "c", "text": "Dubai airport flights arriving"}, "search": {"provider": "gdelt"}}))
    assert result.articles_considered == 1
    assert result.queries[-1].startswith("Bing News RSS:")


def test_video_evidence_retains_exact_timestamp():
    candidate = MediaCandidate(candidate_id="v1", article_url="https://news.example/story", source_name="News", media_type="video", media_url="https://cdn.example/clip.mp4", local_path="/tmp/clip.mp4", title="Road")
    result = CheckResponse.model_validate({"claim_id": "c", "affected": True, "relation": "REFINES", "relevance": 0.8, "confidence": 0.9, "observable_facts": ["Water covers part of the road."], "evidence": [{"evidence_id": "v1", "frame_id": "v1_24400", "timestamp_seconds": 24.4, "frame_url": "/media/frames/v1_24400.jpg", "observation": "Water covers part of the road."}], "suggested_update": {"action": "QUALIFY", "old_text": "Road open", "new_text": "Road partly flooded", "reason": "Visible water"}, "explanation": "A later frame shows water.", "limitations": []})
    evidence = to_evidence(candidate, result)
    assert evidence.timestamp_seconds == 24.4
    assert evidence.frames[0].frame_id == "v1_24400"


def test_at_most_two_queries_and_ten_ranked_articles(monkeypatch):
    articles = [NewsArticle(url=f"https://news.example/{index}", title=f"Story {index}", source_name="News") for index in range(10)]
    calls = []
    monkeypatch.setattr("app.services.search_and_check.build_queries", lambda request: QueryPlan(queries=["one", "two", "three"], entities=[], locations=[], event_terms=[]))

    def search(query, **kwargs):
        calls.append(query)
        return articles

    monkeypatch.setattr("app.services.search_and_check.search_news", search)
    monkeypatch.setattr("app.services.search_and_check.fetch_article", lambda article: None)
    request = SearchRequest.model_validate({"report_id": "r", "topic": "Goma", "claim": {"claim_id": "c", "text": "Road open"}, "search": {"provider": "gdelt"}})
    result = search_and_check(request)
    assert calls == ["one", "two"]
    assert result.articles_considered == 10


def test_socialimage_candidate_without_article_html(monkeypatch, tmp_path):
    article = RetrievedArticle(url="https://news.example/goma", title="Goma road flooded", source_name="News", socialimage="https://cdn.example/goma.jpg")
    path = tmp_path / "image.jpg"
    path.write_bytes(b"image")
    monkeypatch.setattr("app.services.search_and_check.build_queries", lambda request: QueryPlan(queries=["text", "visual"], entities=[], locations=["Goma"], event_terms=[]))
    monkeypatch.setattr("app.services.search_and_check.search_news", lambda *args, **kwargs: [article])
    monkeypatch.setattr("app.services.search_and_check.fetch_article", lambda item: None)
    monkeypatch.setattr("app.services.search_and_check.download_media", lambda item, claim_id: item.model_copy(update={"local_path": str(path)}))
    monkeypatch.setattr("app.services.retrieval.candidate_ranker.text_similarity", lambda claim, text: 0.8)
    monkeypatch.setattr("app.services.retrieval.candidate_ranker.image_visual_score", lambda claim, path: None)
    monkeypatch.setattr("app.services.search_and_check.cached_check", lambda request, media_url: CheckResponse.model_validate({"claim_id": "c", "affected": False, "relation": "INSUFFICIENT", "relevance": 0.7, "confidence": 0.3, "observable_facts": [], "evidence": [], "suggested_update": {"action": "KEEP", "old_text": "Road open", "new_text": "Road open", "reason": "Unclear"}, "explanation": "Unclear", "limitations": []}))
    result = search_and_check(SearchRequest.model_validate({"report_id": "r", "topic": "Goma flooding", "claim": {"claim_id": "c", "text": "Road open"}, "search": {"provider": "gdelt"}}))
    assert result.media_candidates == 1
    assert result.candidate_previews[0].media_url == article.socialimage
    assert result.aggregate.insufficient_count == 1
    assert result.checked_media_count == 1
    assert result.checks[0].relation == "INSUFFICIENT"
    assert result.evidence == []
    assert result.message.startswith("Images were checked")


def test_local_reranking_limits_media_downloads_to_ten(monkeypatch):
    articles = [NewsArticle(url=f"https://news.example/{i}", title="Goma road", source_name="News") for i in range(3)]
    monkeypatch.setattr("app.services.search_and_check.build_queries", lambda request: QueryPlan(queries=["text", "visual"], entities=[], locations=["Goma"], event_terms=[]))
    monkeypatch.setattr("app.services.search_and_check.search_news", lambda *args, **kwargs: articles)
    monkeypatch.setattr("app.services.retrieval.candidate_ranker.text_similarity", lambda claim, text: 0.8)

    def parse(article):
        candidates = [MediaCandidate(candidate_id=f"{article.url}-{i}", article_url=article.url, source_name="News", media_type="image" if i < 3 else "video", media_url=f"{article.url}/{i}.{'jpg' if i < 3 else 'mp4'}", title="Goma road") for i in range(5)]
        return ParsedArticle(article, "", candidates)

    monkeypatch.setattr("app.services.search_and_check.fetch_article", parse)
    downloads = []
    monkeypatch.setattr("app.services.search_and_check.download_media", lambda item, claim_id: downloads.append(item) or None)
    result = search_and_check(SearchRequest.model_validate({"report_id": "r", "topic": "Goma", "claim": {"claim_id": "c", "text": "Goma road"}, "search": {"provider": "gdelt"}}))
    assert result.media_candidates == 15
    assert len(downloads) == 10


def test_image_collage_entries_skip_article_fetch(monkeypatch):
    visual = RetrievedArticle(url="https://news.example/visual", title="Goma road flood", source_name="News", socialimage="https://cdn.example/visual.jpg", image_collage=True)
    article = NewsArticle(url="https://news.example/text", title="Goma road flood", source_name="News")
    monkeypatch.setattr("app.services.search_and_check.build_queries", lambda request: QueryPlan(queries=["text", "visual"], entities=[], locations=["Goma"], event_terms=[]))
    monkeypatch.setattr("app.services.search_and_check.search_news", lambda query, **kwargs: [visual] if query == "visual" else [article])
    fetched = []
    monkeypatch.setattr("app.services.search_and_check.fetch_article", lambda item: fetched.append(item.url) or None)
    monkeypatch.setattr("app.services.search_and_check.download_media", lambda item, claim_id: None)
    monkeypatch.setattr("app.services.retrieval.candidate_ranker.text_similarity", lambda claim, text: 0.8)
    result = search_and_check(SearchRequest.model_validate({"report_id": "r", "topic": "Goma", "claim": {"claim_id": "c", "text": "Goma road"}, "search": {"provider": "gdelt"}}))
    assert fetched == [article.url]
    assert result.media_candidates == 1
    assert result.candidate_previews[0].media_url == visual.socialimage


def test_different_event_result_is_excluded_from_aggregation(monkeypatch, tmp_path):
    article = RetrievedArticle(url="https://news.example/apartment-fire", title="Spokane Valley apartment fire", source_name="News", socialimage="https://cdn.example/fire.jpg", image_collage=True)
    path = tmp_path / "fire.jpg"
    path.write_bytes(b"image")
    monkeypatch.setattr("app.services.search_and_check.build_queries", lambda request: QueryPlan(queries=["text", "visual"], entities=[], locations=["Spokane"], event_terms=[]))
    monkeypatch.setattr("app.services.search_and_check.search_news", lambda query, **kwargs: [article] if query == "visual" else [])
    monkeypatch.setattr("app.services.search_and_check.fetch_article", lambda article: (_ for _ in ()).throw(AssertionError("article fetch called")))
    monkeypatch.setattr("app.services.search_and_check.download_media", lambda item, claim_id: item.model_copy(update={"local_path": str(path)}))
    monkeypatch.setattr("app.services.retrieval.candidate_ranker.text_similarity", lambda claim, text: 0.8)
    monkeypatch.setattr("app.services.retrieval.candidate_ranker.image_visual_score", lambda claim, path: None)
    monkeypatch.setattr("app.services.search_and_check.cached_check", lambda request, media_url: CheckResponse.model_validate({"claim_id": "c", "affected": False, "relation": "IRRELEVANT", "relevance": 0.0, "confidence": 0.9, "observable_facts": [], "evidence": [], "suggested_update": {"action": "KEEP", "old_text": "No homes were destroyed.", "new_text": "No homes were destroyed.", "reason": "Different event"}, "explanation": "Different event", "limitations": [], "event_match": False, "event_match_reason": "Apartment fire differs from wildfire."}))
    result = search_and_check(SearchRequest.model_validate({"report_id": "r", "topic": "Spokane wildfire", "claim": {"claim_id": "c", "text": "No homes were destroyed in the Spokane wildfire."}, "search": {"provider": "gdelt"}}))
    assert result.claim_status == "UNCERTAIN"
    assert result.evidence == []
    assert result.aggregate.irrelevant_count == 1
