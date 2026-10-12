import tempfile
import types
from pathlib import Path
from unittest.mock import patch

from app.models.schemas import SearchRequest
from app.services import search_and_check as search_module
from app.services.retrieval import cache, free_search
from app.services.retrieval.news_search import RetrievedArticle


RSS = b'''<rss xmlns:media="http://search.yahoo.com/mrss/"><channel><item>
<title>Tankers near Hormuz</title><link>https://news.example/tankers</link>
<pubDate>Thu, 19 Mar 2026 12:00:00 GMT</pubDate><source>Example News</source>
<description><![CDATA[<p>Three tankers reported.</p>]]></description>
<media:content url="https://cdn.example/ship.jpg" />
</item></channel></rss>'''


def test_google_news_rss_parses_date_source_and_image():
    rows = free_search.parse_google_news_rss(RSS)
    assert len(rows) == 1
    assert rows[0].source_name == "Example News"
    assert rows[0].published_at.isoformat() == "2026-03-19T12:00:00+00:00"
    assert rows[0].socialimage == "https://cdn.example/ship.jpg"


def test_google_news_wrapper_is_not_publisher_provenance():
    assert free_search.is_google_news_url("https://news.google.com/rss/articles/encoded?oc=5")
    assert not free_search.is_google_news_url("https://www.reuters.com/world/story")


def test_google_news_rss_caches_and_uses_exclusive_next_day_cutoff():
    class Response:
        content = RSS

        def raise_for_status(self):
            pass

    from datetime import datetime, timezone
    with tempfile.TemporaryDirectory() as folder, \
         patch.object(cache, "CACHE_DIR", Path(folder)), \
         patch.object(free_search.requests, "get", return_value=Response()) as get:
        first = free_search.search_google_news_rss("Hormuz tanker", published_before=datetime(2026, 3, 19, 23, 59, tzinfo=timezone.utc))
        second = free_search.search_google_news_rss("Hormuz tanker", published_before=datetime(2026, 3, 19, 23, 59, tzinfo=timezone.utc))
    assert first == second
    assert get.call_count == 1
    assert "before:2026-03-20" in get.call_args.kwargs["params"]["q"]


def test_ddgs_images_and_video_thumbnails_preserve_source_links():
    images = free_search.parse_ddgs_images([{"url": "https://news.example/story",
                                             "image": "https://cdn.example/ship.jpg", "title": "Ship"}])
    videos = free_search.parse_video_thumbnails([{"id": "abcdefghijk", "title": "Tanker video",
                                                  "upload_date": "20260319"}])
    assert images[0].socialimage == "https://cdn.example/ship.jpg"
    assert images[0].url == "https://news.example/story"
    assert videos[0].url == "https://www.youtube.com/watch?v=abcdefghijk"
    assert videos[0].socialimage.endswith("/hqdefault.jpg")
    assert videos[0].published_at.hour == 23


def test_optional_free_backends_map_results_without_live_network():
    class FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def images(self, query, **kwargs):
            assert kwargs["backend"] == "bing"
            return [{"url": "https://news.example/story", "image": "https://cdn.example/ship.jpg"}]

    class FakeYoutubeDL:
        def __init__(self, options):
            assert options["skip_download"] is True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, query, download):
            assert query.startswith("ytsearch") and download is False
            return {"entries": [{"id": "abcdefghijk", "title": "Tanker"}]}

    with tempfile.TemporaryDirectory() as folder, \
         patch.object(cache, "CACHE_DIR", Path(folder)), \
         patch.dict("sys.modules", {"ddgs": types.SimpleNamespace(DDGS=FakeDDGS),
                                    "yt_dlp": types.SimpleNamespace(YoutubeDL=FakeYoutubeDL)}):
        image_rows = free_search.search_ddgs_images("Hormuz tanker")
        video_rows = free_search.search_video_thumbnails("Hormuz tanker")
    assert len(image_rows) == len(video_rows) == 1


def test_free_provider_uses_article_image_then_optional_fallbacks():
    request = SearchRequest.model_validate({"report_id": "live", "topic": "Strait of Hormuz",
                                            "claim": {"claim_id": "c", "text": "A tanker crossed the strait."},
                                            "search": {"provider": "free"}})
    article = RetrievedArticle(url="https://news.example/story", title="Hormuz tanker",
                               source_name="News", socialimage="https://cdn.example/ship.jpg")
    fallback = RetrievedArticle(url="https://other.example/story", title="Hormuz ship",
                                source_name="Other", socialimage="https://cdn.other.example/ship.jpg",
                                image_collage=True)
    calls = []
    with patch.object(search_module, "search_google_news_rss", side_effect=lambda *a, **k: calls.append("rss") or [article]), \
         patch.object(search_module, "search_ddgs_images", side_effect=lambda *a, **k: calls.append("ddgs") or [fallback]), \
         patch.object(search_module, "search_video_thumbnails", return_value=[]), \
         patch.object(search_module, "fetch_article", return_value=None), \
         patch.object(search_module, "candidate_score", return_value=0.8), \
         patch.object(search_module, "download_media", return_value=None), \
         patch("app.services.retrieval.candidate_ranker.text_similarity", return_value=0.8):
        result = search_module.search_and_check(request)
    assert calls.index("rss") < calls.index("ddgs")
    assert result.media_candidates == 2
    assert any(query.startswith("Google News RSS") for query in result.queries)


def test_auto_without_credentials_uses_free_news_path():
    request = SearchRequest.model_validate({"report_id": "live", "topic": "Hormuz",
                                            "claim": {"claim_id": "c", "text": "A tanker is visible."}})
    with patch.object(search_module, "google_configured", return_value=False), \
         patch.object(search_module, "searxng_configured", return_value=False), \
         patch.object(search_module, "search_google_news_rss", return_value=[]) as rss, \
         patch.object(search_module, "search_ddgs_images", return_value=[]), \
         patch.object(search_module, "search_video_thumbnails", return_value=[]), \
         patch.object(search_module, "search_duckduckgo", side_effect=AssertionError("DuckDuckGo must not run")):
        result = search_module.search_and_check(request)
    assert rss.called
    assert result.queries[0].startswith("Google News RSS")


def test_free_provider_discards_google_news_wrapper_image():
    request = SearchRequest.model_validate({"report_id": "live", "topic": "Hormuz",
                                            "claim": {"claim_id": "c", "text": "A tanker is visible."},
                                            "search": {"provider": "free"}})
    wrapper = RetrievedArticle(url="https://news.google.com/rss/articles/encoded", title="Google News",
                               source_name="Reuters", socialimage="https://lh3.googleusercontent.com/logo.jpg")
    with patch.object(search_module, "search_google_news_rss", return_value=[wrapper]), \
         patch.object(search_module, "fetch_article", return_value=None), \
         patch.object(search_module, "search_ddgs_images", return_value=[]), \
         patch.object(search_module, "search_video_thumbnails", return_value=[]), \
         patch("app.services.retrieval.candidate_ranker.text_similarity", return_value=0.8):
        result = search_module.search_and_check(request)
    assert result.media_candidates == 0
    assert result.candidate_previews == []
