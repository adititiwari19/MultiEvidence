from datetime import datetime, timezone

from app.models.schemas import NewsArticle
from app.services.retrieval.article_parser import fetch_article, parse_article_html


ARTICLE = NewsArticle(url="https://news.example/story", title="Fallback", source_name="News", published_at=datetime(2026, 3, 1, tzinfo=timezone.utc))
HTML = '''<html><head><meta property="og:title" content="Road update"><meta property="article:published_time" content="2026-03-17T12:00:00Z"><meta property="og:image" content="/hero.jpg"><meta name="twitter:image" content="/hero.jpg"><meta property="og:video" content="https://cdn.example/clip.mp4"></head><body><article><p>The road has flooded.</p><img src="/tiny.png" width="16" height="16"><figure><img src="/road.jpg" width="640" height="480"><figcaption>Flooded road</figcaption></figure><video><source src="/scene.webm"></video><iframe src="https://video.example/embed/123"></iframe></article></body></html>'''


def test_article_metadata_and_media_extraction():
    parsed = parse_article_html(ARTICLE, HTML)
    assert parsed.article.title == "Road update"
    assert parsed.article.published_at.day == 17
    assert "flooded" in parsed.text
    urls = {item.media_url for item in parsed.candidates}
    assert "https://news.example/hero.jpg" in urls
    assert "https://news.example/road.jpg" in urls
    assert "https://cdn.example/clip.mp4" in urls
    assert "https://news.example/tiny.png" not in urls
    assert len(parsed.candidates) <= 5
    assert next(item for item in parsed.candidates if item.media_url.endswith("road.jpg")).caption == "Flooded road"


def test_article_fetch_cached(monkeypatch):
    monkeypatch.setattr("app.services.retrieval.article_parser.read_json", lambda group, key: None)
    saved = []
    monkeypatch.setattr("app.services.retrieval.article_parser.write_json", lambda group, key, value: saved.append(value))

    class Response:
        headers = {"Content-Type": "text/html"}
        content = HTML.encode()
        text = HTML

        def raise_for_status(self):
            pass

    monkeypatch.setattr("app.services.retrieval.article_parser.requests.get", lambda *args, **kwargs: Response())
    parsed = fetch_article(ARTICLE)
    assert parsed is not None and parsed.candidates
    assert saved[0]["article"]["title"] == "Road update"
