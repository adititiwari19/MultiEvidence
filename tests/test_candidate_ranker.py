from datetime import datetime, timezone

from app.models.schemas import MediaCandidate, SearchRequest
from app.services.retrieval.candidate_ranker import canonical_media_url, choose_candidates, dedupe_media_candidates, is_site_asset, rank_candidates


def test_relevance_ranking_and_limits(monkeypatch):
    monkeypatch.setattr("app.services.retrieval.candidate_ranker.text_similarity", lambda claim, text: 0.8 if "Goma" in text else 0.1)
    request = SearchRequest.model_validate({"report_id": "r", "topic": "Goma flooding", "claim": {"claim_id": "c", "text": "Goma road open"}, "search": {"max_images": 1, "max_videos": 1, "published_after": "2026-03-01T00:00:00Z"}})
    items = [
        MediaCandidate(candidate_id="old", article_url="https://n/old", source_name="n", media_type="image", media_url="https://n/old.jpg", title="Goma old", published_at=datetime(2026, 2, 1, tzinfo=timezone.utc)),
        MediaCandidate(candidate_id="weak", article_url="https://n/weak", source_name="n", media_type="image", media_url="https://n/weak.jpg", title="Other news"),
        MediaCandidate(candidate_id="good", article_url="https://n/good", source_name="n", media_type="image", media_url="https://n/good.jpg", title="Goma road flooding"),
        MediaCandidate(candidate_id="video", article_url="https://n/video", source_name="n", media_type="video", media_url="https://n/video.mp4", title="Goma road update"),
    ]
    ranked = rank_candidates(request, items, ["Goma"])
    assert "old" not in [item.candidate_id for item in ranked]
    selected = choose_candidates(request, items, ["Goma"])
    assert [item.candidate_id for item in selected] == ["good", "video"]


def test_crop_resize_variants_keep_largest_but_preserve_media_id():
    base = "https://cdn.example/photos/road.jpg"
    items = [
        MediaCandidate(candidate_id="small", article_url="https://news.example/a", source_name="News", media_type="image", media_url=base + "?id=42&crop=center&width=320&height=240", title="Road"),
        MediaCandidate(candidate_id="large", article_url="https://news.example/a", source_name="News", media_type="image", media_url=base + "?id=42&resize=1600x900&quality=90", title="Road"),
        MediaCandidate(candidate_id="different", article_url="https://news.example/a", source_name="News", media_type="image", media_url=base + "?id=43&width=640&height=480", title="Road"),
    ]
    assert canonical_media_url(items[0].media_url) == canonical_media_url(items[1].media_url)
    assert canonical_media_url(items[0].media_url) != canonical_media_url(items[2].media_url)
    assert [item.candidate_id for item in dedupe_media_candidates(items)] == ["large", "different"]


def test_site_assets_and_known_tiny_images_are_removed():
    def image(name, url, metadata=None):
        return MediaCandidate(candidate_id=name, article_url="https://news.example/a", source_name="News", media_type="image", media_url=url, title="Road", metadata=metadata or {})

    items = [
        image("loader", "https://cdn.example/assets/loader.gif"),
        image("avatar", "https://cdn.example/users/avatar.jpg"),
        image("logo", "https://cdn.example/brand/logo.png"),
        image("tiny", "https://cdn.example/photos/tiny.jpg", {"width": 199, "height": 600}),
        image("google_news", "https://news.example/assets/google_news.svg"),
        image("badge", "https://news.example/assets/google_preferred_source_badge_dark.webp"),
        image("unsupported_gif", "https://cdn.example/banner.gif"),
        image("large", "https://cdn.example/photos/news-scene.jpg", {"width": 1200, "height": 800}),
        image("unknown", "https://cdn.example/photos/another-scene.jpg"),
    ]
    assert all(is_site_asset(item) for item in items[:7])
    assert not is_site_asset(items[7]) and not is_site_asset(items[8])
    assert [item.candidate_id for item in dedupe_media_candidates(items)] == ["large", "unknown"]
