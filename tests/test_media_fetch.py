from io import BytesIO
import logging

from PIL import Image

from app.models.schemas import MediaCandidate
from app.services.retrieval.media_fetch import download_media, safe_log_url


def candidate(url="https://cdn.example/road.jpg", media_type="image", metadata=None):
    return MediaCandidate(candidate_id="c1", article_url="https://news.example/story", source_name="News", media_type=media_type, media_url=url, title="Road", metadata=metadata or {})


def test_download_image_and_cache(monkeypatch, tmp_path):
    monkeypatch.setattr("app.services.retrieval.media_fetch.RETRIEVED_DIR", tmp_path)
    buffer = BytesIO()
    Image.new("RGB", (240, 240), "blue").save(buffer, format="JPEG")
    calls = []

    class Response:
        headers = {"Content-Type": "image/jpeg"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield buffer.getvalue()

    def get(*args, **kwargs):
        calls.append(args)
        return Response()

    monkeypatch.setattr("app.services.retrieval.media_fetch.requests.get", get)
    first = download_media(candidate(), "claim_1")
    second = download_media(candidate(), "claim_1")
    assert first.local_path == second.local_path
    assert len(calls) == 1


def test_embedded_video_is_unavailable_without_download(monkeypatch):
    monkeypatch.setattr("app.services.retrieval.media_fetch.requests.get", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("download attempted")))
    item = download_media(candidate("https://video.example/embed/123", "video", {"embedded": True}), "c")
    assert item.unavailable_for_visual_verification


def test_download_failure_logs_url_type_and_message(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr("app.services.retrieval.media_fetch.RETRIEVED_DIR", tmp_path)

    class Response:
        headers = {"Content-Type": "application/octet-stream"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

    monkeypatch.setattr("app.services.retrieval.media_fetch.requests.get", lambda *args, **kwargs: Response())
    with caplog.at_level(logging.WARNING):
        assert download_media(candidate("https://cdn.example/road.jpg?image_id=42"), "c") is None
    assert "media_download_failed url=https://cdn.example/road.jpg?image_id=42" in caplog.text
    assert "error_type=ValueError" in caplog.text
    assert "error=Unsupported media MIME type" in caplog.text


def test_media_error_url_redacts_secret_query_values():
    safe = safe_log_url("https://cdn.example/image.jpg?image_id=42&token=private-value")
    assert "image_id=42" in safe
    assert "private-value" not in safe
