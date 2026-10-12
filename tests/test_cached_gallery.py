from unittest.mock import patch

from fastapi import HTTPException

from app.main import cached_evidence_pdf, cached_media
from app.models.schemas import SearchRequest
from app.services.cached_gallery import gallery, match_gallery
from app.services.search_and_check import search_and_check
from scripts.build_cached_gallery import build


def test_gallery_covers_saved_topics_and_resolves_swapped_pdfs():
    rows = build()["entries"]
    assert len(rows) == 32
    assert sum(len(row["media"]) for row in rows) == 84
    assert rows[0]["source_id"] == "S24722"
    assert rows[0]["media"][0]["filename"] == "claim1.png"
    assert rows[8]["evidence_pdf"] == "t2_evidence.pdf"
    assert rows[16]["evidence_pdf"] == "t1_evidence.pdf"
    assert all(row["source_url"].startswith("http") and row["media"] for row in rows)


def test_every_saved_query_hits_its_own_cached_topic():
    for row in gallery():
        for alias in row.aliases:
            hit = match_gallery(alias)
            assert hit and (hit.checkpoint, hit.claim_number) == (row.checkpoint, row.claim_number)
    assert match_gallery("T3 claim 8").claim_number == 8
    assert match_gallery("G7 warship escorts").claim_number == 5
    assert match_gallery("oil prices surge", "T0").claim_number == 8
    assert match_gallery("The road remains open") is None
    assert match_gallery("Strait of Hormuz") is None


def test_cached_request_skips_live_retrieval_and_does_not_invent_verification():
    request = SearchRequest.model_validate({"report_id": "demo", "topic": "Strait of Hormuz",
                                            "claim": {"claim_id": "c", "text": "Three Iranian tankers leave the Strait carrying 5 million barrels"}})
    with patch("app.services.search_and_check.search_google_news_rss", side_effect=AssertionError("Live search ran")):
        result = search_and_check(request)
    assert result.cache_hit
    assert result.cached_gallery[0].checkpoint == "T0"
    assert result.cached_gallery[0].claim_number == 3
    assert result.cached_gallery[0].verification_status == "CANDIDATE_UNVERIFIED"
    assert result.claim_status == "UNCERTAIN"
    assert result.verified_evidence_count == 0


def test_cached_media_is_served_with_valid_type_and_html_is_excluded():
    image = cached_media("t0", "claim1_2")
    assert image.media_type == "image/jpeg"
    assert cached_evidence_pdf("t0_evidence.pdf").media_type == "application/pdf"
    try:
        cached_media("t0", "claim3_img3.html")
    except HTTPException as exc:
        assert exc.status_code == 404
    else:
        raise AssertionError("HTML was exposed as cached media")


def test_unmatched_query_uses_existing_pipeline():
    request = SearchRequest.model_validate({
        "report_id": "demo", "topic": "Road conditions",
        "claim": {"claim_id": "outside", "text": "The road remains open."},
        "search": {"provider": "free"},
    })
    with patch("app.services.search_and_check.search_google_news_rss", return_value=[]) as live, \
         patch("app.services.search_and_check.search_ddgs_images", return_value=[]), \
         patch("app.services.search_and_check.search_video_thumbnails", return_value=[]):
        result = search_and_check(request)
    assert result.cache_hit is False
    assert live.called
