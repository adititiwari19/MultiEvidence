from app.services import relevance


def test_weighted_score_and_threshold(monkeypatch):
    monkeypatch.setattr(relevance, "semantic_similarity", lambda claim, context: 0.2)
    assert relevance.relevance_score("road open", "road blocked") == 0.29
    assert relevance.is_irrelevant(0.29)
    assert not relevance.is_irrelevant(0.30)


def test_metadata_relevance_skips_vlm(monkeypatch, tmp_path):
    from PIL import Image

    from app.models.schemas import CheckRequest
    from app.services.verifier import check

    path = tmp_path / "unrelated.jpg"
    Image.new("RGB", (8, 8)).save(path)
    monkeypatch.setattr("app.services.verifier.assess_relevance", lambda request: 0.1)
    monkeypatch.setattr("app.services.verifier.verify_with_openrouter", lambda *_: (_ for _ in ()).throw(AssertionError("VLM called")))
    request = CheckRequest.model_validate({"report_id": "r", "topic": "weather", "claim": {"claim_id": "c", "text": "A road is open."}, "evidence": [{"evidence_id": "e", "type": "image", "local_path": str(path)}]})
    result = check(request)
    assert result.relation.value == "IRRELEVANT"
    assert result.suggested_update.action == "KEEP"


def test_generic_filename_does_not_reject_visual_evidence():
    from app.models.schemas import CheckRequest

    request = CheckRequest.model_validate({"report_id": "r", "topic": "Measles response", "claim": {"claim_id": "c", "text": "The road is open."}, "evidence": [{"evidence_id": "e", "type": "video", "local_path": "./data/media/example.mp4"}]})
    assert relevance.assess_relevance(request) == relevance.RELEVANCE_THRESHOLD
