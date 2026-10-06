import pytest
from pydantic import ValidationError

from app.models.schemas import CheckRequest, CheckResponse


def test_contract_schema():
    request = CheckRequest.model_validate({
        "report_id": "r1", "topic": "road", "claim": {"claim_id": "c1", "text": "Road open"},
        "evidence": [{"evidence_id": "e1", "type": "image", "local_path": "data/media/a.jpg"}],
    })
    assert request.claim.claim_id == "c1"
    with pytest.raises(ValidationError):
        CheckRequest.model_validate({**request.model_dump(), "evidence": []})
    with pytest.raises(ValidationError):
        CheckResponse.model_validate({"claim_id": "c1", "relation": "UNKNOWN"})


def test_event_match_fields_are_backward_compatible():
    response = CheckResponse.model_validate({"claim_id": "c1", "affected": False, "relation": "INSUFFICIENT", "relevance": 0.5, "confidence": 0.5, "observable_facts": [], "evidence": [], "suggested_update": {"action": "KEEP", "old_text": "Road open", "new_text": "Road open", "reason": "Unclear"}, "explanation": "Unclear", "limitations": []})
    assert response.event_match == "uncertain"
    assert response.event_match_reason == ""
