from pathlib import Path

from PIL import Image

from app.models.schemas import CheckRequest
from app.services.openrouter_client import ModelResult
from app.services.verifier import check


def test_image_endpoint(monkeypatch, tmp_path: Path):
    image_path = tmp_path / "road.jpg"
    Image.new("RGB", (32, 32), "blue").save(image_path)

    def mock_vlm(prompt, images):
        assert len(images) == 1
        assert images[0][1].startswith("data:image/jpeg;base64,")
        return ModelResult.model_validate({
            "relation": "INSUFFICIENT", "confidence": 0.8,
            "observable_facts": ["A blue image is shown."],
            "evidence": [{"frame_id": "e1_image", "observation": "A blue image is shown."}],
            "explanation": "The image does not establish road conditions.",
            "suggested_new_text": "The road is open.", "limitations": ["Road is not visible."],
        })

    monkeypatch.setattr("app.services.verifier.verify_with_openrouter", mock_vlm)
    payload = {
        "report_id": "r1", "topic": "Road", "claim": {"claim_id": "c1", "text": "The road is open."},
        "evidence": [{"evidence_id": "e1", "type": "image", "local_path": str(image_path)}],
    }
    body = check(CheckRequest.model_validate(payload)).model_dump(mode="json")
    assert body["relation"] == "INSUFFICIENT"
    assert body["suggested_update"]["action"] == "KEEP"
    assert body["evidence"][0]["frame_url"] == "/media/frames/e1_image.jpg"
