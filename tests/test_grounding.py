from PIL import Image

from app.models.schemas import CheckRequest, EvidenceReference
from app.services.grounding import Point, choose_sam_mask, ground_reference, parse_points, select_targets
from app.services.openrouter_client import ModelResult
from app.services.verifier import check


def test_ground_reference_draws_colored_points(monkeypatch, tmp_path):
    Image.new("RGB", (100, 60), "white").save(tmp_path / "frame.jpg")
    monkeypatch.setattr("app.services.grounding.FRAME_DIR", tmp_path)
    monkeypatch.setattr("app.services.grounding.ask_molmo2", lambda claim, observation, path: [Point(text="road", claim_text="road", x=0.5, y=0.5)])
    monkeypatch.setattr("app.services.grounding.sam2_predictor", lambda: None)
    ref = EvidenceReference(evidence_id="e", frame_id="frame", frame_url="/media/frames/frame.jpg", observation="Water covers the road.")
    result = ground_reference("The road is open.", ref)
    assert result.overlay_url and (tmp_path / result.overlay_url.rsplit("/", 1)[-1]).is_file()
    assert result.groundings[0].text == "road"
    assert result.groundings[0].claim_text == "road"
    assert result.groundings[0].mask_url is None


def test_ground_reference_writes_sam2_mask(monkeypatch, tmp_path):
    import numpy as np

    Image.new("RGB", (20, 20), "white").save(tmp_path / "frame.jpg")
    monkeypatch.setattr("app.services.grounding.FRAME_DIR", tmp_path)
    monkeypatch.setattr("app.services.grounding.ask_molmo2", lambda *args: [Point(text="road", x=0.5, y=0.5)])

    class Predictor:
        def set_image(self, image):
            assert image.shape[:2] == (20, 20)

        def predict(self, **kwargs):
            mask = np.zeros((1, 20, 20), dtype=bool)
            mask[0, 8:12, 8:12] = True
            return mask, np.array([0.9]), None

    monkeypatch.setattr("app.services.grounding.sam2_predictor", lambda: Predictor())
    ref = EvidenceReference(evidence_id="e", frame_id="frame", frame_url="/media/frames/frame.jpg", observation="A road is visible.")
    result = ground_reference("A road is visible.", ref)
    assert result.groundings[0].mask_url
    assert result.groundings[0].bbox == (8, 8, 11, 11)
    assert (tmp_path / result.groundings[0].mask_url.rsplit("/", 1)[-1]).is_file()


def test_molmo2_point_formats():
    assert parse_points('{"points":[{"text":"road","x":0.5,"y":0.25}]}', "The road is visible.")[0].x == 0.5
    assert parse_points('<point x="50" y="25" alt="road">', "The road is visible.")[0].y == 0.25
    assert parse_points('<point x=50 y=25 alt="flooded road"/>', "The road is visible.")[0].text == "road"
    assert parse_points('<point x=50 y=25/>', "The building is visible.", target="building")[0].text == "building"
    assert parse_points('(50, 25)', "The building is visible.", target="building")[0].x == 0.5
    assert parse_points('<points coords="0 500 250"/>', "The building is visible.", target="building")[0].y == 0.25
    assert parse_points('<points coords="0 500 250"/>', "The building has charred walls and roof.", target="charred walls")[0].text == "charred walls"


def test_sam_mask_prefers_tighter_similarly_scored_region():
    import numpy as np

    masks = np.zeros((3, 20, 20), dtype=bool)
    masks[0, :, :] = True
    masks[1, 9:11, 9:11] = True
    masks[2, 5:15, 5:15] = True
    chosen = choose_sam_mask(masks, np.array([0.414, 0.031, 0.412]))
    assert np.array_equal(chosen, masks[2])


def test_targets_prioritize_claim_entities():
    assert select_targets("A building has visible fire damage.", "The building has charred walls and roof.") == ["building with charred walls and roof", "charred walls", "charred roof"]
    assert select_targets("Flood water covers the road.", "Flood water covers the road.") == ["Flood", "water", "road"]


def test_qualified_target_links_its_claim_entity(monkeypatch, tmp_path):
    Image.new("RGB", (20, 20), "white").save(tmp_path / "frame.jpg")
    monkeypatch.setattr("app.services.grounding.FRAME_DIR", tmp_path)
    monkeypatch.setattr("app.services.grounding.ask_molmo2", lambda *args: [Point(text="building with charred walls and roof", x=0.5, y=0.5)])
    monkeypatch.setattr("app.services.grounding.sam2_predictor", lambda: None)
    ref = EvidenceReference(evidence_id="e", frame_id="frame", frame_url="/media/frames/frame.jpg", observation="The building has charred walls and roof.")
    result = ground_reference("A building has visible fire damage.", ref)
    assert result.groundings[0].claim_text == "building"


def test_grounding_runs_only_for_direct_visual_evidence(monkeypatch, tmp_path):
    image = tmp_path / "image.jpg"
    Image.new("RGB", (20, 20), "blue").save(image)
    monkeypatch.setattr("app.services.verifier.MOLMO2_MODEL", "test-molmo2")
    monkeypatch.setattr("app.services.verifier.assess_relevance", lambda request: 0.8)
    monkeypatch.setattr("app.services.verifier.event_match", lambda request: (True, "Matched source."))
    monkeypatch.setattr("app.services.verifier.verify_with_openrouter", lambda prompt, images: ModelResult.model_validate({"relation": "SUPPORTS", "direct_visual_evidence": True, "confidence": 0.8, "observable_facts": ["A road is visible."], "evidence": [{"frame_id": "e_image", "observation": "A road is visible."}], "explanation": "Road visible.", "suggested_new_text": "The road is visible.", "limitations": []}))
    calls = []
    monkeypatch.setattr("app.services.grounding.ground_reference", lambda claim, reference: calls.append(reference.frame_id) or reference.model_copy(update={"overlay_url": "/media/frames/overlay.jpg"}))
    request = CheckRequest.model_validate({"report_id": "r", "topic": "road", "claim": {"claim_id": "c", "text": "The road is visible."}, "evidence": [{"evidence_id": "e", "type": "image", "local_path": str(image)}], "ground": True})
    result = check(request)
    assert calls == ["e_image"]
    assert result.evidence[0].overlay_url == "/media/frames/overlay.jpg"


def test_insufficient_claim_can_ground_visible_object_without_becoming_support(monkeypatch, tmp_path):
    image = tmp_path / "image.jpg"
    Image.new("RGB", (20, 20), "blue").save(image)
    monkeypatch.setattr("app.services.verifier.MOLMO2_MODEL", "test-molmo2")
    monkeypatch.setattr("app.services.verifier.assess_relevance", lambda request: 0.8)
    monkeypatch.setattr("app.services.verifier.event_match", lambda request: ("uncertain", "Event identity not established."))
    monkeypatch.setattr("app.services.verifier.verify_with_openrouter", lambda prompt, images: ModelResult.model_validate({"relation": "INSUFFICIENT", "direct_visual_evidence": False, "confidence": 0.75, "observable_facts": ["A tanker is visible."], "evidence": [], "explanation": "A photo cannot establish three Iranian tankers or five million barrels.", "suggested_new_text": "A tanker is visible.", "limitations": ["Count and cargo are not visible."]}))
    calls = []
    monkeypatch.setattr("app.services.grounding.ground_reference", lambda claim, reference: calls.append(reference.observation) or reference.model_copy(update={"overlay_url": "/media/frames/tanker_overlay.jpg"}))
    request = CheckRequest.model_validate({"report_id": "r", "topic": "Hormuz", "claim": {"claim_id": "c", "text": "Three Iranian tankers carry five million barrels. A tanker is visible."}, "evidence": [{"evidence_id": "e", "type": "image", "local_path": str(image)}], "ground": True})
    result = check(request)
    assert result.relation.value == "INSUFFICIENT"
    assert calls == ["A tanker is visible."]
    assert result.evidence[0].overlay_url == "/media/frames/tanker_overlay.jpg"
