import json
import logging

import pytest
from PIL import Image

from app.models.schemas import CheckRequest
from app.services.openrouter_client import ModelReference, ModelResult, OpenRouterOutputError
from app.services.verifier import check, resolve_frame_reference
from app.services.video import Frame


def image_request(tmp_path, context=None):
    path = tmp_path / "evidence.jpg"
    Image.new("RGB", (240, 240), "gray").save(path)
    request = CheckRequest.model_validate({
        "report_id": "r", "topic": "Spokane wildfire",
        "claim": {"claim_id": "c", "text": "No homes were destroyed in the Spokane wildfire."},
        "evidence": [{"evidence_id": "e", "type": "image", "local_path": str(path), "context_text": context}],
    })
    return request


@pytest.mark.parametrize("context,relation", [
    ("Spokane wildfire: politician speaks at a podium. Article says 649 homes were destroyed.", "CONTRADICTS"),
    ("Spokane wildfire reconstruction site with a building under construction.", "REFINES"),
])
def test_indirect_image_cannot_verify_claim(monkeypatch, tmp_path, context, relation):
    monkeypatch.setattr("app.services.verifier.assess_relevance", lambda request: 0.85)
    monkeypatch.setattr("app.services.verifier.verify_with_openrouter", lambda prompt, images: ModelResult.model_validate({
        "relation": relation, "confidence": 0.99, "direct_visual_evidence": False,
        "observable_facts": ["The article claims destroyed homes."],
        "evidence": [{"frame_id": "e_image", "observation": "A podium or construction site is visible."}],
        "explanation": "The article reports destruction.", "suggested_new_text": "Homes were destroyed.", "limitations": [],
    }))
    result = check(image_request(tmp_path, context))
    assert result.event_match is True
    assert result.relation.value == "INSUFFICIENT"
    assert result.confidence <= 0.75
    assert result.suggested_update.action == "KEEP"
    assert result.observable_facts == []


def test_different_event_is_irrelevant_before_vlm(monkeypatch, tmp_path):
    monkeypatch.setattr("app.services.verifier.verify_with_openrouter", lambda *_: (_ for _ in ()).throw(AssertionError("VLM should not run")))
    result = check(image_request(tmp_path, "Spokane Valley apartment fire damages one building."))
    assert result.event_match is False
    assert "building fire" in result.event_match_reason
    assert result.relation.value == "IRRELEVANT"
    assert result.evidence == []


def test_article_context_cannot_supply_missing_direct_visual_confirmation(monkeypatch, tmp_path):
    monkeypatch.setattr("app.services.verifier.assess_relevance", lambda request: 0.85)
    monkeypatch.setattr("app.services.verifier.verify_with_openrouter", lambda prompt, images: ModelResult.model_validate({
        "relation": "CONTRADICTS", "confidence": 0.97,
        "observable_facts": ["The article reports 649 destroyed homes."],
        "evidence": [{"frame_id": "e_image", "observation": "A politician speaks at a podium."}],
        "explanation": "The article reports destroyed homes.", "suggested_new_text": "Homes were destroyed.", "limitations": [],
    }))
    result = check(image_request(tmp_path, "Spokane wildfire article: 649 homes destroyed; politician at podium."))
    assert result.relation.value == "INSUFFICIENT"
    assert result.suggested_update.action == "KEEP"


def test_uncertain_event_match_reduces_confidence(monkeypatch, tmp_path):
    monkeypatch.setattr("app.services.verifier.assess_relevance", lambda request: 0.85)
    monkeypatch.setattr("app.services.verifier.verify_with_openrouter", lambda prompt, images: ModelResult.model_validate({
        "relation": "SUPPORTS", "confidence": 0.95, "direct_visual_evidence": True,
        "observable_facts": ["Homes are visible."],
        "evidence": [{"frame_id": "e_image", "observation": "Homes are visible."}],
        "explanation": "Directly visible.", "suggested_new_text": "No homes were destroyed.", "limitations": [],
    }))
    result = check(image_request(tmp_path))
    assert result.event_match == "uncertain"
    assert result.confidence == 0.76


def test_default_confidence_is_capped_at_point_nine(monkeypatch, tmp_path):
    monkeypatch.setattr("app.services.verifier.assess_relevance", lambda request: 0.85)
    monkeypatch.setattr("app.services.verifier.verify_with_openrouter", lambda prompt, images: ModelResult.model_validate({
        "relation": "SUPPORTS", "confidence": 0.99,
        "observable_facts": ["Homes are visible."],
        "evidence": [{"frame_id": "e_image", "observation": "Homes are visible."}],
        "explanation": "Homes are visible.", "suggested_new_text": "No homes were destroyed.", "limitations": [],
    }))
    result = check(image_request(tmp_path))
    assert result.confidence <= 0.90


def test_direct_unambiguous_evidence_can_exceed_point_nine(monkeypatch, tmp_path):
    monkeypatch.setattr("app.services.verifier.assess_relevance", lambda request: 0.85)
    monkeypatch.setattr("app.services.verifier.verify_with_openrouter", lambda prompt, images: ModelResult.model_validate({
        "relation": "CONTRADICTS", "confidence": 0.96, "direct_visual_evidence": True,
        "observable_facts": ["Destroyed homes are visible."],
        "evidence": [{"frame_id": "e_image", "observation": "Destroyed homes are visible."}],
        "explanation": "The homes are visibly destroyed.", "suggested_new_text": "Homes were destroyed.", "limitations": [],
    }))
    result = check(image_request(tmp_path, "Spokane wildfire: homes visibly destroyed near Spokane."))
    assert result.event_match is True
    assert result.relation.value == "CONTRADICTS"
    assert result.confidence == 0.96


def test_eight_frame_indices_keep_exact_sent_order_and_timestamps(monkeypatch, tmp_path, caplog):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"test")
    timestamps = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]
    frames = []
    for index, timestamp in enumerate(timestamps):
        path = tmp_path / f"frame_{index}.jpg"
        Image.new("RGB", (16, 16), "blue").save(path)
        frames.append(Frame(f"v_{index}", timestamp, path))
    sent_order = [frames[index] for index in [3, 0, 7, 1, 6, 2, 5, 4]]
    monkeypatch.setattr("app.services.verifier.assess_relevance", lambda request: 0.9)
    monkeypatch.setattr("app.services.verifier.extract_frames", lambda path, evidence_id: frames)
    monkeypatch.setattr("app.services.verifier.rank_frames", lambda claim, candidates: sent_order)

    def model(prompt, images):
        metadata = json.loads(prompt.split("Input:\n", 1)[1])
        assert [entry["frame_id"] for entry in metadata["frames"]] == [frame.frame_id for frame in sent_order]
        assert [frame_id for frame_id, _ in images] == [frame.frame_id for frame in sent_order]
        return ModelResult.model_validate({
            "relation": "REFINES", "confidence": 0.87, "direct_visual_evidence": True,
            "observable_facts": ["The roadway changes over time."],
            "evidence": [
                {"frame_index": 0, "observation": "Observation at input zero."},
                {"frame_id": "frame 7", "observation": "Observation at input seven."},
                {"frame_index": 4, "observation": "Observation at input four."},
            ],
            "explanation": "The selected frames show a change.", "suggested_new_text": "Road conditions changed.", "limitations": [],
        })

    monkeypatch.setattr("app.services.verifier.verify_with_openrouter", model)
    request = CheckRequest.model_validate({
        "report_id": "r", "topic": "Goma road flooding", "claim": {"claim_id": "c", "text": "The Goma road is open."},
        "evidence": [{"evidence_id": "v", "type": "video", "local_path": str(video_path), "context_text": "Goma road flooding"}],
    })
    with caplog.at_level(logging.INFO):
        result = check(request)
    assert [(ref.frame_id, ref.timestamp_seconds, ref.observation) for ref in result.evidence] == [
        ("v_3", 15.0, "Observation at input zero."),
        ("v_4", 20.0, "Observation at input seven."),
        ("v_6", 30.0, "Observation at input four."),
    ]
    assert "frame_index=0 frame_id=v_3 timestamp_seconds=15.0" in caplog.text
    assert "frame_index=7 frame_id=v_4 timestamp_seconds=20.0" in caplog.text


def test_conflicting_frame_id_and_index_are_rejected():
    frames = [("v_0", 10.0, "/media/frames/v_0.jpg"), ("v_1", 25.0, "/media/frames/v_1.jpg")]
    with pytest.raises(OpenRouterOutputError):
        resolve_frame_reference(ModelReference(frame_id="frame 0", frame_index=1, observation="Mismatch"), frames)
