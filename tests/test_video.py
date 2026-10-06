import subprocess
from pathlib import Path

from app.models.schemas import CheckRequest
from app.services.openrouter_client import ModelResult
from app.services.video import extract_frames, probe_video, sample_timestamps
from app.services.verifier import check


def make_video(path: Path):
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=blue:s=32x32:r=1:d=11", "-c:v", "mpeg4", "-y", str(path)], check=True)


def test_video_timestamps_and_cap(tmp_path):
    path = tmp_path / "sample.mp4"
    make_video(path)
    duration, fps = probe_video(path)
    assert duration == 11
    assert fps == 1
    frames = extract_frames(path, "e1")
    assert [f.timestamp_seconds for f in frames] == [0, 5, 10]
    assert all(f.path.is_file() for f in frames)
    assert len(sample_timestamps(9999)) == 120
    assert sample_timestamps(9999)[-1] == 595


def test_video_endpoint(monkeypatch, tmp_path):
    path = tmp_path / "sample.mp4"
    make_video(path)
    monkeypatch.setattr("app.services.verifier.rank_frames", lambda claim, frames: frames[:2])

    def mock_vlm(prompt, images):
        assert len(images) == 2
        assert '"timestamp_seconds": 5' in prompt
        return ModelResult.model_validate({
            "relation": "REFINES", "confidence": 0.87,
            "observable_facts": ["The frame shows a blue view."],
            "evidence": [{"frame_id": "e1_5000", "observation": "Blue view."}],
            "explanation": "The new frame adds a qualification.",
            "suggested_new_text": "The road is open in some conditions.", "limitations": [],
        })

    monkeypatch.setattr("app.services.verifier.verify_with_openrouter", mock_vlm)
    request = CheckRequest.model_validate({
        "report_id": "r1", "topic": "Road", "claim": {"claim_id": "c1", "text": "The road is open."},
        "evidence": [{"evidence_id": "e1", "type": "video", "local_path": str(path)}],
    })
    result = check(request)
    assert result.relation.value == "REFINES"
    assert result.evidence[0].timestamp_seconds == 5
    assert result.suggested_update.action == "QUALIFY"
