import asyncio
import json

from PIL import Image

from app.main import app
from app.services.openrouter_client import ModelResult


def asgi_request(method: str, path: str, payload: dict | None = None):
    async def run():
        body = json.dumps(payload).encode() if payload is not None else b""
        sent = []
        first = True

        async def receive():
            nonlocal first
            if first:
                first = False
                return {"type": "http.request", "body": body, "more_body": False}
            await asyncio.sleep(3600)

        async def send(message):
            sent.append(message)

        scope = {"type": "http", "asgi": {"version": "3.0"}, "method": method, "path": path, "raw_path": path.encode(), "query_string": b"", "headers": [(b"content-type", b"application/json")], "client": ("test", 1), "server": ("test", 80), "scheme": "http", "root_path": ""}
        await asyncio.wait_for(app(scope, receive, send), timeout=5)
        return sent[0]["status"], json.loads(sent[1]["body"])

    return asyncio.run(run())


def test_health_route():
    status, body = asgi_request("GET", "/health")
    assert status == 200
    assert body == {"status": "ok"}


def test_endpoint_with_mocked_openrouter(monkeypatch, tmp_path):
    path = tmp_path / "road.jpg"
    Image.new("RGB", (8, 8), "gray").save(path)
    monkeypatch.setattr("app.services.verifier.assess_relevance", lambda request: 0.6)
    monkeypatch.setattr("app.services.verifier.verify_with_openrouter", lambda prompt, images: ModelResult.model_validate({"relation": "SUPPORTS", "confidence": 0.9, "observable_facts": ["Gray pixels are visible."], "evidence": [{"frame_id": "e_image", "observation": "Gray pixels."}], "explanation": "Only gray pixels are visible.", "suggested_new_text": "The road is open.", "limitations": []}))
    status, body = asgi_request("POST", "/api/v1/multimodal/check-evidence", {"report_id": "r", "topic": "road", "claim": {"claim_id": "c", "text": "The road is open."}, "evidence": [{"evidence_id": "e", "type": "image", "local_path": str(path)}]})
    assert status == 200
    assert body["claim_id"] == "c"
    assert body["relation"] == "SUPPORTS"
    assert body["evidence"][0]["frame_url"] == "/media/frames/e_image.jpg"
