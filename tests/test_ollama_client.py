import json
from unittest.mock import patch

import pytest
import requests

from app.services import ollama_client
from app.services.openrouter_client import OpenRouterOutputError, OpenRouterRequestError


class FakeResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {"message": {"content": json.dumps({
            "relation": "SUPPORTS", "confidence": 0.8,
            "observable_facts": ["A ship is visible."],
            "evidence": [{"frame_index": 0, "observation": "A ship is visible."}],
            "explanation": "The image directly shows a ship.",
            "suggested_new_text": "A ship is visible in the strait.",
            "limitations": [], "direct_visual_evidence": True,
        })}}


def test_local_ollama_sends_frame_and_validates_result():
    with patch.object(ollama_client.requests, "post", return_value=FakeResponse()) as post:
        result = ollama_client.verify_with_ollama("Check the claim", [("frame_0", "data:image/jpeg;base64,QUJD")])
    payload = post.call_args.kwargs["json"]
    assert payload["messages"][0]["images"] == ["QUJD"]
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["options"]["num_ctx"] == 8192
    assert result.relation.value == "SUPPORTS"


def test_ollama_out_of_memory_error_gives_actionable_message():
    response = requests.Response()
    response.status_code = 500
    response._content = b'{"error":"CUDA error: out of memory"}'
    with patch.object(ollama_client.requests, "post", return_value=response):
        with pytest.raises(OpenRouterRequestError, match="CUDA_VISIBLE_DEVICES=-1"):
            ollama_client.verify_with_ollama("Check", [("frame_0", "data:image/jpeg;base64,QUJD")])


def test_empty_thinking_response_retries_without_echoing_thoughts():
    class ThinkingResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "", "thinking": "private reasoning"},
                    "done_reason": "length", "eval_count": 1536}

    with patch.object(ollama_client.requests, "post", side_effect=[ThinkingResponse(), FakeResponse()]) as post:
        result = ollama_client.verify_with_ollama("Check", [("frame_0", "data:image/jpeg;base64,QUJD")])
    assert result.relation.value == "SUPPORTS"
    assert post.call_count == 2
    assert all("private reasoning" not in str(message) for message in post.call_args.kwargs["json"]["messages"])


def test_repeated_empty_response_reports_finish_reason():
    class EmptyResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "", "thinking": "hidden"}, "done_reason": "length", "eval_count": 1536}

    with patch.object(ollama_client.requests, "post", return_value=EmptyResponse()):
        with pytest.raises(OpenRouterOutputError, match="empty content.*done_reason=length"):
            ollama_client.verify_with_ollama("Check", [("frame_0", "data:image/jpeg;base64,QUJD")])
