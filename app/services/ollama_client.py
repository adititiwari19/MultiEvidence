"""Local Ollama vision verification with the existing strict result schema."""

import json

import requests
from pydantic import ValidationError

from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_NUM_CTX
from app.services.openrouter_client import OpenRouterOutputError, OpenRouterRequestError, parse_model_json


def verify_with_ollama(prompt: str, images: list[tuple[str, str]]):
    encoded = [url.split(",", 1)[1] for _, url in images if url.startswith("data:") and "," in url]
    if len(encoded) != len(images):
        raise ValueError("Ollama verifier requires base64 data URLs for every frame")
    messages = [{"role": "user", "content": prompt, "images": encoded}]
    last_error = ""
    for attempt in range(2):
        try:
            response = requests.post(
                OLLAMA_BASE_URL.rstrip("/") + "/api/chat",
                json={"model": OLLAMA_MODEL, "messages": messages, "format": "json",
                      "think": False, "stream": False,
                      "options": {"temperature": 0, "num_ctx": OLLAMA_NUM_CTX,
                                  "num_predict": 1536}}, timeout=300,
            )
            response.raise_for_status()
            payload = response.json()
            message = payload["message"]
            raw = message.get("content") or ""
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            try:
                detail = exc.response.json().get("error", "") if exc.response is not None else ""
            except (ValueError, TypeError, AttributeError):
                detail = ""
            detail = str(detail)[:400]
            hint = " Restart Ollama with CUDA_VISIBLE_DEVICES=-1 or select a GPU with enough free memory." if "out of memory" in detail.lower() else ""
            raise OpenRouterRequestError(f"Local Ollama HTTP {status}: {detail or 'request failed'}.{hint}") from exc
        except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
            raise OpenRouterRequestError(f"Local Ollama request failed: {type(exc).__name__}") from exc
        if not raw.strip():
            last_error = (f"empty content (done_reason={payload.get('done_reason', 'unknown')}, "
                          f"thinking_chars={len(message.get('thinking') or '')}, "
                          f"eval_count={payload.get('eval_count', 'unknown')})")
            if attempt == 0:
                messages.append({"role": "user", "content": "Your reply was empty. Disable reasoning and return one complete JSON object with all required fields. Cite a supplied frame index when evidence is direct."})
                continue
            break
        try:
            return parse_model_json(raw)
        except (ValueError, ValidationError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            if attempt == 0:
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": "Return one valid JSON object with every required field, allowed relation, and cited frame index. Do not add markdown."})
    hint = " Use a non-thinking vision model such as qwen2.5vl:7b." if "empty content" in last_error and "done_reason=length" in last_error else ""
    raise OpenRouterOutputError(f"Local Ollama returned invalid JSON after one repair attempt: {last_error}.{hint}")
