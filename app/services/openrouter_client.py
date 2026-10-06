import json
import os

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.config import OPENROUTER_BASE_URL, VLM_MODEL
from app.models.schemas import Relation


class ModelReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frame_id: str | None = None
    frame_index: int | None = Field(default=None, ge=0)
    observation: str

    @model_validator(mode="after")
    def has_frame_reference(self):
        if self.frame_id is None and self.frame_index is None:
            raise ValueError("A frame_id or frame_index is required")
        return self


class ModelResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    relation: Relation
    confidence: float = Field(ge=0, le=1)
    observable_facts: list[str]
    evidence: list[ModelReference]
    explanation: str
    suggested_new_text: str
    limitations: list[str]
    direct_visual_evidence: bool | None = None


class OpenRouterOutputError(Exception):
    pass


class OpenRouterRequestError(Exception):
    pass


def parse_model_json(content: str) -> ModelResult:
    data = json.loads(content)
    result = ModelResult.model_validate(data)
    if result.relation == Relation.IRRELEVANT:
        raise ValueError("IRRELEVANT is reserved for the relevance gate")
    if result.relation != Relation.INSUFFICIENT and not result.evidence:
        raise ValueError("Verified relations require at least one cited frame")
    return result


def verify_with_openrouter(prompt: str, images: list[tuple[str, str]]) -> ModelResult:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OPENROUTER_API_KEY is missing")
    from openai import OpenAI

    client = OpenAI(api_key=key, base_url=OPENROUTER_BASE_URL)
    content = [{"type": "text", "text": prompt}]
    content.extend({"type": "image_url", "image_url": {"url": url}} for _, url in images)
    messages = [{"role": "user", "content": content}]
    last_error = ""
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=VLM_MODEL,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0,
            )
        except Exception as exc:
            raise OpenRouterRequestError(f"OpenRouter request failed: {type(exc).__name__}") from exc
        raw = response.choices[0].message.content or ""
        try:
            return parse_model_json(raw)
        except (ValueError, ValidationError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            if attempt == 0:
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": "Your previous response was invalid. Return only one valid JSON object matching every requested field and allowed value. Do not add markdown."})
    raise OpenRouterOutputError(f"VLM returned invalid JSON after one repair attempt: {last_error}")
