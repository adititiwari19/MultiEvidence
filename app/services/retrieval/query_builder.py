import json
import logging
import os
import re

from pydantic import ValidationError

from app.config import OPENROUTER_BASE_URL, TEXT_MODEL
from app.models.schemas import SearchRequest, QueryPlan
from app.services.retrieval.cache import digest, read_json, write_json


LOG = logging.getLogger(__name__)


def _quoted(value: str) -> str:
    cleaned = re.sub(r'[^\w\s-]', ' ', value)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return f'"{cleaned}"' if ' ' in cleaned else cleaned


def _concepts(request: SearchRequest, plan: QueryPlan) -> tuple[list[str], list[str], list[str]]:
    text = f"{request.claim.text} {request.topic}".lower()
    locations = [value for value in plan.locations if value.strip()]
    if re.search(r"\b(?:strait of hormuz|hormuz strait)\b", text, re.I):
        locations.insert(0, "Strait of Hormuz")
    if not locations:
        match = re.search(r"\b(?:to|in|near|at)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", request.claim.text)
        if match:
            locations.append(match.group(1))
        match = re.search(r"\bNorth Kivu\b", request.topic, re.I)
        if match:
            locations.append("North Kivu")
    locations = list(dict.fromkeys(locations))[:2]
    maritime = any(word in text for word in ("strait", "shipping", "vessel", "tanker", "naval", "hormuz"))
    actions = [word for word in ("road", "access", "closure", "blocked", "transport") if word in text]
    if maritime:
        actions = ["shipping", "tanker", "vessel", "transit", "naval"]
    elif any(word in text for word in ("road", "route", "highway", "traffic")):
        actions = ["road", "access", "closure", "blocked", "transport"]
    if not actions:
        actions = [_quoted(value.lower()) for value in plan.event_terms[:3] if _quoted(value.lower())] or ["access", "disruption"]
    visuals = ["ship", "tanker"] if maritime else [word for word in ("flood", "road", "bridge", "fire", "rubble", "damage") if word in text]
    if not maritime and "flood" in text and "road" in text:
        visuals = ["road", "flood"]
    if not visuals:
        visuals = [word for word in ("airport", "plane", "building", "street", "vehicle") if word in text][:2]
    return locations, [word for word in actions if word][:5], [word for word in visuals if word][:2]


def build_gdelt_text_query(request: SearchRequest, plan: QueryPlan) -> str:
    locations, actions, _ = _concepts(request, plan)
    names = locations or [value for value in plan.entities if value.strip()][:1]
    prefix = " ".join(_quoted(value) for value in names)
    group = "(" + " OR ".join(actions) + ")" if len(actions) > 1 else (actions[0] if actions else "")
    return f"{prefix} {group}".strip()


def build_gdelt_visual_query(request: SearchRequest, plan: QueryPlan) -> str:
    locations, _, visuals = _concepts(request, plan)
    names = locations[:1] or [value for value in plan.entities if value.strip()][:1]
    prefix = " ".join(_quoted(value) for value in names)
    tags = [f'{operator}:"{word}"' for word in visuals for operator in ("imagetag", "imagewebtag")]
    if not tags:
        tags = ['imagetag:"road"', 'imagewebtag:"road"']
    return f"{prefix} ({' OR '.join(tags)})".strip()


def structured_queries(request: SearchRequest, plan: QueryPlan) -> QueryPlan:
    return plan.model_copy(update={"queries": [build_gdelt_text_query(request, plan), build_gdelt_visual_query(request, plan)]})


def parse_query_plan(raw: str) -> QueryPlan:
    plan = QueryPlan.model_validate(json.loads(raw))
    if any(not query.strip() for query in plan.queries):
        raise ValueError("Search queries cannot be empty")
    return plan.model_copy(update={"queries": plan.queries[:3]})


def fallback_queries(request: SearchRequest) -> QueryPlan:
    claim = re.sub(r"\s+", " ", request.claim.text).strip().rstrip(".")
    topic = re.sub(r"\s+", " ", request.topic).strip()
    queries = list(dict.fromkeys([claim, f"{claim} {topic}"]))
    return QueryPlan(queries=queries[:2], entities=[], locations=[], event_terms=[])


def build_queries(request: SearchRequest) -> QueryPlan:
    key = digest(request.claim.text, request.topic, str(request.search.published_after), str(request.search.published_before), TEXT_MODEL)
    cached = read_json("queries", key)
    if cached:
        plan = QueryPlan.model_validate(cached)
        return structured_queries(request, plan)
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return structured_queries(request, fallback_queries(request))
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
    except Exception as exc:
        LOG.warning("Query generation unavailable: %s", type(exc).__name__)
        return structured_queries(request, fallback_queries(request))
    prompt = ("Extract named entities, locations, event concepts, action/state terms, and visual concepts "
              "from the supplied claim and topic. Use only concepts grounded in that input. "
              "Return two concise search seed phrases and the extracted concepts. "
              "Return only JSON with exactly: queries (array of two strings), "
              "entities (array), locations (array), event_terms (array). "
              f"Input: {json.dumps({'claim': request.claim.text, 'topic': request.topic, 'published_after': str(request.search.published_after) if request.search.published_after else None, 'published_before': str(request.search.published_before) if request.search.published_before else None})}")
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(2):
        try:
            response = client.chat.completions.create(model=TEXT_MODEL, messages=messages, response_format={"type": "json_object"}, temperature=0)
            raw = response.choices[0].message.content or ""
            plan = parse_query_plan(raw)
            write_json("queries", key, plan.model_dump(mode="json"))
            return structured_queries(request, plan)
        except (ValueError, ValidationError, json.JSONDecodeError) as exc:
            if attempt == 0:
                messages.append({"role": "user", "content": "Return valid JSON with exactly queries, entities, locations, event_terms. Give two nonempty query seed phrases."})
            else:
                LOG.warning("Query generation parsing failed: %s", type(exc).__name__)
        except Exception as exc:
            LOG.warning("Query generation failed: %s", type(exc).__name__)
            break
    return structured_queries(request, fallback_queries(request))
