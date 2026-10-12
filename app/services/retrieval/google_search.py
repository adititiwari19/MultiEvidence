"""Google Programmable Search discovery for article pages and direct images."""

import re
import time
from datetime import datetime, timezone
from urllib.parse import urlsplit

import requests

from app.config import GOOGLE_CSE_API_KEY, GOOGLE_CSE_CX, HTTP_TIMEOUT
from app.models.schemas import QueryPlan, SearchRequest
from app.services.retrieval.cache import digest, read_json, write_json
from app.services.retrieval.news_search import RetrievedArticle, canonical_url


GOOGLE_CSE_URL = "https://www.googleapis.com/customsearch/v1"
CACHE_SECONDS = 1800
COMMON_WORDS = {"about", "after", "again", "along", "amid", "because", "before", "could", "from", "have", "into", "remain", "remains", "report", "reports", "their", "there", "these", "those", "through", "under", "while", "would"}


def google_configured() -> bool:
    return bool(GOOGLE_CSE_API_KEY and GOOGLE_CSE_CX)


def google_queries(request: SearchRequest, plan: QueryPlan) -> list[str]:
    text = f"{request.topic} {request.claim.text}"
    if re.search(r"\b(?:strait of hormuz|hormuz strait)\b", text, re.I):
        anchor = '"Strait of Hormuz"'
        visual = "ship tanker vessel"
    else:
        location = next((part.strip() for part in plan.locations if part.strip()), request.topic.strip())
        anchor = f'"{location}"' if " " in location else location
        visual = "photo image"
    claim_words = [word.lower() for word in re.findall(r"[A-Za-z]{4,}", request.claim.text)]
    anchor_words = set(re.findall(r"[a-z]+", anchor.lower()))
    selected = []
    for word in claim_words:
        if word in anchor_words or word in COMMON_WORDS or word in selected:
            continue
        selected.append(word)
        if len(selected) == 4:
            break
    specific = " ".join(selected) or "news"
    return list(dict.fromkeys([f"{anchor} {visual}", f"{anchor} {specific}"]))


def _date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _metadata(item: dict) -> dict:
    page = item.get("pagemap") or {}
    tags = page.get("metatags") or []
    return tags[0] if isinstance(tags, list) and tags and isinstance(tags[0], dict) else {}


def parse_google_results(payload: dict, *, image: bool) -> list[RetrievedArticle]:
    rows = []
    for item in payload.get("items") or []:
        if not isinstance(item, dict):
            continue
        info = item.get("image") or {}
        article_url = info.get("contextLink") if image else item.get("link")
        media_url = item.get("link") if image else None
        if not isinstance(article_url, str) or urlsplit(article_url).scheme not in {"http", "https"}:
            continue
        tags = _metadata(item)
        if not image:
            page = item.get("pagemap") or {}
            images = page.get("cse_image") or []
            media_url = tags.get("og:image") or (images[0].get("src") if images and isinstance(images[0], dict) else None)
        if media_url and (not isinstance(media_url, str) or urlsplit(media_url).scheme not in {"http", "https"}):
            media_url = None
        if image and not media_url:
            continue
        published = next((_date(tags.get(key)) for key in ("article:published_time", "datepublished", "datePublished", "pubdate", "date") if _date(tags.get(key))), None)
        published = published or _date(item.get("date"))
        rows.append(RetrievedArticle(
            url=canonical_url(article_url), title=item.get("title") or "",
            source_name=item.get("displayLink") or urlsplit(article_url).netloc,
            published_at=published, snippet=item.get("snippet") or "",
            socialimage=media_url, image_collage=image,
        ))
    return rows


def search_google(query: str, *, image: bool, max_results: int = 10, timespan: str | None = None) -> list[RetrievedArticle]:
    if not google_configured():
        raise ValueError("Google search requires GOOGLE_CSE_API_KEY and GOOGLE_CSE_CX")
    if max_results <= 0:
        return []
    key = digest("google-cse-v1", GOOGLE_CSE_CX, query.strip().lower(), str(image), str(min(10, max_results)), timespan or "")
    cached = read_json("google_search", key)
    if cached and 0 <= time.time() - cached.get("cached_at", 0) < CACHE_SECONDS:
        return [RetrievedArticle.model_validate(row) for row in cached["results"]]
    params = {"key": GOOGLE_CSE_API_KEY, "cx": GOOGLE_CSE_CX, "q": query, "num": min(10, max_results)}
    if image:
        params["searchType"] = "image"
    if timespan:
        params["dateRestrict"] = {"24h": "d1", "3days": "d3", "1week": "w1", "3months": "m3"}[timespan]
    response = requests.get(GOOGLE_CSE_URL, params=params, timeout=HTTP_TIMEOUT,
                            headers={"User-Agent": "MultiEvidence/1.0"})
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Google search returned a non-object response")
    results = parse_google_results(payload, image=image)
    write_json("google_search", key, {"cached_at": time.time(), "results": [row.model_dump(mode="json") for row in results]})
    return results
