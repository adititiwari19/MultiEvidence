"""SearXNG JSON search adapter for a self-hosted, keyless search service."""

import time
from urllib.parse import urlsplit

import requests

from app.config import HTTP_TIMEOUT, SEARXNG_BASE_URL
from app.services.retrieval.cache import digest, read_json, write_json
from app.services.retrieval.news_search import RetrievedArticle, canonical_url


CACHE_SECONDS = 6 * 3600


def searxng_configured() -> bool:
    return bool(SEARXNG_BASE_URL)


def parse_searxng_results(payload: dict, *, image: bool, max_results: int = 10) -> list[RetrievedArticle]:
    rows = []
    for item in payload.get("results") or []:
        if not isinstance(item, dict):
            continue
        page = item.get("url")
        media = item.get("img_src") if image else item.get("thumbnail")
        if not isinstance(page, str) or urlsplit(page).scheme not in {"http", "https"}:
            continue
        if image and (not isinstance(media, str) or urlsplit(media).scheme not in {"http", "https"} or media == page):
            continue
        if not image and (not isinstance(media, str) or urlsplit(media).scheme not in {"http", "https"}):
            media = None
        rows.append(RetrievedArticle(url=canonical_url(page), title=item.get("title") or "",
                                     source_name=urlsplit(page).netloc, snippet=item.get("content") or "",
                                     socialimage=media, image_collage=image))
        if len(rows) >= max_results:
            break
    return rows


def search_searxng(query: str, *, image: bool, max_results: int = 10) -> list[RetrievedArticle]:
    if not searxng_configured():
        raise ValueError("SearXNG search requires SEARXNG_BASE_URL")
    if max_results <= 0:
        return []
    max_results = min(30, max_results)
    key = digest("searxng-v1", SEARXNG_BASE_URL, query.strip().lower(), str(image), str(max_results))
    cached = read_json("searxng_search", key)
    if cached and 0 <= time.time() - cached.get("cached_at", 0) < CACHE_SECONDS:
        return [RetrievedArticle.model_validate(row) for row in cached["results"]]
    response = requests.get(SEARXNG_BASE_URL.rstrip("/") + "/search",
                            params={"q": query, "format": "json", "categories": "images" if image else "general"},
                            timeout=HTTP_TIMEOUT, headers={"User-Agent": "MultiEvidence/1.0"})
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("SearXNG returned a non-object response")
    results = parse_searxng_results(payload, image=image, max_results=max_results)
    write_json("searxng_search", key, {"cached_at": time.time(),
                                       "results": [row.model_dump(mode="json") for row in results]})
    return results
