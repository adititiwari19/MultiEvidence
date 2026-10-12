"""Keyless DuckDuckGo web and image discovery.

DuckDuckGo's public HTML and image endpoints are unofficial and may change or
rate limit. Keep the requests few and cache successful responses.
"""

import re
import time
import logging
from datetime import datetime, timezone
from html import unescape
from urllib.parse import parse_qs, urlsplit

import requests
from bs4 import BeautifulSoup

from app.config import HTTP_TIMEOUT
from app.services.retrieval.cache import digest, read_json, write_json
from app.services.retrieval.news_search import RetrievedArticle, canonical_url


HTML_URL = "https://html.duckduckgo.com/html/"
TOKEN_URL = "https://duckduckgo.com/"
IMAGES_URL = "https://duckduckgo.com/i.js"
CACHE_SECONDS = 6 * 3600
LOG = logging.getLogger(__name__)
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; MultiEvidence/1.0)", "Accept-Language": "en-US,en;q=0.9"}


def _url(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    value = unescape(value.strip())
    if value.startswith("//"):
        value = "https:" + value
    parts = urlsplit(value)
    if parts.netloc.endswith("duckduckgo.com") and parts.path.startswith("/l/"):
        value = parse_qs(parts.query).get("uddg", [""])[0]
        parts = urlsplit(value)
    return canonical_url(value) if parts.scheme in {"http", "https"} and parts.netloc else None


def _date(value: str | None) -> datetime | None:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2}))?", value):
        return None
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)


def parse_web_results(html: str, *, max_results: int = 10) -> list[RetrievedArticle]:
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for block in soup.select(".result"):
        link = block.select_one(".result__a")
        url = _url(link.get("href") if link else None)
        if not url:
            continue
        snippet = block.select_one(".result__snippet")
        rows.append(RetrievedArticle(url=url, title=link.get_text(" ", strip=True),
                                     source_name=urlsplit(url).netloc,
                                     snippet=snippet.get_text(" ", strip=True) if snippet else ""))
        if len(rows) >= max_results:
            break
    return rows


def parse_image_results(payload: dict, *, max_results: int = 10) -> list[RetrievedArticle]:
    rows = []
    for item in payload.get("results") or []:
        if not isinstance(item, dict):
            continue
        page = _url(item.get("url"))
        image = _url(item.get("image"))
        if not page or not image or page == image:
            continue
        rows.append(RetrievedArticle(url=page, title=unescape(item.get("title") or ""),
                                     source_name=urlsplit(page).netloc,
                                     published_at=_date(item.get("published_at")),
                                     snippet=unescape(item.get("description") or ""),
                                     socialimage=image, image_collage=True))
        if len(rows) >= max_results:
            break
    return rows


def _image_token(html: str) -> str:
    match = re.search(r"vqd['\"]?\s*(?:=|:)\s*['\"]?([\w-]+)", html)
    if not match:
        raise ValueError("DuckDuckGo image token was not found; its page format may have changed")
    return match.group(1)


def search_duckduckgo(query: str, *, image: bool, max_results: int = 10) -> list[RetrievedArticle]:
    if max_results <= 0:
        return []
    max_results = min(30, max_results)
    key = digest("duckduckgo-v1", query.strip().lower(), str(image), str(max_results))
    cached = read_json("duckduckgo_search", key)
    if cached and cached.get("results") and 0 <= time.time() - cached.get("cached_at", 0) < CACHE_SECONDS:
        return [RetrievedArticle.model_validate(row) for row in cached["results"]]
    session = requests.Session()
    if image:
        token_response = session.get(TOKEN_URL, params={"q": query, "iax": "images", "ia": "images"},
                                     headers=HEADERS, timeout=HTTP_TIMEOUT)
        token_response.raise_for_status()
        token = _image_token(token_response.text)
        response = session.get(IMAGES_URL, params={"q": query, "vqd": token, "o": "json", "f": ",,,", "p": "1"},
                               headers={**HEADERS, "Referer": token_response.url}, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("DuckDuckGo image search returned a non-object response")
        results = parse_image_results(payload, max_results=max_results)
        LOG.info("DuckDuckGo images query=%r raw_results=%d usable_results=%d", query,
                 len(payload.get("results") or []), len(results))
    else:
        response = session.get(HTML_URL, params={"q": query}, headers=HEADERS, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        results = parse_web_results(response.text, max_results=max_results)
    if results:
        write_json("duckduckgo_search", key, {"cached_at": time.time(),
                                              "results": [row.model_dump(mode="json") for row in results]})
    return results
