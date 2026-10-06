import fcntl
import json
import logging
import os
import random
import re
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests

from app.config import CACHE_DIR, HTTP_TIMEOUT
from app.models.schemas import NewsArticle
from app.services.retrieval.cache import digest, read_json, write_json


LOG = logging.getLogger(__name__)
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
TRACKING = {"fbclid", "gclid", "mc_cid", "mc_eid"}
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; MultiEvidence/1.0)"}
GDELT_SPACING_SECONDS = max(10.0, float(os.getenv("GDELT_MIN_INTERVAL_SECONDS", "10")))
GDELT_BACKOFF_SECONDS = tuple(float(value.strip()) for value in os.getenv("GDELT_429_BACKOFF_SECONDS", "15,30,60").split(",") if value.strip())[:3]
GDELT_MAX_429_RETRIES = min(3, max(0, int(os.getenv("GDELT_429_MAX_RETRIES", "3"))))
GDELT_BACKOFF_JITTER_SECONDS = max(0.0, float(os.getenv("GDELT_BACKOFF_JITTER_SECONDS", "1")))
SEARCH_CACHE_SECONDS = 1800
_REQUEST_STATE: ContextVar[dict | None] = ContextVar("gdelt_request_state", default=None)


def _gdelt_log(level: int, message: str, *args) -> None:
    LOG.log(level, "timestamp_utc=%s " + message, datetime.now(timezone.utc).isoformat(timespec="milliseconds"), *args)


class RetrievedArticle(NewsArticle):
    socialimage: str | None = None
    language: str | None = None
    source_country: str | None = None
    image_collage: bool = False


@contextmanager
def gdelt_request_scope():
    """Share 429 suppression across the text and visual searches for one claim."""
    token = _REQUEST_STATE.set({"disabled": False, "status": "NOT_STARTED"})
    try:
        yield
    finally:
        _REQUEST_STATE.reset(token)


def search_status() -> str:
    """Status of the most recent search inside a diagnostic request scope."""
    state = _REQUEST_STATE.get()
    return state.get("status", "NOT_STARTED") if state else "NOT_STARTED"


def _now() -> float:
    return time.monotonic()


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _jitter() -> float:
    return random.uniform(0.0, GDELT_BACKOFF_JITTER_SECONDS)


def _limiter_state(handle) -> dict:
    handle.seek(0)
    raw = handle.read().strip()
    try:
        state = json.loads(raw) if raw else {}
        return state if isinstance(state, dict) else {}
    except json.JSONDecodeError:
        # Read the timestamp written by older versions of the limiter.
        try:
            return {"last_request": float(raw)}
        except ValueError:
            return {}


def _write_limiter_state(handle, state: dict) -> None:
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps(state))
    handle.flush()
    os.fsync(handle.fileno())


def _reserve_gdelt(query: str) -> float:
    """Wait for shared spacing and 429 cooldown before each HTTP request."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    total_wait = 0.0
    while True:
        with (CACHE_DIR / "gdelt_rate.lock").open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                state = _limiter_state(handle)
                current = _now()
                last = float(state.get("last_request", -GDELT_SPACING_SECONDS))
                blocked = float(state.get("blocked_until", 0.0))
                # A monotonic clock reset must not create an indefinite wait.
                next_request = last + GDELT_SPACING_SECONDS if last <= current else current
                wait = max(0.0, next_request - current, blocked - current)
                if not wait:
                    state["last_request"] = current
                    _write_limiter_state(handle, state)
                    _gdelt_log(logging.INFO, "source=GDELT query=%r status=requesting rate_limit_wait=%.3f", query, total_wait)
                    return total_wait
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
        _gdelt_log(logging.INFO, "source=GDELT query=%r status=waiting rate_limit_wait=%.3f", query, wait)
        _sleep(wait)
        total_wait += wait


def _record_gdelt_429(delay: float) -> None:
    """Publish a 429 cooldown so other CLI/server processes see it."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with (CACHE_DIR / "gdelt_rate.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            state = _limiter_state(handle)
            state["last_429_at"] = _now()
            state["blocked_until"] = max(float(state.get("blocked_until", 0.0)), _now() + delay)
            _write_limiter_state(handle, state)
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def canonical_url(url: str) -> str:
    parts = urlsplit(url)
    query = urlencode([(key, value) for key, value in parse_qsl(parts.query) if not key.lower().startswith("utm_") and key.lower() not in TRACKING])
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/") or "/", query, ""))


def parse_gdelt_results(payload: dict) -> list[NewsArticle]:
    articles = []
    for item in payload.get("articles", []):
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        if not url or urlsplit(url).scheme not in {"http", "https"}:
            continue
        timestamp = None
        value = item.get("seendate") or item.get("date")
        if value:
            try:
                timestamp = datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                pass
        image = item.get("socialimage")
        if image and urlsplit(image).scheme not in {"http", "https"}:
            image = None
        articles.append(RetrievedArticle(url=canonical_url(url), title=item.get("title") or "", source_name=item.get("domain") or urlsplit(url).netloc, published_at=timestamp, snippet=item.get("snippet") or "", socialimage=image, language=item.get("language"), source_country=item.get("sourcecountry")))
    return articles


def parse_gdelt_images(payload: dict) -> list[RetrievedArticle]:
    """Map ImageCollageInfo entries to article records carrying direct image URLs."""
    entries = payload.get("images") or payload.get("imagecollageinfo") or []
    if isinstance(entries, dict):
        entries = entries.get("images") or entries.get("data") or []
    if not isinstance(entries, list):
        raise ValueError("GDELT image list is malformed")
    results = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        item = {str(key).lower(): value for key, value in item.items()}
        image = next((item.get(key) for key in ("imageurl", "image_url", "image", "socialimage", "imgurl") if item.get(key)), None)
        article = next((item.get(key) for key in ("articleurl", "article_url", "source_article_url", "pageurl", "sourceurl", "source_url", "webpageurl", "link") if item.get(key)), None)
        if not image and article:
            image = item.get("url")
        if not article and image:
            article = item.get("url")
        if not isinstance(image, str) or not isinstance(article, str) or urlsplit(image).scheme not in {"http", "https"} or urlsplit(article).scheme not in {"http", "https"} or image == article:
            continue
        published = None
        date = item.get("seendate") or item.get("date") or item.get("published_at")
        if date:
            try:
                published = datetime.strptime(date, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            except (TypeError, ValueError):
                try:
                    published = datetime.fromisoformat(str(date).replace("Z", "+00:00"))
                except ValueError:
                    pass
        results.append(RetrievedArticle(url=canonical_url(article), title=item.get("title") or item.get("caption") or "", source_name=item.get("domain") or urlsplit(article).netloc, published_at=published, snippet=item.get("snippet") or "", socialimage=image, language=item.get("language"), source_country=item.get("sourcecountry"), image_collage=True))
    return results


def _failure(query: str, response: requests.Response | None, url: str, reason: str, wait: float = 0.0, full_body: bool = False, attempt: int = 1) -> None:
    status = response.status_code if response is not None else "none"
    final_url = response.url if response is not None else url
    body = (response.text if full_body else response.text[:300]) if response is not None else "<no response>"
    body = body.replace("\\", "\\\\").replace("\r", "\\r").replace("\n", "\\n")
    _gdelt_log(logging.WARNING, "source=GDELT query=%r attempt=%d status=%s results=0 rate_limit_wait=%.3f request_url=%s body=%s reason=%s", query, attempt, status, wait, final_url, body, reason)


def _get(url: str, params: dict, gdelt_state: dict) -> tuple[requests.Response | None, float, int]:
    # requests encodes query values once; do not quote query text before passing params.
    request_url = requests.Request("GET", url, params=params).prepare().url or url
    query = params.get("query") or ""
    retries_429 = 0
    retries_5xx = 0
    total_wait = 0.0
    attempt = 0
    while True:
        if gdelt_state["disabled"]:
            gdelt_state["status"] = "RATE_LIMITED"
            return None, total_wait, attempt
        attempt += 1
        wait = _reserve_gdelt(query)
        total_wait += wait
        _gdelt_log(logging.INFO, "source=GDELT query=%r attempt=%d status=sending results=0 rate_limit_wait=%.3f request_url=%s", query, attempt, wait, request_url)
        try:
            response = requests.get(url, params=params, timeout=HTTP_TIMEOUT, headers=HEADERS)
        except requests.RequestException as exc:
            gdelt_state["status"] = "NETWORK_ERROR"
            _failure(query, None, request_url, type(exc).__name__, wait, attempt=attempt)
            return None, total_wait, attempt
        if response.status_code == 429:
            gdelt_state["status"] = "RATE_LIMITED"
            _failure(query, response, request_url, "HTTP_429", wait, attempt=attempt)
            if retries_429 < GDELT_MAX_429_RETRIES:
                base = GDELT_BACKOFF_SECONDS[min(retries_429, len(GDELT_BACKOFF_SECONDS) - 1)] if GDELT_BACKOFF_SECONDS else 60.0
                delay = base + _jitter()
                _record_gdelt_429(delay)
                _gdelt_log(logging.WARNING, "source=GDELT query=%r attempt=%d status=429 results=0 rate_limit_wait=%.3f backoff_wait=%.3f request_url=%s", query, attempt, wait, delay, request_url)
                retries_429 += 1
                continue
            gdelt_state["disabled"] = True
            _record_gdelt_429((GDELT_BACKOFF_SECONDS[-1] if GDELT_BACKOFF_SECONDS else 60.0) + _jitter())
            return None, total_wait, attempt
        if response.status_code >= 500:
            gdelt_state["status"] = "HTTP_ERROR"
            _failure(query, response, request_url, "HTTP_5XX", wait, attempt=attempt)
            if retries_5xx < 2:
                _sleep(0.5 * (2 ** retries_5xx))
                retries_5xx += 1
                continue
            return None, total_wait, attempt
        if response.status_code >= 400:
            gdelt_state["status"] = "HTTP_ERROR"
            _failure(query, response, request_url, "HTTP_ERROR", wait, attempt=attempt)
            return None, total_wait, attempt
        gdelt_state["status"] = "OK"
        return response, total_wait, attempt


def _gdelt(query: str, max_records: int, published_after: datetime | None, state: dict, timespan: str = "1week", enddatetime: datetime | None = None, visual: bool = False) -> list[NewsArticle]:
    params = {"query": query, "mode": "ImageCollageInfo" if visual else "artlist", "format": "json", "sort": "datedesc", "maxrecords": min(250, max(1, max_records))}
    if published_after:
        after = published_after.replace(tzinfo=timezone.utc) if published_after.tzinfo is None else published_after.astimezone(timezone.utc)
        params["startdatetime"] = after.strftime("%Y%m%d%H%M%S")
    elif not enddatetime:
        params["timespan"] = timespan
    if enddatetime:
        end = enddatetime.replace(tzinfo=timezone.utc) if enddatetime.tzinfo is None else enddatetime.astimezone(timezone.utc)
        params["enddatetime"] = end.strftime("%Y%m%d%H%M%S")
    _gdelt_log(logging.INFO, "source=GDELT query=%r request_url=%s", query, requests.Request("GET", GDELT_URL, params=params).prepare().url)
    response, wait, attempt = _get(GDELT_URL, params, state)
    if response is None:
        state["failed"] = True
        return []
    try:
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("GDELT JSON root is not an object")
        if (visual and "images" not in payload and "imagecollageinfo" not in payload) or (not visual and "articles" not in payload):
            raise ValueError("GDELT JSON is missing expected results")
        results = (parse_gdelt_images(payload) if visual else parse_gdelt_results(payload))[:max_records]
        _gdelt_log(logging.INFO, "source=GDELT query=%r attempt=%d status=%s results=%d rate_limit_wait=%.3f request_url=%s", query, attempt, response.status_code, len(results), wait, response.url)
        return results
    except (ValueError, KeyError, TypeError) as exc:
        state["failed"] = True
        state["status"] = "QUERY_ERROR"
        _failure(query, response, response.url, "QUERY_ERROR", wait, full_body=True, attempt=attempt)
        return []


def search_news(query: str, max_results: int = 10, published_after: datetime | None = None, *, max_records: int = 100, timespan: str = "1week", enddatetime: datetime | None = None) -> list[NewsArticle]:
    if max_results <= 0:
        return []
    cutoff = None
    if published_after:
        cutoff = published_after.replace(tzinfo=timezone.utc) if published_after.tzinfo is None else published_after.astimezone(timezone.utc)
    max_records = min(250, max(1, max_records))
    visual = any(operator in query.lower() for operator in ("imagetag:", "imagewebtag:", "imageocrmeta:"))
    cache_key = digest("gdelt-doc2-v3", query.strip().lower(), "ImageCollageInfo" if visual else "artlist", cutoff.isoformat() if cutoff else "", enddatetime.isoformat() if enddatetime else "", timespan, str(max_records))
    cached = read_json("news_search", cache_key)
    if cached:
        age = time.time() - cached.get("cached_at", 0)
        if 0 <= age < SEARCH_CACHE_SECONDS:
            try:
                results = [RetrievedArticle.model_validate(item) for item in cached["results"]]
                LOG.info("source=Cache query=%r status=cached results=%d rate_limit_wait=0.0", query, len(results))
                state = _REQUEST_STATE.get()
                if state is not None:
                    state["status"] = "CACHED"
                return results[:max_results]
            except (KeyError, TypeError, ValueError):
                pass
    fetch_limit = max_records
    gdelt_state = _REQUEST_STATE.get() or {"disabled": False}

    def recent(items: list[NewsArticle]) -> list[NewsArticle]:
        if cutoff is None:
            return items
        return [item for item in items if item.published_at is None or item.published_at.astimezone(timezone.utc) >= cutoff]

    gdelt_state.pop("failed", None)
    if gdelt_state["disabled"]:
        gdelt_state["failed"] = True
        results = []
    else:
        results = recent(_gdelt(query, fetch_limit, published_after, state=gdelt_state, timespan=timespan, enddatetime=enddatetime, visual=visual))
    results = dedupe_articles(results, limit=fetch_limit)
    if not gdelt_state.get("failed"):
        write_json("news_search", cache_key, {"cached_at": time.time(), "results": [item.model_dump(mode="json") for item in results]})
    return results[:max_results]


def dedupe_articles(articles: list[NewsArticle], limit: int = 30) -> list[NewsArticle]:
    seen = set()
    output = []
    for article in articles:
        key = (canonical_url(article.url), article.socialimage) if isinstance(article, RetrievedArticle) and article.image_collage else canonical_url(article.url)
        if key not in seen:
            seen.add(key)
            output.append(article)
        if len(output) >= limit:
            break
    return output
