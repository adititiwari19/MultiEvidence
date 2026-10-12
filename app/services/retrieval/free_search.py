"""Keyless news, image, and video-thumbnail discovery.

News article pages are the primary image source. Optional ddgs and yt-dlp
provide fallback image links and public video thumbnails.
"""

import logging
import re
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit
from xml.etree import ElementTree

import requests
from bs4 import BeautifulSoup

from app.config import HTTP_TIMEOUT
from app.services.retrieval.cache import digest, read_json, write_json
from app.services.retrieval.news_search import RetrievedArticle, canonical_url


LOG = logging.getLogger(__name__)
GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"
CACHE_SECONDS = 6 * 3600


def is_google_news_url(url: str) -> bool:
    """An RSS wrapper is not a publisher page with image provenance."""
    host = (urlsplit(url).hostname or "").lower()
    return host == "news.google.com"


def _http_url(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    parts = urlsplit(value)
    return canonical_url(value) if parts.scheme in {"http", "https"} and parts.netloc else None


def parse_google_news_rss(xml: bytes, *, max_results: int = 20) -> list[RetrievedArticle]:
    root = ElementTree.fromstring(xml)
    if root.find("channel") is None:
        raise ValueError("Google News did not return an RSS channel")
    rows = []
    for item in root.findall("./channel/item"):
        link = _http_url(item.findtext("link"))
        if not link:
            continue
        published = None
        try:
            published = parsedate_to_datetime(item.findtext("pubDate") or "").astimezone(timezone.utc)
        except (TypeError, ValueError, IndexError):
            pass
        source = next((node.text for node in item if node.tag.lower().endswith("source") and node.text), None)
        description = BeautifulSoup(item.findtext("description") or "", "html.parser").get_text(" ", strip=True)
        image = next((_http_url(node.get("url")) for node in item
                      if node.tag.lower().endswith(("content", "thumbnail")) and _http_url(node.get("url"))), None)
        rows.append(RetrievedArticle(url=link, title=item.findtext("title") or "",
                                     source_name=source or urlsplit(link).netloc,
                                     published_at=published, snippet=description[:800], socialimage=image))
        if len(rows) >= max_results:
            break
    return rows


def search_google_news_rss(query: str, *, max_results: int = 20,
                           published_after: datetime | None = None,
                           published_before: datetime | None = None,
                           timespan: str = "1week") -> list[RetrievedArticle]:
    if max_results <= 0:
        return []
    terms = [query]
    if published_after:
        terms.append("after:" + published_after.strftime("%Y-%m-%d"))
    if published_before:
        terms.append("before:" + (published_before.date() + timedelta(days=1)).isoformat())
    if not published_after and not published_before:
        terms.append("when:" + {"24h": "1d", "3days": "3d", "1week": "7d", "3months": "90d"}[timespan])
    full_query = " ".join(terms)
    key = digest("google-news-rss-v1", full_query.lower(), str(max_results))
    cached = read_json("google_news_rss", key)
    if cached and cached.get("results") and 0 <= time.time() - cached.get("cached_at", 0) < CACHE_SECONDS:
        return [RetrievedArticle.model_validate(row) for row in cached["results"]]
    response = requests.get(GOOGLE_NEWS_RSS_URL, params={"q": full_query, "hl": "en-US", "gl": "US", "ceid": "US:en"},
                            headers={"User-Agent": "Mozilla/5.0 (compatible; MultiEvidence/1.0)"}, timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    if len(response.content) > 2_000_000:
        raise ValueError("Google News RSS response is too large")
    rows = parse_google_news_rss(response.content, max_results=max_results)
    if rows:
        write_json("google_news_rss", key, {"cached_at": time.time(),
                                            "results": [row.model_dump(mode="json") for row in rows]})
    LOG.info("Google News RSS query=%r results=%d", full_query, len(rows))
    return rows


def parse_ddgs_images(items, *, max_results: int = 10) -> list[RetrievedArticle]:
    rows = []
    for item in items:
        if not isinstance(item, dict):
            continue
        page = _http_url(item.get("url"))
        image = _http_url(item.get("image"))
        if not page or not image or page == image:
            continue
        rows.append(RetrievedArticle(url=page, title=item.get("title") or "",
                                     source_name=item.get("source") or urlsplit(page).netloc,
                                     snippet="", socialimage=image, image_collage=True))
        if len(rows) >= max_results:
            break
    return rows


def search_ddgs_images(query: str, *, max_results: int = 10) -> list[RetrievedArticle]:
    if max_results <= 0:
        return []
    try:
        from ddgs import DDGS
    except ImportError as exc:
        raise RuntimeError("ddgs is not installed; run python -m pip install ddgs") from exc
    key = digest("ddgs-images-v1", query.strip().lower(), str(max_results))
    cached = read_json("ddgs_images", key)
    if cached and cached.get("results") and 0 <= time.time() - cached.get("cached_at", 0) < CACHE_SECONDS:
        return [RetrievedArticle.model_validate(row) for row in cached["results"]]
    with DDGS() as client:
        try:
            raw = client.images(query, max_results=max_results, backend="bing")
        except TypeError:
            raw = client.images(query, max_results=max_results)
        rows = parse_ddgs_images(raw, max_results=max_results)
    if rows:
        write_json("ddgs_images", key, {"cached_at": time.time(),
                                        "results": [row.model_dump(mode="json") for row in rows]})
    return rows


def parse_video_thumbnails(entries, *, max_results: int = 5) -> list[RetrievedArticle]:
    rows = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        video_id = item.get("id")
        page = _http_url(item.get("webpage_url") or item.get("url"))
        if not page and isinstance(video_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            page = f"https://www.youtube.com/watch?v={video_id}"
        thumb = _http_url(item.get("thumbnail"))
        if not thumb and isinstance(video_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            thumb = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"
        if not page or not thumb:
            continue
        date = item.get("upload_date")
        published = None
        if isinstance(date, str) and re.fullmatch(r"\d{8}", date):
            try:
                published = datetime.strptime(date, "%Y%m%d").replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
            except ValueError:
                pass
        rows.append(RetrievedArticle(url=page, title=item.get("title") or "",
                                     source_name="YouTube thumbnail", published_at=published,
                                     snippet=(item.get("description") or "")[:800],
                                     socialimage=thumb, image_collage=True))
        if len(rows) >= max_results:
            break
    return rows


def search_video_thumbnails(query: str, *, max_results: int = 5) -> list[RetrievedArticle]:
    if max_results <= 0:
        return []
    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:
        raise RuntimeError("yt-dlp is not installed; run python -m pip install yt-dlp") from exc
    key = digest("yt-dlp-thumbnails-v1", query.strip().lower(), str(max_results))
    cached = read_json("video_thumbnails", key)
    if cached and cached.get("results") and 0 <= time.time() - cached.get("cached_at", 0) < CACHE_SECONDS:
        return [RetrievedArticle.model_validate(row) for row in cached["results"]]
    with YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True,
                    "extract_flat": True, "socket_timeout": HTTP_TIMEOUT}) as client:
        info = client.extract_info(f"ytsearch{max_results}:{query}", download=False)
    rows = parse_video_thumbnails((info or {}).get("entries") or [], max_results=max_results)
    if rows:
        write_json("video_thumbnails", key, {"cached_at": time.time(),
                                             "results": [row.model_dump(mode="json") for row in rows]})
    return rows
