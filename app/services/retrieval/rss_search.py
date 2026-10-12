"""Public news RSS fallback when GDELT rejects a search."""

import logging
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import parse_qs, urlsplit
from xml.etree import ElementTree

import requests
from bs4 import BeautifulSoup

from app.config import HTTP_TIMEOUT
from app.models.schemas import QueryPlan, SearchRequest
from app.services.retrieval.news_search import RetrievedArticle, canonical_url


LOG = logging.getLogger(__name__)
BING_NEWS_RSS_URL = "https://www.bing.com/news/search"
WINDOWS = {"24h": timedelta(days=1), "3days": timedelta(days=3), "1week": timedelta(weeks=1), "3months": timedelta(days=90)}
EVENT_WORDS = ("shipping", "tanker", "vessel", "attack", "closure", "flood", "fire", "flight", "road", "damage")


def bing_query(request: SearchRequest, plan: QueryPlan) -> str:
    place = next((value.strip() for value in plan.locations + plan.entities if value.strip()), request.topic.strip())
    place = re.sub(r"[^\w\s-]", " ", place)
    place = re.sub(r"\s+", " ", place).strip()
    topic = f"{request.claim.text} {request.topic}".lower()
    event = next((word for word in EVENT_WORDS if word in topic and word not in place.lower()), "")
    return f'"{place}" {event}'.strip() if place else re.sub(r"\s+", " ", request.claim.text).strip()


def _publisher_url(link: str) -> str | None:
    parts = urlsplit(link)
    if parts.scheme not in {"http", "https"}:
        return None
    if parts.netloc.lower().endswith("bing.com"):
        target = parse_qs(parts.query).get("url", [""])[0]
        return canonical_url(target) if urlsplit(target).scheme in {"http", "https"} else None
    return canonical_url(link)


def parse_bing_rss(xml: bytes, *, cutoff: datetime | None = None) -> list[RetrievedArticle]:
    root = ElementTree.fromstring(xml)
    if root.find("channel") is None:
        raise ValueError("Bing News did not return an RSS channel")
    articles = []
    for item in root.findall("./channel/item"):
        link = _publisher_url(item.findtext("link") or "")
        if not link:
            continue
        published = None
        if item.findtext("pubDate"):
            try:
                published = parsedate_to_datetime(item.findtext("pubDate")).astimezone(timezone.utc)
            except (TypeError, ValueError, IndexError):
                pass
        if cutoff and published and published < cutoff:
            continue
        source = next((node.text for node in item if node.tag.lower().endswith("source") and node.text), None)
        description = BeautifulSoup(item.findtext("description") or "", "html.parser").get_text(" ", strip=True)
        articles.append(RetrievedArticle(url=link, title=item.findtext("title") or "", source_name=source or urlsplit(link).netloc, published_at=published, snippet=description[:800]))
    return articles


def search_bing_news(query: str, *, max_results: int, published_after: datetime | None, timespan: str) -> list[RetrievedArticle]:
    cutoff = published_after or datetime.now(timezone.utc) - WINDOWS[timespan]
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=timezone.utc)
    response = requests.get(BING_NEWS_RSS_URL, params={"q": query, "format": "rss"}, headers={"User-Agent": "Mozilla/5.0 (compatible; MultiEvidence/1.0)"}, timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    if len(response.content) > 2_000_000:
        raise ValueError("Bing News RSS response is too large")
    articles = parse_bing_rss(response.content, cutoff=cutoff)
    LOG.info("source=BingNewsRSS query=%r status=%s results=%d", query, response.status_code, len(articles))
    return articles[:max_results]
