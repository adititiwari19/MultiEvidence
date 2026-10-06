import logging
import re
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urljoin, urlsplit

import requests
from bs4 import BeautifulSoup

from app.config import HTTP_TIMEOUT
from app.models.schemas import MediaCandidate, NewsArticle
from app.services.retrieval.cache import digest, read_json, write_json


LOG = logging.getLogger(__name__)


@dataclass
class ParsedArticle:
    article: NewsArticle
    text: str
    candidates: list[MediaCandidate]


def parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def absolute_media_url(base: str, value: str | None) -> str | None:
    if not value:
        return None
    url = urljoin(base, value.strip())
    return url if urlsplit(url).scheme in {"http", "https"} else None


def parse_article_html(article: NewsArticle, html: str) -> ParsedArticle:
    soup = BeautifulSoup(html, "html.parser")

    def meta(*names: str) -> str | None:
        for name in names:
            tag = soup.find("meta", attrs={"property": name}) or soup.find("meta", attrs={"name": name})
            if tag and tag.get("content"):
                return tag["content"].strip()
        return None

    title = meta("og:title", "twitter:title") or (soup.title.get_text(" ", strip=True) if soup.title else article.title)
    published_at = parse_date(meta("article:published_time", "datePublished", "date", "pubdate"))
    if not published_at:
        time_tag = soup.find("time", attrs={"datetime": True})
        published_at = parse_date(time_tag.get("datetime")) if time_tag else article.published_at
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    main = soup.find("article") or soup.find("main") or soup.body or soup
    text = re.sub(r"\s+", " ", " ".join(p.get_text(" ", strip=True) for p in main.find_all("p"))).strip()[:4000]
    image_entries = []
    for key in ("og:image", "twitter:image"):
        value = meta(key)
        if value:
            image_entries.append((value, "", {"hero": True, "width": meta("og:image:width"), "height": meta("og:image:height")}))
    for tag in main.find_all("img"):
        try:
            width = int(tag.get("width") or 0)
            height = int(tag.get("height") or 0)
        except ValueError:
            width = height = 0
        if (width and width < 200) or (height and height < 200):
            continue
        source = tag.get("src") or tag.get("data-src")
        if not source and tag.get("srcset"):
            source = tag["srcset"].split(",")[0].strip().split(" ")[0]
        caption = tag.get("alt") or ""
        figure = tag.find_parent("figure")
        if figure and figure.find("figcaption"):
            caption = figure.find("figcaption").get_text(" ", strip=True) or caption
        image_entries.append((source, caption, {"width": width, "height": height}))
    video_entries = []
    for key in ("og:video", "og:video:url", "twitter:player:stream"):
        value = meta(key)
        if value:
            video_entries.append((value, {"embedded": False}))
    for tag in main.find_all(["video", "source", "iframe"]):
        value = tag.get("src") or tag.get("data-src")
        if value:
            video_entries.append((value, {"embedded": tag.name == "iframe"}))
    seen = set()
    candidates = []

    def add(media_type: str, value: str | None, caption: str, metadata: dict):
        url = absolute_media_url(article.url, value)
        if not url or url in seen:
            return
        seen.add(url)
        candidates.append(MediaCandidate(candidate_id=digest(article.url, url)[:16], article_url=article.url, source_name=article.source_name, published_at=published_at, media_type=media_type, media_url=url, title=title, article_text_excerpt=text[:800], caption=caption, metadata=metadata))

    for url, caption, metadata in image_entries[:5]:
        add("image", url, caption, metadata)
    for url, metadata in video_entries[:5]:
        add("video", url, "", metadata)
    # Keep at most five, reserving one place for an available video link.
    if len(candidates) > 5:
        videos = [item for item in candidates if item.media_type == "video"]
        images = [item for item in candidates if item.media_type == "image"]
        candidates = images[:4] + videos[:1] if videos else images[:5]
    return ParsedArticle(article=article.model_copy(update={"title": title, "published_at": published_at}), text=text, candidates=candidates)


def fetch_article(article: NewsArticle) -> ParsedArticle | None:
    key = digest(article.url)
    cached = read_json("articles", key)
    if cached:
        return ParsedArticle(article=NewsArticle.model_validate(cached["article"]), text=cached["text"], candidates=[MediaCandidate.model_validate(item) for item in cached["candidates"]])
    try:
        response = requests.get(article.url, timeout=HTTP_TIMEOUT, headers={"User-Agent": "MultiEvidence/1.0"})
        response.raise_for_status()
        if "html" not in response.headers.get("Content-Type", "text/html").lower() or len(response.content) > 2_000_000:
            return None
        parsed = parse_article_html(article, response.text)
        write_json("articles", key, {"article": parsed.article.model_dump(mode="json"), "text": parsed.text, "candidates": [candidate.model_dump(mode="json") for candidate in parsed.candidates]})
        return parsed
    except (requests.RequestException, ValueError, UnicodeError) as exc:
        LOG.warning("Article fetch failed: %s", type(exc).__name__)
        return None
