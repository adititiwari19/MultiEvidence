import logging
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from app.models.schemas import MediaCandidate, SearchRequest
from app.services.relevance import entity_overlap, semantic_similarity


LOG = logging.getLogger(__name__)
PRESENTATION_PARAMS = {"crop", "resize", "quality", "q", "width", "w", "height", "h", "fit", "dpr", "auto", "format"}
SITE_ASSET = re.compile(r"(?:^|[/_.-])(loader|logo|avatar|favicon|sprite|icon|placeholder|tracking|pixel)(?:[/_.-]|$)", re.I)


def canonical_media_url(url: str) -> str:
    parts = urlsplit(url)
    query = urlencode(sorted((key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True) if key.lower() not in PRESENTATION_PARAMS))
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, query, ""))


def media_dimensions(item: MediaCandidate) -> tuple[int | None, int | None]:
    params = dict((key.lower(), value) for key, value in parse_qsl(urlsplit(item.media_url).query))

    def number(value):
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    width = number(item.metadata.get("width")) or number(params.get("width")) or number(params.get("w"))
    height = number(item.metadata.get("height")) or number(params.get("height")) or number(params.get("h"))
    if not width or not height:
        match = re.search(r"(\d{2,5})[xX](\d{2,5})", params.get("resize", "") or params.get("crop", ""))
        if match:
            width = width or int(match.group(1))
            height = height or int(match.group(2))
    return width, height


def is_site_asset(item: MediaCandidate) -> bool:
    if item.media_type != "image":
        return False
    if SITE_ASSET.search(urlsplit(item.media_url).path):
        return True
    width, height = media_dimensions(item)
    return (width is not None and width < 200) or (height is not None and height < 200)


def dedupe_media_candidates(candidates: list[MediaCandidate], limit: int = 40) -> list[MediaCandidate]:
    selected: dict[str, MediaCandidate] = {}
    for item in candidates:
        if is_site_asset(item):
            continue
        key = canonical_media_url(item.media_url) if item.media_type == "image" else item.media_url
        previous = selected.get(key)
        if previous is None:
            selected[key] = item
        else:
            old_width, old_height = media_dimensions(previous)
            width, height = media_dimensions(item)
            old_size = (old_width or 0) * (old_height or 0), max(old_width or 0, old_height or 0)
            new_size = (width or 0) * (height or 0), max(width or 0, height or 0)
            if new_size > old_size:
                selected[key] = item
    return list(selected.values())[:limit]


@lru_cache(maxsize=4096)
def text_similarity(claim: str, text: str) -> float:
    if not text:
        return 0.0
    try:
        return semantic_similarity(claim, text[:800])
    except Exception:
        return entity_overlap(claim, text)


@lru_cache(maxsize=512)
def image_visual_score(claim: str, local_path: str) -> float | None:
    try:
        from sentence_transformers import util

        from app.services.video import cached_image_embedding, cached_text_embedding

        return max(0.0, min(1.0, float(util.cos_sim([cached_text_embedding(claim)], [cached_image_embedding(Path(local_path))]).item())))
    except Exception as exc:
        LOG.warning("Image visual ranking unavailable: %s", type(exc).__name__)
        return None


def candidate_score(request: SearchRequest, item: MediaCandidate, locations: list[str] | None = None) -> float:
    claim = request.claim.text
    title = text_similarity(claim, item.title)
    article = text_similarity(claim, item.article_text_excerpt)
    caption = text_similarity(claim, item.caption)
    context = f"{item.title} {item.caption} {item.article_text_excerpt}"
    overlap = entity_overlap(claim, context)
    location = max((entity_overlap(value, context) for value in locations or []), default=0.0)
    cutoff = request.search.published_after or request.claim.report_timestamp
    time_score = 0.5
    if cutoff and item.published_at:
        before = item.published_at.replace(tzinfo=timezone.utc) if item.published_at.tzinfo is None else item.published_at
        threshold = cutoff.replace(tzinfo=timezone.utc) if cutoff.tzinfo is None else cutoff
        time_score = 1.0 if before >= threshold else 0.0
    score = 0.35 * article + 0.25 * title + 0.15 * caption + 0.15 * overlap + 0.05 * location + 0.05 * time_score
    if item.media_type == "image" and item.local_path and Path(item.local_path).is_file():
        visual = image_visual_score(claim, item.local_path)
        if visual is not None:
            score = 0.75 * score + 0.25 * visual
    return round(max(0.0, min(1.0, score)), 4)


def rank_candidates(request: SearchRequest, candidates: list[MediaCandidate], locations: list[str] | None = None) -> list[MediaCandidate]:
    cutoff = request.search.published_after
    eligible = []
    for item in candidates:
        if cutoff and item.published_at:
            published = item.published_at.replace(tzinfo=timezone.utc) if item.published_at.tzinfo is None else item.published_at
            after = cutoff.replace(tzinfo=timezone.utc) if cutoff.tzinfo is None else cutoff
            if published < after:
                continue
        eligible.append(item)
    return sorted(eligible, key=lambda item: candidate_score(request, item, locations), reverse=True)


def choose_candidates(request: SearchRequest, candidates: list[MediaCandidate], locations: list[str] | None = None) -> list[MediaCandidate]:
    counts = {"image": 0, "video": 0}
    limits = {"image": request.search.max_images, "video": request.search.max_videos}
    selected = []
    for item in rank_candidates(request, candidates, locations):
        if counts[item.media_type] >= limits[item.media_type]:
            continue
        if candidate_score(request, item, locations) < 0.20:
            continue
        selected.append(item)
        counts[item.media_type] += 1
    return selected
