"""Inspect keyless media discovery without invoking a vision model."""

import argparse
import json
import logging
from datetime import datetime, timedelta, timezone

from app.models.schemas import MediaCandidate
from app.services.retrieval.article_parser import fetch_article
from app.services.retrieval.cache import digest
from app.services.retrieval.candidate_ranker import dedupe_media_candidates
from app.services.retrieval.free_search import is_google_news_url, search_ddgs_images, search_google_news_rss, search_video_thumbnails


LOG = logging.getLogger(__name__)


def _date(value: str | None, *, end_of_day: bool = False) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use YYYY-MM-DD for publication dates") from exc
    return parsed + timedelta(days=1, microseconds=-1) if end_of_day else parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--timespan", choices=["24h", "3days", "1week", "3months"], default="3months")
    parser.add_argument("--max-pages", type=int, default=10)
    parser.add_argument("--published-after", metavar="YYYY-MM-DD", help="Earliest publication date")
    parser.add_argument("--published-before", metavar="YYYY-MM-DD", help="Historical report cutoff date")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    try:
        upper = _date(args.published_before, end_of_day=True)
        lower = _date(args.published_after) or (upper - timedelta(days={"24h": 1, "3days": 3, "1week": 7, "3months": 90}[args.timespan]) if upper else None)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if lower and upper and lower > upper:
        parser.error("--published-after must be on or before --published-before")

    def eligible(value: datetime | None) -> bool:
        if upper and value is None:
            return False
        if value is None:
            return True
        value = value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
        return (lower is None or value >= lower) and (upper is None or value <= upper)

    articles = search_google_news_rss(args.query, max_results=max(1, min(30, args.max_pages)),
                                      published_after=lower, published_before=upper, timespan=args.timespan)
    candidates = []
    for article in articles:
        if not eligible(article.published_at):
            continue
        if article.socialimage and not is_google_news_url(article.url):
            candidates.append(MediaCandidate(candidate_id=digest(article.url, article.socialimage)[:16],
                                             article_url=article.url, source_name=article.source_name,
                                             published_at=article.published_at, media_type="image",
                                             media_url=article.socialimage, title=article.title))
        try:
            parsed = fetch_article(article)
        except Exception as exc:
            LOG.warning("Article skipped: %s", type(exc).__name__)
            continue
        if parsed and not is_google_news_url(parsed.article.url):
            candidates.extend(item for item in parsed.candidates if item.media_type == "image" and eligible(item.published_at))
    source_counts = {"google_news_articles": len(articles), "ddgs_images": 0, "video_thumbnails": 0}
    if len(dedupe_media_candidates(candidates)) < 5:
        for name, discover in (("ddgs_images", search_ddgs_images),
                               ("video_thumbnails", search_video_thumbnails)):
            try:
                rows = discover(args.query, max_results=10 if name == "ddgs_images" else 5)
            except Exception as exc:
                LOG.warning("%s unavailable: %s", name, type(exc).__name__)
                continue
            source_counts[name] = len(rows)
            for row in rows:
                if upper and row.published_at is None:
                    try:
                        parsed = fetch_article(row)
                        if parsed and not is_google_news_url(parsed.article.url):
                            row = row.model_copy(update={"url": parsed.article.url,
                                                         "published_at": parsed.article.published_at})
                    except Exception as exc:
                        LOG.warning("Fallback source skipped: %s", type(exc).__name__)
                if row.socialimage and eligible(row.published_at) and not is_google_news_url(row.url):
                    candidates.append(MediaCandidate(candidate_id=digest(row.url, row.socialimage)[:16],
                                                     article_url=row.url, source_name=row.source_name,
                                                     published_at=row.published_at, media_type="image",
                                                     media_url=row.socialimage, title=row.title))
            if len(dedupe_media_candidates(candidates)) >= 5:
                break
    candidates = dedupe_media_candidates(candidates, limit=30)
    print(json.dumps({"query": args.query, "published_after": lower.isoformat() if lower else None,
                      "published_before": upper.isoformat() if upper else None,
                      "sources": source_counts, "image_candidates": len(candidates),
                      "images": [{"title": item.title, "source_url": item.article_url,
                                  "image_url": item.media_url, "published_at": item.published_at.isoformat() if item.published_at else None}
                                 for item in candidates]}, indent=2))


if __name__ == "__main__":
    main()
