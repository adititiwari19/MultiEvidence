import logging
from datetime import timedelta, timezone
from pathlib import Path

from app.config import FRAME_DIR, MOLMO2_MODEL, OLLAMA_MODEL, SAM2_CHECKPOINT, SAM2_MODEL, SAM2_MODEL_CONFIG, VERIFIER_BACKEND, VLM_MODEL
from app.models.schemas import AggregateStatus, CandidatePreview, CheckRequest, CheckResponse, EvidenceItem, MediaCandidate, Relation, SearchRequest, SearchResponse, SuggestedUpdate
from app.services.aggregator import aggregate, aggregate_patch, select_evidence
from app.services.cached_gallery import match_gallery
from app.services.retrieval.article_parser import fetch_article
from app.services.retrieval.cache import digest, read_json, write_json
from app.services.retrieval.candidate_ranker import candidate_score, choose_candidates, dedupe_media_candidates, rank_candidates
from app.services.retrieval.duckduckgo_search import search_duckduckgo
from app.services.retrieval.free_search import is_google_news_url, search_ddgs_images, search_google_news_rss, search_video_thumbnails
from app.services.retrieval.google_search import google_configured, google_queries, search_google
from app.services.retrieval.media_fetch import download_media
from app.services.retrieval.news_search import dedupe_articles, gdelt_request_scope, search_news, search_status
from app.services.retrieval.query_builder import build_queries, fallback_queries
from app.services.retrieval.rss_search import bing_query, search_bing_news
from app.services.retrieval.searxng_search import searxng_configured, search_searxng
from app.services.verifier import check
from app.services.openrouter_client import OpenRouterOutputError, OpenRouterRequestError


LOG = logging.getLogger(__name__)


def cached_check(request: CheckRequest, media_url: str) -> CheckResponse:
    item = request.evidence[0]
    key = digest("visual-first-v2", media_url, request.claim.text, request.topic, item.context_text or "", str(request.claim.report_timestamp), str(item.published_at), VERIFIER_BACKEND, OLLAMA_MODEL if VERIFIER_BACKEND == "ollama" else VLM_MODEL, str(request.ground), MOLMO2_MODEL if request.ground else "", SAM2_MODEL if request.ground else "", SAM2_CHECKPOINT if request.ground else "", SAM2_MODEL_CONFIG if request.ground else "")
    cached = read_json("verification", key)
    if cached:
        result = CheckResponse.model_validate(cached)
        if all((FRAME_DIR / Path(ref.frame_url).name).is_file() and (not ref.overlay_url or (FRAME_DIR / Path(ref.overlay_url).name).is_file()) for ref in result.evidence):
            return result
    result = check(request)
    write_json("verification", key, result.model_dump(mode="json"))
    return result


def to_evidence(candidate: MediaCandidate, response: CheckResponse) -> EvidenceItem:
    reference = response.evidence[0] if response.evidence else None
    observation = reference.observation if reference else (response.observable_facts[0] if response.observable_facts else response.explanation)
    return EvidenceItem(evidence_id=candidate.candidate_id, relation=response.relation.value, confidence=response.confidence, media_type=candidate.media_type, source_name=candidate.source_name, source_url=candidate.article_url, published_at=candidate.published_at, media_url=candidate.media_url, local_path=candidate.local_path or "", observation=observation, timestamp_seconds=reference.timestamp_seconds if reference else None, frame_url=reference.frame_url if reference else None, suggested_update=response.suggested_update, frames=response.evidence, event_match=response.event_match, event_match_reason=response.event_match_reason, explanation=response.explanation, limitations=response.limitations)


def search_and_check(request: SearchRequest, *, timespan: str | None = None) -> SearchResponse:
    cached = match_gallery(request.claim.text, request.search.checkpoint)
    if cached:
        old_text = request.claim.text
        return SearchResponse(
            claim_id=request.claim.claim_id,
            queries=[f"Local Hormuz gallery: {cached.checkpoint} claim {cached.claim_number}"],
            articles_considered=1, media_candidates=len(cached.media), filtered_candidates=0,
            verified_evidence_count=0, checked_media_count=0, claim_status="UNCERTAIN",
            confidence=0, aggregate=AggregateStatus(claim_status="UNCERTAIN", confidence=0),
            evidence=[], checks=[], cache_hit=True, cached_gallery=[cached],
            suggested_update=SuggestedUpdate(action="KEEP", old_text=old_text, new_text=old_text,
                                             reason="The saved images are candidates; their full claims have not been visually verified."),
            message="Matched the local Hormuz gallery. Images and source links are cached; assessment notes describe their visible limits.",
        )
    if request.ground and not MOLMO2_MODEL:
        raise ValueError("ground=true requires MOLMO2_MODEL")
    provider = request.search.provider
    if provider == "auto":
        provider = "google" if google_configured() else "searxng" if searxng_configured() else "free"
    use_google = provider == "google"
    if request.search.provider == "google" and not google_configured():
        raise ValueError("Google search requires GOOGLE_CSE_API_KEY and GOOGLE_CSE_CX")
    if provider == "searxng" and not searxng_configured():
        raise ValueError("SearXNG search requires SEARXNG_BASE_URL")
    plan = fallback_queries(request) if provider in {"google", "searxng", "duckduckgo", "free"} else build_queries(request)
    timespan = timespan or request.search.timespan
    found = []
    retrieval_failures = []
    upper = request.search.published_before or request.claim.report_timestamp
    lower = request.search.published_after
    if upper and lower:
        upper_utc = upper.replace(tzinfo=timezone.utc) if upper.tzinfo is None else upper.astimezone(timezone.utc)
        lower_utc = lower.replace(tzinfo=timezone.utc) if lower.tzinfo is None else lower.astimezone(timezone.utc)
        if lower_utc > upper_utc:
            raise ValueError("published_after must not be later than published_before or report_timestamp")
    if upper and not lower:
        windows = {"24h": 1, "3days": 3, "1week": 7, "3months": 90}
        lower = upper - timedelta(days=windows[timespan])

    def within_window(item):
        if item.published_at is None:
            return upper is None
        date = item.published_at.replace(tzinfo=timezone.utc) if item.published_at.tzinfo is None else item.published_at.astimezone(timezone.utc)
        if lower:
            start = lower.replace(tzinfo=timezone.utc) if lower.tzinfo is None else lower.astimezone(timezone.utc)
            if date < start:
                return False
        if upper:
            end = upper.replace(tzinfo=timezone.utc) if upper.tzinfo is None else upper.astimezone(timezone.utc)
            if date > end:
                return False
        return True

    if provider in {"google", "searxng", "duckduckgo", "free"}:
        google_seeds = google_queries(request, plan)
        queries = []
        for query in google_seeds[:2]:
            for image in (() if provider == "free" else (True, False)):
                kind = "Images" if image else "Web"
                provider_label = {"google": "Google", "searxng": "SearXNG", "duckduckgo": "DuckDuckGo"}[provider]
                queries.append(f"{provider_label} {kind}: {query}")
                try:
                    if use_google:
                        found.extend(search_google(query, image=image, max_results=10,
                                                   timespan=timespan if not upper and not request.search.published_after else None))
                    elif provider == "searxng":
                        found.extend(search_searxng(query, image=image, max_results=10))
                    else:
                        found.extend(search_duckduckgo(query, image=image, max_results=10))
                except Exception as exc:
                    LOG.warning("%s %s discovery failed: %s", provider_label, kind, type(exc).__name__)
                    retrieval_failures.append({"google": "GOOGLE_ERROR", "searxng": "SEARXNG_ERROR", "duckduckgo": "DUCKDUCKGO_ERROR"}[provider])
            if provider == "free":
                queries.append(f"Google News RSS: {query}")
                try:
                    found.extend(search_google_news_rss(query, max_results=15, published_after=lower,
                                                        published_before=upper, timespan=timespan))
                except Exception as exc:
                    LOG.warning("Google News RSS discovery failed: %s", type(exc).__name__)
                    retrieval_failures.append("GOOGLE_NEWS_ERROR")
    else:
        with gdelt_request_scope():
            for query in plan.queries[:2]:
                try:
                    found.extend(search_news(query, max_results=100, published_after=lower, enddatetime=upper, timespan=timespan))
                except Exception as exc:
                    LOG.warning("News discovery skipped: %s", type(exc).__name__)
                    retrieval_failures.append("NETWORK_ERROR")
                status = search_status()
                if status in {"RATE_LIMITED", "QUERY_ERROR", "HTTP_ERROR", "NETWORK_ERROR"}:
                    retrieval_failures.append(status)
                if status == "RATE_LIMITED":
                    break
        queries = plan.queries.copy()
        if not found and retrieval_failures:
            fallback_query = bing_query(request, plan)
            queries.append(f"Bing News RSS: {fallback_query}")
            try:
                found = search_bing_news(fallback_query, max_results=30, published_after=lower, timespan=timespan)
            except Exception as exc:
                LOG.warning("Bing News RSS discovery failed: %s: %s", type(exc).__name__, exc)
                retrieval_failures.append("FALLBACK_ERROR")
    if provider in {"duckduckgo", "searxng"} and upper:
        # Search results have no trustworthy publication date. Read a bounded
        # number of source pages and use only dates published by those pages.
        source_dates = {}
        for item in found:
            if item.url in source_dates or len(source_dates) >= 12 or item.image_collage:
                continue
            try:
                parsed = fetch_article(item)
                source_dates[item.url] = parsed.article.published_at if parsed else None
            except Exception as exc:
                LOG.warning("Source date fetch skipped: %s", type(exc).__name__)
                source_dates[item.url] = None
        found = [item.model_copy(update={"published_at": source_dates[item.url]})
                 if item.url in source_dates and source_dates[item.url] else item for item in found]
    found = dedupe_articles([item for item in found if within_window(item)], limit=200)

    def article_score(article):
        probe = MediaCandidate(candidate_id="ranking", article_url=article.url, source_name=article.source_name, published_at=article.published_at, media_type="image", media_url=article.url, title=article.title, article_text_excerpt=article.snippet)
        return candidate_score(request, probe, plan.locations)

    limit = min(10, request.search.max_articles)
    visual_articles = sorted((item for item in found if getattr(item, "image_collage", False)), key=article_score, reverse=True)[:limit]
    text_articles = sorted((item for item in found if not getattr(item, "image_collage", False)), key=article_score, reverse=True)[:limit]
    articles = visual_articles + text_articles
    candidates = []
    for article in articles:
        image = getattr(article, "socialimage", None)
        if image and not (provider == "free" and is_google_news_url(article.url)):
            candidates.append(MediaCandidate(candidate_id=digest(article.url, image)[:16], article_url=article.url, source_name=article.source_name, published_at=article.published_at, media_type="image", media_url=image, title=article.title, article_text_excerpt=article.snippet, metadata={"discovered_image": True, "language": getattr(article, "language", None), "source_country": getattr(article, "source_country", None)}))
        if getattr(article, "image_collage", False):
            continue
        try:
            parsed = fetch_article(article)
        except Exception as exc:
            LOG.warning("Article skipped: %s", type(exc).__name__)
            continue
        if not parsed:
            continue
        if provider == "free" and is_google_news_url(parsed.article.url):
            continue
        for candidate in parsed.candidates[:5]:
            candidates.append(candidate)
    if provider == "free" and len(dedupe_media_candidates(candidates)) < 5:
        fallback_query = google_queries(request, plan)[0]
        for label, discover, maximum in (("ddgs images", search_ddgs_images, 10),
                                         ("yt-dlp thumbnails", search_video_thumbnails, 5)):
            queries.append(f"{label}: {fallback_query}")
            try:
                rows = discover(fallback_query, max_results=maximum)
            except Exception as exc:
                LOG.warning("%s fallback unavailable: %s", label, type(exc).__name__)
                retrieval_failures.append("FREE_FALLBACK_ERROR")
                continue
            if upper:
                dated = []
                for row in rows:
                    if row.published_at is None:
                        try:
                            parsed = fetch_article(row)
                            if parsed:
                                row = row.model_copy(update={"published_at": parsed.article.published_at})
                        except Exception as exc:
                            LOG.warning("Fallback source date skipped: %s", type(exc).__name__)
                    dated.append(row)
                rows = dated
            for row in rows:
                if not within_window(row) or not row.socialimage:
                    continue
                articles.append(row)
                candidates.append(MediaCandidate(candidate_id=digest(row.url, row.socialimage)[:16],
                                                 article_url=row.url, source_name=row.source_name,
                                                 published_at=row.published_at, media_type="image",
                                                 media_url=row.socialimage, title=row.title,
                                                 article_text_excerpt=row.snippet,
                                                 metadata={"discovered_image": True,
                                                           "video_thumbnail": label == "yt-dlp thumbnails"}))
            if len(dedupe_media_candidates(candidates)) >= 5:
                break
    candidates = dedupe_media_candidates(candidates, limit=30 if provider == "free" else 40)
    previews = [CandidatePreview(media_type=item.media_type, media_url=item.media_url, title=item.title, source_url=item.article_url) for item in candidates[:12]]
    shortlisted = choose_candidates(request, candidates, plan.locations)[:7 if provider == "free" else 10]
    downloaded = []
    for candidate in shortlisted:
        try:
            fetched = download_media(candidate, request.claim.claim_id)
            if fetched and fetched.local_path and not fetched.unavailable_for_visual_verification:
                downloaded.append(fetched)
        except Exception as exc:
            LOG.warning("Media skipped: %s", type(exc).__name__)
    ranked = rank_candidates(request, downloaded, plan.locations)
    evidence = []
    checks = []
    insufficient_count = 0
    irrelevant_count = 0
    for candidate in ranked:
        context = " ".join(part for part in [candidate.title, candidate.caption, candidate.article_text_excerpt] if part)
        manual_request = CheckRequest(report_id=request.report_id, topic=request.topic, claim=request.claim, evidence=[{"evidence_id": candidate.candidate_id, "type": candidate.media_type, "local_path": candidate.local_path, "published_at": candidate.published_at, "source_name": candidate.source_name, "source_url": candidate.article_url, "media_url": candidate.media_url, "context_text": context, "caption": candidate.caption}], ground=request.ground)
        try:
            result = cached_check(manual_request, candidate.media_url)
        except Exception as exc:
            if (request.ground or VERIFIER_BACKEND == "ollama") and isinstance(exc, (OpenRouterOutputError, OpenRouterRequestError, ValueError, RuntimeError, ImportError)):
                raise
            LOG.warning("Visual verification skipped: %s", type(exc).__name__)
            continue
        checked = to_evidence(candidate, result)
        checks.append(checked)
        if result.relation == Relation.INSUFFICIENT:
            insufficient_count += 1
        if result.relation == Relation.IRRELEVANT or result.event_match is False:
            irrelevant_count += 1
        if result.relation in {Relation.SUPPORTS, Relation.CONTRADICTS, Relation.REFINES} and result.confidence >= 0.65:
            if result.event_match is not False:
                evidence.append(checked)
    selected = select_evidence(evidence)
    status = aggregate(evidence, insufficient_count, irrelevant_count)
    patch = aggregate_patch(request.claim.text, status, selected)
    message = None
    if not selected:
        if use_google and not found and "GOOGLE_ERROR" in retrieval_failures:
            message = "Google search failed or returned no usable results; no evidence was evaluated. Check the server log."
        elif provider == "duckduckgo" and not found and "DUCKDUCKGO_ERROR" in retrieval_failures:
            message = "DuckDuckGo search failed or returned no usable results; no evidence was evaluated. Check the server log."
        elif provider == "searxng" and not found and "SEARXNG_ERROR" in retrieval_failures:
            message = "SearXNG search failed or returned no usable results; no evidence was evaluated. Check the server log."
        elif provider == "free" and not found and "GOOGLE_NEWS_ERROR" in retrieval_failures and not candidates:
            message = "Google News RSS failed and free media fallbacks found no usable images; check the server log."
        elif provider == "duckduckgo" and upper and not found:
            message = "No DuckDuckGo result had a usable publication date within the historical window; use dated source articles for this report."
        elif provider == "searxng" and upper and not found:
            message = "No SearXNG result had a usable publication date within the historical window; use dated source articles for this report."
        elif not found and "FALLBACK_ERROR" in retrieval_failures:
            message = "GDELT retrieval failed and Bing News RSS was unavailable; no evidence was evaluated. Check the server log."
        elif not found and "RATE_LIMITED" in retrieval_failures:
            message = "GDELT rate-limited retrieval and the fallback found no articles; no evidence was evaluated."
        elif not found and retrieval_failures:
            message = "Evidence retrieval failed; no evidence was evaluated. Check the GDELT errors in the server log."
        elif articles and not candidates:
            message = "Article links were found, but no usable image or video links were extracted; visual verification and grounding did not run."
        elif checks and insufficient_count:
            message = "Images were checked, but they did not establish the full claim. See each image's visible observation and limits below."
        else:
            message = "No sufficiently relevant multimodal evidence was found."
    return SearchResponse(claim_id=request.claim.claim_id, queries=queries, articles_considered=len(articles), media_candidates=len(candidates), filtered_candidates=len(candidates) - len(ranked), verified_evidence_count=len(selected), checked_media_count=len(checks), claim_status=status.claim_status, confidence=status.confidence, aggregate=status, evidence=selected, checks=checks, candidate_previews=previews, suggested_update=patch, message=message)
