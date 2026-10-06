import logging
from pathlib import Path

from app.config import FRAME_DIR, VLM_MODEL
from app.models.schemas import CandidatePreview, CheckRequest, CheckResponse, EvidenceItem, MediaCandidate, Relation, SearchRequest, SearchResponse
from app.services.aggregator import aggregate, aggregate_patch, select_evidence
from app.services.retrieval.article_parser import fetch_article
from app.services.retrieval.cache import digest, read_json, write_json
from app.services.retrieval.candidate_ranker import candidate_score, choose_candidates, dedupe_media_candidates, rank_candidates
from app.services.retrieval.media_fetch import download_media
from app.services.retrieval.news_search import dedupe_articles, gdelt_request_scope, search_news
from app.services.retrieval.query_builder import build_queries
from app.services.verifier import check


LOG = logging.getLogger(__name__)


def cached_check(request: CheckRequest, media_url: str) -> CheckResponse:
    item = request.evidence[0]
    key = digest("visual-first-v2", media_url, request.claim.text, request.topic, item.context_text or "", str(request.claim.report_timestamp), str(item.published_at), VLM_MODEL)
    cached = read_json("verification", key)
    if cached:
        result = CheckResponse.model_validate(cached)
        if all((FRAME_DIR / Path(ref.frame_url).name).is_file() for ref in result.evidence):
            return result
    result = check(request)
    write_json("verification", key, result.model_dump(mode="json"))
    return result


def to_evidence(candidate: MediaCandidate, response: CheckResponse) -> EvidenceItem:
    reference = response.evidence[0] if response.evidence else None
    observation = reference.observation if reference else (response.observable_facts[0] if response.observable_facts else response.explanation)
    return EvidenceItem(evidence_id=candidate.candidate_id, relation=response.relation.value, confidence=response.confidence, media_type=candidate.media_type, source_name=candidate.source_name, source_url=candidate.article_url, published_at=candidate.published_at, media_url=candidate.media_url, local_path=candidate.local_path or "", observation=observation, timestamp_seconds=reference.timestamp_seconds if reference else None, frame_url=reference.frame_url if reference else None, suggested_update=response.suggested_update, frames=response.evidence, event_match=response.event_match, event_match_reason=response.event_match_reason)


def search_and_check(request: SearchRequest, *, timespan: str = "1week") -> SearchResponse:
    plan = build_queries(request)
    found = []
    cutoff = request.search.published_after or request.claim.report_timestamp
    with gdelt_request_scope():
        for query in plan.queries[:2]:
            try:
                found.extend(search_news(query, max_results=100, published_after=cutoff, timespan=timespan))
            except Exception as exc:
                LOG.warning("News discovery skipped: %s", type(exc).__name__)
    found = dedupe_articles(found, limit=200)

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
        if image:
            candidates.append(MediaCandidate(candidate_id=digest(article.url, image)[:16], article_url=article.url, source_name=article.source_name, published_at=article.published_at, media_type="image", media_url=image, title=article.title, article_text_excerpt=article.snippet, metadata={"gdelt_socialimage": True, "language": getattr(article, "language", None), "source_country": getattr(article, "source_country", None)}))
        if getattr(article, "image_collage", False):
            continue
        try:
            parsed = fetch_article(article)
        except Exception as exc:
            LOG.warning("Article skipped: %s", type(exc).__name__)
            continue
        if not parsed:
            continue
        for candidate in parsed.candidates[:5]:
            candidates.append(candidate)
    candidates = dedupe_media_candidates(candidates, limit=40)
    previews = [CandidatePreview(media_type=item.media_type, media_url=item.media_url, title=item.title, source_url=item.article_url) for item in candidates[:12]]
    shortlisted = choose_candidates(request, candidates, plan.locations)[:10]
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
    insufficient_count = 0
    irrelevant_count = 0
    for candidate in ranked:
        context = " ".join(part for part in [candidate.title, candidate.caption, candidate.article_text_excerpt] if part)
        manual_request = CheckRequest(report_id=request.report_id, topic=request.topic, claim=request.claim, evidence=[{"evidence_id": candidate.candidate_id, "type": candidate.media_type, "local_path": candidate.local_path, "published_at": candidate.published_at, "source_name": candidate.source_name, "source_url": candidate.article_url, "media_url": candidate.media_url, "context_text": context, "caption": candidate.caption}])
        try:
            result = cached_check(manual_request, candidate.media_url)
        except Exception as exc:
            LOG.warning("Visual verification skipped: %s", type(exc).__name__)
            continue
        if result.relation == Relation.INSUFFICIENT:
            insufficient_count += 1
        if result.relation == Relation.IRRELEVANT or result.event_match is False:
            irrelevant_count += 1
        if result.relation in {Relation.SUPPORTS, Relation.CONTRADICTS, Relation.REFINES} and result.confidence >= 0.65:
            if result.event_match is not False:
                evidence.append(to_evidence(candidate, result))
    selected = select_evidence(evidence)
    status = aggregate(evidence, insufficient_count, irrelevant_count)
    patch = aggregate_patch(request.claim.text, status, selected)
    return SearchResponse(claim_id=request.claim.claim_id, queries=plan.queries, articles_considered=len(articles), media_candidates=len(candidates), filtered_candidates=len(candidates) - len(ranked), verified_evidence_count=len(selected), claim_status=status.claim_status, confidence=status.confidence, aggregate=status, evidence=selected, candidate_previews=previews, suggested_update=patch, message="No sufficiently relevant multimodal evidence was found." if not selected else None)
