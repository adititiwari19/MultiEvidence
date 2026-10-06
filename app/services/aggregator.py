from datetime import datetime, timezone
from urllib.parse import urlsplit

from app.models.schemas import AggregateStatus, EvidenceItem, SuggestedUpdate
from app.services.retrieval.news_search import canonical_url


def utc_date(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def source_bonus(item: EvidenceItem) -> int:
    domain = urlsplit(item.source_url).netloc.lower()
    return int(domain.endswith((".gov", ".int", ".edu")))


def sort_key(item: EvidenceItem):
    relation_strength = {"CONTRADICTS": 3, "REFINES": 2, "SUPPORTS": 1}
    date = utc_date(item.published_at) or datetime.min.replace(tzinfo=timezone.utc)
    return relation_strength.get(item.relation, 0), item.confidence, date.timestamp(), source_bonus(item)


def select_evidence(items: list[EvidenceItem], limit: int = 5) -> list[EvidenceItem]:
    useful = sorted((item for item in items if item.relation in {"SUPPORTS", "CONTRADICTS", "REFINES"} and item.event_match is not False and item.confidence >= 0.65), key=sort_key, reverse=True)
    selected = []
    seen_sources = set()
    for item in useful:
        source = urlsplit(item.source_url).netloc.lower() or item.source_name.lower()
        if source in seen_sources:
            continue
        selected.append(item)
        seen_sources.add(source)
        if len(selected) == limit:
            return selected
    for item in useful:
        if item not in selected:
            selected.append(item)
        if len(selected) == limit:
            break
    return selected


def aggregate(items: list[EvidenceItem], insufficient_count: int = 0, irrelevant_count: int = 0) -> AggregateStatus:
    insufficient_count += sum(item.relation == "INSUFFICIENT" for item in items)
    irrelevant_count += sum(item.relation == "IRRELEVANT" or item.event_match is False for item in items)
    independent = {}
    for item in items:
        if item.relation not in {"SUPPORTS", "CONTRADICTS", "REFINES"} or item.event_match is False or item.confidence < 0.65:
            continue
        source = canonical_url(item.source_url) if item.source_url else item.source_name.lower()
        previous = independent.get(source)
        if previous is None or (item.confidence, sort_key(item)) > (previous.confidence, sort_key(previous)):
            independent[source] = item
    useful = list(independent.values())
    support = [item for item in useful if item.relation == "SUPPORTS"]
    contradict = [item for item in useful if item.relation == "CONTRADICTS"]
    refine = [item for item in useful if item.relation == "REFINES"]
    status = "UNCERTAIN"
    confidence = 0.0
    if contradict:
        strongest = max(contradict, key=lambda item: item.confidence)
        if support:
            latest_support = max((utc_date(item.published_at) for item in support if item.published_at), default=None)
            contradiction_date = utc_date(strongest.published_at)
            newer = bool(contradiction_date and latest_support and contradiction_date > latest_support)
            if strongest.confidence >= 0.8 and newer:
                status, confidence = "CONTRADICTED", strongest.confidence
            else:
                status, confidence = "UNCERTAIN", round(min(strongest.confidence, max(item.confidence for item in support)) * 0.5, 4)
        elif strongest.confidence >= 0.75:
            status, confidence = "CONTRADICTED", strongest.confidence
    elif refine:
        status, confidence = "REFINED", max(item.confidence for item in refine)
    elif support:
        status, confidence = "SUPPORTED", max(item.confidence for item in support)
    return AggregateStatus(claim_status=status, confidence=confidence, support_count=len(support), contradict_count=len(contradict), refine_count=len(refine), insufficient_count=insufficient_count, irrelevant_count=irrelevant_count)


def aggregate_patch(old_text: str, status: AggregateStatus, items: list[EvidenceItem]) -> SuggestedUpdate:
    desired = {"CONTRADICTED": ("CONTRADICTS", "REPLACE"), "REFINED": ("REFINES", "QUALIFY")}.get(status.claim_status)
    if desired:
        matches = [item for item in items if item.relation == desired[0] and item.suggested_update.action == desired[1]]
        if matches:
            return max(matches, key=lambda item: item.confidence).suggested_update
    return SuggestedUpdate(action="KEEP", old_text=old_text, new_text=old_text, reason="No sufficiently strong visual update was found.")
