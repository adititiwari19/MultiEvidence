from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Claim(StrictModel):
    claim_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    report_timestamp: datetime | None = None


class EvidenceInput(StrictModel):
    evidence_id: str = Field(min_length=1)
    type: Literal["image", "video"]
    url: str | None = None
    local_path: str | None = None
    published_at: datetime | None = None
    source_name: str | None = None
    source_url: str | None = None
    media_url: str | None = None
    context_text: str | None = None
    caption: str | None = None

    @model_validator(mode="after")
    def has_location(self):
        if not self.local_path and not self.url:
            raise ValueError("Evidence needs local_path or url")
        return self


class CheckRequest(StrictModel):
    report_id: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    claim: Claim
    evidence: list[EvidenceInput] = Field(min_length=1, max_length=1)
    ground: bool = False


class Relation(str, Enum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    REFINES = "REFINES"
    INSUFFICIENT = "INSUFFICIENT"
    IRRELEVANT = "IRRELEVANT"


class GroundingTarget(StrictModel):
    text: str
    claim_text: str | None = None
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    color: str
    mask_url: str | None = None
    bbox: tuple[int, int, int, int] | None = None


class EvidenceReference(StrictModel):
    evidence_id: str
    frame_id: str
    timestamp_seconds: float | None = None
    frame_url: str
    observation: str
    overlay_url: str | None = None
    groundings: list[GroundingTarget] = Field(default_factory=list)


class SuggestedUpdate(StrictModel):
    action: Literal["KEEP", "REPLACE", "QUALIFY"]
    old_text: str
    new_text: str
    reason: str


class CheckResponse(StrictModel):
    claim_id: str
    affected: bool
    relation: Relation
    relevance: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    observable_facts: list[str]
    evidence: list[EvidenceReference]
    suggested_update: SuggestedUpdate
    explanation: str
    limitations: list[str]
    event_match: bool | Literal["uncertain"] = "uncertain"
    event_match_reason: str = ""


class SearchOptions(StrictModel):
    provider: Literal["auto", "free", "google", "searxng", "duckduckgo", "gdelt"] = "auto"
    checkpoint: Literal["T0", "T1", "T2", "T3"] | None = None
    max_articles: int = Field(default=20, ge=1, le=30)
    max_images: int = Field(default=8, ge=0, le=8)
    max_videos: int = Field(default=5, ge=0, le=5)
    published_after: datetime | None = None
    published_before: datetime | None = None
    timespan: Literal["24h", "3days", "1week", "3months"] = "1week"


class SearchRequest(StrictModel):
    report_id: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    claim: Claim
    search: SearchOptions = Field(default_factory=SearchOptions)
    ground: bool = False


class QueryPlan(StrictModel):
    queries: list[str] = Field(min_length=2, max_length=5)
    entities: list[str]
    locations: list[str]
    event_terms: list[str]


class NewsArticle(StrictModel):
    url: str
    title: str
    source_name: str
    published_at: datetime | None = None
    snippet: str = ""


class MediaCandidate(StrictModel):
    candidate_id: str
    article_url: str
    source_name: str
    published_at: datetime | None = None
    media_type: Literal["image", "video"]
    media_url: str
    title: str
    article_text_excerpt: str = ""
    caption: str = ""
    metadata: dict = Field(default_factory=dict)
    local_path: str | None = None
    unavailable_for_visual_verification: bool = False


class EvidenceItem(StrictModel):
    evidence_id: str
    relation: Literal["SUPPORTS", "CONTRADICTS", "REFINES", "INSUFFICIENT", "IRRELEVANT"]
    confidence: float = Field(ge=0, le=1)
    media_type: Literal["image", "video"]
    source_name: str
    source_url: str
    published_at: datetime | None = None
    media_url: str
    local_path: str
    observation: str
    timestamp_seconds: float | None = None
    frame_url: str | None = None
    suggested_update: SuggestedUpdate
    frames: list[EvidenceReference] = Field(default_factory=list)
    event_match: bool | Literal["uncertain"] = "uncertain"
    event_match_reason: str = ""
    explanation: str = ""
    limitations: list[str] = Field(default_factory=list)


class AggregateStatus(StrictModel):
    claim_status: Literal["SUPPORTED", "CONTRADICTED", "REFINED", "UNCERTAIN"]
    confidence: float = Field(ge=0, le=1)
    support_count: int = 0
    contradict_count: int = 0
    refine_count: int = 0
    insufficient_count: int = 0
    irrelevant_count: int = 0


class CandidatePreview(StrictModel):
    media_type: Literal["image", "video"]
    media_url: str
    title: str
    source_url: str


class CachedMedia(StrictModel):
    filename: str
    media_type: Literal["image", "video"]
    media_url: str
    caption: str


class CachedGalleryEntry(StrictModel):
    checkpoint: Literal["T0", "T1", "T2", "T3"]
    claim_number: int
    claim: str
    article_title: str
    aliases: list[str]
    source_id: str
    source_name: str
    source_url: str
    source_record_date: str
    evidence_pdf: str
    evidence_pdf_url: str
    related_links: list[str]
    assessment: str
    verification_status: Literal["CANDIDATE_UNVERIFIED"]
    media: list[CachedMedia]


class SearchResponse(StrictModel):
    claim_id: str
    queries: list[str]
    articles_considered: int
    media_candidates: int
    filtered_candidates: int
    verified_evidence_count: int
    checked_media_count: int = 0
    claim_status: Literal["SUPPORTED", "CONTRADICTED", "REFINED", "UNCERTAIN"]
    confidence: float = Field(ge=0, le=1)
    aggregate: AggregateStatus
    evidence: list[EvidenceItem]
    checks: list[EvidenceItem] = Field(default_factory=list)
    cache_hit: bool = False
    cached_gallery: list[CachedGalleryEntry] = Field(default_factory=list)
    candidate_previews: list[CandidatePreview] = Field(default_factory=list)
    suggested_update: SuggestedUpdate
    message: str | None = None
