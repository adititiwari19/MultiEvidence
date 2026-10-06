import logging
import re
from functools import lru_cache
from pathlib import Path

from app.config import EMBEDDING_MODEL, RELEVANCE_THRESHOLD
from app.models.schemas import CheckRequest


LOG = logging.getLogger(__name__)
STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "of", "and", "or", "as", "for", "with", "from", "by", "it", "this", "that"}
GENERIC_NAMES = {"image", "video", "example", "sample", "upload", "media", "photo", "file"}


def words(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9]+", text.lower()) if len(word) > 2 and word not in STOPWORDS}


def entity_overlap(claim: str, context: str) -> float:
    tokens = words(claim)
    return len(tokens & words(context)) / len(tokens) if tokens else 0.0


def semantic_similarity(claim: str, context: str) -> float:
    from sentence_transformers import util

    model = text_model()
    embeddings = model.encode([claim, context], convert_to_tensor=True)
    return max(0.0, min(1.0, float(util.cos_sim(embeddings[0], embeddings[1]).item())))


@lru_cache(maxsize=1)
def text_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def relevance_score(claim: str, context: str) -> float:
    return round(0.70 * semantic_similarity(claim, context) + 0.30 * entity_overlap(claim, context), 4)


def assess_relevance(request: CheckRequest) -> float:
    item = request.evidence[0]
    if item.context_text:
        try:
            return relevance_score(request.claim.text, item.context_text[:1500])
        except Exception as exc:
            LOG.warning("Relevance embedding unavailable: %s", type(exc).__name__)
            return RELEVANCE_THRESHOLD
    stem = Path(item.local_path or item.url or "").stem.replace("_", " ").replace("-", " ")
    if not words(stem) or words(stem) <= GENERIC_NAMES:
        return RELEVANCE_THRESHOLD
    context = " ".join(part for part in [request.topic, item.source_name or "", stem] if part).strip()
    try:
        return relevance_score(request.claim.text, context)
    except Exception as exc:
        LOG.warning("Relevance embedding unavailable: %s", type(exc).__name__)
        return RELEVANCE_THRESHOLD


def is_irrelevant(score: float) -> bool:
    return score < RELEVANCE_THRESHOLD
