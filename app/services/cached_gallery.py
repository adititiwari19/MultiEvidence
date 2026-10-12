"""Match known Hormuz demo prompts to the local, source-linked media gallery."""

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from app.models.schemas import CachedGalleryEntry


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "demo" / "hormuz" / "cached_gallery.json"
HISTORY = ROOT / "bengal_parisa_T3" / "History"
STOPWORDS = {"a", "an", "and", "about", "amid", "are", "by", "did", "do", "for", "from",
             "how", "in", "is", "of", "on", "or", "the", "their", "to", "was", "what",
             "when", "where", "which", "who", "why", "with"}


def tokens(value: str) -> set[str]:
    value = unicodedata.normalize("NFKD", value).casefold()
    value = "".join(char for char in value if not unicodedata.combining(char))
    words = re.findall(r"[a-z0-9]+", value)
    return {word[:-1] if len(word) > 4 and word.endswith("s") else word
            for word in words if word not in STOPWORDS and (len(word) > 2 or any(char.isdigit() for char in word))}


@lru_cache(maxsize=1)
def gallery() -> list[CachedGalleryEntry]:
    if not MANIFEST.is_file():
        return []
    payload = json.loads(MANIFEST.read_text())
    entries = []
    for row in payload["entries"]:
        folder = HISTORY / row["checkpoint"].lower()
        media = [item for item in row["media"] if (folder / item["filename"]).is_file()]
        if not media:
            continue
        entries.append(CachedGalleryEntry.model_validate({
            **row,
            "media": media,
            "evidence_pdf_url": "/evidence-docs/" + row["evidence_pdf"],
        }))
    return entries


def match_gallery(query: str, checkpoint: str | None = None) -> CachedGalleryEntry | None:
    entries = [row for row in gallery() if not checkpoint or row.checkpoint == checkpoint]
    if not entries:
        return None
    explicit = re.search(r"\b(t[0-3])\s*(?:claim|query|article|#)\s*([1-8])\b", query, re.I)
    if explicit:
        return next((row for row in entries if row.checkpoint.lower() == explicit.group(1).lower()
                     and row.claim_number == int(explicit.group(2))), None)
    query_words = tokens(query)
    if len(query_words) < 3:
        return None
    scored = []
    for row in entries:
        best = 0.0
        for alias in row.aliases:
            alias_words = tokens(alias)
            shared = len(query_words & alias_words)
            if not shared or not alias_words:
                continue
            if query_words == alias_words:
                best = 1.0
                break
            precision = shared / len(query_words)
            recall = shared / len(alias_words)
            score = 2 * precision * recall / (precision + recall)
            if query_words <= alias_words:
                score = max(score, 0.75 + 0.25 * recall)
            best = max(best, score)
        scored.append((best, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_row = scored[0]
    next_score = scored[1][0] if len(scored) > 1 else 0.0
    return best_row if best_score >= 0.58 and (best_score == 1.0 or best_score - next_score >= 0.07) else None
