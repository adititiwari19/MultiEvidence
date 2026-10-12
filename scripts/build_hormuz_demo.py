"""Build a static, source-linked Hormuz image evidence handoff.

Run from the repository root after article sites and the selected verifier are reachable.
The output is a draft for human provenance review, never a claim of approval.
"""

import argparse
import hashlib
import json
import logging
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from app.config import OLLAMA_MODEL, VERIFIER_BACKEND, VLM_MODEL, ensure_directories
from app.models.schemas import CheckRequest, MediaCandidate, NewsArticle, SearchRequest
from app.services.retrieval.candidate_ranker import dedupe_media_candidates, rank_candidates
from app.services.retrieval.media_fetch import download_media
from app.services.verifier import check


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "demo/hormuz/plan.json"
DEFAULT_OUTPUT = ROOT / "demo/hormuz/output"
SOURCE_FILE = ROOT / "bengal_parisa_T3/02_T3_all_available_evidence.json"
RELATION_PRIORITY = {"SUPPORTS": 0, "REFINES": 1, "CONTRADICTS": 2, "INSUFFICIENT": 3, "IRRELEVANT": 4}
LOG = logging.getLogger(__name__)


def load_plan(path: Path) -> tuple[dict, dict]:
    plan = json.loads(path.read_text())
    sources = {row["source_id"]: row for row in json.loads(SOURCE_FILE.read_text())}
    for target in plan["targets"]:
        report = (ROOT / plan["report_root"] / f'{target["checkpoint"]}_report.md').read_text()
        if target["claim"] not in report:
            raise ValueError(f'Claim is not verbatim in {target["checkpoint"]} report: {target["claim_id"]}')
        cutoff = report.split("Evidence cutoff: ", 1)[1].splitlines()[0]
        target["cutoff"] = cutoff
        for source_id in target["source_ids"]:
            if source_id not in sources:
                raise ValueError(f"Unknown source: {source_id}")
            source = sources[source_id]
            if source["checkpoint"] > target["checkpoint"]:
                raise ValueError(f"Future checkpoint source: {source_id}")
            published = datetime.fromisoformat(source["date"]).replace(tzinfo=timezone.utc)
            if published > datetime.fromisoformat(cutoff.replace("Z", "+00:00")):
                raise ValueError(f"Source published after report cutoff: {source_id}")
    return plan, sources


def source_candidates(source: dict, overrides: dict, offline: bool) -> tuple[list[MediaCandidate], list[str]]:
    source_id = source["source_id"]
    errors = []
    published_at = datetime.fromisoformat(source["date"]).replace(tzinfo=timezone.utc)
    candidates = []
    if source_id in overrides:
        for entry in overrides[source_id]:
            path = Path(entry["local_path"]).expanduser().resolve()
            if not path.is_file():
                errors.append(f"{source_id}: missing override file {path}")
                continue
            media_url = entry.get("media_url")
            if not media_url or not media_url.startswith(("http://", "https://")):
                errors.append(f"{source_id}: override requires an original http(s) media_url")
                continue
            candidates.append(MediaCandidate(
                candidate_id=f'{source_id}-{hashlib.sha256(media_url.encode()).hexdigest()[:12]}',
                article_url=source["url"], source_name=source["outlet"], published_at=published_at,
                media_type="image", media_url=media_url, title=source["title"],
                article_text_excerpt=source.get("retrieved_snippet") or "",
                caption=entry.get("caption") or "", local_path=str(path),
                metadata={"local_override": True},
            ))
        return dedupe_media_candidates(candidates), errors
    if offline:
        return [], [f"{source_id}: no local override supplied"]
    try:
        from app.services.retrieval.article_parser import fetch_article
    except ImportError as exc:
        return [], [f"{source_id}: article parser dependency missing: {exc.name}"]
    article = NewsArticle(url=source["url"], title=source["title"], source_name=source["outlet"],
                          published_at=published_at, snippet=source.get("retrieved_snippet") or "")
    parsed = fetch_article(article)
    if not parsed:
        return [], [f"{source_id}: article unavailable"]
    for item in parsed.candidates:
        if item.media_type != "image":
            continue
        # The report source record gives the source date; the current page may have changed.
        candidates.append(item.model_copy(update={"candidate_id": f'{source_id}-{item.candidate_id}',
                                           "published_at": published_at}))
    if not candidates:
        errors.append(f"{source_id}: no article images found")
    return dedupe_media_candidates(candidates), errors


def verify_candidate(target: dict, source: dict, candidate: MediaCandidate) -> dict:
    if not candidate.local_path:
        candidate = download_media(candidate, target["claim_id"])
    if not candidate or not candidate.local_path:
        raise RuntimeError("image download failed")
    request = CheckRequest.model_validate({
        "report_id": target["checkpoint"], "topic": "Iran conflict and Strait of Hormuz",
        "claim": {"claim_id": target["claim_id"], "text": target["claim"], "report_timestamp": target["cutoff"]},
        "evidence": [{"evidence_id": candidate.candidate_id, "type": "image",
                      "local_path": candidate.local_path, "published_at": candidate.published_at,
                      "source_name": source["outlet"], "source_url": source["url"],
                      "media_url": candidate.media_url, "caption": candidate.caption,
                      "context_text": " ".join((source["title"], source.get("retrieved_snippet") or ""))[:1500]}],
    })
    result = check(request)
    return {
        "checkpoint": target["checkpoint"], "claim_id": target["claim_id"], "claim": target["claim"],
        "report_cutoff": target["cutoff"], "source_id": source["source_id"],
        "source_title": source["title"], "source_outlet": source["outlet"],
        "source_url": source["url"], "source_record_date": source["date"],
        "media_url": candidate.media_url, "caption": candidate.caption,
        "image_original_path": candidate.local_path,
        "relation": result.relation.value, "confidence": result.confidence,
        "observation": result.observable_facts, "explanation": result.explanation,
        "limitations": result.limitations, "event_match": result.event_match,
        "event_match_reason": result.event_match_reason,
        "suggested_update": result.suggested_update.model_dump(mode="json"),
        "model": OLLAMA_MODEL if VERIFIER_BACKEND == "ollama" else VLM_MODEL,
        "review_status": "PENDING_HUMAN_PROVENANCE_REVIEW",
    }


def select_entries(entries: list[dict], limit: int = 5) -> list[dict]:
    """Prefer distinct sources and retain one honest insufficient example if present."""
    ranked = sorted(entries, key=lambda row: (RELATION_PRIORITY[row["relation"]], -row["confidence"]))
    chosen = []
    seen = set()
    caution = next((row for row in ranked if row["relation"] in {"INSUFFICIENT", "IRRELEVANT"}), None)
    substantive_limit = limit - int(caution is not None)
    for row in ranked:
        if row["relation"] in {"INSUFFICIENT", "IRRELEVANT"}:
            continue
        if row["source_id"] not in seen:
            chosen.append(row)
            seen.add(row["source_id"])
        if len(chosen) >= substantive_limit:
            break
    for row in ranked:
        if len(chosen) >= substantive_limit:
            break
        if row not in chosen and row["relation"] not in {"INSUFFICIENT", "IRRELEVANT"}:
            chosen.append(row)
    if caution is not None and caution not in chosen:
        chosen.append(caution)
    for row in ranked:
        if len(chosen) >= limit:
            break
        if row not in chosen:
            chosen.append(row)
    return chosen


def build(plan_path: Path, output: Path, overrides_path: Path | None = None, offline: bool = False) -> dict:
    ensure_directories()
    plan, sources = load_plan(plan_path)
    overrides = json.loads(overrides_path.read_text()) if overrides_path else {}
    output.mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir(exist_ok=True)
    rows = defaultdict(list)
    errors = []
    candidate_cache = {}
    for target in plan["targets"]:
        pool = []
        for source_id in target["source_ids"]:
            source = sources[source_id]
            if source_id not in candidate_cache:
                candidate_cache[source_id] = source_candidates(source, overrides, offline)
            candidates, source_errors = candidate_cache[source_id]
            errors.extend(source_errors)
            pool.extend(candidates)
        request = SearchRequest.model_validate({
            "report_id": target["checkpoint"], "topic": plan["topic"],
            "claim": {"claim_id": target["claim_id"], "text": target["claim"],
                      "report_timestamp": target["cutoff"]},
            "search": {"max_images": 8, "max_videos": 0, "published_before": target["cutoff"]},
        })
        # These sources were already selected from the report's supplied
        # evidence. Rank them, but do not discard a source solely because a
        # generic lexical score misses a maritime synonym.
        shortlist = rank_candidates(request, dedupe_media_candidates(pool), ["Strait of Hormuz"])[:8]
        downloaded = []
        for candidate in shortlist:
            source_id = candidate.candidate_id.split("-", 1)[0]
            try:
                fetched = candidate if candidate.local_path else download_media(candidate, target["claim_id"])
                if fetched and fetched.local_path:
                    downloaded.append(fetched)
                else:
                    errors.append(f"{source_id}: image download failed")
            except Exception as exc:
                errors.append(f"{source_id}: {type(exc).__name__}: {exc}")
        for candidate in rank_candidates(request, downloaded, ["Strait of Hormuz"]):
            source_id = candidate.candidate_id.split("-", 1)[0]
            try:
                rows[target["checkpoint"]].append(verify_candidate(target, sources[source_id], candidate))
            except Exception as exc:
                LOG.warning("Verification failed for %s: %s", source_id, type(exc).__name__)
                errors.append(f"{source_id}: {type(exc).__name__}: {exc}")
    selected = []
    coverage = {}
    for checkpoint in ("T0", "T1", "T2", "T3"):
        picked = select_entries(rows[checkpoint])
        coverage[checkpoint] = {"selected": len(picked), "target_minimum": 2,
                                "count_met": len(picked) >= 2, "demo_ready": False}
        for row in picked:
            original = Path(row.pop("image_original_path"))
            digest = hashlib.sha256(original.read_bytes()).hexdigest()[:16]
            name = f'{checkpoint}_{row["source_id"]}_{digest}{original.suffix.lower()}'
            destination = output / "images" / name
            shutil.copyfile(original, destination)
            row["image_path"] = f"images/{name}"
            row["sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
            selected.append(row)
    if not selected:
        state = "NO_VERIFIED_IMAGES"
    elif not all(item["count_met"] for item in coverage.values()):
        state = "PARTIAL_COVERAGE"
    else:
        state = "PENDING_HUMAN_REVIEW"
    manifest = {"topic": plan["topic"], "source_bundle": "bengal_parisa_T3",
                "state": state, "model": VLM_MODEL, "coverage": coverage,
                "images": selected, "errors": sorted(set(errors))}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def finalize(output: Path) -> dict:
    """Mark a human-reviewed bundle ready only when coverage and files are complete."""
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    counts = {checkpoint: 0 for checkpoint in ("T0", "T1", "T2", "T3")}
    for row in manifest["images"]:
        if row["review_status"] != "APPROVED":
            raise ValueError(f'Image still needs review: {row["checkpoint"]} {row["source_id"]}')
        image_path = (output / row["image_path"]).resolve()
        if not image_path.is_relative_to((output / "images").resolve()) or not image_path.is_file():
            raise ValueError(f'Image file missing or outside bundle: {row["image_path"]}')
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f'Image file changed after verification: {row["image_path"]}')
        counts[row["checkpoint"]] += 1
    for checkpoint, count in counts.items():
        if not 2 <= count <= 5:
            raise ValueError(f"{checkpoint} needs 2–5 approved images; found {count}")
        manifest["coverage"][checkpoint]["selected"] = count
        manifest["coverage"][checkpoint]["count_met"] = True
        manifest["coverage"][checkpoint]["demo_ready"] = True
    manifest["state"] = "READY"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overrides", type=Path, help="JSON mapping source IDs to local image files and original URLs")
    parser.add_argument("--offline", action="store_true", help="Use only local overrides; do not fetch article pages")
    parser.add_argument("--finalize", action="store_true", help="Validate manually approved images and mark the bundle READY")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    manifest = finalize(args.output) if args.finalize else build(args.plan, args.output, args.overrides, args.offline)
    print(json.dumps({"state": manifest["state"], "coverage": manifest["coverage"],
                      "image_count": len(manifest["images"]), "error_count": len(manifest["errors"]),
                      "manifest": str(args.output / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
