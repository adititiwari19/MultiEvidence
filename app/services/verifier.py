import json
import logging
import re
import time
from pathlib import Path

from app.config import MOLMO2_MODEL, OLLAMA_MODEL, VERIFIER_BACKEND, VLM_MODEL
from app.models.schemas import CheckRequest, CheckResponse, EvidenceReference, Relation, SuggestedUpdate
from app.services.image_verifier import prepare_image
from app.services.openrouter_client import OpenRouterOutputError, verify_with_openrouter
from app.services.ollama_client import verify_with_ollama
from app.services.patcher import make_patch
from app.services.relevance import assess_relevance, is_irrelevant
from app.services.video import extract_frames, rank_frames


LOG = logging.getLogger(__name__)
PROMPT = (Path(__file__).parents[1] / "prompts" / "verification.txt").read_text()
EVENT_TYPES = {
    "wildfire": r"\b(?:wildfire|forest fire|bushfire)\b",
    "building fire": r"\b(?:apartment|house|building|warehouse|structure) fire\b",
    "flood": r"\bflood(?:s|ed|ing)?\b",
    "earthquake": r"\bearthquake\b",
    "landslide": r"\blandslide\b",
}


def event_match(request: CheckRequest) -> tuple[bool | str, str]:
    """Use source context to reject clear event mismatches without claiming visual proof."""
    item = request.evidence[0]
    context = " ".join(part for part in (item.caption, item.context_text, item.source_url) if part)
    if not context.strip():
        return "uncertain", "No event-specific source context was supplied."
    claim_text = f"{request.claim.text} {request.topic}"
    claim_types = {name for name, pattern in EVENT_TYPES.items() if re.search(pattern, claim_text, re.I)}
    context_types = {name for name, pattern in EVENT_TYPES.items() if re.search(pattern, context, re.I)}
    if claim_types and context_types and claim_types.isdisjoint(context_types):
        return False, f"Source context describes {', '.join(sorted(context_types))}, while the claim concerns {', '.join(sorted(claim_types))}."
    claim_places = {word.lower() for word in re.findall(r"\b[A-Z][a-z]{2,}\b", claim_text) if word.lower() not in {"the", "no", "a", "road", "flood", "response"}}
    context_places = {word.lower() for word in re.findall(r"\b[A-Z][a-z]{2,}\b", context) if word.lower() not in {"the", "no", "news", "road", "flood", "response"}}
    if claim_places and context_places and claim_places.isdisjoint(context_places) and claim_types and context_types:
        return False, "The named places in the claim and source context do not match."
    if item.published_at and request.claim.report_timestamp and abs((item.published_at.date() - request.claim.report_timestamp.date()).days) > 180:
        return "uncertain", "The source and report dates are far apart, so event identity is uncertain."
    if claim_types & context_types and (not claim_places or claim_places & context_places):
        return True, "Source context matches the claim's event type and available place names."
    if claim_places & context_places:
        return True, "Source context shares a named place with the claim; event identity is plausible."
    return "uncertain", "The supplied metadata does not establish that this is the same event."


def resolve_frame_reference(ref, frame_order: list[tuple[str, float | None, str]]) -> tuple[str, float | None, str]:
    by_id = {entry[0]: entry for entry in frame_order}
    index = ref.frame_index
    if index is None and ref.frame_id not in by_id and ref.frame_id:
        match = re.fullmatch(r"(?:frame[ _-]*)?(\d+)", ref.frame_id, re.I)
        if match:
            index = int(match.group(1))
    if index is not None:
        if index >= len(frame_order):
            raise OpenRouterOutputError(f"VLM cited out-of-range frame index: {index}")
        entry = frame_order[index]
        if ref.frame_id and ref.frame_id in by_id and ref.frame_id != entry[0]:
            raise OpenRouterOutputError("VLM frame ID and index disagree")
        if ref.frame_id and ref.frame_id not in by_id:
            named_index = re.fullmatch(r"(?:frame[ _-]*)?(\d+)", ref.frame_id, re.I)
            if named_index and int(named_index.group(1)) != index:
                raise OpenRouterOutputError("VLM frame ID and index disagree")
        return entry
    if ref.frame_id in by_id:
        return by_id[ref.frame_id]
    raise OpenRouterOutputError(f"VLM cited an unknown frame: {ref.frame_id}")


def check(request: CheckRequest) -> CheckResponse:
    started = time.monotonic()
    if request.ground and not MOLMO2_MODEL:
        raise ValueError("ground=true requires MOLMO2_MODEL")
    item = request.evidence[0]
    if not item.local_path:
        raise ValueError("Remote evidence URLs are not supported; provide local_path")
    path = Path(item.local_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Evidence file not found: {path}")
    match, match_reason = event_match(request)
    relevance = 0.0 if match is False else assess_relevance(request)
    if match is False or is_irrelevant(relevance):
        reason = match_reason if match is False else "Evidence metadata is unrelated to the claim."
        response = CheckResponse(claim_id=request.claim.claim_id, affected=False, relation=Relation.IRRELEVANT, relevance=relevance, confidence=min(0.9, round(1 - relevance, 4)), observable_facts=[], evidence=[], suggested_update=SuggestedUpdate(action="KEEP", old_text=request.claim.text, new_text=request.claim.text, reason=reason), explanation=reason, limitations=["Visual content was not verified."], event_match=match, event_match_reason=match_reason)
        LOG.info(json.dumps({"claim_id": request.claim.claim_id, "evidence_id": item.evidence_id, "modality": item.type, "relevance": relevance, "candidate_frames": 0, "selected_frames": 0, "model": None, "latency_seconds": round(time.monotonic() - started, 3), "relation": response.relation.value, "confidence": response.confidence}))
        return response
    if item.type == "image":
        frame_id, data_url, frame_url = prepare_image(path, item.evidence_id)
        frame_order = [(frame_id, None, frame_url)]
        images = [(frame_id, data_url)]
        candidate_count = 1
    else:
        candidates = extract_frames(path, item.evidence_id)
        if not candidates:
            raise ValueError("No video frames were extracted")
        selected = rank_frames(request.claim.text, candidates)
        frame_order = [(frame.frame_id, frame.timestamp_seconds, frame.url) for frame in selected]
        images = [(frame.frame_id, frame.data_url()) for frame in selected]
        candidate_count = len(candidates)
    for index, (frame_id, timestamp, _) in enumerate(frame_order):
        LOG.info("frame_index=%d frame_id=%s timestamp_seconds=%s", index, frame_id, timestamp)
    metadata = {"claim": request.claim.text, "topic": request.topic, "published_at": item.published_at.isoformat() if item.published_at else None, "report_timestamp": request.claim.report_timestamp.isoformat() if request.claim.report_timestamp else None, "source_name": item.source_name, "source_url": item.source_url, "article_context": item.context_text[:1000] if item.context_text else None, "caption": item.caption, "event_match": match, "event_match_reason": match_reason, "frames": [{"frame_index": index, "frame_id": frame_id, "timestamp_seconds": timestamp} for index, (frame_id, timestamp, _) in enumerate(frame_order)]}
    verification_prompt = PROMPT + "\nInput:\n" + json.dumps(metadata)
    if VERIFIER_BACKEND == "ollama":
        result = verify_with_ollama(verification_prompt, images)
    elif VERIFIER_BACKEND == "openrouter":
        result = verify_with_openrouter(verification_prompt, images)
    else:
        raise ValueError(f"Unsupported VERIFIER_BACKEND: {VERIFIER_BACKEND}")
    if result.relation in {Relation.SUPPORTS, Relation.CONTRADICTS, Relation.REFINES} and (result.direct_visual_evidence is False or (result.direct_visual_evidence is None and bool(item.context_text))):
        result = result.model_copy(update={"relation": Relation.INSUFFICIENT, "confidence": min(result.confidence, 0.75), "observable_facts": [], "evidence": [], "explanation": "The supplied media does not directly show the claimed state.", "suggested_new_text": request.claim.text})
    direct_relation = result.relation in {Relation.SUPPORTS, Relation.CONTRADICTS, Relation.REFINES} and result.direct_visual_evidence is True and bool(result.evidence)
    confidence = min(result.confidence, 1.0 if direct_relation else 0.90)
    if result.direct_visual_evidence is False:
        confidence = min(confidence, 0.75)
    if match == "uncertain":
        confidence = round(confidence * 0.8, 4)
    result = result.model_copy(update={"confidence": confidence})
    references = []
    for ref in result.evidence:
        frame_id, timestamp, url = resolve_frame_reference(ref, frame_order)
        references.append(EvidenceReference(evidence_id=item.evidence_id, frame_id=frame_id, timestamp_seconds=timestamp, frame_url=url, observation=ref.observation))
    if result.relation == Relation.INSUFFICIENT and item.type == "image" and not references and result.observable_facts:
        frame_id, timestamp, url = frame_order[0]
        references.append(EvidenceReference(evidence_id=item.evidence_id, frame_id=frame_id, timestamp_seconds=timestamp, frame_url=url, observation=result.observable_facts[0]))
    if request.ground and references and (direct_relation or result.relation == Relation.INSUFFICIENT and bool(result.observable_facts)):
        from app.services.grounding import ground_reference

        references = [ground_reference(request.claim.text, reference) for reference in references]
    patch = make_patch(request.claim.text, result)
    response = CheckResponse(claim_id=request.claim.claim_id, affected=patch.action != "KEEP", relation=result.relation, relevance=relevance, confidence=result.confidence, observable_facts=result.observable_facts, evidence=references, suggested_update=patch, explanation=result.explanation, limitations=result.limitations, event_match=match, event_match_reason=match_reason)
    LOG.info(json.dumps({"claim_id": request.claim.claim_id, "evidence_id": item.evidence_id, "modality": item.type, "relevance": response.relevance, "candidate_frames": candidate_count, "selected_frames": len(images), "model": OLLAMA_MODEL if VERIFIER_BACKEND == "ollama" else VLM_MODEL, "latency_seconds": round(time.monotonic() - started, 3), "relation": response.relation.value, "confidence": response.confidence}))
    return response
