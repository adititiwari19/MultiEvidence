from datetime import datetime, timezone

from app.models.schemas import EvidenceItem, SuggestedUpdate
from app.services.aggregator import aggregate, aggregate_patch, select_evidence


def item(evidence_id, relation, confidence, day, source):
    action = "REPLACE" if relation == "CONTRADICTS" else "QUALIFY" if relation == "REFINES" else "KEEP"
    return EvidenceItem(evidence_id=evidence_id, relation=relation, confidence=confidence, media_type="image", source_name=source, source_url=f"https://{source}/story", published_at=datetime(2026, 3, day, tzinfo=timezone.utc), media_url=f"https://{source}/image.jpg", local_path="/tmp/image.jpg", observation="Visible condition", suggested_update=SuggestedUpdate(action=action, old_text="Road open", new_text="Road blocked" if action != "KEEP" else "Road open", reason="Visual evidence"))


def test_recent_contradiction_and_patch():
    old_support = item("a", "SUPPORTS", 0.85, 1, "source-a.org")
    new_contradict = item("b", "CONTRADICTS", 0.92, 17, "source-b.org")
    result = aggregate([old_support, new_contradict], insufficient_count=1)
    assert result.claim_status == "CONTRADICTED"
    assert result.support_count == 1 and result.contradict_count == 1 and result.insufficient_count == 1
    assert aggregate_patch("Road open", result, [old_support, new_contradict]).action == "REPLACE"


def test_conflicting_recent_evidence_is_uncertain():
    items = [item("a", "SUPPORTS", 0.9, 17, "source-a.org"), item("b", "CONTRADICTS", 0.85, 17, "source-b.org")]
    result = aggregate(items)
    assert result.claim_status == "UNCERTAIN"
    assert aggregate_patch("Road open", result, items).action == "KEEP"


def test_select_caps_and_prefers_sources():
    items = [item(str(i), "SUPPORTS", 0.9 - i * 0.01, 17, "same.org") for i in range(6)]
    items.append(item("different", "REFINES", 0.7, 17, "other.org"))
    selected = select_evidence(items)
    assert len(selected) == 5
    assert any(item.source_name == "other.org" for item in selected)


def test_duplicate_media_from_one_article_counts_as_one_source():
    first = item("image-a", "SUPPORTS", 0.88, 17, "source.org")
    duplicate = item("image-b", "SUPPORTS", 0.86, 17, "source.org").model_copy(update={"source_url": "https://source.org/story?utm_source=gallery"})
    result = aggregate([first, duplicate])
    assert result.claim_status == "SUPPORTED"
    assert result.support_count == 1
    assert result.confidence == 0.88
    assert len(select_evidence([first, duplicate])) == 2  # Both may still be displayed.


def test_irrelevant_and_insufficient_do_not_affect_claim_status():
    support = item("support", "SUPPORTS", 0.84, 17, "source.org")
    irrelevant = item("different-event", "IRRELEVANT", 0.99, 17, "other.org").model_copy(update={"event_match": False})
    insufficient = item("unclear", "INSUFFICIENT", 0.90, 17, "third.org")
    result = aggregate([support, irrelevant, insufficient])
    assert result.claim_status == "SUPPORTED"
    assert (result.support_count, result.contradict_count, result.refine_count, result.insufficient_count, result.irrelevant_count) == (1, 0, 0, 1, 1)
    assert [entry.evidence_id for entry in select_evidence([support, irrelevant, insufficient])] == ["support"]
