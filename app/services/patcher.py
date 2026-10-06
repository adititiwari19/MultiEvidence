from app.models.schemas import Relation, SuggestedUpdate
from app.services.openrouter_client import ModelResult


def make_patch(old_text: str, result: ModelResult) -> SuggestedUpdate:
    if result.relation == Relation.CONTRADICTS and result.confidence >= 0.75 and result.suggested_new_text.strip():
        action = "REPLACE"
    elif result.relation == Relation.REFINES and result.suggested_new_text.strip():
        action = "QUALIFY"
    else:
        action = "KEEP"
    new_text = result.suggested_new_text.strip() if action != "KEEP" else old_text
    return SuggestedUpdate(action=action, old_text=old_text, new_text=new_text, reason=result.explanation)
