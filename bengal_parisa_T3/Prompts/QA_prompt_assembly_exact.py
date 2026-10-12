def _prompt(
    message: str,
    state: Dict[str, Any],
    history: List[Dict[str, str]],
    memory_retrieval: Dict[str, Any] | None = None,
) -> str:
    return f"""\
CURRENT REPORT
--------------
{_report_text(state.get("current_report"))}

AVAILABLE EVIDENCE
------------------
{_source_text(state.get("sources", []))}

RELEVANT PERSISTENT MEMORY
--------------------------
{_memory_text(memory_retrieval)}

RECENT DIALOGUE
---------------
{_history_text(history)}

ANALYST QUESTION
----------------
{message}
"""
