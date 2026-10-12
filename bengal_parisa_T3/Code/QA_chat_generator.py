from __future__ import annotations

import re
import time
from typing import Any, Dict, List

import requests


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


def _report_text(report: Dict[str, Any] | None) -> str:
    if not report:
        return "(No report available.)"

    parts = []
    for sec in report.get("sections", []):
        title = sec.get("title", "")
        if title:
            parts.append(f"## {title}")
        for p in sec.get("paragraphs", []):
            parts.append(str(p))
    return "\n".join(parts)


def _source_text(sources: List[Dict[str, Any]]) -> str:
    if not sources:
        return "(No sources available.)"

    lines = []
    for s in sources:
        sid = s.get("id", "?")
        title = s.get("title", "Source")
        date = s.get("date", "")
        snippet = s.get("snippet", "")
        lines.append(
            f"[{sid}] {title}"
            + (f" ({date})" if date else "")
            + f"\n{snippet}"
        )
    return "\n\n".join(lines)



def _memory_text(memory_retrieval: Dict[str, Any] | None) -> str:
    if not memory_retrieval:
        return "(No relevant persistent memory retrieved.)"

    results = memory_retrieval.get("results", [])

    if not results:
        return "(No relevant persistent memory retrieved.)"

    lines = []

    for item in results[:5]:
        text = (item.get("memory") or "").strip()

        if not text:
            continue

        meta = item.get("metadata") or {}
        mem_type = meta.get("memory_type", "memory")

        lines.append(f"[{mem_type}] {text}")

    return "\n".join(lines) if lines else "(No relevant persistent memory retrieved.)"


def _history_text(history: List[Dict[str, str]]) -> str:
    if not history:
        return "(No prior dialogue.)"

    # Keep the live prompt compact.
    lines = []
    for turn in history[-8:]:
        role = "Analyst" if turn.get("role") == "user" else "Assistant"
        text = (turn.get("content") or "").strip()
        if text:
            lines.append(f"{role}: {text}")
    return "\n".join(lines)


SYSTEM_INSTRUCTIONS = """\
You are BENGAL, an analyst-facing assistant working over a living,
evidence-grounded report.

Answer the analyst's question using the supplied report and evidence.

Requirements:
- Ground factual claims in the supplied evidence.
- Cite supporting source IDs inline, for example [S1] or [S1][S2].
- Never invent a source ID.
- If sources disagree, explicitly describe the disagreement.
- If the available evidence is insufficient, say so.
- Do not call an answer "verified", "correct", or "trustworthy".
- Persistent memory is prior context, not an authoritative source.
- For current factual claims, prefer the supplied evidence and cite its source IDs.
- Be concise and useful to an analyst.
"""


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


def _extract_openai_text(data: Dict[str, Any]) -> str:
    # Raw Responses API response: find output_text content blocks.
    texts = []

    for item in data.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                text = content.get("text")
                if text:
                    texts.append(text)

    if texts:
        return "\n".join(texts).strip()

    # Defensive fallback in case the representation changes.
    if isinstance(data.get("output_text"), str):
        return data["output_text"].strip()

    raise RuntimeError("OpenAI response did not contain text output")


def _validated_sources(answer: str, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id = {str(s.get("id")): s for s in sources}

    cited = []
    seen = set()

    for sid in re.findall(r"\[([A-Za-z]+\d+)\]", answer):
        if sid in by_id and sid not in seen:
            cited.append(by_id[sid])
            seen.add(sid)

    return cited


def generate_chat_answer(
    *,
    provider: str,
    api_key: str,
    model: str,
    message: str,
    state: Dict[str, Any],
    history: List[Dict[str, str]],
    memory_retrieval: Dict[str, Any] | None = None,
) -> Dict[str, Any]:

    if provider not in {"openai", "openrouter"}:
        raise ValueError(f"Unsupported API provider: {provider}")

    if not api_key:
        raise ValueError("API key is required")

    if not model:
        raise ValueError("Model name is required")

    prompt = _prompt(
        message,
        state,
        history,
        memory_retrieval,
    )
    started = time.time()

    if provider == "openai":
        r = requests.post(
            OPENAI_RESPONSES_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "instructions": SYSTEM_INSTRUCTIONS,
                "input": prompt,
            },
            timeout=90,
        )

        if not r.ok:
            try:
                msg = r.json().get("error", {}).get("message", r.text[:500])
            except Exception:
                msg = r.text[:500]
            raise RuntimeError(f"OpenAI API {r.status_code}: {msg}")

        data = r.json()
        answer = _extract_openai_text(data)
        usage = data.get("usage", {})

    else:
        r = requests.post(
            OPENROUTER_CHAT_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=90,
        )

        if not r.ok:
            try:
                msg = r.json().get("error", {}).get("message", r.text[:500])
            except Exception:
                msg = r.text[:500]
            raise RuntimeError(f"OpenRouter API {r.status_code}: {msg}")

        data = r.json()
        answer = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {})

    return {
        "answer": answer,
        "provider": provider,
        "model": model,
        "latency_s": time.time() - started,
        "usage": usage,
        "sources": _validated_sources(answer, state.get("sources", [])),
    }
