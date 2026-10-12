# Strait of Hormuz — BENGAL QA and Auditor Examples

## Materials

- **01_T3_current_report.md** — complete English report at T3.
- **02_T3_all_available_evidence.json** — 32 source records collected across T0–T3, with retrieved snippets and source text when available from the backend.
- **03_T3_new_evidence_only.json** — eight new sources at the T3 checkpoint.
- **04_T3_QA_context_fixture.json** — the T3 report, reconstructed cumulative evidence, prior T2 report, and latest changes. This is a candidate-testing fixture, **not a captured full live QA request**.
- **History/** — all four report versions, changes, and eight new sources at each checkpoint.
- **Prompts/** — exact QA system instructions, dynamic prompt assembly, and both active auditor prompt formats.
- **Code/** — QA answer generator and the Auditor adapter.

## How QA and the auditor work

We plan to finish the summary timeline first, then open the chat at T3 (March 19).

For most questions, the QA agent gets the current report, available source snippets, recent chat, and any relevant MemGuard memories. It is asked to cite its sources. It does **not** get every older report as a separate document.

For questions like “What changed since the last update?”, we use the T2 → T3 edits and the new T3 sources instead.

Once the agent answers, the auditor checks the conversation, current report, and selected source snippets. It runs both zero-shot and checkpoint-repeat checks. Right now, either one can trigger “Review suggested”. We're hoping to find a few good examples where the flag is actually helpful, without false alarms.

The older reports are included in `History/` in case you want to compare them. 
