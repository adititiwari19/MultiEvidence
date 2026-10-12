from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


class ParisaDialogAuditor:

    def __init__(self):
        repo = Path(os.environ["PARISA_REPO"]).resolve()

        if not (repo / "common" / "detect.py").exists():
            raise RuntimeError(
                f"PARISA_REPO must contain common/detect.py: {repo}"
            )

        sys.path.insert(0, str(repo))

        from common.detect import HallucinationDetector

        api_key = (
            os.environ.get("PARISA_API_KEY")
            or os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("API_KEY")
        )

        if not api_key:
            raise RuntimeError(
                "Set PARISA_API_KEY, OPENROUTER_API_KEY, or API_KEY"
            )

        model = os.environ.get(
            "PARISA_MODEL",
            "openai/gpt-5.4-mini-20260317",
        )

        reasoning = os.environ.get(
            "PARISA_REASONING",
            "medium",
        )

        if reasoning == "none":
            reasoning = None

        self.zs = HallucinationDetector(
            api_key=api_key,
            model=model,
            prompt_format="zs",
            reasoning_effort=reasoning,
        )

        self.ckp = HallucinationDetector(
            api_key=api_key,
            model=model,
            prompt_format="checkpoint-repeat",
            reasoning_effort=reasoning,
        )

    @staticmethod
    def format_dialogue(
        turns: List[Dict[str, str]]
    ) -> str:
        lines = []

        for t in turns:
            role = t.get("role", "")
            content = (t.get("content") or "").strip()

            if not content:
                continue

            prefix = (
                "User"
                if role == "user"
                else "Assistant"
            )

            lines.append(f"{prefix}: {content}")

        return "\n".join(lines)

    @staticmethod
    def format_report(
        report: Optional[Dict[str, Any]]
    ) -> str:
        if not report:
            return ""

        parts = []

        title = report.get("title") or report.get("topic")
        if title:
            parts.append(str(title))

        for sec in report.get("sections", []):
            sec_title = sec.get("title")

            if sec_title:
                parts.append(f"{sec_title}:")

            for p in sec.get("paragraphs", []):
                parts.append(str(p))

        # Keep auditor prompt bounded.
        text = "\n".join(parts)
        return text[:5000]

    @staticmethod
    def format_evidence(
        evidence: Optional[List[Dict[str, Any]]]
    ) -> str:
        if not evidence:
            return ""

        lines = []

        # Current demo is tiny, but bound this for future use.
        for s in evidence[-12:]:
            sid = str(s.get("id", "?"))
            title = str(s.get("title", "Source"))
            date = str(s.get("date", "") or "")
            snippet = str(s.get("snippet", "") or "")

            header = f"[{sid}] {title}"
            if date:
                header += f" ({date})"

            lines.append(header)

            if snippet:
                lines.append(snippet[:800])

        return "\n".join(lines)

    def build_audit_input(
        self,
        turns: List[Dict[str, str]],
        *,
        evidence: Optional[List[Dict[str, Any]]] = None,
        report: Optional[Dict[str, Any]] = None,
    ) -> str:

        dialogue = self.format_dialogue(turns)
        report_text = self.format_report(report)
        evidence_text = self.format_evidence(evidence)

        pieces = [
            "GROUNDING CONTEXT PROVIDED TO THE ASSISTANT",
            "The assistant had access to the following report and "
            "evidence when producing its response.",
        ]

        if report_text:
            pieces.extend([
                "",
                "CURRENT REPORT:",
                report_text,
            ])

        if evidence_text:
            pieces.extend([
                "",
                "AVAILABLE EVIDENCE:",
                evidence_text,
            ])

        pieces.extend([
            "",
            "DIALOGUE TO AUDIT:",
            dialogue,
        ])

        return "\n".join(pieces)

    def audit(
        self,
        turns: List[Dict[str, str]],
        *,
        evidence: Optional[List[Dict[str, Any]]] = None,
        report: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        audit_input = self.build_audit_input(
            turns,
            evidence=evidence,
            report=report,
        )

        r_zs = self.zs.detect(audit_input)
        r_ckp = self.ckp.detect(audit_input)

        flagged = (
            r_zs.label == 1
            or r_ckp.label == 1
        )

        # Prefer checkpoint+repeat explanation for presentation.
        primary = (
            r_ckp
            if r_ckp.label == 1
            else r_zs
            if r_zs.label == 1
            else r_ckp
        )

        hal_type = (
            primary.hal_type
            if primary.hal_type
            and primary.hal_type != "none"
            else "none"
        )

        explanation = (
            primary.explanation
            if flagged and primary.explanation
            else "No issue flagged."
        )

        return {
            "flagged": flagged,
            "label": 1 if flagged else 0,
            "type": hal_type,
            "explanation": explanation,

            # Available for debugging/reproducibility.
            # The UI does not expose these implementation details.
            "details": {
                "zero_shot": r_zs.to_dict(),
                "checkpoint_repeat": r_ckp.to_dict(),
            },

            "latency_s": (
                float(r_zs.latency_s or 0.0)
                + float(r_ckp.latency_s or 0.0)
            ),
        }
