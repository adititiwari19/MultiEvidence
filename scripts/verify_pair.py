#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from app.models.schemas import CheckRequest
from app.services.verifier import check


def main() -> None:
    parser = argparse.ArgumentParser(description="Check one report claim against a local image or video")
    parser.add_argument("--claim", required=True)
    parser.add_argument("--media", required=True)
    parser.add_argument("--type", choices=["image", "video"], required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--claim-id", default="cli_claim")
    parser.add_argument("--evidence-id", default="cli_evidence")
    parser.add_argument("--source-name")
    parser.add_argument("--published-at")
    parser.add_argument("--report-timestamp")
    args = parser.parse_args()
    request = CheckRequest.model_validate({
        "report_id": "cli", "topic": args.topic,
        "claim": {"claim_id": args.claim_id, "text": args.claim, "report_timestamp": args.report_timestamp},
        "evidence": [{"evidence_id": args.evidence_id, "type": args.type, "local_path": str(Path(args.media).resolve()), "published_at": args.published_at, "source_name": args.source_name}],
    })
    print(json.dumps(check(request).model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
