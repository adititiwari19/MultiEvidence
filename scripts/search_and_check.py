#!/usr/bin/env python3
import argparse
import json
import logging

from app.models.schemas import SearchRequest
from app.services.search_and_check import search_and_check


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Find and check public multimedia evidence for one claim")
    parser.add_argument("--claim", required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--published-after")
    parser.add_argument("--published-before", help="Historical cutoff; excludes later articles")
    parser.add_argument("--report-timestamp", help="Timestamp of the report claim; also sets a historical cutoff")
    parser.add_argument("--timespan", default="1week", choices=["24h", "3days", "1week", "3months"], help="GDELT window when --published-after is omitted")
    parser.add_argument("--claim-id", default="cli_claim")
    parser.add_argument("--report-id", default="cli_report")
    parser.add_argument("--max-articles", type=int, default=20)
    parser.add_argument("--provider", choices=["auto", "free", "google", "searxng", "duckduckgo", "gdelt"], default="auto")
    args = parser.parse_args()
    request = SearchRequest.model_validate({"report_id": args.report_id, "topic": args.topic, "claim": {"claim_id": args.claim_id, "text": args.claim, "report_timestamp": args.report_timestamp}, "search": {"provider": args.provider, "max_articles": args.max_articles, "published_after": args.published_after, "published_before": args.published_before}})
    print(json.dumps(search_and_check(request, timespan=args.timespan).model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
