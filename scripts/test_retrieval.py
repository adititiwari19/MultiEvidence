#!/usr/bin/env python3
import argparse
import json
import logging

from app.models.schemas import SearchRequest
from app.services.retrieval.news_search import gdelt_request_scope, search_news, search_status
from app.services.retrieval.query_builder import build_queries


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Print raw news search results without article or media processing")
    parser.add_argument("--query", help="Run one GDELT DOC query directly")
    parser.add_argument("--claim", help="Build structured text and visual queries for a claim")
    parser.add_argument("--topic", default="")
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--timespan", default="1week", choices=["24h", "3days", "1week", "3months"])
    args = parser.parse_args()
    if not args.query and not args.claim:
        parser.error("provide --query or --claim")
    if args.claim:
        request = SearchRequest.model_validate({"report_id": "diagnostic", "topic": args.topic or args.claim, "claim": {"claim_id": "diagnostic", "text": args.claim}})
        queries = build_queries(request).queries[:2]
        print(f"TEXT QUERY: {queries[0]}\nVISUAL QUERY: {queries[1]}", flush=True)
    else:
        queries = [args.query]
    with gdelt_request_scope():
        for query in queries:
            results = search_news(query, max_results=args.max_results, timespan=args.timespan)
            print(json.dumps({"query": query, "status": search_status(), "count": len(results), "results": [{"title": item.title, "article_url": item.url, "socialimage": getattr(item, "socialimage", None), "published_at": item.published_at.isoformat() if item.published_at else None, "source_name": item.source_name} for item in results]}, indent=2))


if __name__ == "__main__":
    main()
