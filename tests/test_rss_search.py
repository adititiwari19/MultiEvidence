from datetime import datetime, timezone

from app.models.schemas import QueryPlan, SearchRequest
from app.services.retrieval.rss_search import bing_query, parse_bing_rss, search_bing_news


FEED = b'''<?xml version="1.0"?><rss version="2.0"><channel><title>News</title><item>
<title>Dubai airport flight update</title>
<link>https://www.bing.com/news/apiclick.aspx?ref=FexRss&amp;url=https%3A%2F%2Fpublisher.example%2Fstory%3Futm_source%3Dbing</link>
<description>&lt;b&gt;Flights&lt;/b&gt; are arriving</description>
<pubDate>Thu, 08 Oct 2026 12:00:00 GMT</pubDate><source>Publisher</source>
</item></channel></rss>'''


def test_rss_parses_publisher_link_and_date():
    articles = parse_bing_rss(FEED, cutoff=datetime(2026, 10, 1, tzinfo=timezone.utc))
    assert len(articles) == 1
    assert articles[0].url == "https://publisher.example/story"
    assert articles[0].source_name == "Publisher"
    assert articles[0].snippet == "Flights are arriving"
    assert parse_bing_rss(FEED, cutoff=datetime(2026, 10, 9, tzinfo=timezone.utc)) == []


def test_bing_query_uses_place_and_event():
    request = SearchRequest.model_validate({"report_id": "r", "topic": "Dubai airport", "claim": {"claim_id": "c", "text": "Flights are arriving at Dubai airport"}})
    plan = QueryPlan(queries=["one", "two"], entities=[], locations=["Dubai airport"], event_terms=[])
    assert bing_query(request, plan) == '"Dubai airport" flight'


def test_rss_search_uses_one_feed_request(monkeypatch):
    calls = []

    class Response:
        status_code = 200
        content = FEED

        def raise_for_status(self):
            pass

    def get(url, **kwargs):
        calls.append((url, kwargs["params"]))
        return Response()

    monkeypatch.setattr("app.services.retrieval.rss_search.requests.get", get)
    results = search_bing_news('"Dubai airport" flight', max_results=10, published_after=datetime(2026, 10, 1, tzinfo=timezone.utc), timespan="1week")
    assert len(results) == 1
    assert calls[0][1] == {"q": '"Dubai airport" flight', "format": "rss"}
