import json
import sys
from types import SimpleNamespace

from scripts import search_and_check as full_cli
from scripts import test_retrieval as diagnostic


def test_diagnostic_distinguishes_query_error_from_empty_results(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["test_retrieval", "--query", "Goma road", "--timespan", "3months"])
    seen = []
    monkeypatch.setattr(diagnostic, "search_news", lambda query, **kwargs: seen.append(kwargs) or [])
    monkeypatch.setattr(diagnostic, "search_status", lambda: "QUERY_ERROR")
    diagnostic.main()
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "QUERY_ERROR"
    assert output["count"] == 0
    assert seen[0]["timespan"] == "3months"


def test_full_cli_passes_timespan(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["search_and_check", "--claim", "Goma road open", "--topic", "North Kivu", "--timespan", "3months"])
    seen = []
    monkeypatch.setattr(full_cli, "search_and_check", lambda request, **kwargs: seen.append(kwargs) or SimpleNamespace(model_dump=lambda **options: {"claim_id": request.claim.claim_id}))
    full_cli.main()
    assert json.loads(capsys.readouterr().out)["claim_id"] == "cli_claim"
    assert seen == [{"timespan": "3months"}]
