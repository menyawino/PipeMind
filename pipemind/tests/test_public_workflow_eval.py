from __future__ import annotations

from pipemind.tools.evaluate_public_workflows import check_thresholds, summarize_results


def test_summarize_results_aggregates_rates():
    results = [
        {
            "parse_ok": True,
            "ingestion_mode": "compiled",
            "bootstrap_mode": "native",
            "tool_count": 10,
            "agent_ready_rule_count": 8,
            "composition_ready_rule_count": 6,
            "mcp_app_ok": True,
        },
        {
            "parse_ok": True,
            "ingestion_mode": "source-fallback",
            "bootstrap_mode": "source-fallback",
            "tool_count": 6,
            "agent_ready_rule_count": 3,
            "composition_ready_rule_count": 2,
            "mcp_app_ok": False,
        },
        {
            "parse_ok": False,
            "mcp_app_ok": False,
        },
    ]

    summary = summarize_results(results)

    assert summary["repo_count"] == 3
    assert summary["parse_success_count"] == 2
    assert summary["compiled_import_count"] == 1
    assert summary["compiled_import_rate"] == 1 / 3
    assert summary["tool_count"] == 16
    assert summary["agent_ready_rule_count"] == 11
    assert summary["agent_ready_coverage"] == 11 / 16
    assert summary["composition_ready_rule_count"] == 8
    assert summary["mcp_app_success_rate"] == 0.5


def test_check_thresholds_reports_failures():
    summary = {
        "compiled_import_rate": 0.75,
        "agent_ready_coverage": 0.80,
    }

    acceptance = check_thresholds(summary, min_compiled_import_rate=0.80, min_agent_ready_coverage=0.85)

    assert acceptance["passed"] is False
    assert acceptance["checks"]["compiled_import_rate"] is False
    assert acceptance["checks"]["agent_ready_coverage"] is False