"""
Lexplain — MCP Tool: compare_precedents_v1
MIT License | See README for MCP provenance contract.

Compares ingredient_report to each precedent (none for prototype) and
computes prosecution/defense viability scores from ingredient satisfaction.
"""
import json
import os
from typing import Any, Dict, List

from utils.provenance import make_provenance


def compare_precedents_v1(
    case_id: str,
    ingredient_report_path: str,
    precedent_matches_path: str,
    cases_dir: str,
    tenant_id: str = "local_dev",
    input_refs: List[str] | None = None,
    _mcp_trace_id: str = "",
    _mcp_tool_version: str = "1.0.0",
    **_: Any,
) -> Dict[str, Any]:
    with open(ingredient_report_path, "r", encoding="utf-8") as f:
        ingredient_report = json.load(f)
    with open(precedent_matches_path, "r", encoding="utf-8") as f:
        precedent_matches = json.load(f)

    evaluations = ingredient_report.get("statute_evaluations", [])
    if evaluations:
        avg_score = sum(e["overall_score"] for e in evaluations) / len(evaluations)
    else:
        avg_score = 0.0

    # Heuristic: prosecution viability tracks average ingredient satisfaction;
    # defense viability is inverse weighted
    prosecution_viability = round(min(avg_score * 1.1, 1.0), 3)
    defense_viability = round(max(1.0 - avg_score * 0.9, 0.0), 3)

    provenance = make_provenance(
        tool_name="compare_precedents_v1",
        tool_version=_mcp_tool_version,
        input_refs=input_refs or [ingredient_report_path, precedent_matches_path],
        trace_id=_mcp_trace_id,
    )

    comparator_report = {
        "case_id": case_id,
        "tenant_id": tenant_id,
        "prosecution_viability": prosecution_viability,
        "defense_viability": defense_viability,
        "precedent_comparisons": [],  # empty — no precedents in prototype
        "note": "Viability computed from ingredient satisfaction aggregate (no precedents in prototype).",
        "provenance": provenance,
    }

    out_path = os.path.join(cases_dir, case_id, "comparator_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparator_report, f, indent=2, ensure_ascii=False)

    return {"result_ref": out_path, "provenance": provenance}
