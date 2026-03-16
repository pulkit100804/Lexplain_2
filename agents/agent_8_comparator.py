"""
Lexplain — Agent 8: Precedent Comparator
MIT License | See README for MCP provenance contract.

Calls MCP tool compare_precedents_v1 to compute prosecution/defense viability.
"""
import os
from typing import Any, Dict

from mcp.mcp_client import MCPClient


def run(case_id: str, case_dir: str, cases_dir: str) -> Dict[str, Any]:
    ingredient_report_path = os.path.join(case_dir, "ingredient_report.json")
    precedent_matches_path = os.path.join(case_dir, "precedent_matches.json")
    client = MCPClient()
    response = client.call("compare_precedents_v1", {
        "case_id": case_id,
        "ingredient_report_path": ingredient_report_path,
        "precedent_matches_path": precedent_matches_path,
        "cases_dir": cases_dir,
        "input_refs": [ingredient_report_path, precedent_matches_path],
    })
    result = response["result"]
    print(f"[Agent 8] Comparator report saved → {result['result_ref']}")
    return {"comparator_report_path": result["result_ref"]}
