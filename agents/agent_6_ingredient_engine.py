"""
Lexplain — Agent 6: Ingredient Satisfaction Engine
MIT License | See README for MCP provenance contract.

Calls MCP tool evaluate_ingredients_v1 to check statute ingredient satisfaction.
"""
import os
from typing import Any, Dict

from mcp.mcp_client import MCPClient


def run(case_id: str, case_dir: str, cases_dir: str) -> Dict[str, Any]:
    statute_candidates_path = os.path.join(case_dir, "statute_candidates.json")
    fact_graph_path = os.path.join(case_dir, "fact_event_graph.json")
    client = MCPClient()
    response = client.call("evaluate_ingredients_v1", {
        "case_id": case_id,
        "statute_candidates_path": statute_candidates_path,
        "fact_graph_path": fact_graph_path,
        "cases_dir": cases_dir,
        "input_refs": [statute_candidates_path, fact_graph_path],
    })
    result = response["result"]
    print(f"[Agent 6] Ingredient report saved → {result['result_ref']}")
    return {"ingredient_report_path": result["result_ref"]}
