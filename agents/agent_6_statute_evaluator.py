"""
Lexplain — Agent 6: Statute Ingredient Evaluator
MIT License | See README for MCP provenance contract.

Calls MCP tool evaluate_statute_ingredients_v1 to evaluate
candidate statutes' ingredients against legal signals and case facts.
"""
import os
from typing import Any, Dict

from mcp.mcp_client import MCPClient


def run(case_id: str, case_dir: str, cases_dir: str) -> Dict[str, Any]:
    legal_signal_graph_path = os.path.join(case_dir, "legal_signal_graph.json")
    client = MCPClient()
    response = client.call("evaluate_statute_ingredients_v1", {
        "case_id": case_id,
        "legal_signal_graph_path": legal_signal_graph_path,
        "cases_dir": cases_dir,
        "input_refs": [legal_signal_graph_path],
    })
    result = response["result"]
    print(f"[Agent 6] Statute evaluation complete: {result['evaluation_count']} statutes evaluated")
    return {"statute_evaluation_path": result["result_ref"]}
