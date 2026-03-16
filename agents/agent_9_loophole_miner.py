"""
Lexplain — Agent 9: Loophole Miner
MIT License | See README for MCP provenance contract.

Calls MCP tool mine_loopholes_v1 to identify weak/missing evidence.
"""
import os
from typing import Any, Dict

from mcp.mcp_client import MCPClient


def run(case_id: str, case_dir: str, cases_dir: str) -> Dict[str, Any]:
    ingredient_report_path = os.path.join(case_dir, "ingredient_report.json")
    role_tagged_graph_path = os.path.join(case_dir, "role_tagged_graph.json")
    client = MCPClient()
    response = client.call("mine_loopholes_v1", {
        "case_id": case_id,
        "ingredient_report_path": ingredient_report_path,
        "role_tagged_graph_path": role_tagged_graph_path,
        "cases_dir": cases_dir,
        "input_refs": [ingredient_report_path, role_tagged_graph_path],
    })
    result = response["result"]
    print(f"[Agent 9] Loophole map saved → {result['result_ref']}")
    return {"loophole_map_path": result["result_ref"]}
