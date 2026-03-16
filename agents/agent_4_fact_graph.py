"""
Lexplain — Agent 4: Fact Event Graph Builder
MIT License | See README for MCP provenance contract.

Calls MCP tool build_fact_graph_v1 to extract actors, events, dates.
"""
import os
from typing import Any, Dict

from mcp.mcp_client import MCPClient


def run(case_id: str, case_dir: str, cases_dir: str) -> Dict[str, Any]:
    role_tagged_path = os.path.join(case_dir, "role_tagged_graph.json")
    client = MCPClient()
    response = client.call("build_fact_graph_v1", {
        "case_id": case_id,
        "role_tagged_graph_path": role_tagged_path,
        "cases_dir": cases_dir,
        "input_refs": [role_tagged_path],
    })
    result = response["result"]
    print(f"[Agent 4] Fact event graph built → {result['result_ref']}")
    return {"fact_graph_path": result["result_ref"]}
