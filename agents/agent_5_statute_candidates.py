"""
Lexplain — Agent 5: Statute Candidate Generator
MIT License | See README for MCP provenance contract.

Calls MCP tool identify_statutes_v1 to map facts to candidate IPC sections.
"""
import os
from typing import Any, Dict

from mcp.mcp_client import MCPClient


def run(case_id: str, case_dir: str, cases_dir: str) -> Dict[str, Any]:
    fact_graph_path = os.path.join(case_dir, "fact_event_graph.json")
    client = MCPClient()
    response = client.call("identify_statutes_v1", {
        "case_id": case_id,
        "fact_graph_path": fact_graph_path,
        "cases_dir": cases_dir,
        "input_refs": [fact_graph_path],
    })
    result = response["result"]
    candidates = result.get("candidates", [])
    print(f"[Agent 5] Statute candidates identified → {len(candidates)} candidates")
    return {"statute_candidates_path": result["result_ref"], "candidates": candidates}
