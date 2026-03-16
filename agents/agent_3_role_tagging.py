"""
Lexplain — Agent 3: Role Tagging
MIT License | See README for MCP provenance contract.

Calls MCP tool tag_legal_roles_v1 to tag document graph nodes with legal roles.
"""
import os
from typing import Any, Dict

from mcp.mcp_client import MCPClient


def run(case_id: str, case_dir: str, cases_dir: str) -> Dict[str, Any]:
    doc_graph_path = os.path.join(case_dir, "document_graph.json")
    client = MCPClient()
    response = client.call("tag_legal_roles_v1", {
        "case_id": case_id,
        "document_graph_path": doc_graph_path,
        "cases_dir": cases_dir,
        "input_refs": [doc_graph_path],
    })
    result = response["result"]
    print(f"[Agent 3] Role tagging complete → {result['result_ref']}")
    return {"role_tagged_graph_path": result["result_ref"]}
