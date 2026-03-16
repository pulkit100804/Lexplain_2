"""
Lexplain — Agent 7: Retrieval / Precedent Search (PLACEHOLDER)
MIT License | See README for MCP provenance contract.

Calls MCP tool search_precedents_v1 (placeholder — always returns empty list).
RAG intentionally NOT implemented. Plug in Elastic/Qdrant retrieval later.
"""
import os
from typing import Any, Dict

from mcp.mcp_client import MCPClient


def run(case_id: str, case_dir: str, cases_dir: str) -> Dict[str, Any]:
    fact_graph_path = os.path.join(case_dir, "fact_event_graph.json")
    client = MCPClient()
    response = client.call("search_precedents_v1", {
        "case_id": case_id,
        "query_text": "",  # placeholder — no real query
        "cases_dir": cases_dir,
        "input_refs": [fact_graph_path],
    })
    result = response["result"]
    print(f"[Agent 7] Precedent search (placeholder) → {result['result_ref']} (0 matches)")
    return {"precedent_matches_path": result["result_ref"]}
