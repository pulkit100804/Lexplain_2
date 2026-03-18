"""
Lexplain — Agent 5: Legal Signal Extractor
MIT License | See README for MCP provenance contract.

Calls MCP tool extract_legal_signals_v1 to transform raw case facts
into structured actors, events, legal signals, and candidate sections.
"""
import os
from typing import Any, Dict

from mcp.mcp_client import MCPClient


def run(case_id: str, case_dir: str, cases_dir: str) -> Dict[str, Any]:
    fact_graph_path = os.path.join(case_dir, "fact_event_graph.json")
    client = MCPClient()
    response = client.call("extract_legal_signals_v1", {
        "case_id": case_id,
        "fact_graph_path": fact_graph_path,
        "cases_dir": cases_dir,
        "input_refs": [fact_graph_path],
    })
    result = response["result"]
    print(
        f"[Agent 5] Legal signals extracted: "
        f"{result['signal_count']} signals, "
        f"{result['candidate_count']} candidate sections"
    )
    return {
        "legal_signal_graph_path": result["result_ref"],
        "candidate_count": result["candidate_count"],
    }
