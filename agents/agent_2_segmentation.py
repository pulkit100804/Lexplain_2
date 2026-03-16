"""
Lexplain — Agent 2: Segmentation / Document Graph
MIT License | See README for MCP provenance contract.

Calls MCP tool extract_sentences_v1 to build the document graph.
"""
import os
from typing import Any, Dict

from mcp.mcp_client import MCPClient


def run(case_id: str, case_dir: str, cases_dir: str) -> Dict[str, Any]:
    norm_path = os.path.join(case_dir, "normalized_text.txt")
    client = MCPClient()
    response = client.call("extract_sentences_v1", {
        "case_id": case_id,
        "normalized_text_path": norm_path,
        "cases_dir": cases_dir,
        "input_refs": [norm_path],
    })
    result = response["result"]
    doc_graph_path = result["result_ref"]
    print(f"[Agent 2] Document graph built → {doc_graph_path} ({result['node_count']} nodes)")
    return {"document_graph_path": doc_graph_path, "node_count": result["node_count"]}
