"""
Lexplain — MCP Tool: extract_sentences_v1
MIT License | See README for MCP provenance contract.

Splits normalized text into sentence/clause nodes using rule-based SBD.
"""
import json
import os
from typing import Any, Dict, List

from utils.text_utils import split_sentences
from utils.provenance import make_provenance


def extract_sentences_v1(
    case_id: str,
    normalized_text_path: str,
    cases_dir: str,
    tenant_id: str = "local_dev",
    input_refs: List[str] | None = None,
    _mcp_trace_id: str = "",
    _mcp_tool_version: str = "1.0.0",
    **_: Any,
) -> Dict[str, Any]:
    """Build document_graph.json from normalized text."""
    with open(normalized_text_path, "r", encoding="utf-8") as f:
        text = f.read()

    sentences = split_sentences(text)
    nodes: List[Dict[str, Any]] = []
    offset = 0
    for i, sent in enumerate(sentences):
        start = text.find(sent, offset)
        end = start + len(sent) if start >= 0 else offset + len(sent)
        nodes.append({
            "node_id": f"n{i:04d}",
            "text": sent,
            "page_no": 0,
            "span": [max(start, offset), end],
        })
        offset = end

    # simple adjacency edges
    edges: List[Dict[str, Any]] = []
    for i in range(len(nodes) - 1):
        edges.append({
            "from_node": nodes[i]["node_id"],
            "to_node": nodes[i + 1]["node_id"],
            "edge_type": "adjacent",
        })

    provenance = make_provenance(
        tool_name="extract_sentences_v1",
        tool_version=_mcp_tool_version,
        input_refs=input_refs or [normalized_text_path],
        trace_id=_mcp_trace_id,
    )

    document_graph = {
        "case_id": case_id,
        "tenant_id": tenant_id,
        "nodes": nodes,
        "edges": edges,
        "node_count": len(nodes),
        "provenance": provenance,
    }

    out_path = os.path.join(cases_dir, case_id, "document_graph.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(document_graph, f, indent=2, ensure_ascii=False)

    return {"result_ref": out_path, "node_count": len(nodes), "provenance": provenance}
