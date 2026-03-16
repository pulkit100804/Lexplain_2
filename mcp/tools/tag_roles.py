"""
Lexplain — MCP Tool: tag_legal_roles_v1
MIT License | See README for MCP provenance contract.

Tags document graph nodes with legal roles using deterministic keyword rules.
"""
import json
import os
from typing import Any, Dict, List

from utils.provenance import make_provenance

_ROLE_RULES: List[Dict[str, Any]] = [
    {"role": "allegation", "keywords": ["stated", "alleged", "claims", "submits", "contends"]},
    {"role": "evidence", "keywords": ["complainant", "witness", "exhibit", "produced", "affidavit"]},
    {"role": "order", "keywords": ["ordered", "directed", "hereby", "court directs", "judgment"]},
    {"role": "finding", "keywords": ["found", "held", "observed", "noted", "concluded"]},
    {"role": "fact", "keywords": ["on", "at", "when", "after", "before", "during"]},
]


def _tag_node(text: str) -> Dict[str, Any]:
    text_lower = text.lower()
    for rule in _ROLE_RULES:
        for kw in rule["keywords"]:
            if kw in text_lower:
                return {"role": rule["role"], "confidence": 0.9, "matched_keyword": kw}
    return {"role": "unclassified", "confidence": 0.5, "matched_keyword": None}


def tag_legal_roles_v1(
    case_id: str,
    document_graph_path: str,
    cases_dir: str,
    tenant_id: str = "local_dev",
    input_refs: List[str] | None = None,
    _mcp_trace_id: str = "",
    _mcp_tool_version: str = "1.0.0",
    **_: Any,
) -> Dict[str, Any]:
    with open(document_graph_path, "r", encoding="utf-8") as f:
        doc_graph = json.load(f)

    tagged_nodes = []
    for node in doc_graph["nodes"]:
        tag = _tag_node(node["text"])
        tagged_nodes.append({**node, "role_tag": tag["role"], "confidence": tag["confidence"], "matched_keyword": tag["matched_keyword"]})

    provenance = make_provenance(
        tool_name="tag_legal_roles_v1",
        tool_version=_mcp_tool_version,
        input_refs=input_refs or [document_graph_path],
        trace_id=_mcp_trace_id,
    )

    role_tagged_graph = {
        "case_id": case_id,
        "tenant_id": tenant_id,
        "nodes": tagged_nodes,
        "edges": doc_graph.get("edges", []),
        "provenance": provenance,
    }

    out_path = os.path.join(cases_dir, case_id, "role_tagged_graph.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(role_tagged_graph, f, indent=2, ensure_ascii=False)

    return {"result_ref": out_path, "provenance": provenance}
