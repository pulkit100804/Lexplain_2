"""
Lexplain — MCP Tool: search_precedents_v1
MIT License | See README for MCP provenance contract.

PLACEHOLDER — RAG intentionally left as placeholder.
Returns empty list. Plug in Elastic/Qdrant retrieval here in a future sprint.

TODO: Replace this stub with a real vector search / BM25 retrieval pipeline.
      Suggested integration points:
        - Elasticsearch client: from elasticsearch import Elasticsearch
        - Qdrant client: from qdrant_client import QdrantClient
        - Embed query using sentence-transformers and do nearest-neighbour search
"""
import json
import os
from typing import Any, Dict, List

from utils.provenance import make_provenance


def search_precedents_v1(
    case_id: str,
    query_text: str,
    cases_dir: str,
    tenant_id: str = "local_dev",
    input_refs: List[str] | None = None,
    _mcp_trace_id: str = "",
    _mcp_tool_version: str = "1.0.0",
    **_: Any,
) -> Dict[str, Any]:
    """
    PLACEHOLDER: Returns empty precedent list.
    RAG / precedent retrieval is intentionally NOT implemented in this prototype.
    """
    provenance = make_provenance(
        tool_name="search_precedents_v1",
        tool_version=_mcp_tool_version,
        input_refs=input_refs or [],
        trace_id=_mcp_trace_id,
        model_version=None,
    )
    precedent_matches = {
        "case_id": case_id,
        "tenant_id": tenant_id,
        "matches": [],  # RAG placeholder — always empty in prototype
        "note": "RAG intentionally left as placeholder; plug your retrieval later.",
        "provenance": provenance,
    }
    out_path = os.path.join(cases_dir, case_id, "precedent_matches.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(precedent_matches, f, indent=2, ensure_ascii=False)

    return {"result_ref": out_path, "matches": [], "provenance": provenance}
