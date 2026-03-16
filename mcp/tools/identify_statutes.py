"""
Lexplain — MCP Tool: identify_statutes_v1
MIT License | See README for MCP provenance contract.

Maps events from the fact graph to candidate IPC sections using keyword rules.
"""
import json
import os
from typing import Any, Dict, List

from utils.provenance import make_provenance

# Rule-based keyword → statute mapping
_STATUTE_RULES: List[Dict[str, Any]] = [
    {
        "statute_id": "IPC_420",
        "name": "Cheating and dishonestly inducing delivery of property",
        "keywords": ["cheat", "fraud", "dishonest", "deceiv", "misrepresent"],
    },
    {
        "statute_id": "IPC_506",
        "name": "Punishment for criminal intimidation",
        "keywords": ["threat", "intimidat", "menace", "coerce", "frighten"],
    },
    {
        "statute_id": "IPC_302",
        "name": "Punishment for murder",
        "keywords": ["murder", "killed", "kill", "death", "homicide"],
    },
    {
        "statute_id": "IPC_307",
        "name": "Attempt to murder",
        "keywords": ["attempt to murder", "attempt to kill", "assault with intention"],
    },
    {
        "statute_id": "IPC_376",
        "name": "Punishment for rape",
        "keywords": ["rape", "sexual assault", "outrag", "molestation"],
    },
    {
        "statute_id": "IPC_379",
        "name": "Punishment for theft",
        "keywords": ["theft", "stole", "stolen", "rob", "pickpocket"],
    },
    {
        "statute_id": "IPC_354",
        "name": "Assault or criminal force to outrage modesty",
        "keywords": ["outrage modesty", "assault modesty", "indecent"],
    },
    {
        "statute_id": "IPC_323",
        "name": "Punishment for voluntarily causing hurt",
        "keywords": ["hurt", "beat", "assault", "injury", "physical harm"],
    },
]


def identify_statutes_v1(
    case_id: str,
    fact_graph_path: str,
    cases_dir: str,
    tenant_id: str = "local_dev",
    input_refs: List[str] | None = None,
    _mcp_trace_id: str = "",
    _mcp_tool_version: str = "1.0.0",
    **_: Any,
) -> Dict[str, Any]:
    with open(fact_graph_path, "r", encoding="utf-8") as f:
        fact_graph = json.load(f)

    # Collect all text; use .get() to guard against nodes missing 'text'
    all_text = " ".join(n.get("text", "").lower() for n in fact_graph.get("nodes", []))

    candidates: List[Dict[str, Any]] = []
    for rule in _STATUTE_RULES:
        matched_kws = [kw for kw in rule["keywords"] if kw in all_text]
        if matched_kws:
            candidates.append({
                "statute_id": rule["statute_id"],
                "name": rule["name"],
                "match_reason": f"Keywords found: {', '.join(matched_kws)}",
                "matched_keywords": matched_kws,
                "confidence": min(0.5 + 0.1 * len(matched_kws), 0.95),
            })

    provenance = make_provenance(
        tool_name="identify_statutes_v1",
        tool_version=_mcp_tool_version,
        input_refs=input_refs or [fact_graph_path],
        trace_id=_mcp_trace_id,
    )

    statute_candidates = {
        "case_id": case_id,
        "tenant_id": tenant_id,
        "candidates": candidates,
        "candidate_count": len(candidates),
        "provenance": provenance,
    }

    out_path = os.path.join(cases_dir, case_id, "statute_candidates.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(statute_candidates, f, indent=2, ensure_ascii=False)

    return {"result_ref": out_path, "candidates": candidates, "provenance": provenance}
