"""
Lexplain — MCP Tool: mine_loopholes_v1
MIT License | See README for MCP provenance contract.

Identifies legal loopholes by inspecting ingredient satisfaction thresholds.
"""
import json
import os
from typing import Any, Dict, List

from utils.provenance import make_provenance

_WEAK_THRESHOLD = 0.4
_PARTIAL_THRESHOLD = 0.7


def mine_loopholes_v1(
    case_id: str,
    ingredient_report_path: str,
    role_tagged_graph_path: str,
    cases_dir: str,
    tenant_id: str = "local_dev",
    input_refs: List[str] | None = None,
    _mcp_trace_id: str = "",
    _mcp_tool_version: str = "1.0.0",
    **_: Any,
) -> Dict[str, Any]:
    with open(ingredient_report_path, "r", encoding="utf-8") as f:
        ingredient_report = json.load(f)
    with open(role_tagged_graph_path, "r", encoding="utf-8") as f:
        role_tagged = json.load(f)

    missing_evidence: List[str] = []
    weak_ingredients: List[str] = []
    recommended_actions: List[str] = []

    for evaluation in ingredient_report.get("statute_evaluations", []):
        statute_id = evaluation["statute_id"]
        for ing in evaluation.get("ingredients", []):
            if ing["status"] == "not_satisfied":
                missing_evidence.append(f"{statute_id}: ingredient '{ing['ingredient_id']}' not found in case text")
                recommended_actions.append(f"Gather evidence to establish '{ing['ingredient_id']}' for {statute_id}")
            elif ing["score"] < _WEAK_THRESHOLD:
                weak_ingredients.append(f"{statute_id}: ingredient '{ing['ingredient_id']}' score={ing['score']:.2f}")
                recommended_actions.append(f"Strengthen '{ing['ingredient_id']}' evidence for {statute_id}")

    # Check for contradictory nodes (nodes with conflicting role tags)
    nodes = role_tagged.get("nodes", [])
    contradictory_nodes: List[str] = []
    role_map: Dict[str, List[str]] = {}
    for node in nodes:
        role = node.get("role_tag", "unclassified")
        role_map.setdefault(role, []).append(node["node_id"])

    # Heuristic: if allegation and finding both exist, flag potential contradiction
    if "allegation" in role_map and "finding" in role_map:
        if len(role_map["allegation"]) > 2 and len(role_map["finding"]) > 2:
            contradictory_nodes.append("Multiple allegation and finding nodes — potential factual contradiction")

    provenance = make_provenance(
        tool_name="mine_loopholes_v1",
        tool_version=_mcp_tool_version,
        input_refs=input_refs or [ingredient_report_path, role_tagged_graph_path],
        trace_id=_mcp_trace_id,
    )

    loophole_map = {
        "case_id": case_id,
        "tenant_id": tenant_id,
        "missing_evidence": missing_evidence,
        "contradictory_nodes": contradictory_nodes,
        "weak_ingredients": weak_ingredients,
        "recommended_next_actions": list(set(recommended_actions)),
        "provenance": provenance,
    }

    out_path = os.path.join(cases_dir, case_id, "loophole_map.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(loophole_map, f, indent=2, ensure_ascii=False)

    return {"result_ref": out_path, "provenance": provenance}
