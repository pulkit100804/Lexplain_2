"""
Lexplain — MCP Tool: evaluate_ingredients_v1
MIT License | See README for MCP provenance contract.

Checks each statute candidate's legal ingredients against the fact_event_graph.
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from utils.provenance import make_provenance

_KB_PATH = Path(__file__).parents[2] / "knowledge_base" / "ingredients_ipc.json"


def _load_kb() -> List[Dict[str, Any]]:
    if _KB_PATH.exists():
        with open(_KB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _score_ingredient(ingredient: Dict[str, Any], all_text: str) -> Dict[str, Any]:
    """Check if a single ingredient is satisfied by the case text."""
    patterns = ingredient.get("match_patterns", [])
    matching = [p for p in patterns if p.lower() in all_text.lower()]
    if not patterns:
        score = 0.0
        status = "not_satisfied"
    elif len(matching) == len(patterns):
        score = 1.0
        status = "satisfied"
    elif matching:
        score = len(matching) / len(patterns)
        status = "partial"
    else:
        score = 0.0
        status = "not_satisfied"
    return {
        "ingredient_id": ingredient.get("ingredient_id", ingredient.get("name", "unknown")),
        "status": status,
        "supporting_node_ids": [],
        "score": round(score, 3),
        "matched_patterns": matching,
    }


def evaluate_ingredients_v1(
    case_id: str,
    statute_candidates_path: str,
    fact_graph_path: str,
    cases_dir: str,
    tenant_id: str = "local_dev",
    input_refs: List[str] | None = None,
    _mcp_trace_id: str = "",
    _mcp_tool_version: str = "1.0.0",
    **_: Any,
) -> Dict[str, Any]:
    with open(statute_candidates_path, "r", encoding="utf-8") as f:
        statute_candidates = json.load(f)
    with open(fact_graph_path, "r", encoding="utf-8") as f:
        fact_graph = json.load(f)

    kb = _load_kb()
    kb_by_section = {entry["section_id"]: entry for entry in kb}

    all_text = " ".join(n["text"] for n in fact_graph.get("nodes", []))
    results: List[Dict[str, Any]] = []

    for candidate in statute_candidates.get("candidates", []):
        statute_id = candidate["statute_id"]
        kb_entry = kb_by_section.get(statute_id, {})
        ingredients = kb_entry.get("ingredients", [])
        if not ingredients:
            # Use candidate match keywords as a simple proxy ingredient
            ingredients = [{"ingredient_id": f"{statute_id}_kw", "name": "keyword_match", "match_patterns": candidate.get("matched_keywords", [])}]

        ingredient_results = [_score_ingredient(ing, all_text) for ing in ingredients]
        overall_score = (sum(r["score"] for r in ingredient_results) / len(ingredient_results)) if ingredient_results else 0.0

        results.append({
            "statute_id": statute_id,
            "name": candidate["name"],
            "ingredients": ingredient_results,
            "overall_score": round(overall_score, 3),
        })

    provenance = make_provenance(
        tool_name="evaluate_ingredients_v1",
        tool_version=_mcp_tool_version,
        input_refs=input_refs or [statute_candidates_path, fact_graph_path],
        trace_id=_mcp_trace_id,
    )

    ingredient_report = {
        "case_id": case_id,
        "tenant_id": tenant_id,
        "statute_evaluations": results,
        "provenance": provenance,
    }

    out_path = os.path.join(cases_dir, case_id, "ingredient_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ingredient_report, f, indent=2, ensure_ascii=False)

    return {"result_ref": out_path, "provenance": provenance}
