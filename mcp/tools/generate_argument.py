"""
Lexplain — MCP Tool: generate_argument_v1
MIT License | See README for MCP provenance contract.

Calls Gemini (via MCP's Gemini adapter) to generate role-conditioned legal arguments.
Falls back to deterministic templating if Gemini is unavailable.

Agents MUST NOT call Gemini directly — all Gemini access is through this tool via MCPClient.
"""
import json
import os
import re
import uuid
from typing import Any, Callable, Dict, List, Optional

from utils.provenance import make_provenance

_SYSTEM_PROMPT = """You are a legal drafting assistant. Use only the structured inputs. Do not hallucinate facts or cite cases not present in "precedents". Produce three sections: (A) prosecution arguments, (B) defense arguments, (C) neutral assessment. Provide inline citation blocks referencing precedent ids and supporting_node_ids. Output must include a JSON section mapping: { "point_id": "supporting_node_ids", "precedent_ids": [...], "confidence": 0.00 }."""


def _build_fact_summary(fact_graph: Dict[str, Any], ingredient_report: Dict[str, Any]) -> Dict[str, Any]:
    """Build a condensed fact summary <= 800 tokens."""
    events = fact_graph.get("events", [])[:10]  # limit to 10 events
    actors = fact_graph.get("actors", [])[:10]
    evaluations = ingredient_report.get("statute_evaluations", [])

    satisfied = []
    violated = []
    for ev in evaluations:
        for ing in ev.get("ingredients", []):
            entry = {"statute": ev["statute_id"], "ingredient": ing["ingredient_id"], "score": ing["score"]}
            if ing["status"] == "satisfied":
                satisfied.append(entry)
            else:
                violated.append(entry)

    return {
        "events_summary": [{"event_type": e["event_type"], "actors": e["actors"]} for e in events],
        "key_actors": actors,
        "statutes_triggered": [e["statute_id"] for e in evaluations],
        "satisfied_ingredients": satisfied[:15],
        "violated_ingredients": violated[:15],
    }


def _fallback_arguments(
    case_id: str,
    fact_summary: Dict[str, Any],
    loophole_map: Dict[str, Any],
    role: str,
) -> List[Dict[str, Any]]:
    """Template-based fallback when Gemini is not available."""
    statutes = fact_summary.get("statutes_triggered", [])
    satisfied = fact_summary.get("satisfied_ingredients", [])
    violated = fact_summary.get("violated_ingredients", [])
    missing = loophole_map.get("missing_evidence", [])

    arguments: List[Dict[str, Any]] = []

    if role in ("prosecution", "both"):
        if statutes:
            arguments.append({
                "id": "P1",
                "title": f"Prima facie case under {', '.join(statutes[:3])}",
                "text": (
                    f"The facts disclose a prima facie case under {', '.join(statutes)}. "
                    f"The following ingredients are satisfied: "
                    + (", ".join(f"{s['ingredient']} ({s['statute']})" for s in satisfied[:5]) or "see evidence.")
                ),
                "supporting_node_ids": [],
                "precedent_ids": [],
                "confidence": round(sum(s["score"] for s in satisfied) / max(len(satisfied), 1), 2),
            })
        if not statutes:
            arguments.append({
                "id": "P1",
                "title": "Insufficient statutory triggers identified",
                "text": "No clear statutory violations identified from provided facts. Further investigation required.",
                "supporting_node_ids": [],
                "precedent_ids": [],
                "confidence": 0.1,
            })

    if role in ("defense", "both"):
        arguments.append({
            "id": "D1",
            "title": "Incomplete proof of essential ingredients",
            "text": (
                "The prosecution has not established all essential ingredients. "
                + (f"The following are absent or weak: {', '.join(v['ingredient'] for v in violated[:5])}." if violated else "Further examination needed.")
            ),
            "supporting_node_ids": [],
            "precedent_ids": [],
            "confidence": round(1.0 - sum(s["score"] for s in satisfied) / max(len(satisfied) + len(violated), 1), 2),
        })
        if missing:
            arguments.append({
                "id": "D2",
                "title": "Missing evidence critical to prosecution case",
                "text": f"Critical evidentiary gaps exist: {'; '.join(missing[:3])}",
                "supporting_node_ids": [],
                "precedent_ids": [],
                "confidence": 0.7,
            })

    return arguments


def generate_argument_v1(
    case_id: str,
    fact_graph_path: str,
    ingredient_report_path: str,
    loophole_map_path: str,
    comparator_report_path: str,
    cases_dir: str,
    tenant_id: str = "local_dev",
    role: str = "both",
    input_refs: List[str] | None = None,
    _mcp_trace_id: str = "",
    _mcp_tool_version: str = "1.0.0",
    _gemini_caller: Optional[Callable[..., str]] = None,
    **_: Any,
) -> Dict[str, Any]:
    with open(fact_graph_path, "r", encoding="utf-8") as f:
        fact_graph = json.load(f)
    with open(ingredient_report_path, "r", encoding="utf-8") as f:
        ingredient_report = json.load(f)
    with open(loophole_map_path, "r", encoding="utf-8") as f:
        loophole_map = json.load(f)
    with open(comparator_report_path, "r", encoding="utf-8") as f:
        comparator_report = json.load(f)

    fact_summary = _build_fact_summary(fact_graph, ingredient_report)
    model_version: Optional[str] = None
    llm_used = False
    raw_gemini_text = ""

    # Attempt Gemini call via MCP adapter
    if _gemini_caller is not None:
        user_input = json.dumps({
            "case_id": case_id,
            "tenant_id": tenant_id,
            "fact_summary": fact_summary,
            "satisfied_ingredients": fact_summary["satisfied_ingredients"],
            "violated_ingredients": fact_summary["violated_ingredients"],
            "precedents": [],
            "loophole_map": loophole_map,
            "role": role,
        }, ensure_ascii=False)

        raw_gemini_text = _gemini_caller(_SYSTEM_PROMPT, user_input, temperature=0.2, max_tokens=2048)
        if raw_gemini_text:
            llm_used = True
            model_version = "gemini-1.5-flash"

    # Parse Gemini response or use fallback
    arguments: List[Dict[str, Any]] = []
    if llm_used and raw_gemini_text:
        # Try to extract JSON from Gemini's response.
        # Limitation: greedy match to capture the outermost JSON object; nested
        # objects are handled by json.loads. If Gemini returns multiple top-level
        # objects only the first is used — see fallback branch below for resilience.
        json_match = re.search(r'\{[\s\S]*\}', raw_gemini_text)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                arguments = parsed.get("arguments", [])
            except json.JSONDecodeError:
                pass
        if not arguments:
            # Wrap full Gemini text as a single argument
            arguments = [{
                "id": "G1",
                "title": "Gemini Legal Analysis",
                "text": raw_gemini_text[:3000],
                "supporting_node_ids": [],
                "precedent_ids": [],
                "confidence": 0.75,
            }]
    else:
        arguments = _fallback_arguments(case_id, fact_summary, loophole_map, role)

    provenance = make_provenance(
        tool_name="generate_argument_v1",
        tool_version=_mcp_tool_version,
        input_refs=input_refs or [fact_graph_path, ingredient_report_path, loophole_map_path],
        trace_id=_mcp_trace_id,
        model_version=model_version,
    )

    prosecution_v = comparator_report.get("prosecution_viability", 0.5)
    defense_v = comparator_report.get("defense_viability", 0.5)
    viability_assessment = {
        "prosecution_viability": prosecution_v,
        "defense_viability": defense_v,
        "note": "Generated by Gemini" if llm_used else "Template fallback (Gemini unavailable or GEMINI_API_KEY not set)",
    }

    argument_package = {
        "case_id": case_id,
        "tenant_id": tenant_id,
        "role_requested": role,
        "llm_used": llm_used,
        "arguments": arguments,
        "viability_assessment": viability_assessment,
        "raw_gemini_text": raw_gemini_text if llm_used else None,
        "provenance": provenance,
    }

    out_path = os.path.join(cases_dir, case_id, "argument_package.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(argument_package, f, indent=2, ensure_ascii=False)

    return {"result_ref": out_path, "llm_used": llm_used, "provenance": provenance}
