"""
Lexplain — MCP Tool: build_fact_graph_v1
MIT License | See README for MCP provenance contract.

Extracts actors, actions, objects, dates from role-tagged graph nodes
using simple regex + heuristics. No external NLP dependency.
"""
import json
import os
import re
import uuid
from typing import Any, Dict, List

from utils.provenance import make_provenance

_DATE_RE = re.compile(
    r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},?\s+\d{4}|\d{1,2}\s+"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r",?\s+\d{4})\b",
    re.IGNORECASE,
)

_ACTION_WORDS = [
    "cheated", "threatened", "assaulted", "stole", "murdered", "abducted",
    "defrauded", "intimidated", "attacked", "robbed", "harassed", "filed",
    "alleged", "submitted", "claimed", "stated", "ordered",
]

_ACTOR_INDICATORS = ["accused", "complainant", "victim", "petitioner", "respondent", "defendant", "plaintiff", "witness"]


def _extract_dates(text: str) -> List[str]:
    return _DATE_RE.findall(text)


def _extract_actors(text: str) -> List[str]:
    text_lower = text.lower()
    found = []
    for ind in _ACTOR_INDICATORS:
        if ind in text_lower:
            found.append(ind)
    return found


def _extract_actions(text: str) -> List[str]:
    text_lower = text.lower()
    return [w for w in _ACTION_WORDS if w in text_lower]


def build_fact_graph_v1(
    case_id: str,
    role_tagged_graph_path: str,
    cases_dir: str,
    tenant_id: str = "local_dev",
    input_refs: List[str] | None = None,
    _mcp_trace_id: str = "",
    _mcp_tool_version: str = "1.0.0",
    **_: Any,
) -> Dict[str, Any]:
    with open(role_tagged_graph_path, "r", encoding="utf-8") as f:
        tagged = json.load(f)

    events: List[Dict[str, Any]] = []
    all_actors: List[str] = []

    for node in tagged["nodes"]:
        text = node["text"]
        actors = _extract_actors(text)
        actions = _extract_actions(text)
        dates = _extract_dates(text)
        all_actors.extend(actors)
        if actions or actors:
            events.append({
                "event_id": f"evt_{uuid.uuid4().hex[:8]}",
                "node_ids": [node["node_id"]],
                "event_type": actions[0] if actions else "mentioned",
                "actors": actors,
                "objects": [],
                "dates": list(set(dates)),
            })

    provenance = make_provenance(
        tool_name="build_fact_graph_v1",
        tool_version=_mcp_tool_version,
        input_refs=input_refs or [role_tagged_graph_path],
        trace_id=_mcp_trace_id,
    )

    fact_graph = {
        "case_id": case_id,
        "tenant_id": tenant_id,
        "nodes": tagged["nodes"],
        "events": events,
        "actors": list(set(all_actors)),
        "provenance": provenance,
    }

    out_path = os.path.join(cases_dir, case_id, "fact_event_graph.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fact_graph, f, indent=2, ensure_ascii=False)

    return {"result_ref": out_path, "provenance": provenance}
