"""
Lexplain — Agent 10: Argument Generator
MIT License | See README for MCP provenance contract.

Calls MCP tool generate_argument_v1 which invokes Gemini (if available)
or falls back to deterministic templating.
Agents MUST NOT call Gemini directly.
"""
import os
from typing import Any, Dict

from mcp.mcp_client import MCPClient


def run(case_id: str, case_dir: str, cases_dir: str, role: str = "both") -> Dict[str, Any]:
    fact_graph_path = os.path.join(case_dir, "fact_event_graph.json")
    ingredient_report_path = os.path.join(case_dir, "ingredient_report.json")
    loophole_map_path = os.path.join(case_dir, "loophole_map.json")
    comparator_report_path = os.path.join(case_dir, "comparator_report.json")
    client = MCPClient()
    response = client.call("generate_argument_v1", {
        "case_id": case_id,
        "fact_graph_path": fact_graph_path,
        "ingredient_report_path": ingredient_report_path,
        "loophole_map_path": loophole_map_path,
        "comparator_report_path": comparator_report_path,
        "cases_dir": cases_dir,
        "role": role,
        "input_refs": [fact_graph_path, ingredient_report_path, loophole_map_path, comparator_report_path],
    })
    result = response["result"]
    llm_used = result.get("llm_used", False)
    print(f"[Agent 10] Argument package saved → {result['result_ref']} (LLM={'Gemini' if llm_used else 'fallback'})")
    return {"argument_package_path": result["result_ref"], "llm_used": llm_used}
