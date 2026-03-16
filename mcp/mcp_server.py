"""
Lexplain — MCP Server (local module).
MIT License | See README for MCP provenance contract.

Provides a lightweight local Model Context Protocol server that:
- Maintains a ToolRegistry
- Executes tool calls with provenance tracking
- Hosts a Gemini adapter (calls google.generativeai)
- Enforces that agents NEVER call Gemini directly

Environment variables:
  GEMINI_API_KEY  – required for live Gemini calls; if absent, fallback is used
"""
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from mcp.tool_registry import ToolRegistry
from utils.provenance import make_provenance

# ── Gemini adapter ───────────────────────────────────────────────────────────
_GEMINI_MODEL = "gemini-1.5-flash"


def _call_gemini(system_prompt: str, user_json: str, temperature: float = 0.2, max_tokens: int = 2048) -> str:
    """
    Call Gemini via google.generativeai.
    Returns raw text. Falls back to empty string on any error.
    Agents must NEVER call this directly — use MCPServer.call_tool('generate_argument_v1', ...).
    """
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return ""
    try:
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            _GEMINI_MODEL,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )
        response = model.generate_content(f"{system_prompt}\n\n{user_json}")
        return response.text or ""
    except (ImportError, ValueError, RuntimeError, OSError) as exc:
        print(f"[MCP] Gemini call failed: {exc}")
        return ""


# ── MCP Server ────────────────────────────────────────────────────────────────

class MCPServer:
    """
    Local MCP server.  Agents interact via MCPClient which delegates here.
    """

    def __init__(self) -> None:
        self.registry = ToolRegistry()
        self._register_tools()

    # ── Tool registration ────────────────────────────────────────────────────

    def _register_tools(self) -> None:
        from mcp.tools.extract_sentences import extract_sentences_v1
        from mcp.tools.tag_roles import tag_legal_roles_v1
        from mcp.tools.build_fact_graph import build_fact_graph_v1
        from mcp.tools.identify_statutes import identify_statutes_v1
        from mcp.tools.evaluate_ingredients import evaluate_ingredients_v1
        from mcp.tools.search_precedents import search_precedents_v1
        from mcp.tools.compare_precedents import compare_precedents_v1
        from mcp.tools.mine_loopholes import mine_loopholes_v1
        from mcp.tools.generate_argument import generate_argument_v1

        self.registry.register("extract_sentences_v1", extract_sentences_v1, "1.0.0")
        self.registry.register("tag_legal_roles_v1", tag_legal_roles_v1, "1.0.0")
        self.registry.register("build_fact_graph_v1", build_fact_graph_v1, "1.0.0")
        self.registry.register("identify_statutes_v1", identify_statutes_v1, "1.0.0")
        self.registry.register("evaluate_ingredients_v1", evaluate_ingredients_v1, "1.0.0")
        self.registry.register("search_precedents_v1", search_precedents_v1, "1.0.0")
        self.registry.register("compare_precedents_v1", compare_precedents_v1, "1.0.0")
        self.registry.register("mine_loopholes_v1", mine_loopholes_v1, "1.0.0")
        self.registry.register("generate_argument_v1", generate_argument_v1, "1.0.0")

    # ── Public call_tool ─────────────────────────────────────────────────────

    def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a registered tool with provenance tracking.
        Returns { result: Any, provenance: {...} }.
        """
        fn, version = self.registry.get(tool_name)
        trace_id = str(uuid.uuid4())
        # Inject MCP-level context into params
        params["_mcp_trace_id"] = trace_id
        params["_mcp_tool_version"] = version
        params["_gemini_caller"] = _call_gemini
        result = fn(**params)
        return {
            "result": result,
            "provenance": make_provenance(
                tool_name=tool_name,
                tool_version=version,
                input_refs=params.get("input_refs", []),
                trace_id=trace_id,
            ),
        }
