"""
Lexplain — MCP Client (thin wrapper used by agents).
MIT License | See README for MCP provenance contract.

Agents use MCPClient.call() to invoke tools through the MCP server.
Agents MUST NOT call Gemini or any LLM directly.
"""
from typing import Any, Dict

from mcp.mcp_server import MCPServer

# Singleton server instance shared across all agents in the same process
_SERVER_INSTANCE: MCPServer | None = None


def get_server() -> MCPServer:
    global _SERVER_INSTANCE
    if _SERVER_INSTANCE is None:
        _SERVER_INSTANCE = MCPServer()
    return _SERVER_INSTANCE


class MCPClient:
    """Thin client used by agents to call MCP-registered tools."""

    def __init__(self, tenant_id: str = "local_dev") -> None:
        self.tenant_id = tenant_id
        self._server = get_server()

    def call(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate tool call to MCPServer and return the response."""
        params.setdefault("tenant_id", self.tenant_id)
        return self._server.call_tool(tool_name, params)
