"""
Lexplain — MCP Tool Registry.
MIT License | See README for MCP provenance contract.

Maintains a mapping of tool_name -> (callable, version).
"""
from typing import Any, Callable, Dict, Tuple


class ToolRegistry:
    """Registry mapping tool names to callables and version strings."""

    def __init__(self) -> None:
        self._registry: Dict[str, Tuple[Callable[..., Any], str]] = {}

    def register(self, name: str, fn: Callable[..., Any], version: str = "1.0.0") -> None:
        """Register a tool by name."""
        self._registry[name] = (fn, version)

    def get(self, name: str) -> Tuple[Callable[..., Any], str]:
        """Return (callable, version) for the named tool."""
        if name not in self._registry:
            raise KeyError(f"Tool '{name}' not registered in MCPToolRegistry")
        return self._registry[name]

    def list_tools(self) -> Dict[str, str]:
        """Return {tool_name: version} for all registered tools."""
        return {name: ver for name, (_, ver) in self._registry.items()}
