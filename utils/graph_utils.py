"""
Lexplain — Graph utility functions.
MIT License | See README for MCP provenance contract.
"""
from typing import Any, Dict, List


def build_adjacency_edges(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build simple sequential adjacency edges between nodes."""
    edges: List[Dict[str, Any]] = []
    for i in range(len(nodes) - 1):
        edges.append({
            "from_node": nodes[i]["node_id"],
            "to_node": nodes[i + 1]["node_id"],
            "edge_type": "adjacent",
        })
    return edges
