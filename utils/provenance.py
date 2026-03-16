"""
Lexplain — Provenance utilities.
MIT License | See README for MCP provenance contract.
"""
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def make_provenance(
    tool_name: str,
    tool_version: str,
    input_refs: Optional[List[str]] = None,
    model_version: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a provenance dict conforming to the MCP provenance contract."""
    return {
        "trace_id": trace_id or str(uuid.uuid4()),
        "tool_name": tool_name,
        "tool_version": tool_version,
        "model_version": model_version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_refs": input_refs or [],
    }
