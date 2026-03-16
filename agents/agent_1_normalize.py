"""
Lexplain — Agent 1: Normalize
MIT License | See README for MCP provenance contract.

Cleans raw text: unifies whitespace, normalizes punctuation, expands abbreviations.
"""
import json
import os
from typing import Any, Dict

from mcp.mcp_client import MCPClient
from utils.text_utils import clean_text, normalize_abbreviations
from utils.provenance import make_provenance


def run(case_id: str, case_dir: str, cases_dir: str) -> Dict[str, Any]:
    raw_path = os.path.join(case_dir, "raw.txt")
    with open(raw_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    normalized = normalize_abbreviations(clean_text(raw_text))

    norm_path = os.path.join(case_dir, "normalized_text.txt")
    with open(norm_path, "w", encoding="utf-8") as f:
        f.write(normalized)

    provenance = make_provenance(
        tool_name="agent_1_normalize",
        tool_version="1.0.0",
        input_refs=[raw_path],
    )
    doc_ref = {
        "case_id": case_id,
        "document_text_ref": norm_path,
        "char_count": len(normalized),
        "provenance": provenance,
    }
    ref_path = os.path.join(case_dir, "normalized_ref.json")
    with open(ref_path, "w", encoding="utf-8") as f:
        json.dump(doc_ref, f, indent=2)

    print(f"[Agent 1] Normalized text saved → {norm_path}")
    return {"normalized_text_path": norm_path, "ref_path": ref_path}
