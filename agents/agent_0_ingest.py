"""
Lexplain — Agent 0: Ingest
MIT License | See README for MCP provenance contract.

Accepts raw text input, creates the case folder, saves raw.txt and metadata.json.
"""
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

from utils.provenance import make_provenance


def run(case_text: str, cases_dir: str, tenant_id: str = "local_dev") -> Dict[str, Any]:
    """Create case folder and persist raw.txt + metadata.json."""
    case_id = f"case_{uuid.uuid4().hex[:12]}"
    case_dir = os.path.join(cases_dir, case_id)
    os.makedirs(case_dir, exist_ok=True)

    # Save raw text
    raw_path = os.path.join(case_dir, "raw.txt")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(case_text)

    provenance = make_provenance(
        tool_name="agent_0_ingest",
        tool_version="1.0.0",
        input_refs=["stdin"],
    )

    metadata = {
        "case_id": case_id,
        "tenant_id": tenant_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "char_count": len(case_text),
        "word_count": len(case_text.split()),
        "provenance": provenance,
    }
    meta_path = os.path.join(case_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[Agent 0] Case ingested → case_id={case_id}")
    return {"case_id": case_id, "case_dir": case_dir, "raw_path": raw_path}
