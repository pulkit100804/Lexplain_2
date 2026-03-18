"""
Lexplain — Centralized IPC Knowledge-Base loader.
MIT License | See README for MCP provenance contract.

Single source of truth for loading the dict-shaped ingredients_ipc.json.
Both identify_statutes_v1 and evaluate_ingredients_v1 MUST use this module.

KB path resolution order:
  1. LEXPLAIN_KB_PATH environment variable
  2. knowledge_base/ingredients_ipc.json relative to repo root

The KB JSON root may be:
  - A DICT keyed by section number ("295A", "300", …) with section objects as values  [NEW]
  - A LIST of section objects (legacy format)                                          [OLD]

In both cases this module normalises to dict[section_id, section_obj].
"""
import functools
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_KB_REL = "knowledge_base/ingredients_ipc.json"


def _resolve_kb_path() -> Path:
    """Return the absolute path to the KB file."""
    env = os.getenv("LEXPLAIN_KB_PATH", "")
    if env:
        return Path(env).resolve()
    return _REPO_ROOT / _DEFAULT_KB_REL


@functools.lru_cache(maxsize=1)
def load_ipc_kb() -> Dict[str, Dict[str, Any]]:
    """Load and normalise the IPC KB.  Cached across an entire process run."""
    kb_path = _resolve_kb_path()
    if not kb_path.exists():
        return {}

    with open(kb_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    # --- normalise to {section_id: section_obj} -------------------------
    normalised: Dict[str, Dict[str, Any]] = {}

    if isinstance(raw, dict):
        # New dict-shaped KB: keys are section numbers like "295A"
        # Or it might be nested under a law code like {"IPC": {"295A": {...}}}
        for key, value in raw.items():
            if not isinstance(value, dict):
                continue
            # Check if this value is itself a section object
            if "ingredients" in value or "section_id" in value or "heading" in value:
                sid = value.get("section_id", key)
                normalised[str(sid)] = value
            else:
                # It is likely a nested container (e.g. "IPC": {...})
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, dict):
                        sid = subvalue.get("section_id", subkey)
                        normalised[str(sid)] = subvalue

    elif isinstance(raw, list):
        # Legacy list-shaped KB
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            sid = entry.get("section_id", "")
            if sid:
                normalised[str(sid)] = entry

    return normalised


def get_section(section_id: str) -> Optional[Dict[str, Any]]:
    """Return the section object for *section_id*, or None."""
    return load_ipc_kb().get(section_id)


def iter_sections() -> Iterable[Dict[str, Any]]:
    """Yield every section object in the KB."""
    return load_ipc_kb().values()


def section_ids() -> list[str]:
    """Return a sorted list of all section IDs in the KB."""
    return sorted(load_ipc_kb().keys())
