"""
Lexplain — Offence Category Index.
MIT License | See README for MCP provenance contract.

Builds an inverted index from the IPC Knowledge Base:
    { offence_category → [section_id, ...] }

Used by Agent 5 to map extracted legal signals to candidate statute sections.
"""
import functools
from typing import Dict, List, Set

from utils.kb_loader import load_ipc_kb


# Definition / non-offence element types that indicate a section is NOT a crime section.
_DEFINITION_ELEMENT_TYPES = frozenset({
    "definition", "definition_component", "definition_by_reference",
    "definition_element", "naming", "jurisdiction", "jurisdiction_scope",
    "interpretation", "rule_of_interpretation", "exclusivity", "rule",
    "liability", "defined_entity_type", "authority_source",
    "employment_status", "provision",
})


def _is_offence_section(sec: dict) -> bool:
    """Return True if a section describes an actual offence (not a definition)."""
    # Must have ingredients
    ingredients = sec.get("ingredients", [])
    if not ingredients:
        return False
    # Must NOT be a pure definition section (all ingredients are definitions)
    if all(ing.get("element_type", "") in _DEFINITION_ELEMENT_TYPES for ing in ingredients):
        return False
    # Must have an offence_category
    cats = sec.get("offence_category", [])
    if not cats:
        return False
    # Skip repealed sections
    status = sec.get("status", {})
    if status.get("repealed", False):
        return False
    return True


@functools.lru_cache(maxsize=1)
def build_category_index() -> Dict[str, List[str]]:
    """Build { offence_category → [section_id, ...] } from the KB."""
    kb = load_ipc_kb()
    index: Dict[str, List[str]] = {}
    for sid, sec in kb.items():
        if not _is_offence_section(sec):
            continue
        for cat in sec.get("offence_category", []):
            cat_lower = cat.strip().lower()
            if cat_lower not in index:
                index[cat_lower] = []
            index[cat_lower].append(sid)
    return index


@functools.lru_cache(maxsize=1)
def get_all_categories() -> List[str]:
    """Return a sorted list of all offence categories in the KB."""
    return sorted(build_category_index().keys())


def find_sections_by_categories(categories: List[str]) -> List[str]:
    """Given a list of offence categories, return unique section IDs that match."""
    index = build_category_index()
    seen: Set[str] = set()
    result: List[str] = []
    for cat in categories:
        cat_lower = cat.strip().lower()
        for sid in index.get(cat_lower, []):
            if sid not in seen:
                seen.add(sid)
                result.append(sid)
    return result
