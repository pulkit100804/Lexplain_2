"""
Lexplain — MCP Tool: identify_statutes_v1
MIT License | See README for MCP provenance contract.

Maps case facts to candidate IPC sections using PURE DETERMINISTIC
keyword matching:
  1. Tokenize both sides (lowercase → stopword removal → stemming)
  2. Compute token overlap:  score = |overlap| / |statute_tokens|
  3. IDF-boost rare/legal terms
  4. Threshold + sort → top N candidates

Filters OUT:
  - Definition/non-offence sections (IPC 1-52, etc.)
  - Sections with empty ingredients (punishment-reference sections)

NO LLM calls.  Agent 6 handles LLM validation downstream.
Agents MUST NOT call any LLM directly — use MCPClient only.
"""
import json
import os
from typing import Any, Dict, List, Optional

from utils.kb_loader import iter_sections, load_ipc_kb
from utils.provenance import make_provenance
from utils.tokenizer import compute_idf, tokenize

# ── Configurable constants ────────────────────────────────────────────────
_TOP_N = 15            # max candidates forwarded to Agent 6
_MIN_SCORE = 0.01      # minimum score to keep a candidate

# Element types that indicate a section is a definition/non-offence section.
# If ALL ingredients in a section have element_types in this set → skip it.
_DEFINITION_ELEMENT_TYPES = frozenset({
    "definition", "definition_component", "definition_by_reference",
    "definition_element", "naming", "jurisdiction", "jurisdiction_scope",
    "interpretation", "rule_of_interpretation", "exclusivity", "rule",
    "liability", "defined_entity_type", "authority_source",
    "employment_status", "provision",
})


# ── Helpers ───────────────────────────────────────────────────────────────

def _is_definition_section(sec: Dict[str, Any]) -> bool:
    """Return True if this section is a definition/non-offence section."""
    ingredients = sec.get("ingredients", [])
    if not ingredients:
        return False  # empty-ingredient sections are handled separately
    return all(
        ing.get("element_type", "") in _DEFINITION_ELEMENT_TYPES
        for ing in ingredients
    )


def _has_legal_ingredients(sec: Dict[str, Any]) -> bool:
    """Return True if this section has real legal ingredients to evaluate."""
    ingredients = sec.get("ingredients", [])
    return len(ingredients) > 0


def _section_to_text(sec: Dict[str, Any]) -> str:
    """Build a single searchable text blob from a KB section object."""
    parts: list[str] = []
    parts.append(sec.get("heading", ""))
    parts.append(sec.get("canonical_text", ""))
    parts.append(sec.get("name", ""))
    for ing in sec.get("ingredients", []):
        parts.append(ing.get("text", ""))
        parts.append(ing.get("description", ""))
        norm = ing.get("normalized", {})
        if isinstance(norm, dict):
            parts.append(" ".join(str(v) for v in norm.values()))
    return " ".join(p for p in parts if p)


def _build_case_text(fact_graph: Dict[str, Any]) -> str:
    """Concatenate all node texts from the fact graph."""
    return " ".join(n.get("text", "") for n in fact_graph.get("nodes", []))


# ── Main tool function ────────────────────────────────────────────────────

def identify_statutes_v1(
    case_id: str,
    fact_graph_path: str,
    cases_dir: str,
    tenant_id: str = "local_dev",
    input_refs: List[str] | None = None,
    _mcp_trace_id: str = "",
    _mcp_tool_version: str = "1.0.0",
    _gemini_caller=None,       # accepted but NOT used (pure deterministic)
    **_: Any,
) -> Dict[str, Any]:
    # ── Load inputs ───────────────────────────────────────────────────────
    with open(fact_graph_path, "r", encoding="utf-8") as f:
        fact_graph = json.load(f)

    case_text = _build_case_text(fact_graph)
    kb = load_ipc_kb()

    # ── Tokenize case text ────────────────────────────────────────────────
    case_tokens = set(tokenize(case_text))

    # ── Strict LLM Keyword Extraction ─────────────────────────────────────
    llm_keywords = []
    if _gemini_caller is not None and case_text.strip():
        llm_prompt = """\
You are a legal keyword extractor. Your task is to extract critical legal concepts and factual keywords from the provided case facts.

STRICT RULES:
1. DO NOT invent facts or hallucinate details not present in the text.
2. Extract only single words or short compound phrases (e.g., "cheating", "false promise", "murder", "intent").
3. Return ONLY strict JSON in the exact format shown below, with no markdown formatting or extra text.

{
  "keywords": ["keyword1", "keyword2", "keyword3"]
}"""
        raw_llm = _gemini_caller(llm_prompt, f"Case Facts:\n{case_text[:5000]}", temperature=0.1, max_tokens=1024)
        if raw_llm:
            try:
                import re
                json_match = re.search(r'\{[\s\S]*\}', raw_llm)
                if json_match:
                    parsed = json.loads(json_match.group())
                    words = parsed.get("keywords", [])
                    if isinstance(words, list):
                        for w in words:
                            llm_keywords.extend(tokenize(str(w)))
            except Exception:
                pass

    # Merge LLM tokens with base case tokens
    case_tokens.update(llm_keywords)

    if not case_tokens:
        candidates: List[Dict[str, Any]] = []
    else:
        # ── Filter sections ──────────────────────────────────────────────
        sections = []
        for sec in iter_sections():

            # Rule 7: Remove definition sections
            if _is_definition_section(sec):
                continue
            # Remove sections with no legal ingredients
            if not _has_legal_ingredients(sec):
                continue
            sections.append(sec)

        # ── Tokenize all statute texts & compute IDF ─────────────────────
        all_section_tokens: List[List[str]] = []
        for sec in sections:
            all_section_tokens.append(tokenize(_section_to_text(sec)))

        idf = compute_idf(all_section_tokens)

        # ── Score each statute via token overlap + IDF boost ─────────────
        scored: List[tuple] = []
        for sec, sec_tokens_list in zip(sections, all_section_tokens):
            sec_tokens = set(sec_tokens_list)
            if not sec_tokens:
                continue

            # Core overlap
            overlap = case_tokens & sec_tokens
            if not overlap:
                continue

            # Base score: fraction of statute tokens matched
            overlap_score = len(overlap) / len(sec_tokens)

            # IDF boost: sum IDF of matched terms (normalized)
            idf_sum = sum(idf.get(w, 1.0) for w in overlap)
            max_possible_idf = sum(idf.get(w, 1.0) for w in sec_tokens) or 1.0
            idf_score = idf_sum / max_possible_idf

            # Combined: 50% overlap + 50% IDF-weighted overlap
            score = 0.5 * overlap_score + 0.5 * idf_score

            if score >= _MIN_SCORE:
                scored.append((sec, score, sorted(overlap)))

        # Sort by score descending, take top N
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:_TOP_N]

        # Build output candidates
        candidates = []
        for rank, (sec, score, matched) in enumerate(scored, 1):
            candidates.append({
                "statute_id": sec.get("section_id", ""),
                "name": sec.get("heading", sec.get("name", "")),
                "matched_terms": matched,
                "score": round(score, 4),
                "final": {
                    "score": round(score, 4),
                    "source": "deterministic",
                    "rank": rank,
                },
            })

    # ── Provenance ────────────────────────────────────────────────────────
    provenance = make_provenance(
        tool_name="identify_statutes_v1",
        tool_version=_mcp_tool_version,
        input_refs=input_refs or [fact_graph_path],
        trace_id=_mcp_trace_id,
        model_version=None,   # pure deterministic — no LLM
    )

    statute_candidates = {
        "case_id": case_id,
        "tenant_id": tenant_id,
        "candidates": candidates,
        "candidate_count": len(candidates),
        "provenance": provenance,
    }

    out_path = os.path.join(cases_dir, case_id, "statute_candidates.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(statute_candidates, f, indent=2, ensure_ascii=False)

    return {"result_ref": out_path, "candidates": candidates, "provenance": provenance}
