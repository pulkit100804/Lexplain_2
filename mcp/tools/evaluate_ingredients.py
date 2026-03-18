"""
Lexplain — MCP Tool: evaluate_ingredients_v1
MIT License | See README for MCP provenance contract.

Legal charge evaluation engine.  NOT a text matcher.

Architecture:
  1. Load candidate statutes + their KB ingredients (legal conditions)
  2. Single constrained LLM call to validate ingredients against case facts
  3. Core ingredient gating: if ANY core ingredient is not_satisfied → score = 0
  4. Partial penalty: partial status → score * 0.25
  5. Not-applicable filter: if ALL ingredients not_satisfied → score 0
  6. Fallback: deterministic token overlap if LLM fails

STRICT RULES:
  - Gemini MUST NOT invent facts, sections, or ingredients
  - Gemini validates ONLY against existing KB ingredients and provided case facts
  - Prefer underconfidence over hallucination
  - No keyword_proxy — if section has no ingredients, it's not_applicable

Agents MUST NOT call any LLM directly — use MCPClient only.
"""
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

from utils.kb_loader import get_section
from utils.provenance import make_provenance
from utils.tokenizer import tokenize

# ── Configurable constants ────────────────────────────────────────────────
_MAX_STATUTES_PER_CALL = 8  # if > this many candidates, split into parallel calls

# Status thresholds
_THRESH_STRONG = 0.70
_THRESH_PLAUSIBLE = 0.40
_THRESH_WEAK = 0.10

# Core element types — if ANY core ingredient is not_satisfied → overall score = 0
CORE_ELEMENT_TYPES = frozenset({
    "actus_reus", "act_element", "act", "act_or_omission",
    "mens_rea", "mental_element", "mental_state",
})


def _charge_status(score: float) -> str:
    """Map score to legal charge status."""
    if score >= _THRESH_STRONG:
        return "strong"
    if score >= _THRESH_PLAUSIBLE:
        return "plausible"
    if score >= _THRESH_WEAK:
        return "weak"
    return "not_applicable"


def _ingredient_status(score: float) -> str:
    """Map ingredient-level score to status."""
    if score >= 0.65:
        return "satisfied"
    if score >= 0.30:
        return "partial"
    return "not_satisfied"


def _is_core_ingredient(element_type: str) -> bool:
    """Check if an ingredient is a core legal element."""
    return element_type in CORE_ELEMENT_TYPES


# ── Deterministic fallback: token overlap per ingredient ──────────────────

def _det_score_ingredient(ingredient_text: str, case_text: str) -> float:
    """Simple token overlap score for one ingredient vs case text."""
    ing_tokens = set(tokenize(ingredient_text))
    case_tokens = set(tokenize(case_text))
    if not ing_tokens:
        return 0.0
    overlap = ing_tokens & case_tokens
    return len(overlap) / len(ing_tokens)


# ── LLM Validation Prompt ────────────────────────────────────────────────

_LLM_VALIDATION_PROMPT = """\
You are a legal validation engine for the Indian Penal Code.

STRICT RULES — FOLLOW EXACTLY:
1. You MUST NOT invent any facts not present in the case
2. You MUST NOT add new sections or ingredients not in the provided list
3. You MUST NOT assume facts that are not explicitly stated
4. If evidence is missing or unclear → mark as "not_satisfied" or "partial"
5. Prefer underconfidence over hallucination
6. Base your evaluation ONLY on the case facts provided and the KB ingredients listed

You are given:
1. Case facts (from user input — treat as the ONLY source of truth)
2. Candidate IPC sections with their legal ingredients from the knowledge base

For each section, evaluate EACH ingredient against the case facts:
- "satisfied": case facts clearly establish this ingredient
- "partial": some evidence exists but not conclusive
- "not_satisfied": no evidence in the case facts supports this ingredient

Return ONLY strict JSON (no markdown, no extra text):
{
  "evaluations": [
    {
      "statute_id": "420",
      "ingredients": [
        {
          "id": "420.1",
          "status": "satisfied" or "partial" or "not_satisfied",
          "confidence": 0.0 to 1.0,
          "reasoning": "one sentence based ONLY on case facts"
        }
      ],
      "overall_reasoning": "one sentence legal assessment"
    }
  ]
}
"""


def _build_llm_payload(
    candidates: List[Dict[str, Any]],
    case_text: str,
    sections_data: Dict[str, Dict[str, Any]],
) -> str:
    """Build the user payload for the LLM validation call."""
    statutes_info = []
    for cand in candidates:
        sid = cand["statute_id"]
        sec = sections_data.get(sid, {})
        ingredients_info = []
        for ing in sec.get("ingredients", []):
            ingredients_info.append({
                "id": ing.get("id", ing.get("ingredient_id", "")),
                "text": ing.get("text", ing.get("description", "")),
                "element_type": ing.get("element_type", ""),
            })

        statutes_info.append({
            "statute_id": sid,
            "heading": sec.get("heading", cand.get("name", "")),
            "canonical_text": (sec.get("canonical_text", ""))[:400],
            "ingredients": ingredients_info,
        })

    return json.dumps({
        "case_facts": case_text[:5000],
        "candidate_sections": statutes_info,
    }, ensure_ascii=False)


def _call_llm_batch(
    candidates: List[Dict[str, Any]],
    case_text: str,
    sections_data: Dict[str, Dict[str, Any]],
    gemini_caller: Callable[..., str],
) -> Dict[str, Dict[str, Any]]:
    """Make ONE LLM call for a batch of candidates.
    Returns {statute_id: {ingredients: [{id, status, confidence, reasoning}], overall_reasoning}}."""
    user_payload = _build_llm_payload(candidates, case_text, sections_data)
    raw = gemini_caller(_LLM_VALIDATION_PROMPT, user_payload, temperature=0.1, max_tokens=4096)

    result: Dict[str, Dict[str, Any]] = {}
    if not raw:
        return result

    try:
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            parsed = json.loads(json_match.group())
            for entry in parsed.get("evaluations", []):
                sid = str(entry.get("statute_id", ""))
                ing_results = {}
                for ing_eval in entry.get("ingredients", []):
                    ing_id = str(ing_eval.get("id", ""))
                    status = str(ing_eval.get("status", "not_satisfied"))
                    # Clamp to valid values
                    if status not in ("satisfied", "partial", "not_satisfied"):
                        status = "not_satisfied"
                    confidence = float(ing_eval.get("confidence", 0.0))
                    confidence = max(0.0, min(1.0, confidence))
                    ing_results[ing_id] = {
                        "status": status,
                        "confidence": round(confidence, 4),
                        "reasoning": str(ing_eval.get("reasoning", "")),
                    }
                result[sid] = {
                    "ingredients": ing_results,
                    "overall_reasoning": str(entry.get("overall_reasoning", "")),
                }
    except (json.JSONDecodeError, ValueError, KeyError, TypeError):
        pass

    return result


# ── Scoring Engine ────────────────────────────────────────────────────────

def _compute_statute_score(
    ingredient_results: List[Dict[str, Any]],
) -> tuple:
    """Compute overall score with core gating, partial penalty, and weighted scoring.

    Returns (overall_score, charge_status, reason_suffix).
    """
    if not ingredient_results:
        return 0.0, "not_applicable", "no ingredients to evaluate"

    # Check if ALL ingredients are not_satisfied
    all_not_satisfied = all(
        ir["final"]["status"] == "not_satisfied"
        for ir in ingredient_results
    )
    if all_not_satisfied:
        return 0.0, "not_applicable", "all ingredients unsatisfied by case facts"

    # Check core ingredient gating
    core_ingredients = [
        ir for ir in ingredient_results
        if _is_core_ingredient(ir.get("element_type", ""))
    ]

    if core_ingredients:
        core_failed = any(
            ir["final"]["status"] == "not_satisfied"
            for ir in core_ingredients
        )
        if core_failed:
            failed_names = [
                ir["ingredient_id"] for ir in core_ingredients
                if ir["final"]["status"] == "not_satisfied"
            ]
            return 0.0, "not_applicable", f"core ingredient(s) not satisfied: {', '.join(failed_names)}"

    # Weighted scoring: core ingredients weight 2x, non-core weight 1x
    total_weight = 0.0
    weighted_sum = 0.0

    for ir in ingredient_results:
        is_core = _is_core_ingredient(ir.get("element_type", ""))
        weight = 2.0 if is_core else 1.0
        score = ir["final"]["score"]

        # Rule 4: Partial penalty — multiply by 0.25
        if ir["final"]["status"] == "partial":
            score = score * 0.25

        weighted_sum += weight * score
        total_weight += weight

    overall = weighted_sum / total_weight if total_weight > 0 else 0.0
    overall = round(overall, 4)
    status = _charge_status(overall)

    return overall, status, ""


# ── Main tool function ────────────────────────────────────────────────────

def evaluate_ingredients_v1(
    case_id: str,
    statute_candidates_path: str,
    fact_graph_path: str,
    cases_dir: str,
    tenant_id: str = "local_dev",
    input_refs: List[str] | None = None,
    _mcp_trace_id: str = "",
    _mcp_tool_version: str = "1.0.0",
    _gemini_caller: Optional[Callable[..., str]] = None,
    **_: Any,
) -> Dict[str, Any]:
    # ── Load inputs ───────────────────────────────────────────────────────
    with open(statute_candidates_path, "r", encoding="utf-8") as f:
        statute_candidates = json.load(f)
    with open(fact_graph_path, "r", encoding="utf-8") as f:
        fact_graph = json.load(f)

    all_text = " ".join(n.get("text", "") for n in fact_graph.get("nodes", []))

    # ── Load KB sections for all candidates ───────────────────────────────
    candidates = statute_candidates.get("candidates", [])
    sections_data: Dict[str, Dict[str, Any]] = {}
    for cand in candidates:
        sid = cand["statute_id"]
        sec = get_section(sid)
        if sec is not None:
            sections_data[sid] = sec
        else:
            sections_data[sid] = {
                "section_id": sid,
                "heading": cand.get("name", ""),
                "ingredients": [],
            }

    # ── LLM Validation (single call or parallel batches) ──────────────────
    model_version: Optional[str] = None
    llm_results: Dict[str, Dict[str, Any]] = {}

    if _gemini_caller is not None and candidates:
        if len(candidates) <= _MAX_STATUTES_PER_CALL:
            llm_results = _call_llm_batch(
                candidates, all_text, sections_data, _gemini_caller
            )
        else:
            # Split into batches and run in PARALLEL threads
            batches = []
            for i in range(0, len(candidates), _MAX_STATUTES_PER_CALL):
                batches.append(candidates[i : i + _MAX_STATUTES_PER_CALL])

            with ThreadPoolExecutor(max_workers=min(len(batches), 4)) as pool:
                futures = {
                    pool.submit(
                        _call_llm_batch, batch, all_text, sections_data, _gemini_caller
                    ): batch
                    for batch in batches
                }
                for future in as_completed(futures):
                    try:
                        batch_result = future.result(timeout=60)
                        llm_results.update(batch_result)
                    except Exception:
                        pass  # fallback to deterministic for this batch

        if llm_results:
            model_version = "gemini-2.5-flash"

    # ── Build output per statute ──────────────────────────────────────────
    results: List[Dict[str, Any]] = []

    for cand in candidates:
        sid = cand["statute_id"]
        statute_name = cand.get("name", "")
        sec = sections_data.get(sid, {})
        ingredients = sec.get("ingredients", [])

        # NO keyword_proxy — if no ingredients, mark as not_applicable
        if not ingredients:
            results.append({
                "statute_id": sid,
                "name": statute_name,
                "score": 0.0,
                "status": "not_applicable",
                "reason": "section has no legal ingredients in knowledge base",
                "ingredients": [],
                "overall_score": 0.0,
            })
            continue

        # Get LLM evaluation for this statute (if available)
        llm_eval = llm_results.get(sid)
        llm_ings = llm_eval.get("ingredients", {}) if llm_eval else {}

        # Build per-ingredient results
        ingredient_results: List[Dict[str, Any]] = []

        for ing in ingredients:
            ing_id = ing.get("id", ing.get("ingredient_id", "unknown"))
            ing_text = ing.get("text", ing.get("description", ""))
            element_type = ing.get("element_type", "")

            # Check if LLM provided evaluation for this ingredient
            llm_ing = llm_ings.get(ing_id)

            if llm_ing and llm_ing.get("status") in ("satisfied", "partial", "not_satisfied"):
                # Use LLM evaluation
                final_status = llm_ing["status"]
                final_score = llm_ing.get("confidence", 0.0)
                final_source = "llm"
                reasoning = llm_ing.get("reasoning", "")
            else:
                # Deterministic fallback
                det_score = _det_score_ingredient(ing_text, all_text)
                final_score = det_score
                final_status = _ingredient_status(final_score)
                final_source = "deterministic"
                reasoning = ""

            final_score = round(final_score, 4)

            ingredient_results.append({
                "ingredient_id": ing_id,
                "ingredient_text": ing_text,
                "element_type": element_type,
                "is_core": _is_core_ingredient(element_type),
                "final": {
                    "status": final_status,
                    "score": final_score,
                    "source": final_source,
                },
                "reasoning": reasoning,
            })

        # Compute overall score with gating rules
        overall_score, charge_status, gating_reason = _compute_statute_score(ingredient_results)

        # Build reason
        if gating_reason:
            reason = gating_reason
        elif llm_eval and llm_eval.get("overall_reasoning"):
            reason = llm_eval["overall_reasoning"]
        else:
            # Build from ingredient statuses
            satisfied = [ir["ingredient_id"] for ir in ingredient_results if ir["final"]["status"] == "satisfied"]
            missing = [ir["ingredient_id"] for ir in ingredient_results if ir["final"]["status"] == "not_satisfied"]
            parts = []
            if satisfied:
                parts.append(f"satisfied: {', '.join(satisfied)}")
            if missing:
                parts.append(f"missing: {', '.join(missing)}")
            reason = "; ".join(parts) if parts else "evaluated deterministically"

        results.append({
            "statute_id": sid,
            "name": statute_name,
            "score": overall_score,
            "status": charge_status,
            "reason": reason,
            "ingredients": ingredient_results,
            "overall_score": overall_score,
        })

    # Sort by score descending, not_applicable at bottom
    results.sort(key=lambda r: (r["status"] != "not_applicable", r["score"]), reverse=True)

    # ── Provenance ────────────────────────────────────────────────────────
    provenance = make_provenance(
        tool_name="evaluate_ingredients_v1",
        tool_version=_mcp_tool_version,
        input_refs=input_refs or [statute_candidates_path, fact_graph_path],
        trace_id=_mcp_trace_id,
        model_version=model_version,
    )

    ingredient_report = {
        "case_id": case_id,
        "tenant_id": tenant_id,
        "statute_evaluations": results,
        "provenance": provenance,
    }

    out_path = os.path.join(cases_dir, case_id, "ingredient_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ingredient_report, f, indent=2, ensure_ascii=False)

    return {"result_ref": out_path, "provenance": provenance}
