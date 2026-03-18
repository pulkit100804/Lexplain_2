"""
Lexplain — MCP Tool: evaluate_statute_ingredients_v1  (v3 – Hybrid Validator)
MIT License | See README for MCP provenance contract.

7-Step Hybrid Statute Validator
===============================
1. FACT BUILDER   – events + signals → deterministic facts   (NO LLM)
2. DOMAIN EXTRACTOR – fact types → case domains              (NO LLM)
3. STATUTE FILTER – hard-reject statutes outside domains     (NO LLM)
4. INGREDIENT MATCHER – match facts to ingredients           (NO LLM)
5. LLM FALLBACK  – restricted Gemini for ambiguous only      (temp=0)
6. SCORER         – weighted core vs secondary
7. FORMATTER      – evaluated / rejected output

STRICT RULES:
  - NO hallucination, NO invented facts
  - NO assumption of intent or causation
  - Deterministic steps first; LLM only as last resort
"""
import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from utils.kb_loader import get_section
from utils.provenance import make_provenance

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — FACT BUILDER  (rule-based, NO LLM)
# ══════════════════════════════════════════════════════════════════════════════

# Map signal names / categories to canonical fact types
_SIGNAL_TO_FACT_TYPE: Dict[str, str] = {
    # Result signals
    "death_occurred":                    "death",
    "physical_injury_detected":          "injury",
    "bodily_harm":                       "injury",
    "grievous_bodily_harm":              "injury",
    "cause_of_death_indicated":          "death",
    "property_loss":                     "property_loss",
    "property_damage":                   "property_damage",
    "property_found_abandoned":          "property_loss",
    # Relationship signals
    "co_presence_at_location":           "last_seen",
    "last_seen_together":                "last_seen",
    "familial_relationship":             "relationship",
    "employer_employee_relation":        "relationship",
    "dispute_between_actors":            "dispute",
    # Possession signals
    "possession_of_property_of_another": "possession",
    "possession_of_weapon":              "weapon",
    "recovery_of_item_from_person":      "possession",
    "object_recovered_linked_to_actor":  "possession",
    "possession_of_evidence":            "possession",
    # Behavior signals
    "absence_of_explanation":            "no_explanation",
    "unusual_behavior_observed":         "suspicious_behavior",
    "fleeing_from_scene":                "fleeing",
    "concealment_of_evidence":           "concealment",
    "deceptive_conduct":                 "deception",
    "making_threats":                    "threat",
    "use_of_force":                      "force",
    "motive_established":                "motive",
    # Temporal signals
    "temporal_gap_unexplained":          "temporal_gap",
    "events_in_close_succession":        "temporal_proximity",
    # Evidentiary signals
    "forensic_evidence_found":           "forensic",
    "absence_of_direct_evidence":        "no_direct_evidence",
    # Legacy/broad signals
    "sexual_offence_occurred":           "sexual_offence",
    "taking_of_property":                "theft",
    "taking_with_force":                 "robbery",
    "deception_for_gain":                "cheating",
    "unlawful_entry":                    "trespass",
    "breach_of_trust":                   "breach_trust",
    "misappropriation_of_property":      "misappropriation",
    "document_fraud":                    "forgery",
    "currency_fraud":                    "forgery",
    "unlawful_restraint_of_person":      "restraint",
    "defamation_occurred":               "defamation",
    "public_order_disturbance":          "public_order",
    "criminal_intimidation":             "intimidation",
}

# Fallback category → fact type (for signals not in the map above)
_CATEGORY_TO_FACT_TYPE: Dict[str, str] = {
    "result":       "generic_result",
    "relationship": "relationship",
    "possession":   "possession",
    "behavior":     "behavior",
    "temporal":     "temporal_gap",
    "evidentiary":  "forensic",
}


def _build_facts(
    events: List[Dict[str, Any]],
    legal_signals: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Step 1: Convert events + legal signals into a flat list of FACTS.
    Each fact: {"type": str, "text": str, "source": str|list, "actors": list}
    """
    facts: List[Dict[str, Any]] = []
    seen_types: Set[str] = set()

    # ── From legal signals (primary source) ───────────────────────────────
    for sig in legal_signals:
        signal_name = sig.get("signal", "")
        category = sig.get("category", "")
        fact_type = _SIGNAL_TO_FACT_TYPE.get(signal_name)

        if not fact_type and category:
            fact_type = _CATEGORY_TO_FACT_TYPE.get(category, "generic_fact")

        if not fact_type:
            continue

        actors = sig.get("actors", [])
        objects = sig.get("related_objects", [])
        sources = sig.get("source_events", [])

        # Build human-readable text
        text_parts = [signal_name.replace("_", " ").capitalize()]
        if objects:
            text_parts.append(f"involving {', '.join(str(o) for o in objects)}")
        if actors:
            text_parts.append(f"(actors: {', '.join(str(a) for a in actors)})")

        facts.append({
            "type": fact_type,
            "text": " ".join(text_parts),
            "source": sources if len(sources) != 1 else sources[0],
            "actors": actors,
        })
        seen_types.add(fact_type)

    # ── From events (supplement if signals didn't cover a key event) ──────
    for event in events:
        desc = str(event.get("description", "")).lower()
        eid = event.get("event_id", "")
        actors = event.get("actors", [])

        # Only add event-derived facts for major categories not already covered
        if "dead" in desc or "death" in desc or "died" in desc or "killed" in desc:
            if "death" not in seen_types:
                facts.append({"type": "death", "text": str(event.get("description", "")), "source": eid, "actors": actors})
                seen_types.add("death")
        if "injur" in desc or "hurt" in desc or "wound" in desc or "blunt" in desc:
            if "injury" not in seen_types:
                facts.append({"type": "injury", "text": str(event.get("description", "")), "source": eid, "actors": actors})
                seen_types.add("injury")
        if "stole" in desc or "took" in desc or "snatch" in desc:
            if "theft" not in seen_types:
                facts.append({"type": "theft", "text": str(event.get("description", "")), "source": eid, "actors": actors})
                seen_types.add("theft")

    return facts


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — DOMAIN EXTRACTOR  (rule-based, NO LLM)
# ══════════════════════════════════════════════════════════════════════════════

_FACT_TYPE_TO_DOMAINS: Dict[str, List[str]] = {
    "death":              ["homicide"],
    "injury":             ["bodily_harm", "homicide"],
    "possession":         ["property"],
    "misappropriation":   ["property"],
    "theft":              ["property"],
    "robbery":            ["property", "bodily_harm"],
    "property_loss":      ["property"],
    "property_damage":    ["property"],
    "weapon":             ["bodily_harm"],
    "force":              ["bodily_harm"],
    "deception":          ["fraud"],
    "cheating":           ["fraud"],
    "forgery":            ["forgery"],
    "trespass":           ["trespass"],
    "restraint":          ["restraint"],
    "sexual_offence":     ["sexual_offence"],
    "defamation":         ["defamation"],
    "public_order":       ["public_order"],
    "threat":             ["intimidation"],
    "intimidation":       ["intimidation"],
    "breach_trust":       ["property", "fraud"],
    # Behavioral / circumstantial facts contribute to 'general' domain
    "last_seen":          ["general"],
    "relationship":       ["general"],
    "dispute":            ["general"],
    "no_explanation":     ["general"],
    "suspicious_behavior":["general"],
    "fleeing":            ["general"],
    "concealment":        ["general"],
    "motive":             ["general"],
    "temporal_gap":       ["general"],
    "temporal_proximity": ["general"],
    "forensic":           ["general"],
    "no_direct_evidence": ["general"],
    "generic_result":     ["general"],
    "generic_fact":       ["general"],
    "behavior":           ["general"],
}


def _extract_domains(facts: List[Dict[str, Any]]) -> Set[str]:
    """Step 2: Derive case domains from fact types."""
    domains: Set[str] = set()
    for fact in facts:
        ft = fact.get("type", "")
        for d in _FACT_TYPE_TO_DOMAINS.get(ft, ["general"]):
            domains.add(d)
    return domains


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — STATUTE FILTER  (rule-based, NO LLM)
# ══════════════════════════════════════════════════════════════════════════════

# Keywords in ingredient text → required domain
_INGREDIENT_DOMAIN_RULES: List[Tuple[List[str], str]] = [
    (["death", "kills", "murder", "homicide", "culpable homicide"],          "homicide"),
    (["hurt", "injury", "grievous", "bodily harm", "wound", "force"],        "bodily_harm"),
    (["property", "possession", "misappropriat", "dishonest", "movable"],    "property"),
    (["cheat", "decei", "fraud", "induce"],                                  "fraud"),
    (["forg", "false document", "counterfeit"],                              "forgery"),
    (["trespass", "entry", "house-break", "lurking"],                        "trespass"),
    (["restrain", "confine", "kidnap", "abduct"],                            "restraint"),
    (["sexual", "rape", "modesty", "outraging"],                             "sexual_offence"),
    (["defam", "imputation"],                                                "defamation"),
    (["riot", "unlawful assembly", "public"],                                "public_order"),
    (["pregnan", "miscarriage", "quick with child", "unborn"],               "miscarriage"),
    (["marriage", "husband", "wife", "dowry"],                               "marriage"),
    (["intimidat", "threaten", "alarm"],                                     "intimidation"),
]


def _infer_statute_domains(ingredients: List[Dict[str, Any]]) -> Set[str]:
    """Analyze ingredient texts to determine which domains a statute requires."""
    domains: Set[str] = set()
    for ing in ingredients:
        text = str(ing.get("text", ing.get("description", ""))).lower()
        for keywords, domain in _INGREDIENT_DOMAIN_RULES:
            if any(kw in text for kw in keywords):
                domains.add(domain)
    # If no specific domain detected, it's a general statute
    if not domains:
        domains.add("general")
    return domains


def _filter_statute(statute_domains: Set[str], case_domains: Set[str]) -> bool:
    """
    Return True if the statute should be KEPT (at least one required domain
    is present in the case). Return False to REJECT.
    """
    # "general" domain statutes always pass (they don't require specific facts)
    if "general" in statute_domains:
        return True
    return bool(statute_domains & case_domains)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — INGREDIENT MATCHER  (rule-based, NO LLM)
# ══════════════════════════════════════════════════════════════════════════════

# Map keyword phrases in ingredient text → required fact types
_INGREDIENT_FACT_RULES: List[Tuple[List[str], str]] = [
    (["causes death", "cause death", "death is caused", "causing death", "commits murder"],   "death"),
    (["hurt", "injury", "bodily harm", "wound", "blunt force"],                                "injury"),
    (["property", "movable property", "misappropriat"],                                        "possession"),
    (["possession", "possessed"],                                                              "possession"),
    (["dishonest", "fraudulent"],                                                              "deception"),
    (["deceased", "dead person", "at the time of his death", "decease"],                       "death"),
    (["assault", "criminal force", "force"],                                                   "force"),
    (["intention", "intent", "knowledge", "knows"],                                            "motive"),
    (["trespass", "entry", "house-break"],                                                     "trespass"),
    (["restrain", "confine"],                                                                  "restraint"),
    (["cheat", "deceiv", "induc", "fraudulent representation"],                                "deception"),
    (["threaten", "alarm", "intimidat"],                                                       "threat"),
    (["miscarriage", "pregnan", "quick with child"],                                           "pregnancy"),
    (["consent"],                                                                              "consent"),
]


def _match_ingredient_to_facts(
    ingredient_text: str,
    element_type: str,
    facts: List[Dict[str, Any]],
    fact_type_set: Set[str],
) -> Tuple[str, float, str, str]:
    """
    Step 4: Try to match an ingredient to existing facts using rules.
    Returns (status, confidence, evidence_text, source).
    """
    text_lower = ingredient_text.lower()

    # Determine which fact types this ingredient needs
    required_fact_types: List[str] = []
    for keywords, fact_type in _INGREDIENT_FACT_RULES:
        if any(kw in text_lower for kw in keywords):
            required_fact_types.append(fact_type)

    if not required_fact_types:
        # Could not determine required facts — needs LLM fallback
        return "needs_llm", 0.0, "", "unresolved"

    # Check if ALL required fact types are present
    all_present = all(ft in fact_type_set for ft in required_fact_types)
    any_present = any(ft in fact_type_set for ft in required_fact_types)

    if all_present:
        # Find the best matching fact for evidence
        evidence_texts = []
        for ft in required_fact_types:
            for fact in facts:
                if fact["type"] == ft:
                    evidence_texts.append(fact["text"])
                    break
        evidence = "; ".join(evidence_texts) if evidence_texts else "Direct fact match"
        return "satisfied", 1.0, evidence, "fact_builder"

    elif any_present:
        # Partial match — some required facts found
        matched = [ft for ft in required_fact_types if ft in fact_type_set]
        missing = [ft for ft in required_fact_types if ft not in fact_type_set]
        evidence_texts = []
        for ft in matched:
            for fact in facts:
                if fact["type"] == ft:
                    evidence_texts.append(fact["text"])
                    break
        evidence = f"Partial: found [{', '.join(matched)}], missing [{', '.join(missing)}]"
        if evidence_texts:
            evidence += f". Evidence: {'; '.join(evidence_texts)}"

        # Weight based on element type
        if element_type in ("actus_reus", "act", "action"):
            return "partially_satisfied", 0.6, evidence, "fact_builder"
        elif element_type in ("mens_rea",):
            return "partially_satisfied", 0.4, evidence, "fact_builder"
        else:
            return "partially_satisfied", 0.5, evidence, "fact_builder"

    else:
        return "not_satisfied", 0.0, f"No facts matching required types: {required_fact_types}", "fact_builder"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — LLM FALLBACK  (restricted, temp=0)
# ══════════════════════════════════════════════════════════════════════════════

_LLM_FALLBACK_PROMPT = """\
You are a STRICT legal fact matcher. You MUST NOT invent or assume any facts.

TASK: Given the ingredient text and the list of AVAILABLE FACTS from the case,
determine if any existing fact supports this ingredient.

RULES:
- You MUST ONLY use facts from the provided list
- You MUST NOT create new facts
- You MUST NOT assume intent, motive, or causation
- If no fact matches, return exactly: NO_MATCH
- If a fact partially matches, return the fact index and confidence

Ingredient: "{ingredient_text}"
Element Type: "{element_type}"

Available Facts:
{facts_json}

Return ONLY valid JSON (no markdown):
{{"match": true/false, "fact_index": <int or null>, "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}}
"""


def _llm_fallback(
    ingredient_text: str,
    element_type: str,
    facts: List[Dict[str, Any]],
    gemini_caller: Optional[Callable[..., str]],
) -> Tuple[str, float, str, str]:
    """
    Step 5: Use LLM as restricted fallback for ambiguous ingredients.
    Only called when rule engine returns 'needs_llm'.
    """
    if not gemini_caller:
        return "not_satisfied", 0.0, "No LLM available and rule engine could not match.", "no_llm"

    # Prepare facts summary for LLM (anonymized indices)
    facts_for_llm = []
    for i, f in enumerate(facts):
        facts_for_llm.append({"index": i, "type": f["type"], "text": f["text"]})

    prompt = _LLM_FALLBACK_PROMPT.format(
        ingredient_text=ingredient_text,
        element_type=element_type,
        facts_json=json.dumps(facts_for_llm, indent=2),
    )

    try:
        raw = gemini_caller(
            prompt,
            "Match the ingredient to existing facts only.",
            temperature=0.0,
            max_tokens=256,
        )

        # Strip markdown if present
        cleaned = raw.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        parsed = json.loads(cleaned)

        if parsed.get("match") is True:
            fact_idx = parsed.get("fact_index")
            confidence = min(float(parsed.get("confidence", 0.5)), 0.8)  # Cap LLM confidence at 0.8
            reasoning = str(parsed.get("reasoning", "LLM matched to existing fact"))

            if fact_idx is not None and 0 <= fact_idx < len(facts):
                evidence = f"LLM matched: {facts[fact_idx]['text']}. {reasoning}"
            else:
                evidence = f"LLM partial: {reasoning}"

            if confidence >= 0.6:
                return "partially_satisfied", confidence, evidence, "llm_fallback"
            else:
                return "weakly_supported", confidence, evidence, "llm_fallback"
        else:
            reasoning = str(parsed.get("reasoning", "No matching fact found"))
            return "not_satisfied", 0.0, f"LLM confirmed no match: {reasoning}", "llm_fallback"

    except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
        return "not_satisfied", 0.0, f"LLM fallback failed: {exc}", "llm_fallback_error"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — SCORER
# ══════════════════════════════════════════════════════════════════════════════

# Core element types get higher weight
_CORE_ELEMENT_TYPES = {"actus_reus", "act", "action", "mens_rea", "consequence", "causation"}
_CORE_WEIGHT = 2.0
_SECONDARY_WEIGHT = 1.0


def _score_statute(ingredient_results: List[Dict[str, Any]]) -> Tuple[float, str]:
    """
    Step 6: Compute weighted score and confidence level.
    """
    if not ingredient_results:
        return 0.0, "low"

    total_weight = 0.0
    weighted_score = 0.0
    core_satisfied = 0
    core_total = 0

    for ing in ingredient_results:
        element_type = ing.get("element_type", "")
        is_core = element_type in _CORE_ELEMENT_TYPES
        weight = _CORE_WEIGHT if is_core else _SECONDARY_WEIGHT

        if is_core:
            core_total += 1

        status = ing.get("status", "not_satisfied")
        conf = float(ing.get("confidence", 0.0))

        if status == "satisfied":
            weighted_score += weight * conf
            if is_core:
                core_satisfied += 1
        elif status in ("partially_satisfied", "weakly_supported"):
            weighted_score += weight * conf
        # not_satisfied contributes 0

        total_weight += weight

    score = round(weighted_score / total_weight, 4) if total_weight > 0 else 0.0

    # Confidence based on core ingredient satisfaction
    if core_total > 0:
        core_ratio = core_satisfied / core_total
        if core_ratio >= 0.8 and score >= 0.7:
            confidence = "high"
        elif core_ratio >= 0.5 or score >= 0.4:
            confidence = "medium"
        else:
            confidence = "low"
    else:
        if score >= 0.7:
            confidence = "high"
        elif score >= 0.4:
            confidence = "medium"
        else:
            confidence = "low"

    return score, confidence


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — MAIN TOOL FUNCTION  (orchestrator + formatter)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_statute_ingredients_v1(
    case_id: str,
    legal_signal_graph_path: str,
    cases_dir: str,
    tenant_id: str = "local_dev",
    input_refs: List[str] | None = None,
    _mcp_trace_id: str = "",
    _mcp_tool_version: str = "1.0.0",
    _gemini_caller: Optional[Callable[..., str]] = None,
    **_: Any,
) -> Dict[str, Any]:
    """
    Hybrid Statute Validator — 7-step pipeline.
    """
    # ── Load inputs ───────────────────────────────────────────────────────
    with open(legal_signal_graph_path, "r", encoding="utf-8") as f:
        signal_graph = json.load(f)

    candidates = signal_graph.get("candidate_sections", [])
    legal_signals = signal_graph.get("legal_signals", [])
    events = signal_graph.get("events", [])

    # ── STEP 1: Build Facts ──────────────────────────────────────────────
    facts = _build_facts(events, legal_signals)
    fact_type_set: Set[str] = {f["type"] for f in facts}

    # ── STEP 2: Extract Domains ──────────────────────────────────────────
    case_domains = _extract_domains(facts)

    # ── Load KB sections ─────────────────────────────────────────────────
    sections_data: Dict[str, Dict[str, Any]] = {}
    for cand in candidates:
        sid = cand["section_id"]
        sec = get_section(sid)
        if sec is not None:
            sections_data[sid] = sec
        else:
            sections_data[sid] = {
                "section_id": sid,
                "heading": cand.get("heading", ""),
                "ingredients": [],
            }

    # ── Evaluate each candidate ──────────────────────────────────────────
    results: List[Dict[str, Any]] = []
    llm_calls_made = 0

    for cand in candidates:
        sid = cand["section_id"]
        sec = sections_data.get(sid, {})
        ingredients = sec.get("ingredients", [])
        heading = sec.get("heading", cand.get("heading", ""))

        if not ingredients:
            results.append({
                "statute_id": sid,
                "name": heading,
                "status": "rejected",
                "score": 0.0,
                "confidence": "low",
                "ingredient_analysis": [],
                "reasoning": ["No ingredients found in knowledge base."],
            })
            continue

        # ── STEP 3: Domain Filter ────────────────────────────────────────
        statute_domains = _infer_statute_domains(ingredients)
        if not _filter_statute(statute_domains, case_domains):
            results.append({
                "statute_id": sid,
                "name": heading,
                "status": "rejected",
                "score": 0.0,
                "confidence": "none",
                "ingredient_analysis": [],
                "reasoning": [
                    f"Hard-rejected: statute requires domains {sorted(statute_domains)}, "
                    f"but case only has {sorted(case_domains)}."
                ],
            })
            continue

        # ── STEP 4 + 5: Ingredient Matching ──────────────────────────────
        ingredient_analysis: List[Dict[str, Any]] = []

        for ing in ingredients:
            ing_id = ing.get("id", ing.get("ingredient_id", ""))
            ing_text = str(ing.get("text", ing.get("description", "")))
            element_type = ing.get("element_type", "")

            # Step 4: Rule-based matching
            status, confidence, evidence, source = _match_ingredient_to_facts(
                ing_text, element_type, facts, fact_type_set
            )

            # Step 5: LLM fallback only for unresolved ingredients
            if status == "needs_llm" and _gemini_caller:
                status, confidence, evidence, source = _llm_fallback(
                    ing_text, element_type, facts, _gemini_caller
                )
                llm_calls_made += 1

            elif status == "needs_llm":
                # No LLM available — mark as unresolved
                status = "not_satisfied"
                confidence = 0.0
                evidence = "Rule engine could not match; no LLM available."
                source = "no_match"

            ingredient_analysis.append({
                "ingredient_id": ing_id,
                "text": ing_text,
                "element_type": element_type,
                "status": status,
                "confidence": round(confidence, 4),
                "evidence": evidence,
                "source": source,
            })

        # ── STEP 6: Score ────────────────────────────────────────────────
        score, confidence_level = _score_statute(ingredient_analysis)

        # ── Build reasoning array ────────────────────────────────────────
        reasoning: List[str] = []
        sat = [ia for ia in ingredient_analysis if ia["status"] == "satisfied"]
        part = [ia for ia in ingredient_analysis if ia["status"] in ("partially_satisfied", "weakly_supported")]
        miss = [ia for ia in ingredient_analysis if ia["status"] == "not_satisfied"]

        if sat:
            reasoning.append(f"{len(sat)}/{len(ingredient_analysis)} ingredients satisfied with direct facts.")
        if part:
            reasoning.append(f"{len(part)} ingredients partially supported.")
        if miss:
            reasoning.append(f"{len(miss)} ingredients not satisfied.")

        # Add fact-based explanations for satisfied core ingredients
        for ia in sat:
            if ia["element_type"] in _CORE_ELEMENT_TYPES:
                reasoning.append(f"Core [{ia['ingredient_id']}]: {ia['evidence']}")

        results.append({
            "statute_id": sid,
            "name": heading,
            "status": "evaluated",
            "score": round(score, 4),
            "confidence": confidence_level,
            "ingredient_analysis": ingredient_analysis,
            "reasoning": reasoning,
        })

    # Sort by score descending
    results.sort(key=lambda r: r["score"], reverse=True)

    # ── Provenance ────────────────────────────────────────────────────────
    provenance = make_provenance(
        tool_name="evaluate_statute_ingredients_v1",
        tool_version="3.0.0",  # v3 Hybrid Validator
        input_refs=input_refs or [legal_signal_graph_path],
        trace_id=_mcp_trace_id,
        model_version="gemini-2.5-flash" if _gemini_caller else "deterministic_only",
    )

    statute_evaluation = {
        "case_id": case_id,
        "tenant_id": tenant_id,
        "statute_evaluations": results,
        "evaluation_count": len(results),
        "rejected_count": sum(1 for r in results if r["status"] == "rejected"),
        "evaluated_count": sum(1 for r in results if r["status"] == "evaluated"),
        "llm_calls_made": llm_calls_made,
        "case_facts": facts,
        "case_domains": sorted(case_domains),
        "provenance": provenance,
    }

    out_path = os.path.join(cases_dir, case_id, "statute_evaluation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(statute_evaluation, f, indent=2, ensure_ascii=False)

    return {
        "result_ref": out_path,
        "evaluation_count": len(results),
        "rejected_count": sum(1 for r in results if r["status"] == "rejected"),
        "evaluated_count": sum(1 for r in results if r["status"] == "evaluated"),
        "llm_calls_made": llm_calls_made,
        "provenance": provenance,
    }
