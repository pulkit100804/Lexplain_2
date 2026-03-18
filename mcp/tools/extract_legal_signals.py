"""
Lexplain — MCP Tool: extract_legal_signals_v1
MIT License | See README for MCP provenance contract.

Legal Signal Extraction Engine v2 — transforms raw case facts into
structured actors, events, HIGH-QUALITY legal signals, AND candidate sections.

Pipeline:
  1. Consolidate text from fact_event_graph nodes
  2. LLM extraction: actors, events, AND direct legal signals (structured)
  3. Deterministic cross-event inference (co-presence, possession chains, temporal gaps)
  4. Map signals → candidate IPC sections via offence_category index
  5. Output legal_signal_graph.json

Signal quality rules:
  - Every signal MUST have legal meaning
  - NO vague labels (other_act, suspicious_movement, unknown_behavior)
  - Signals include actor/object context
  - Cross-event patterns are detected deterministically
"""
import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Set

from utils.kb_loader import load_ipc_kb
from utils.offence_category_index import find_sections_by_categories, get_all_categories
from utils.provenance import make_provenance

# ── Banned signal labels (these are legally meaningless) ──────────────────────

_BANNED_SIGNALS = frozenset({
    "other_act", "other_omission", "suspicious_movement", "suspicious_event",
    "unknown_behavior", "general_act", "miscellaneous", "unspecified",
})

# ── Legal Signal Categories ──────────────────────────────────────────────────

SIGNAL_CATEGORIES = [
    "result",        # outcome affecting a person or property (death, injury, loss)
    "relationship",  # connection between actors (co-presence, proximity, association)
    "possession",    # control or custody of an object (property of another)
    "behavior",      # actions or omissions by actors (concealment, fleeing, no explanation)
    "temporal",      # time-based patterns (unexplained gap, sequence)
    "evidentiary",   # discovery or recovery of objects/facts linked to persons
]

# ── LLM Prompt v2 ────────────────────────────────────────────────────────────

_EXTRACTION_PROMPT = """\
You are Agent 5 of a legal reasoning pipeline (Lexplain).
Your task is to extract LEGALLY MEANINGFUL SIGNALS from structured facts.

You MUST ALWAYS produce signals.
Returning an empty output is NOT allowed.

────────────────────────────────────────────────
IMPORTANT BALANCE RULE (STRICT)
────────────────────────────────────────────────
You must follow BOTH:
1. DO NOT invent facts  
2. DO NOT return empty output  

If unsure, create a weaker but still valid signal (e.g. "co_presence").
Minimum requirement: You MUST output AT LEAST 3 signals if events exist.

────────────────────────────────────────────────
TASK: Extract structured information from the case facts below.
────────────────────────────────────────────────

PART 1 — ACTORS:
Extract every person or entity mentioned.
- role MUST be one of: "accused", "victim", "witness", "complainant", "unknown"
- Only assign "victim" if death/harm is explicit. Only assign "accused" if allegations are explicit.

PART 2 — EVENTS:
Extract every atomic factual event.
- Each event should describe ONE atomic fact.
- "object" = thing involved (weapon, property, body, etc.) — leave empty if none
- "location" and "time" — leave empty if not stated

PART 3 — LEGAL SIGNALS:
This is the MOST IMPORTANT part. From the given events, you MUST extract legal signals.
Even if information is partial, generate the BEST POSSIBLE signals based on available facts.

For EACH event, ask:
- Does this event show a RESULT? (death, injury, loss)
- Does this event connect two actors?
- Does this event show possession or recovery?
- Does this event show behavior (e.g., no explanation)?
- Does this event create a timeline inconsistency?

If YES → generate a signal matching one of the categories: {categories}

Allowed examples for signals:
- "death_occurred" (category: result)
- "physical_injury_detected" (category: result)
- "co_presence_at_location" (category: relationship)
- "last_seen_together" (category: relationship)
- "possession_of_property_of_another" (category: possession)
- "object_recovered_linked_to_actor" (category: evidentiary)
- "absence_of_explanation" (category: behavior)
- "temporal_gap_unexplained" (category: temporal)

If strong signals are not available, output weaker but valid signals like "co_presence_at_location" or "object_recovered_linked_to_actor".

BAD signal labels (NEVER USE THESE):
❌ "other_act" — too vague, has no legal meaning
❌ "suspicious_movement" — vague, not a legal fact
❌ "murder_committed" — this assumes a crime, not a fact

PART 4 — LEGAL CATEGORIES:
Based on the signals extracted, list which broad offence categories from the IPC
are potentially relevant.
- Categories MUST be from this list: {kb_categories}
- Only list categories that have direct evidentiary support.

────────────────────────────────────────────────
Return ONLY strict JSON (no markdown, no extra text):
────────────────────────────────────────────────
{{
  "actors": [
    {{
      "id": "A1",
      "name": "...",
      "role": "accused | victim | witness | complainant | unknown"
    }}
  ],
  "events": [
    {{
      "event_id": "E1",
      "description": "one-sentence factual description",
      "actors": ["A1"],
      "object": "",
      "location": "",
      "time": "",
      "confidence": 0.0
    }}
  ],
  "legal_signals": [
    {{
      "signal": "specific_meaningful_label",
      "category": "result | relationship | possession | behavior | temporal | evidentiary",
      "actors": ["A1", "A2"],
      "related_objects": ["object_name"],
      "source_events": ["E1"],
      "confidence": 0.0
    }}
  ],
  "legal_categories": ["category1", "category2"]
}}

FINAL CHECK:
- Ensure output signals array is NOT empty
- Ensure at least 3 signals exist
- Ensure no vague signals are used
"""

# ── Signal → offence_category mapping for KB lookup ───────────────────────────
# Maps legal signals to KB offence_category values for candidate retrieval.
# This covers both LLM-generated and deterministically-inferred signals.

_SIGNAL_TO_CATEGORIES: Dict[str, List[str]] = {
    # ── Result signals ─────────────────────────────────────────────────
    "death_occurred":                ["murder", "homicide", "culpable_homicide", "death",
                                      "offences_affecting_life", "offence_against_person",
                                      "crime_against_person", "endangering_life"],
    "physical_injury_detected":      ["hurt", "causing_hurt", "causing_injury", "bodily_harm",
                                      "assault", "voluntarily_causing_hurt", "offence_against_person",
                                      "crime_against_person", "violent_crime"],
    "bodily_harm":                   ["hurt", "causing_hurt", "causing_injury", "bodily_harm",
                                      "assault", "voluntarily_causing_hurt", "offence_against_person",
                                      "crime_against_person", "violent_crime"],
    "grievous_bodily_harm":          ["grievous_hurt", "voluntarily_causing_grievous_hurt",
                                      "causing_injury", "bodily_harm", "violent_crime",
                                      "offence_against_person"],
    "property_loss":                 ["theft", "offence_against_property", "property_offence",
                                      "dishonesty_offence", "property_crime"],
    "property_damage":               ["mischief", "property_damage", "offence_against_property"],

    # ── Relationship signals ───────────────────────────────────────────
    "co_presence_at_location":       ["murder", "homicide", "culpable_homicide",
                                      "offence_against_person", "assault"],
    "last_seen_together":            ["murder", "homicide", "culpable_homicide"],
    "familial_relationship":         ["offence_against_women", "cruelty", "domestic_violence"],
    "employer_employee_relation":    ["criminal_breach_of_trust", "dishonesty_offence"],

    # ── Possession signals ─────────────────────────────────────────────
    "possession_of_property_of_another": ["theft", "receiving_stolen_property", "dishonesty",
                                          "property_offence", "dishonesty_offence",
                                          "offence_against_property"],
    "possession_of_weapon":          ["murder", "assault", "violent_crime",
                                      "offence_against_person"],
    "possession_of_evidence":        ["possession", "receiving_stolen_property"],

    # ── Behavior signals ──────────────────────────────────────────────
    "absence_of_explanation":        ["murder", "homicide", "culpable_homicide",
                                      "concealment"],
    "fleeing_from_scene":            ["murder", "homicide", "concealment",
                                      "offence_against_person"],
    "concealment_of_evidence":       ["concealment", "evidence_tampering",
                                      "obstruction_of_justice",
                                      "offence_against_public_justice"],
    "making_threats":                ["criminal_intimidation", "threat", "coercion",
                                      "extortion"],
    "deceptive_conduct":             ["cheating", "fraud", "deception", "property_offence",
                                      "dishonesty", "criminal_breach_of_trust",
                                      "dishonesty_offence", "false_representation"],
    "use_of_force":                  ["assault", "criminal_force", "offence_against_person",
                                      "violent_crime", "robbery"],
    "motive_established":            ["murder", "homicide", "criminal_offence"],
    "conspiracy_established":        ["conspiracy", "criminal_conspiracy", "abetment"],
    "abetment_established":          ["abetment", "conspiracy", "criminal_conspiracy"],

    # ── Temporal signals ───────────────────────────────────────────────
    "temporal_gap_unexplained":      ["murder", "homicide", "culpable_homicide",
                                      "concealment"],
    "events_in_close_succession":    ["murder", "homicide", "assault"],

    # ── Evidentiary signals ────────────────────────────────────────────
    "recovery_of_item_from_person":  ["theft", "receiving_stolen_property", "possession",
                                      "dishonesty_offence", "property_offence"],
    "forensic_evidence_found":       ["murder", "homicide", "assault", "offence_against_person"],

    # ── Legacy / broad signals (for backward compat) ───────────────────
    "sexual_offence_occurred":       ["rape", "sexual_offence", "sexual_offences",
                                      "offences_relating_to_modesty", "crimes_against_women",
                                      "gang_rape", "sexual_exploitation"],
    "harassment_occurred":           ["sexual_harassment", "offence_against_women",
                                      "crimes_against_women"],
    "stalking_occurred":             ["criminal_offence", "crimes_against_women"],
    "taking_of_property":            ["theft", "offence_against_property", "property_offence",
                                      "aggravated_theft", "property_crime"],
    "taking_with_force":             ["robbery", "offence_against_property", "violent_crime",
                                      "property_offence"],
    "coercion_for_property":         ["extortion", "offence_against_property", "property_offence",
                                      "coercion"],
    "deception_for_gain":            ["cheating", "fraud", "deception", "property_offence",
                                      "dishonesty", "criminal_breach_of_trust",
                                      "dishonesty_offence", "false_representation"],
    "unlawful_entry":                ["criminal_trespass", "trespass", "house-trespass",
                                      "house-breaking"],
    "breach_of_trust":               ["criminal_breach_of_trust", "dishonesty",
                                      "offence_against_property"],
    "misappropriation_of_property":  ["dishonesty", "property_offence",
                                      "criminal_breach_of_trust"],
    "public_order_disturbance":      ["public_order", "public_order_offence",
                                      "offence_against_public_order"],
    "offence_against_state":         ["waging_war", "offence_against_state",
                                      "offences_against_the_state", "national_security",
                                      "sedition"],
    "document_fraud":                ["forgery", "false_document", "document_offence",
                                      "counterfeiting"],
    "currency_fraud":                ["counterfeiting", "currency_offence",
                                      "offences_relating_to_coin"],
    "unlawful_restraint_of_person":  ["kidnapping", "abduction", "kidnapping_and_abduction",
                                      "wrongful_confinement", "kidnapping/abduction"],
    "human_trafficking":             ["human_trafficking", "slavery", "forced_labor",
                                      "sexual_exploitation"],
    "evidence_tampering":            ["evidence_tampering", "false_evidence",
                                      "offence_against_public_justice"],
    "false_evidence_given":          ["false_evidence", "perjury",
                                      "offence_against_public_justice"],
    "obstruction_of_justice":        ["obstruction_of_justice",
                                      "offence_against_public_justice"],
    "harbouring_offender":           ["harbouring", "harbouring_an_offender",
                                      "harbouring_offender"],
    "defamation_occurred":           ["defamation"],
    "insult_occurred":               ["defamation", "criminal_intimidation"],
    "hate_speech_occurred":          ["hate_speech", "national_integration"],
    "public_nuisance":               ["public_nuisance", "public_health"],
    "negligent_act":                 ["negligence", "omission/negligence",
                                      "offence_against_person"],
    "rash_act":                      ["negligence", "endangerment", "endangering_life"],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _consolidate_text(fact_graph: Dict[str, Any]) -> str:
    """Combine all node texts from the fact graph into a single case description."""
    return " ".join(n.get("text", "") for n in fact_graph.get("nodes", []))


def _parse_llm_response(raw: str) -> Optional[Dict[str, Any]]:
    """Parse LLM JSON response safely."""
    if not raw:
        return None
    try:
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            return json.loads(json_match.group())
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _validate_signal(sig: Dict[str, Any]) -> bool:
    """Validate that a signal is legally meaningful and not banned."""
    label = sig.get("signal", "").strip().lower()
    if not label:
        return False
    if label in _BANNED_SIGNALS:
        return False
    category = sig.get("category", "").strip().lower()
    if category not in SIGNAL_CATEGORIES:
        return False
    return True


def _infer_cross_event_signals(
    actors: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    existing_signals: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Deterministic cross-event pattern detection.
    Generates additional signals from multi-event relationships.
    """
    inferred: List[Dict[str, Any]] = []
    existing_labels: Set[str] = {s.get("signal", "") for s in existing_signals}

    # ── Co-presence: If 2+ actors share an event at the same location ─────
    for event in events:
        event_actors = event.get("actors", [])
        location = event.get("location", "")
        if len(event_actors) >= 2 and "co_presence_at_location" not in existing_labels:
            inferred.append({
                "signal": "co_presence_at_location",
                "category": "relationship",
                "actors": event_actors,
                "related_objects": [],
                "source_events": [event.get("event_id", "")],
                "confidence": 0.9 if location else 0.7,
            })
            existing_labels.add("co_presence_at_location")

    # ── Possession chain: object of one actor found with another ──────────
    # Build a map of objects mentioned per actor
    actor_objects: Dict[str, List[str]] = {}
    object_events: Dict[str, List[str]] = {}
    for event in events:
        obj = str(event.get("object", "")).strip()
        if not obj:
            continue
        actors_list = event.get("actors", [])
        if isinstance(actors_list, list):
            for raw_actor in actors_list:
                actor_id = str(raw_actor)
                if actor_id not in actor_objects:
                    actor_objects[actor_id] = []
                actor_objects[actor_id].append(obj)
        if obj not in object_events:
            object_events[obj] = []
        object_events[obj].append(str(event.get("event_id", "")))

    # Check if any object is linked to multiple actors (potential possession issue)
    obj_to_actors: Dict[str, Set[str]] = {}
    for actor_id, objs in actor_objects.items():
        for obj in objs:
            if obj not in obj_to_actors:
                obj_to_actors[obj] = set()
            obj_to_actors[obj].add(actor_id)

    for obj, linked_actors in obj_to_actors.items():
        if len(linked_actors) >= 2 and "possession_of_property_of_another" not in existing_labels:
            inferred.append({
                "signal": "possession_of_property_of_another",
                "category": "possession",
                "actors": sorted(linked_actors),
                "related_objects": [obj],
                "source_events": object_events.get(obj, []),
                "confidence": 0.85,
            })
            existing_labels.add("possession_of_property_of_another")

    # ── Temporal gap: if events have time fields with gaps ────────────────
    timed_events = [e for e in events if e.get("time", "").strip()]
    if len(timed_events) >= 2 and "temporal_gap_unexplained" not in existing_labels:
        # If there are timed events with different timestamps, flag potential gap
        times = [e.get("time", "") for e in timed_events]
        if len(set(times)) > 1:
            inferred.append({
                "signal": "temporal_gap_unexplained",
                "category": "temporal",
                "actors": [],
                "related_objects": [],
                "source_events": [e.get("event_id", "") for e in timed_events],
                "confidence": 0.6,
            })
            existing_labels.add("temporal_gap_unexplained")

    # ── Last seen together: if actors co-present before a result event ────
    result_signals = {s.get("signal") for s in existing_signals + inferred
                      if s.get("category") == "result"}
    has_co_presence = "co_presence_at_location" in existing_labels
    if result_signals and has_co_presence and "last_seen_together" not in existing_labels:
        # Find actors from co-presence
        co_presence_actors: List[str] = []
        for s in existing_signals + inferred:
            if s.get("signal") == "co_presence_at_location":
                co_presence_actors = s.get("actors", [])
                break
        inferred.append({
            "signal": "last_seen_together",
            "category": "relationship",
            "actors": co_presence_actors,
            "related_objects": [],
            "source_events": [],
            "confidence": 0.8,
        })
        existing_labels.add("last_seen_together")

    return inferred


def _signals_to_candidate_sections(
    signals: List[Dict[str, Any]],
) -> List[str]:
    """Map legal signals to candidate IPC section IDs via the offence category index."""
    all_categories: List[str] = []
    for sig in signals:
        signal_name = sig.get("signal", "")
        cats = _SIGNAL_TO_CATEGORIES.get(signal_name, [])
        all_categories.extend(cats)

    if not all_categories:
        return []

    return find_sections_by_categories(all_categories)


# ── Main tool function ────────────────────────────────────────────────────────

def extract_legal_signals_v1(
    case_id: str,
    fact_graph_path: str,
    cases_dir: str,
    tenant_id: str = "local_dev",
    input_refs: List[str] | None = None,
    _mcp_trace_id: str = "",
    _mcp_tool_version: str = "2.0.0",
    _gemini_caller: Optional[Callable[..., str]] = None,
    **_: Any,
) -> Dict[str, Any]:
    # ── Load input ────────────────────────────────────────────────────────
    with open(fact_graph_path, "r", encoding="utf-8") as f:
        fact_graph = json.load(f)

    case_text = _consolidate_text(fact_graph)

    # ── LLM extraction ────────────────────────────────────────────────────
    actors: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    legal_signals: List[Dict[str, Any]] = []
    legal_categories: List[str] = []

    if _gemini_caller and case_text.strip():
        # Build prompt with signal categories and KB categories
        all_kb_categories = get_all_categories()
        prompt = _EXTRACTION_PROMPT.format(
            categories=", ".join(SIGNAL_CATEGORIES),
            kb_categories=", ".join(all_kb_categories),
        )
        raw = _gemini_caller(
            prompt,
            f"Case Facts:\n{case_text[:5000]}",
            temperature=0.1,
            max_tokens=4096,
        )
        parsed = _parse_llm_response(raw)
        if parsed:
            actors = parsed.get("actors", [])
            events = parsed.get("events", [])
            legal_categories = parsed.get("legal_categories", [])

            # Extract and validate LLM-generated signals
            raw_signals = parsed.get("legal_signals", [])
            for sig in raw_signals:
                if _validate_signal(sig):
                    legal_signals.append(sig)

    # ── Deterministic cross-event inference ────────────────────────────────
    inferred_signals = _infer_cross_event_signals(actors, events, legal_signals)
    legal_signals.extend(inferred_signals)

    # ── Map signals to candidate sections ─────────────────────────────────
    signal_candidates = _signals_to_candidate_sections(legal_signals)
    llm_candidates: List[str] = find_sections_by_categories(legal_categories) if legal_categories else []

    # Merge and rank candidates
    _MAX_CANDIDATES = 20

    # Collect all relevant categories from signals
    all_signal_cats: Set[str] = set()
    for sig in legal_signals:
        signal_name = str(sig.get("signal", ""))
        for cat in _SIGNAL_TO_CATEGORIES.get(signal_name, []):
            all_signal_cats.add(cat.lower())
    # Add LLM-suggested categories
    for cat in legal_categories:
        all_signal_cats.add(str(cat).lower())

    seen_sids: Set[str] = set()
    scored_candidates: List[Any] = []
    kb = load_ipc_kb()

    all_candidates: List[str] = []
    all_candidates.extend(signal_candidates)
    if llm_candidates:
        all_candidates.extend(llm_candidates)

    for sid in all_candidates:
        if sid in seen_sids:
            continue
        seen_sids.add(sid)
        sec = kb.get(sid, {})
        sec_cats = set(c.lower() for c in sec.get("offence_category", []))
        match_score = len(sec_cats & all_signal_cats)
        scored_candidates.append((match_score, sid, sec))

    # Sort by match score descending, then take top N
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    scored_candidates = scored_candidates[:_MAX_CANDIDATES]

    candidate_sections: List[Dict[str, Any]] = []
    for score, sid, sec in scored_candidates:
        candidate_sections.append({
            "section_id": sid,
            "heading": sec.get("heading", ""),
            "offence_category": sec.get("offence_category", []),
            "relevance_score": score,
        })

    # ── Build output ──────────────────────────────────────────────────────
    provenance = make_provenance(
        tool_name="extract_legal_signals_v1",
        tool_version=_mcp_tool_version,
        input_refs=input_refs or [fact_graph_path],
        trace_id=_mcp_trace_id,
        model_version="gemini-2.5-flash" if _gemini_caller else None,
    )

    legal_signal_graph = {
        "case_id": case_id,
        "tenant_id": tenant_id,
        "actors": actors,
        "events": events,
        "legal_signals": legal_signals,
        "candidate_sections": candidate_sections,
        "candidate_count": len(candidate_sections),
        "provenance": provenance,
    }

    out_path = os.path.join(cases_dir, case_id, "legal_signal_graph.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(legal_signal_graph, f, indent=2, ensure_ascii=False)

    return {
        "result_ref": out_path,
        "candidate_count": len(candidate_sections),
        "signal_count": len(legal_signals),
        "provenance": provenance,
    }
