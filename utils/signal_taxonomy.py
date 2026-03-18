"""
Lexplain — Signal Taxonomy
MIT License | See README for MCP provenance contract.

Maps specific legal signals (output from Agent 5) to universal,
abstract signal types for the generalized inference engine (Agent 6).

Categories:
- result (e.g., death, injury, property damage)
- action (e.g., use of force, taking property)
- relationship (e.g., presence, proximity)
- possession (e.g., control over object, recovery)
- behavior (e.g., flight, concealment, no explanation)
- mental_state (e.g., intent, threat, motive)
- circumstance (e.g., public order, environment)
"""
from typing import Dict, List, Set

# Map specific signals to their abstract taxonomy types
_SIGNAL_TO_TYPES: Dict[str, List[str]] = {
    # ── Result Signals ─────────────────────────────────────────────────
    "death_occurred": ["result"],
    "physical_injury_detected": ["result", "action"],
    "bodily_harm": ["result", "action"],
    "grievous_bodily_harm": ["result", "action"],
    "property_loss": ["result"],
    "property_damage": ["result", "action"],

    # ── Relationship Signals ───────────────────────────────────────────
    "co_presence_at_location": ["relationship"],
    "last_seen_together": ["relationship"],
    "familial_relationship": ["relationship", "circumstance"],
    "employer_employee_relation": ["relationship", "circumstance"],

    # ── Possession Signals ─────────────────────────────────────────────
    "possession_of_property_of_another": ["possession", "action"],
    "possession_of_weapon": ["possession", "action"],
    "possession_of_evidence": ["possession"],

    # ── Behavior Signals ──────────────────────────────────────────────
    "absence_of_explanation": ["behavior", "mental_state"],
    "fleeing_from_scene": ["behavior"],
    "concealment_of_evidence": ["behavior", "action"],
    "making_threats": ["mental_state", "action"],
    "deceptive_conduct": ["action", "behavior", "mental_state"],
    "use_of_force": ["action"],
    "motive_established": ["mental_state"],
    "conspiracy_established": ["mental_state", "relationship"],
    "abetment_established": ["action", "mental_state"],

    # ── Temporal Signals ──────────────────────────────────────────────
    "temporal_gap_unexplained": ["behavior"],
    "events_in_close_succession": ["circumstance"],

    # ── Evidentiary Signals ───────────────────────────────────────────
    "recovery_of_item_from_person": ["possession", "result"],
    "forensic_evidence_found": ["result"],

    # ── Legacy / broad signals (backward compatibility) ────────────────
    "sexual_offence_occurred": ["action", "result"],
    "harassment_occurred": ["action", "behavior"],
    "stalking_occurred": ["action", "behavior"],
    "voyeurism_occurred": ["action", "behavior"],
    "taking_of_property": ["action", "possession"],
    "taking_with_force": ["action", "possession", "result"],
    "taking_with_group_force": ["action", "possession", "circumstance"],
    "coercion_for_property": ["action", "mental_state", "possession"],
    "deception_for_gain": ["action", "behavior", "mental_state"],
    "unlawful_entry": ["action", "behavior"],
    "breach_of_trust": ["behavior", "possession", "mental_state"],
    "misappropriation_of_property": ["action", "possession", "mental_state"],
    "public_order_disturbance": ["circumstance", "action"],
    "offence_against_state": ["action", "mental_state"],
    "document_fraud": ["action", "behavior", "mental_state"],
    "currency_fraud": ["action", "possession", "mental_state"],
    "unlawful_restraint_of_person": ["action", "behavior"],
    "human_trafficking": ["action", "behavior"],
    "evidence_tampering": ["action", "behavior"],
    "false_evidence_given": ["action", "behavior", "mental_state"],
    "obstruction_of_justice": ["action", "behavior"],
    "harbouring_offender": ["behavior", "action"],
    "defamation_occurred": ["action", "result"],
    "insult_occurred": ["action", "behavior"],
    "hate_speech_occurred": ["action", "mental_state"],
    "public_nuisance": ["action", "circumstance"],
    "negligent_act": ["action", "behavior"],
    "rash_act": ["action", "behavior"],
    "evidentiary_recovery": ["possession", "result"],
    "threat_issued": ["mental_state", "action"],
    "criminal_intimidation": ["mental_state", "action"],
    "suspicious_movement": ["behavior"],  # kept for backward compat but banned in new prompt
}


def get_signal_types(signal_name: str) -> List[str]:
    """Return the abstract types for a specific legal signal."""
    return _SIGNAL_TO_TYPES.get(signal_name, [])


def normalize_signals(legal_signals: List[Dict[str, float]]) -> Set[str]:
    """
    Take a list of signals (e.g. from Agent 5) and return a flat set of all
    abstract signal types present in the case.

    Expected input format: [{"signal": "death_occurred", "confidence": 1.0}, ...]
    Returns: {"result", "relationship", "behavior", ...}
    """
    types_present: Set[str] = set()
    for sig in legal_signals:
        sig_name = sig.get("signal", "")
        types = get_signal_types(sig_name)
        if types:
            for t in types:
                types_present.add(t)
        else:
            # For unknown signal names, try to infer from the category field
            # (new v2 signals include a category field)
            category = sig.get("category", "")
            if category:
                # Map signal categories to abstract types
                cat_to_type = {
                    "result": "result",
                    "relationship": "relationship",
                    "possession": "possession",
                    "behavior": "behavior",
                    "temporal": "behavior",       # temporal patterns imply behavioral evidence
                    "evidentiary": "possession",  # evidentiary signals typically involve possession
                }
                mapped = cat_to_type.get(category)
                if mapped:
                    types_present.add(mapped)
                # Also add "action" as a baseline for any signal
                types_present.add("action")
    return types_present
