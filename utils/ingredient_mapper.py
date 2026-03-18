"""
Lexplain — Ingredient Type Mapper
MIT License | See README for MCP provenance contract.

Uses Gemini to map IPC ingredients to required abstract signal types:
["result", "action", "relationship", "possession", "behavior", "mental_state", "circumstance"]

Caches results locally in data/ingredient_type_cache.json to avoid repeated LLM calls.
"""
import json
import os
import re
from typing import Callable, List

# Cache file location
CACHE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "ingredient_type_cache.json"
)

# Allowed abstract types
ALLOWED_TYPES = [
    "result", "action", "relationship", "possession",
    "behavior", "mental_state", "circumstance"
]

_MAPPING_PROMPT = """\
You are an expert legal reasoning system mapping legal ingredients to structural abstract evidence types.

Your task is to analyze the provided legal ingredient and return exactly which abstract evidence types are REQUIRED to prove it.

ALLOWED TYPES:
- "result" : concrete outcome (death, injury, property damage, loss)
- "action" : physical act (using force, taking property, issuing threat)
- "relationship" : connection between actors (presence, proximity, guardianship)
- "possession" : control over object/property/evidence
- "behavior" : conduct implying guilt (fleeing, no explanation, concealment)
- "mental_state" : mens rea (intention, motive, knowledge, threat, dishonesty)
- "circumstance" : environment/context (public order, night-time, rioting)

Ingredient: "{ingredient_text}"
Element Type (if known): "{element_type}"

Return ONLY a JSON array of required types from the allowed list above. Do not include markdown formatting or inside a json block.
Example output format:
["result", "action", "mental_state"]
"""


def _load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def get_required_signal_types(
    ingredient_id: str,
    ingredient_text: str,
    element_type: str,
    gemini_caller: Callable[..., str]
) -> List[str]:
    """
    Get the required abstract signal types for an ingredient.
    Checks cache first. If missing, uses the LLM to classify and then saves to cache.
    """
    cache = _load_cache()

    if ingredient_id in cache:
        return cache[ingredient_id]

    # Handle basic deterministic mapping if we can't call Gemini
    if not gemini_caller:
        return _fallback_mapping(ingredient_text, element_type)

    # Call LLM
    prompt = _MAPPING_PROMPT.format(
        ingredient_text=ingredient_text,
        element_type=element_type
    )
    
    raw = gemini_caller(prompt, "Please formulate the JSON array.", temperature=0.0, max_tokens=128)
    
    # Parse JSON list
    try:
        # strip markdown code blocks if any
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.endswith("```"):
            raw = raw[:-3]
        
        parsed = json.loads(raw.strip())
        if isinstance(parsed, list):
            # Validate types
            valid_types = [t for t in parsed if t in ALLOWED_TYPES]
            
            # Save to cache
            cache[ingredient_id] = valid_types
            _save_cache(cache)
            return valid_types
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback if LLM parsing failed
    fallback = _fallback_mapping(ingredient_text, element_type)
    cache[ingredient_id] = fallback
    _save_cache(cache)
    return fallback


def _fallback_mapping(ingredient_text: str, element_type: str) -> List[str]:
    """Fallback mapping logic if LLM fails or is unavailable."""
    text_lower = ingredient_text.lower()
    types = set()

    if element_type == "actus_reus":
        types.add("action")
        if "death" in text_lower or "hurt" in text_lower or "damage" in text_lower:
            types.add("result")
    elif element_type == "mens_rea":
        types.add("mental_state")
    elif element_type == "circumstance":
        types.add("circumstance")

    if "possession" in text_lower or "property" in text_lower:
        types.add("possession")

    if not types:
        types.add("action")  # default

    return list(types)
