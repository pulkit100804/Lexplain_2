"""
Lexplain — Tests for LLM fallback behaviour.
MIT License.

Validates that when _gemini_caller returns empty/invalid JSON,
the pipeline falls back to deterministic results correctly.
Also validates: core ingredient gating, partial penalty,
not_applicable filtering, and no keyword_proxy.
"""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from utils import kb_loader


# ── Fixtures ──────────────────────────────────────────────────────────────

SAMPLE_KB = {
    "420": {
        "section_id": "420",
        "heading": "Cheating and dishonestly inducing delivery of property",
        "canonical_text": "Whoever cheats and thereby dishonestly induces the person deceived...",
        "ingredients": [
            {"id": "420.1", "text": "cheating or deception", "element_type": "actus_reus", "normalized": {}},
            {"id": "420.2", "text": "dishonest inducement", "element_type": "mens_rea", "normalized": {}},
            {"id": "420.3", "text": "delivery of property", "element_type": "consequence", "normalized": {}},
        ],
        "offence_category": ["property"],
        "punishment": "7 years + fine",
        "status": {"repealed": False, "note": None},
    },
    "44": {
        "section_id": "44",
        "heading": "Injury",
        "canonical_text": "The word 'injury' denotes any harm whatever illegally caused to any person...",
        "ingredients": [
            {"id": "44.1", "text": "injury definition", "element_type": "definition", "normalized": {}},
        ],
        "offence_category": ["definition"],
        "punishment": None,
        "status": {"repealed": False, "note": None},
    },
    "999": {
        "section_id": "999",
        "heading": "Test section with no ingredients",
        "canonical_text": "Punishment section",
        "ingredients": [],
        "offence_category": ["test"],
        "punishment": "fine",
        "status": {"repealed": False, "note": None},
    },
}

SAMPLE_FACT_GRAPH = {
    "case_id": "test_case",
    "nodes": [
        {"node_id": "n1", "text": "The accused cheated and defrauded the complainant of Rs 5 lakh."},
        {"node_id": "n2", "text": "The accused made false representations about a property deal."},
        {"node_id": "n3", "text": "The accused dishonestly induced the complainant to transfer money."},
    ],
    "events": [],
    "actors": ["accused", "complainant"],
}


@pytest.fixture(autouse=True)
def clear_cache():
    kb_loader.load_ipc_kb.cache_clear()
    yield
    kb_loader.load_ipc_kb.cache_clear()


@pytest.fixture
def setup_files(tmp_path, monkeypatch):
    """Create KB, fact graph, and statute_candidates files for testing."""
    kb_path = tmp_path / "kb.json"
    with open(kb_path, "w") as f:
        json.dump(SAMPLE_KB, f)
    monkeypatch.setenv("LEXPLAIN_KB_PATH", str(kb_path))

    case_dir = tmp_path / "cases" / "test_case"
    case_dir.mkdir(parents=True)

    fg_path = str(case_dir / "fact_event_graph.json")
    with open(fg_path, "w") as f:
        json.dump(SAMPLE_FACT_GRAPH, f)

    return {
        "cases_dir": str(tmp_path / "cases"),
        "case_dir": str(case_dir),
        "fact_graph_path": fg_path,
    }


# ── Tests: identify_statutes_v1 ──────────────────────────────────────────

def test_identify_statutes_no_llm(setup_files):
    """Statutes identified deterministically — definition sections excluded."""
    from mcp.tools.identify_statutes import identify_statutes_v1

    result = identify_statutes_v1(
        case_id="test_case",
        fact_graph_path=setup_files["fact_graph_path"],
        cases_dir=setup_files["cases_dir"],
        _gemini_caller=None,
    )

    assert "candidates" in result
    for cand in result["candidates"]:
        assert cand["final"]["source"] == "deterministic"
        assert "score" in cand
        assert "matched_terms" in cand


def test_identify_statutes_filters_definitions(setup_files):
    """Definition sections (like 44 'Injury') should NOT appear in candidates."""
    from mcp.tools.identify_statutes import identify_statutes_v1

    result = identify_statutes_v1(
        case_id="test_case",
        fact_graph_path=setup_files["fact_graph_path"],
        cases_dir=setup_files["cases_dir"],
    )

    candidate_ids = [c["statute_id"] for c in result["candidates"]]
    assert "44" not in candidate_ids, "Definition section 44 should be filtered out"
    assert "999" not in candidate_ids, "Empty-ingredient section should be filtered out"


def test_identify_statutes_llm_failure(setup_files):
    """LLM caller is accepted but NOT used by Agent 5 (pure deterministic)."""
    from mcp.tools.identify_statutes import identify_statutes_v1

    def bad_gemini(system_prompt, user_json, **kw):
        return "NOT VALID JSON AT ALL {{{{"

    result = identify_statutes_v1(
        case_id="test_case",
        fact_graph_path=setup_files["fact_graph_path"],
        cases_dir=setup_files["cases_dir"],
        _gemini_caller=bad_gemini,
    )

    for cand in result["candidates"]:
        assert cand["final"]["source"] == "deterministic"


def test_identify_statutes_llm_empty(setup_files):
    """LLM caller returns empty → Agent 5 is unaffected (pure deterministic)."""
    from mcp.tools.identify_statutes import identify_statutes_v1

    def empty_gemini(system_prompt, user_json, **kw):
        return ""

    result = identify_statutes_v1(
        case_id="test_case",
        fact_graph_path=setup_files["fact_graph_path"],
        cases_dir=setup_files["cases_dir"],
        _gemini_caller=empty_gemini,
    )

    for cand in result["candidates"]:
        assert cand["final"]["source"] == "deterministic"


# ── Tests: evaluate_ingredients_v1 ────────────────────────────────────────

def _make_statute_candidates(case_dir, candidates_list):
    """Helper: write a statute_candidates.json."""
    sc = {"case_id": "test_case", "candidates": candidates_list, "provenance": {}}
    path = os.path.join(case_dir, "statute_candidates.json")
    with open(path, "w") as f:
        json.dump(sc, f)
    return path


def test_evaluate_no_keyword_proxy(setup_files):
    """No keyword_proxy element_type should ever appear in ingredient output."""
    from mcp.tools.evaluate_ingredients import evaluate_ingredients_v1

    sc_path = _make_statute_candidates(setup_files["case_dir"], [
        {"statute_id": "420", "name": "Cheating"},
    ])

    result = evaluate_ingredients_v1(
        case_id="test_case",
        statute_candidates_path=sc_path,
        fact_graph_path=setup_files["fact_graph_path"],
        cases_dir=setup_files["cases_dir"],
    )

    with open(result["result_ref"]) as f:
        report = json.load(f)

    # Check no ingredient has element_type == "keyword_proxy"
    for ev in report["statute_evaluations"]:
        for ing in ev["ingredients"]:
            assert ing.get("element_type") != "keyword_proxy", \
                f"keyword_proxy element_type found in {ing.get('ingredient_id')}"


def test_evaluate_empty_ingredients_not_applicable(setup_files):
    """Sections with no KB ingredients → status = 'not_applicable'."""
    from mcp.tools.evaluate_ingredients import evaluate_ingredients_v1

    sc_path = _make_statute_candidates(setup_files["case_dir"], [
        {"statute_id": "999", "name": "Test section with no ingredients"},
    ])

    result = evaluate_ingredients_v1(
        case_id="test_case",
        statute_candidates_path=sc_path,
        fact_graph_path=setup_files["fact_graph_path"],
        cases_dir=setup_files["cases_dir"],
    )

    with open(result["result_ref"]) as f:
        report = json.load(f)

    ev = report["statute_evaluations"][0]
    assert ev["status"] == "not_applicable"
    assert ev["score"] == 0.0


def test_evaluate_has_charge_status(setup_files):
    """Each statute evaluation must have status and reason fields."""
    from mcp.tools.evaluate_ingredients import evaluate_ingredients_v1

    sc_path = _make_statute_candidates(setup_files["case_dir"], [
        {"statute_id": "420", "name": "Cheating"},
    ])

    result = evaluate_ingredients_v1(
        case_id="test_case",
        statute_candidates_path=sc_path,
        fact_graph_path=setup_files["fact_graph_path"],
        cases_dir=setup_files["cases_dir"],
    )

    with open(result["result_ref"]) as f:
        report = json.load(f)

    for ev in report["statute_evaluations"]:
        assert "status" in ev, "Missing 'status' in evaluation"
        assert ev["status"] in ("strong", "plausible", "weak", "not_applicable")
        assert "reason" in ev, "Missing 'reason' in evaluation"
        assert "score" in ev


def test_evaluate_ingredients_core_marked(setup_files):
    """Core ingredients must be flagged with is_core=True."""
    from mcp.tools.evaluate_ingredients import evaluate_ingredients_v1

    sc_path = _make_statute_candidates(setup_files["case_dir"], [
        {"statute_id": "420", "name": "Cheating"},
    ])

    result = evaluate_ingredients_v1(
        case_id="test_case",
        statute_candidates_path=sc_path,
        fact_graph_path=setup_files["fact_graph_path"],
        cases_dir=setup_files["cases_dir"],
    )

    with open(result["result_ref"]) as f:
        report = json.load(f)

    ev = report["statute_evaluations"][0]
    core_found = False
    for ing in ev["ingredients"]:
        assert "is_core" in ing
        if ing["element_type"] in ("actus_reus", "mens_rea"):
            assert ing["is_core"] is True
            core_found = True
    assert core_found, "Expected at least one core ingredient marked"


def test_evaluate_ingredients_no_llm(setup_files):
    """No LLM → deterministic evaluation only."""
    from mcp.tools.evaluate_ingredients import evaluate_ingredients_v1

    sc_path = _make_statute_candidates(setup_files["case_dir"], [
        {"statute_id": "420", "name": "Cheating"},
    ])

    result = evaluate_ingredients_v1(
        case_id="test_case",
        statute_candidates_path=sc_path,
        fact_graph_path=setup_files["fact_graph_path"],
        cases_dir=setup_files["cases_dir"],
        _gemini_caller=None,
    )

    with open(result["result_ref"]) as f:
        report = json.load(f)

    for ev in report["statute_evaluations"]:
        for ing in ev["ingredients"]:
            assert ing["final"]["source"] == "deterministic"


def test_evaluate_ingredients_llm_failure(setup_files):
    """LLM returns garbage → deterministic fallback."""
    from mcp.tools.evaluate_ingredients import evaluate_ingredients_v1

    sc_path = _make_statute_candidates(setup_files["case_dir"], [
        {"statute_id": "420", "name": "Cheating"},
    ])

    def bad_gemini(system_prompt, user_json, **kw):
        return "INVALID JSON!!!"

    result = evaluate_ingredients_v1(
        case_id="test_case",
        statute_candidates_path=sc_path,
        fact_graph_path=setup_files["fact_graph_path"],
        cases_dir=setup_files["cases_dir"],
        _gemini_caller=bad_gemini,
    )

    with open(result["result_ref"]) as f:
        report = json.load(f)

    for ev in report["statute_evaluations"]:
        for ing in ev["ingredients"]:
            assert ing["final"]["source"] == "deterministic"
