"""
Lexplain — Smoke test: runs the full pipeline with a short sample case.
MIT License | See README for MCP provenance contract.

Run with:  pytest tests/test_pipeline_smoke.py -v
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure project root is on path
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

SAMPLE_CASE = (
    "The complainant alleged that the accused cheated him of Rs. 2,00,000 by making "
    "false representations. The accused dishonestly induced the complainant to transfer "
    "the amount. The accused also threatened the complainant when he asked for the money back."
)

EXPECTED_FILES = [
    "raw.txt",
    "metadata.json",
    "normalized_text.txt",
    "normalized_ref.json",
    "document_graph.json",
    "role_tagged_graph.json",
    "fact_event_graph.json",
    "statute_candidates.json",
    "ingredient_report.json",
    "precedent_matches.json",
    "comparator_report.json",
    "loophole_map.json",
    "argument_package.json",
]


@pytest.fixture(scope="module")
def pipeline_output(tmp_path_factory: pytest.TempPathFactory):
    """Run the full pipeline and return (case_id, case_dir, cases_dir)."""
    cases_dir = str(tmp_path_factory.mktemp("cases"))

    import agents.agent_0_ingest as a0
    import agents.agent_1_normalize as a1
    import agents.agent_2_segmentation as a2
    import agents.agent_3_role_tagging as a3
    import agents.agent_4_fact_graph as a4
    import agents.agent_5_statute_candidates as a5
    import agents.agent_6_ingredient_engine as a6
    import agents.agent_7_retrieval as a7
    import agents.agent_8_comparator as a8
    import agents.agent_9_loophole_miner as a9
    import agents.agent_10_argument_generator as a10

    r0 = a0.run(SAMPLE_CASE, cases_dir)
    case_id = r0["case_id"]
    case_dir = r0["case_dir"]

    a1.run(case_id, case_dir, cases_dir)
    a2.run(case_id, case_dir, cases_dir)
    a3.run(case_id, case_dir, cases_dir)
    a4.run(case_id, case_dir, cases_dir)
    a5.run(case_id, case_dir, cases_dir)
    a6.run(case_id, case_dir, cases_dir)
    a7.run(case_id, case_dir, cases_dir)
    a8.run(case_id, case_dir, cases_dir)
    a9.run(case_id, case_dir, cases_dir)
    a10.run(case_id, case_dir, cases_dir, role="both")

    return case_id, case_dir, cases_dir


def test_all_files_created(pipeline_output):
    _, case_dir, _ = pipeline_output
    for fname in EXPECTED_FILES:
        fpath = os.path.join(case_dir, fname)
        assert os.path.exists(fpath), f"Expected file not found: {fname}"


def test_metadata_structure(pipeline_output):
    case_id, case_dir, _ = pipeline_output
    with open(os.path.join(case_dir, "metadata.json")) as f:
        meta = json.load(f)
    assert meta["case_id"] == case_id
    assert meta["tenant_id"] == "local_dev"
    assert "provenance" in meta


def test_document_graph_structure(pipeline_output):
    _, case_dir, _ = pipeline_output
    with open(os.path.join(case_dir, "document_graph.json")) as f:
        doc_graph = json.load(f)
    assert "nodes" in doc_graph
    assert "edges" in doc_graph
    assert doc_graph["node_count"] > 0
    assert all("node_id" in n and "text" in n for n in doc_graph["nodes"])


def test_role_tagged_graph_structure(pipeline_output):
    _, case_dir, _ = pipeline_output
    with open(os.path.join(case_dir, "role_tagged_graph.json")) as f:
        tagged = json.load(f)
    assert "nodes" in tagged
    assert all("role_tag" in n and "confidence" in n for n in tagged["nodes"])


def test_fact_graph_structure(pipeline_output):
    _, case_dir, _ = pipeline_output
    with open(os.path.join(case_dir, "fact_event_graph.json")) as f:
        fg = json.load(f)
    assert "events" in fg
    assert "actors" in fg
    assert "provenance" in fg


def test_statute_candidates(pipeline_output):
    _, case_dir, _ = pipeline_output
    with open(os.path.join(case_dir, "statute_candidates.json")) as f:
        sc = json.load(f)
    assert "candidates" in sc
    # Sample case mentions cheat/fraud/threat → should trigger at least IPC_420 and IPC_506
    statute_ids = [c["statute_id"] for c in sc["candidates"]]
    assert "IPC_420" in statute_ids, f"Expected IPC_420 in {statute_ids}"
    assert "IPC_506" in statute_ids, f"Expected IPC_506 in {statute_ids}"


def test_ingredient_report(pipeline_output):
    _, case_dir, _ = pipeline_output
    with open(os.path.join(case_dir, "ingredient_report.json")) as f:
        ir = json.load(f)
    assert "statute_evaluations" in ir
    assert len(ir["statute_evaluations"]) > 0
    for ev in ir["statute_evaluations"]:
        assert "statute_id" in ev
        assert "ingredients" in ev
        assert "overall_score" in ev


def test_precedent_matches_empty(pipeline_output):
    _, case_dir, _ = pipeline_output
    with open(os.path.join(case_dir, "precedent_matches.json")) as f:
        pm = json.load(f)
    assert pm["matches"] == [], "RAG placeholder should return empty list"
    assert "note" in pm


def test_comparator_report(pipeline_output):
    _, case_dir, _ = pipeline_output
    with open(os.path.join(case_dir, "comparator_report.json")) as f:
        cr = json.load(f)
    assert "prosecution_viability" in cr
    assert "defense_viability" in cr
    assert 0.0 <= cr["prosecution_viability"] <= 1.0
    assert 0.0 <= cr["defense_viability"] <= 1.0


def test_loophole_map(pipeline_output):
    _, case_dir, _ = pipeline_output
    with open(os.path.join(case_dir, "loophole_map.json")) as f:
        lm = json.load(f)
    assert "missing_evidence" in lm
    assert "contradictory_nodes" in lm
    assert "weak_ingredients" in lm
    assert "recommended_next_actions" in lm


def test_argument_package_structure(pipeline_output):
    _, case_dir, _ = pipeline_output
    with open(os.path.join(case_dir, "argument_package.json")) as f:
        ap = json.load(f)
    assert "case_id" in ap
    assert "arguments" in ap
    assert "viability_assessment" in ap
    assert "provenance" in ap
    # role_requested must match what was passed to agent_10 ('both')
    assert ap["role_requested"] == "both", f"Expected role_requested='both', got {ap.get('role_requested')}"
    # Must have at least one argument (fallback or Gemini)
    assert len(ap["arguments"]) > 0
    for arg in ap["arguments"]:
        assert "id" in arg
        assert "title" in arg
        assert "text" in arg


def test_provenance_in_all_json(pipeline_output):
    _, case_dir, _ = pipeline_output
    json_files_needing_provenance = [
        "metadata.json", "document_graph.json", "role_tagged_graph.json",
        "fact_event_graph.json", "statute_candidates.json", "ingredient_report.json",
        "precedent_matches.json", "comparator_report.json", "loophole_map.json",
        "argument_package.json",
    ]
    for fname in json_files_needing_provenance:
        with open(os.path.join(case_dir, fname)) as f:
            data = json.load(f)
        assert "provenance" in data, f"Missing provenance in {fname}"
        prov = data["provenance"]
        assert "trace_id" in prov
        assert "tool_name" in prov
        assert "timestamp" in prov
