"""
Lexplain — Tests for utils/kb_loader.py
MIT License.

Validates:
- Dict-shaped KB loads and normalizes correctly
- get_section returns correct entries
- iter_sections yields all entries
- Missing KB returns empty dict gracefully
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from utils import kb_loader


# ── Fixtures ──────────────────────────────────────────────────────────────

SAMPLE_DICT_KB = {
    "295A": {
        "section_id": "295A",
        "heading": "Deliberate and malicious acts to outrage religious feelings",
        "canonical_text": "Whoever, with deliberate and malicious intention...",
        "ingredients": [
            {"id": "295A.1", "text": "deliberate act", "element_type": "actus_reus", "normalized": {}},
            {"id": "295A.2", "text": "outrage religious feelings", "element_type": "intent", "normalized": {}},
        ],
        "offence_category": ["religious"],
        "punishment": "Up to 3 years",
        "status": {"repealed": False, "note": None},
    },
    "300": {
        "section_id": "300",
        "heading": "Murder",
        "canonical_text": "Culpable homicide is murder if...",
        "ingredients": [
            {"id": "300.1", "text": "causes death", "element_type": "actus_reus", "normalized": {}},
        ],
        "offence_category": ["against body"],
        "punishment": "Death or imprisonment for life",
        "status": {"repealed": False, "note": None},
    },
}


@pytest.fixture
def dict_kb_file(tmp_path):
    """Write a dict-shaped KB to a temp file and return path."""
    kb_path = tmp_path / "test_kb.json"
    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump(SAMPLE_DICT_KB, f)
    return str(kb_path)


@pytest.fixture(autouse=True)
def clear_kb_cache():
    """Clear the lru_cache before each test so env var changes take effect."""
    kb_loader.load_ipc_kb.cache_clear()
    yield
    kb_loader.load_ipc_kb.cache_clear()


# ── Tests ─────────────────────────────────────────────────────────────────

def test_load_dict_kb(dict_kb_file, monkeypatch):
    monkeypatch.setenv("LEXPLAIN_KB_PATH", dict_kb_file)
    kb = kb_loader.load_ipc_kb()
    assert isinstance(kb, dict)
    assert "295A" in kb
    assert "300" in kb
    assert kb["295A"]["heading"] == "Deliberate and malicious acts to outrage religious feelings"


def test_get_section(dict_kb_file, monkeypatch):
    monkeypatch.setenv("LEXPLAIN_KB_PATH", dict_kb_file)
    sec = kb_loader.get_section("300")
    assert sec is not None
    assert sec["section_id"] == "300"
    assert sec["heading"] == "Murder"


def test_get_section_missing(dict_kb_file, monkeypatch):
    monkeypatch.setenv("LEXPLAIN_KB_PATH", dict_kb_file)
    sec = kb_loader.get_section("999")
    assert sec is None


def test_iter_sections(dict_kb_file, monkeypatch):
    monkeypatch.setenv("LEXPLAIN_KB_PATH", dict_kb_file)
    all_secs = list(kb_loader.iter_sections())
    assert len(all_secs) == 2
    ids = {s["section_id"] for s in all_secs}
    assert ids == {"295A", "300"}


def test_section_ids(dict_kb_file, monkeypatch):
    monkeypatch.setenv("LEXPLAIN_KB_PATH", dict_kb_file)
    ids = kb_loader.section_ids()
    assert ids == ["295A", "300"]


def test_missing_kb_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("LEXPLAIN_KB_PATH", str(tmp_path / "nonexistent.json"))
    kb = kb_loader.load_ipc_kb()
    assert kb == {}


def test_legacy_list_kb(tmp_path, monkeypatch):
    """Verify the loader also handles the old list-shaped KB."""
    legacy = [
        {"section_id": "IPC_420", "name": "Cheating", "ingredients": []},
        {"section_id": "IPC_302", "name": "Murder", "ingredients": []},
    ]
    kb_path = tmp_path / "legacy.json"
    with open(kb_path, "w") as f:
        json.dump(legacy, f)
    monkeypatch.setenv("LEXPLAIN_KB_PATH", str(kb_path))
    kb = kb_loader.load_ipc_kb()
    assert "IPC_420" in kb
    assert "IPC_302" in kb
