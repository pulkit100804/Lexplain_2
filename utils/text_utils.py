"""
Lexplain — Text utility functions.
MIT License | See README for MCP provenance contract.
"""
import re
from typing import List

_ABBREVS = {
    r"\bSec\.": "Section",
    r"\bArt\.": "Article",
    r"\bCr\.P\.C\.": "CrPC",
    r"\bI\.P\.C\.": "IPC",
    r"\bCPC\.": "CPC",
    r"\bvs\.": "versus",
    r"\bv\.": "versus",
    r"\bHon'ble": "Honourable",
    r"\bdt\.": "dated",
    r"\bw\.e\.f\.": "with effect from",
    r"\bp\.a\.": "per annum",
}

_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def clean_text(text: str) -> str:
    """Unify whitespace and normalize punctuation."""
    text = text.strip()
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[''`]", "'", text)
    text = re.sub(r"[\u201c\u201d]", '"', text)
    return text


def normalize_abbreviations(text: str) -> str:
    """Replace common legal abbreviations with expanded forms."""
    for pattern, replacement in _ABBREVS.items():
        text = re.sub(pattern, replacement, text)
    return text


def split_sentences(text: str) -> List[str]:
    """Rule-based sentence/clause boundary detection for legal text."""
    # Split on ". " or "! " or "? " followed by uppercase, or newlines
    sentences: List[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        parts = _SPLIT_RE.split(para)
        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part)
    return sentences
