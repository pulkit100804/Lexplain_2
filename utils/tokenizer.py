"""
Lexplain — Lightweight tokeniser + stemmer.
MIT License | See README for MCP provenance contract.

Pure-Python text normalisation used by the deterministic keyword matcher.
No external dependencies.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List

# ---------------------------------------------------------------------------
# Simple rule-based Porter-style stemmer  (no NLTK needed)
# ---------------------------------------------------------------------------
_STEP2_SUFFIXES = [
    ("ational", "ate"), ("tional", "tion"), ("enci", "ence"),
    ("anci", "ance"), ("izer", "ize"), ("iser", "ise"),
    ("abli", "able"), ("alli", "al"), ("entli", "ent"),
    ("eli", "e"), ("ousli", "ous"), ("ization", "ize"),
    ("isation", "ise"), ("ation", "ate"), ("ator", "ate"),
    ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
    ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"),
    ("biliti", "ble"),
]

_STEP3_SUFFIXES = [
    ("icate", "ic"), ("ative", ""), ("alize", "al"),
    ("alise", "al"), ("iciti", "ic"), ("ical", "ic"),
    ("ful", ""), ("ness", ""),
]


def stem(word: str) -> str:
    """Very lightweight Porter-like stemmer (English-only)."""
    if len(word) <= 3:
        return word
    # Step 1: plurals / past-participle
    if word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("ies"):
        word = word[:-2]
    elif word.endswith("ss"):
        pass
    elif word.endswith("s") and len(word) > 3:
        word = word[:-1]
    if word.endswith("eed"):
        pass
    elif word.endswith("ed") and len(word) > 4:
        word = word[:-2]
    elif word.endswith("ing") and len(word) > 5:
        word = word[:-3]
    # Step 2
    for suffix, replacement in _STEP2_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)] + replacement
            break
    # Step 3
    for suffix, replacement in _STEP3_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)] + replacement
            break
    return word


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should may might can could of in on at to for "
    "with by from as into through during before after above below "
    "between out off over under again further then once that this "
    "these those am its it he she they we you i my your his her our "
    "their what which who whom how when where why all each every both "
    "few more most other some such no nor not only own same so than "
    "too very and but if or because until while about against up down "
    "also just".split()
)


def tokenize(text: str) -> List[str]:
    """Tokenise → lowercase → remove stopwords → stem."""
    raw = _TOKEN_RE.findall(text.lower())
    return [stem(t) for t in raw if t not in _STOPWORDS]


# ---------------------------------------------------------------------------
# IDF computation
# ---------------------------------------------------------------------------

def compute_idf(documents_tokens: List[List[str]]) -> Dict[str, float]:
    """Compute IDF scores across a collection of tokenized documents."""
    n_docs = len(documents_tokens)
    if n_docs == 0:
        return {}
    df: Counter = Counter()
    for tokens in documents_tokens:
        df.update(set(tokens))
    return {
        term: math.log((1 + n_docs) / (1 + freq)) + 1.0
        for term, freq in df.items()
    }
