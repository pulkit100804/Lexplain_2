"""
Lexplain — Deterministic semantic similarity engine.
MIT License | See README for MCP provenance contract.

Provides TF-IDF char-n-gram + stemming + legal-synonym expansion + fuzzy
matching utilities.  NO network calls, NO ML model downloads.  Everything
runs in-process with pure Python + rapidfuzz.

Public API
----------
build_corpus_index(docs)  — build TF-IDF vectors for a list of documents
score_query(query, index) — score *query* against every doc in the index
score_pair(text_a, text_b) — score two texts against each other
extract_evidence_snippets(query, full_text, top_k) — best matching sentences
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Optional: rapidfuzz  (graceful degradation if missing)
# ---------------------------------------------------------------------------
try:
    from rapidfuzz import fuzz as _rfuzz  # type: ignore
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False

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


def _simple_stem(word: str) -> str:
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
# Legal synonym expansion (generic, NOT per-section)
# ---------------------------------------------------------------------------
_LEGAL_SYNONYMS: Dict[str, List[str]] = {
    # property / financial
    "cheat": ["fraud", "deceive", "misrepresent", "defraud", "swindle"],
    "fraud": ["cheat", "deceive", "defraud", "misrepresent"],
    "theft": ["steal", "larceny", "pilfer", "rob"],
    "steal": ["theft", "larceny"],
    "rob": ["robbery", "theft", "loot", "plunder"],
    "extort": ["blackmail", "coerce", "intimidate"],
    # violence
    "murder": ["homicide", "kill", "slay"],
    "kill": ["murder", "slay", "homicide"],
    "hurt": ["injure", "harm", "wound", "assault"],
    "assault": ["attack", "hurt", "batter"],
    "grievous": ["serious", "severe", "grave"],
    # threats / coercion
    "threat": ["threaten", "intimidate", "menace", "coerce"],
    "intimidate": ["threaten", "menace", "coerce", "frighten"],
    "coerce": ["compel", "force", "pressure", "intimidate"],
    # sexual offences
    "rape": ["sexual assault", "outrage", "ravish"],
    "outrage": ["violate", "molest"],
    "modesty": ["dignity", "honour", "honor"],
    # forgery / documents
    "forge": ["fabricate", "counterfeit", "falsify"],
    "counterfeit": ["forge", "fake", "falsify"],
    "document": ["instrument", "record", "writing"],
    # intent / knowledge
    "dishonest": ["fraudulent", "wrongful", "unlawful"],
    "intention": ["intent", "purpose", "design", "motive"],
    "knowledge": ["awareness", "knowing"],
    "negligent": ["careless", "reckless", "negligence"],
    # property
    "property": ["asset", "possession", "belonging", "estate"],
    "deliver": ["transfer", "hand over", "convey"],
    "possess": ["hold", "own", "retain", "custody"],
    # persons / roles
    "accused": ["defendant", "offender", "perpetrator"],
    "complainant": ["victim", "aggrieved", "injured party"],
    "abetment": ["abet", "instigate", "aid", "assist"],
    # general legal
    "offence": ["crime", "offense", "violation", "wrongdoing"],
    "punishment": ["penalty", "sentence", "fine", "imprisonment"],
    "imprison": ["jail", "incarcerate", "confine", "detain"],
    "wrongful": ["illegal", "unlawful", "illicit"],
    "alarm": ["fear", "apprehension", "fright"],
}


def _expand_with_synonyms(tokens: List[str]) -> List[str]:
    """Expand a token list with legal synonyms (deduplicated)."""
    expanded: list[str] = list(tokens)
    seen = set(tokens)
    for tok in tokens:
        for syn in _LEGAL_SYNONYMS.get(tok, []):
            stem = _simple_stem(syn)
            if stem not in seen:
                expanded.append(stem)
                seen.add(stem)
    return expanded


# ---------------------------------------------------------------------------
# Tokenisation helpers
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


def tokenise(text: str, *, stem: bool = True, remove_stops: bool = True) -> List[str]:
    """Tokenise, lower, optionally stem and remove stop-words."""
    raw = _TOKEN_RE.findall(text.lower())
    if remove_stops:
        raw = [t for t in raw if t not in _STOPWORDS]
    if stem:
        raw = [_simple_stem(t) for t in raw]
    return raw


def _char_ngrams(text: str, ns: Sequence[int] = (3, 4, 5)) -> Counter:
    """Return a Counter of character n-grams of the given sizes."""
    text = re.sub(r"\s+", " ", text.lower().strip())
    counter: Counter = Counter()
    for n in ns:
        for i in range(len(text) - n + 1):
            counter[text[i : i + n]] += 1
    return counter


# ---------------------------------------------------------------------------
# TF-IDF index
# ---------------------------------------------------------------------------
@dataclass
class _TfIdfIndex:
    """Lightweight TF-IDF index over a corpus of documents."""
    doc_vectors: List[Dict[str, float]] = field(default_factory=list)
    idf: Dict[str, float] = field(default_factory=dict)
    doc_norms: List[float] = field(default_factory=list)
    raw_tokens: List[List[str]] = field(default_factory=list)


def build_corpus_index(
    documents: List[str],
    *,
    use_synonyms: bool = True,
    use_ngrams: bool = True,
) -> _TfIdfIndex:
    """Build a TF-IDF index from a list of document strings."""
    idx = _TfIdfIndex()
    n_docs = len(documents)
    if n_docs == 0:
        return idx

    # Tokenise + optionally expand
    all_tokens: List[List[str]] = []
    for doc in documents:
        toks = tokenise(doc)
        if use_synonyms:
            toks = _expand_with_synonyms(toks)
        if use_ngrams:
            # add char n-grams as extra tokens (prefixed to avoid collision)
            for ng, cnt in _char_ngrams(doc).items():
                toks.extend([f"_ng_{ng}"] * min(cnt, 3))
        all_tokens.append(toks)
    idx.raw_tokens = all_tokens

    # DF
    df: Counter = Counter()
    for toks in all_tokens:
        df.update(set(toks))

    # IDF (smoothed)
    idx.idf = {
        term: math.log((1 + n_docs) / (1 + freq)) + 1.0
        for term, freq in df.items()
    }

    # TF-IDF vectors + norms
    for toks in all_tokens:
        tf = Counter(toks)
        vec = {t: (1 + math.log(c)) * idx.idf.get(t, 1.0) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        idx.doc_vectors.append(vec)
        idx.doc_norms.append(norm)

    return idx


def _cosine(vec_a: Dict[str, float], norm_a: float,
            vec_b: Dict[str, float], norm_b: float) -> float:
    """Cosine similarity between two sparse vectors."""
    dot = sum(vec_a.get(k, 0.0) * v for k, v in vec_b.items())
    denom = norm_a * norm_b
    return dot / denom if denom > 0 else 0.0


def score_query(
    query: str,
    index: _TfIdfIndex,
    *,
    use_synonyms: bool = True,
    use_ngrams: bool = True,
    fuzzy_weight: float = 0.3,
) -> List[Dict[str, Any]]:
    """Score *query* against every document in the index.

    Returns a list (same order as docs) of
    {"score": float, "matched_terms": list[str], "method": str}.
    """
    if not index.doc_vectors:
        return []

    # Build query vector
    q_toks = tokenise(query)
    if use_synonyms:
        q_toks = _expand_with_synonyms(q_toks)
    if use_ngrams:
        for ng, cnt in _char_ngrams(query).items():
            q_toks.extend([f"_ng_{ng}"] * min(cnt, 3))

    q_tf = Counter(q_toks)
    q_vec = {t: (1 + math.log(c)) * index.idf.get(t, 1.0) for t, c in q_tf.items()}
    q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0

    results: List[Dict[str, Any]] = []
    for i, (d_vec, d_norm) in enumerate(zip(index.doc_vectors, index.doc_norms)):
        tfidf_score = _cosine(q_vec, q_norm, d_vec, d_norm)

        # Fuzzy token-level matching (if rapidfuzz available)
        fuzzy_score = 0.0
        if _HAS_RAPIDFUZZ and index.raw_tokens[i]:
            # get set of non-ngram tokens from doc
            doc_str = " ".join(t for t in index.raw_tokens[i] if not t.startswith("_ng_"))
            query_str = " ".join(t for t in q_toks if not t.startswith("_ng_"))
            if doc_str and query_str:
                fuzzy_score = _rfuzz.token_sort_ratio(query_str, doc_str) / 100.0

        combined = (1.0 - fuzzy_weight) * tfidf_score + fuzzy_weight * fuzzy_score

        # Matched terms = intersection of non-ngram query tokens and doc tokens
        q_set = {t for t in q_toks if not t.startswith("_ng_")}
        d_set = set(index.raw_tokens[i])
        matched = sorted(q_set & d_set)

        method = "tfidf+ngrams"
        if _HAS_RAPIDFUZZ:
            method += "+fuzzy"

        results.append({
            "score": round(min(combined, 1.0), 4),
            "matched_terms": matched,
            "method": method,
        })
    return results


# ---------------------------------------------------------------------------
# Pairwise scoring (convenience for ingredient matching)
# ---------------------------------------------------------------------------

def score_pair(
    text_a: str,
    text_b: str,
    *,
    use_synonyms: bool = True,
    fuzzy_weight: float = 0.35,
) -> Dict[str, Any]:
    """Score two texts against each other.  Returns {score, matched_terms, method}."""
    index = build_corpus_index([text_b], use_synonyms=use_synonyms)
    results = score_query(text_a, index, use_synonyms=use_synonyms, fuzzy_weight=fuzzy_weight)
    if results:
        return results[0]
    return {"score": 0.0, "matched_terms": [], "method": "none"}


# ---------------------------------------------------------------------------
# Evidence snippet extraction
# ---------------------------------------------------------------------------
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def extract_evidence_snippets(
    query: str,
    full_text: str,
    top_k: int = 3,
) -> List[str]:
    """Return the *top_k* sentences from *full_text* most similar to *query*."""
    sentences = _SENT_SPLIT.split(full_text)
    if not sentences:
        return []
    # Also split on newlines
    expanded: list[str] = []
    for s in sentences:
        for part in s.split("\n"):
            part = part.strip()
            if part:
                expanded.append(part)
    if not expanded:
        return []

    index = build_corpus_index(expanded, use_synonyms=True, use_ngrams=False)
    scores = score_query(query, index, use_synonyms=True, use_ngrams=False, fuzzy_weight=0.4)
    ranked = sorted(enumerate(scores), key=lambda x: x[1]["score"], reverse=True)
    return [expanded[i] for i, _ in ranked[:top_k]]
