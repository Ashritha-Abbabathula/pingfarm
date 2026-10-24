"""How we judge retrieval.

A question lists the document that holds the answer (`doc_id`) and one or more
`evidence` phrases that must appear in the passage. A retrieved chunk is
RELEVANT if it comes from that document AND contains at least one evidence
phrase. Because we match on text (not chunk numbers), the same questions can
be used to compare different chunking strategies fairly.
"""
from dataclasses import dataclass


@dataclass
class Question:
    id: str
    lang: str
    text: str
    doc_ids: list[str]
    evidence: list[str]


def is_relevant(chunk_doc_id: str, chunk_text: str, q: Question) -> bool:
    if chunk_doc_id not in q.doc_ids:
        return False
    if not q.evidence:  # document-level label (used for the real KCC questions)
        return True
    low = " ".join(chunk_text.lower().split())  # ignore line breaks / double spaces
    return any(" ".join(e.lower().split()) in low for e in q.evidence)


def first_relevant_rank(hits, q: Question) -> int | None:
    """1-based rank of the first relevant hit, or None."""
    for rank, h in enumerate(hits, start=1):
        if is_relevant(h.chunk.doc_id, h.chunk.text, q):
            return rank
    return None


def hit_at_k(ranks: list[int | None], k: int) -> float:
    return sum(1 for r in ranks if r is not None and r <= k) / max(len(ranks), 1)


def mrr(ranks: list[int | None]) -> float:
    """Mean Reciprocal Rank: average of 1/rank (0 when nothing relevant was found)."""
    return sum((1.0 / r) if r else 0.0 for r in ranks) / max(len(ranks), 1)
