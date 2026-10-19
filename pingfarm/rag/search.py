"""Finding the right chunks for a question.

Three retrievers, one interface:
  dense  : meaning-based search with embeddings + FAISS
  bm25   : keyword search (works only when query and passage share words)
  hybrid : both, merged with Reciprocal Rank Fusion (RRF)
and an optional cross-encoder reranker that re-scores the top candidates.
"""
import re
from dataclasses import dataclass

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from .documents import Chunk
from .embed import Embedder

RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


@dataclass
class Hit:
    chunk: Chunk
    score: float


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def rrf_merge(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


class Retriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder):
        self.chunks = chunks
        self.embedder = embedder
        vectors = embedder.embed_passages([c.text for c in chunks])
        self.index = faiss.IndexFlatIP(vectors.shape[1])  # exact inner-product search
        self.index.add(vectors)
        self.bm25 = BM25Okapi([_tokens(c.text) for c in chunks])
        self._reranker = None  # loaded the first time it is needed

    def _get_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(RERANK_MODEL)
        return self._reranker

    # ---- the three retrievers, each returns chunk indices best-first ----
    def _dense(self, query: str, n: int) -> list[int]:
        q = self.embedder.embed_queries([query])
        _, ids = self.index.search(q, min(n, len(self.chunks)))
        return [int(i) for i in ids[0] if i >= 0]

    def _bm25(self, query: str, n: int) -> list[int]:
        scores = self.bm25.get_scores(_tokens(query))
        return [int(i) for i in np.argsort(scores)[::-1][:n]]

    def search(self, query: str, mode: str = "hybrid", k: int = 5, rerank: bool = False,
               candidates: int = 20) -> list[Hit]:
        """Return the top-k chunks. `candidates` is how many the reranker gets to see."""
        pool = max(k, candidates) if rerank else k
        if mode == "dense":
            ranked = [(i, 0.0) for i in self._dense(query, pool)]
        elif mode == "bm25":
            ranked = [(i, 0.0) for i in self._bm25(query, pool)]
        elif mode == "hybrid":
            ranked = rrf_merge([self._dense(query, 50), self._bm25(query, 50)])[:pool]
        else:
            raise ValueError(f"unknown mode {mode!r}")

        if rerank:
            pairs = [(query, self.chunks[i].text) for i, _ in ranked]
            scores = self._get_reranker().predict(pairs)
            ranked = sorted(
                ((i, float(s)) for (i, _), s in zip(ranked, scores)),
                key=lambda t: t[1],
                reverse=True,
            )
        return [Hit(self.chunks[i], s) for i, s in ranked[:k]]
