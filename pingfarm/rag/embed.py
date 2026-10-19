"""Turning text into vectors.

We use a multilingual E5 model so a Hindi or Telugu question lands close to an
English passage that answers it (cross-lingual retrieval). E5 models expect a
prefix: "query: " on questions and "passage: " on documents.
"""
import numpy as np

DEFAULT_MODEL = "intfloat/multilingual-e5-small"


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def _encode(self, texts: list[str], prefix: str, batch_size: int = 32) -> np.ndarray:
        vecs = self.model.encode(
            [prefix + t for t in texts],
            batch_size=batch_size,
            normalize_embeddings=True,  # unit length => dot product == cosine
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype="float32")

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, "passage: ")

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, "query: ")
