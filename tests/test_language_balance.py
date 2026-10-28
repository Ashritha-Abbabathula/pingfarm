"""The language-balanced search and the script check (no model downloads)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pingfarm.rag.documents import Chunk
from pingfarm.rag.search import Retriever, script_language

HINDI_PAGE = "टमाटर की खेती"
ENGLISH_PAGE = "Tomato late blight management"
HINDI_QUESTION = "टमाटर में झुलसा रोग"


class FakeEmbedder:
    """Dimensions: [is_hindi, is_english, topic]. A Hindi question sits closest to the Hindi page."""

    vectors = {
        HINDI_PAGE: [1.0, 0.0, 0.5],
        ENGLISH_PAGE: [0.0, 1.0, 1.0],
        HINDI_QUESTION: [1.0, 0.0, 1.0],
    }

    def embed_passages(self, texts):
        return np.array([self.vectors[t] for t in texts], dtype="float32")

    def embed_queries(self, texts):
        return np.array([self.vectors[t] for t in texts], dtype="float32")


def _retriever():
    chunks = [Chunk("vik_hi", 0, HINDI_PAGE, "hi page", "u"), Chunk("tnau_en", 0, ENGLISH_PAGE, "en page", "u")]
    return Retriever(chunks, FakeEmbedder())


def test_script_language():
    assert script_language(HINDI_PAGE) == "hi"
    assert script_language("టమాటా ఆకు ముడత") == "te"
    assert script_language(ENGLISH_PAGE) == "en"
    assert script_language("") == "en"


def test_plain_dense_search_prefers_same_language_page():
    assert _retriever().search(HINDI_QUESTION, mode="dense", k=1)[0].chunk.doc_id == "vik_hi"


def test_balanced_search_lets_the_english_page_compete():
    hits = _retriever().search(HINDI_QUESTION, mode="dense", k=2, balance_languages=True)
    assert hits[0].chunk.doc_id == "tnau_en"
    assert {h.chunk.doc_id for h in hits} == {"vik_hi", "tnau_en"}
