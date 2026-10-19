"""Fast tests that need no model downloads."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pingfarm.evalkit.metrics import Question, first_relevant_rank, hit_at_k, is_relevant, mrr
from pingfarm.rag.chunking import fixed_chunks, section_chunks
from pingfarm.rag.documents import Chunk, Document
from pingfarm.rag.search import Hit, rrf_merge


def _doc(text):
    return Document("d1", "Title", "Org", "http://x", text)


def test_fixed_chunks_overlap_and_coverage():
    text = " ".join(f"w{i}" for i in range(100))
    chunks = fixed_chunks(_doc(text), size=40, overlap=10)
    assert chunks[0].text.split()[-10:] == chunks[1].text.split()[:10]
    assert "w99" in chunks[-1].text


def test_section_chunks_respect_headings_and_size():
    text = "Early Blight\nBrown spots on leaves.\nLate Blight\nWater soaked patches appear."
    chunks = section_chunks(_doc(text), max_words=8)
    assert len(chunks) == 2
    assert chunks[0].text.startswith("Early Blight")


def test_rrf_prefers_items_ranked_high_in_both_lists():
    merged = rrf_merge([[1, 2, 3], [3, 1, 2]])
    assert merged[0][0] == 1  # 1st in one list and 2nd in the other beats 3


def test_relevance_needs_right_doc_and_evidence():
    q = Question("q1", "en", "?", ["d1"], ["mancozeb"])
    assert is_relevant("d1", "Spray Mancozeb weekly", q)
    assert not is_relevant("d2", "Spray Mancozeb weekly", q)
    assert not is_relevant("d1", "Spray water", q)


def test_metrics():
    q = Question("q1", "en", "?", ["d1"], ["x"])
    hits = [Hit(Chunk("d9", 0, "x", "", ""), 0), Hit(Chunk("d1", 1, "has x", "", ""), 0)]
    assert first_relevant_rank(hits, q) == 2
    assert hit_at_k([2, None, 1], 1) == 1 / 3
    assert abs(mrr([2, None, 1]) - 0.5) < 1e-9
