"""Two ways to cut a document into chunks. Comparing them is Experiment 1.

fixed_chunks   : slide a window of N words, overlapping by M words.
section_chunks : respect the document's own structure. Split at headings and
                 blank lines, then glue small pieces together up to N words.
"""
import re

from .documents import Chunk, Document


def _words(text: str) -> list[str]:
    return text.split()


def fixed_chunks(doc: Document, size: int = 150, overlap: int = 30) -> list[Chunk]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = _words(doc.text)
    step = size - overlap
    chunks = []
    for i, start in enumerate(range(0, max(len(words), 1), step)):
        piece = words[start : start + size]
        if not piece:
            break
        chunks.append(Chunk(doc.id, i, " ".join(piece), doc.title, doc.url))
        if start + size >= len(words):
            break
    return chunks


_CONNECTORS = {"a", "an", "the", "of", "and", "or", "in", "on", "is", "are", "to", "by",
               "for", "with", "as", "at", "from", "that", "which", "be", "was", "has", "have"}

_HEADING = re.compile(r"^(?:[A-Z][A-Za-z /&,\-()]{2,60}|\d+[.)]\s+.{2,80})$")


def _blocks(text: str) -> list[str]:
    """Split text into blocks: a short title-like line starts a new block."""
    blocks, current = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        words = line.split()
        # A wrapped PDF line like "Potato is also a" must not count as a heading,
        # so we reject lines that end in a connecting word.
        looks_like_heading = (
            len(words) <= 8
            and _HEADING.match(line)
            and not line.endswith(".")
            and words[-1].lower() not in _CONNECTORS
        )
        if looks_like_heading and current:
            blocks.append(" ".join(current))
            current = []
        current.append(line)
    if current:
        blocks.append(" ".join(current))
    return blocks


def section_chunks(doc: Document, max_words: int = 200) -> list[Chunk]:
    chunks, buffer, count = [], [], 0

    def flush():
        nonlocal buffer, count
        if buffer:
            chunks.append(Chunk(doc.id, len(chunks), " ".join(buffer), doc.title, doc.url))
            buffer, count = [], 0

    for block in _blocks(doc.text):
        n = len(block.split())
        if n > max_words:  # a single huge block: cut it by words
            flush()
            words = block.split()
            for start in range(0, len(words), max_words):
                buffer = words[start : start + max_words]
                count = len(buffer)
                flush()
            continue
        if count + n > max_words:
            flush()
        buffer.append(block)
        count += n
    flush()
    return chunks


CHUNKERS = {
    "fixed": fixed_chunks,
    "section": section_chunks,
}


def chunk_all(docs: list[Document], strategy: str = "section", header: bool = False,
              **kwargs) -> list[Chunk]:
    """Chunk every document. With header=True each chunk starts with its document's
    header ("Tomato disease: Late blight ..."), so a chunk that never names its crop
    still carries that context ("contextual chunk headers")."""
    fn = CHUNKERS[strategy]
    out: list[Chunk] = []
    for d in docs:
        for c in fn(d, **kwargs):
            if header and d.header:
                c = Chunk(c.doc_id, c.chunk_id, f"{d.header}. {c.text}", c.title, c.url)
            out.append(c)
    return out
