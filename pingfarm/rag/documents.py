"""Loading the knowledge base from data/raw."""
import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    org: str
    url: str
    text: str
    header: str = ""  # short "what is this document about" label, e.g. "Tomato: Late blight"


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: int
    text: str
    title: str
    url: str


def load_documents(raw_dir: Path = DEFAULT_RAW) -> list[Document]:
    """Read every <id>.txt together with its <id>.meta.json."""
    docs = []
    for meta_path in sorted(Path(raw_dir).glob("*.meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        text_path = meta_path.with_name(meta["id"] + ".txt")
        if not text_path.exists():
            continue
        text = text_path.read_text(encoding="utf-8")
        docs.append(
            Document(
                id=meta["id"],
                title=meta["title"],
                org=meta["org"],
                url=meta["url"],
                text=text,
                header=_make_header(meta, text),
            )
        )
    return docs


def _make_header(meta: dict, text: str) -> str:
    """TNAU pages are one disease each and never repeat the crop name in the body,
    so we build 'Tomato: Late blight: Phytophthora infestans' from the page itself.
    Longer articles just use their title."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if meta["org"].startswith("Tamil Nadu") and len(lines) >= 2:
        crop = meta["crops"][0].title()
        disease = lines[1] if "::" in lines[0] else lines[0]
        return f"{crop} disease: {disease}"
    return meta["title"]
