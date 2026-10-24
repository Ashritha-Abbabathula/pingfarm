"""Download the knowledge-base documents listed in data/sources.json.

Each document is saved as plain text in data/raw/<id>.txt next to a
<id>.meta.json file (title, org, url). We keep only the list of sources in git,
not the downloaded text, so we never re-publish other people's documents.

Run:  python scripts/fetch_corpus.py
"""
import json
import re
import sys
import time
import urllib.request
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
UA = {"User-Agent": "Mozilla/5.0 (PingFarm student project; contact: check4ashritha@gmail.com)"}


class _TextExtractor(HTMLParser):
    """Collects visible text, one block per line, skipping scripts and styles."""

    SKIP = {"script", "style", "noscript", "head"}
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "table", "ul", "ol"}

    def __init__(self):
        super().__init__()
        self.parts, self._skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self._skip = max(0, self._skip - 1)
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(raw: bytes) -> str:
    parser = _TextExtractor()
    parser.feed(raw.decode("utf-8", errors="replace"))
    text = "".join(parser.parts)
    lines = [re.sub(r"[ \t\xa0]+", " ", ln).strip() for ln in text.splitlines()]
    # Drop website menus/footers: they would pollute search results.
    noise = ("Home |", "©", "Photo courtesy", "Acknowledgement", "Updated on",
             "http", "Source of Images", "Source of Information")
    return "\n".join(ln for ln in lines if ln and not ln.startswith(noise))


def vikaspedia_to_text(raw: bytes) -> str:
    """Vikaspedia is a JavaScript (Next.js) site: the real article sits in JSON inside the page."""
    page = raw.decode("utf-8", errors="replace")
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', page, flags=re.S)
    if not m:
        raise ValueError("no embedded page data found")
    props = json.loads(m.group(1))["props"]["pageProps"]["ssrPageContent"]
    title = (props.get("title") or "").strip()
    body = html_to_text((props.get("content") or "").encode("utf-8"))
    return (title + "\n" + body).strip()


def pdf_to_text(raw: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(raw))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n\n".join(pages)


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    sources = json.loads((ROOT / "data" / "sources.json").read_text(encoding="utf-8"))
    failed = 0
    for src in sources:
        try:
            req = urllib.request.Request(src["url"], headers=UA)
            data = None
            for attempt in range(5):  # the network sometimes fails certificate checks briefly: wait, never disable checks
                try:
                    data = urllib.request.urlopen(req, timeout=60).read()
                    break
                except Exception:
                    time.sleep(3 * (attempt + 1))
            if data is None:
                raise RuntimeError("download kept failing")
            kind = src["type"]
            text = pdf_to_text(data) if kind == "pdf" else vikaspedia_to_text(data) if kind == "vikaspedia" else html_to_text(data)
            (RAW / f"{src['id']}.txt").write_text(text, encoding="utf-8")
            (RAW / f"{src['id']}.meta.json").write_text(
                json.dumps(src, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"ok    {src['id']:32s} {len(text):>8,d} chars")
        except Exception as exc:  # keep going; report at the end
            failed += 1
            print(f"FAIL  {src['id']}: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
