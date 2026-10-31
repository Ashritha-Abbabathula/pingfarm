"""Sort every real-question retrieval miss (hit@3) by what went wrong. Writes results/error_analysis.csv.

Uses the app's setup (fixed-200 chunks + header, dense, reranker, language-balanced search).
This is an automatic first sort, not a judgement: "not in the top 20" can mean a retrieval failure OR a page that
does not really answer the question, so read the printed examples before drawing conclusions.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from pingfarm.evalkit.metrics import first_relevant_rank
from pingfarm.evalkit.run_eval import load_sets
from pingfarm.rag.chunking import chunk_all
from pingfarm.rag.documents import load_documents
from pingfarm.rag.embed import Embedder
from pingfarm.rag.search import Retriever

docs = load_documents()
crops = {}
for meta in (ROOT / "data" / "raw").glob("*.meta.json"):
    m = json.loads(meta.read_text(encoding="utf-8"))
    crops[m["id"]] = set(m.get("crops", []))

real_rows = [json.loads(l) for l in (ROOT / "data" / "eval" / "real_questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
topic_of = {r["id"]: r["topic"] for r in real_rows if r["answerable"]}

retr = Retriever(chunk_all(docs, "fixed", header=True, size=200, overlap=40), Embedder())
rows = []
for q in load_sets()["real"]:
    hits = retr.search(q.text, mode="dense", k=20, rerank=True, balance_languages=True)
    rank = first_relevant_rank(hits, q)  # rank within the top 20, or None
    top3 = [h.chunk.doc_id for h in hits[:3]]
    asked = {c for c in ("tomato", "potato") if re.search(c, q.text.lower())}
    top1_crops = crops.get(top3[0], set())
    wrong_crop = bool(asked) and bool(top1_crops) and not (asked & top1_crops)
    if rank is not None and rank <= 3:
        kind = "hit"
    elif rank is not None and rank <= 5:
        kind = "near miss (rank 4-5)"
    elif rank is not None:
        kind = "ranked low (rank 6-20)"
    else:
        kind = "not in top 20"
    rows.append({"id": q.id, "topic": topic_of.get(q.id), "text": q.text, "accepted": ";".join(q.doc_ids), "rank": rank,
                 "kind": kind, "wrong_crop_top1": wrong_crop, "top3": ";".join(top3)})

df = pd.DataFrame(rows)
df.to_csv(ROOT / "results" / "error_analysis.csv", index=False)
pd.set_option("display.width", 200)
print("questions:", len(df))
print(df["kind"].value_counts().to_string(), "\n")
miss = df[df["kind"] != "hit"]
print("misses:", len(miss), "| of these the top-1 page was for the WRONG crop:", int(miss["wrong_crop_top1"].sum()))
print("\nmisses by topic:")
print(miss.groupby("topic").size().sort_values(ascending=False).to_string())
print("\nmisses by kind and topic (top rows):")
print(miss.groupby(["kind", "topic"]).size().sort_values(ascending=False).head(15).to_string())
print("\nexamples of 'not in top 20':")
for _, r in miss[miss["kind"] == "not in top 20"].head(12).iterrows():
    print(f"  [{r['topic']}] {r['text'][:70]!r} accepted={r['accepted']} got={r['top3']}")
