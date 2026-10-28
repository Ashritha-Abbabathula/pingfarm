"""Measure the language-balanced search against normal search (writes results/language_balance.csv)."""
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pingfarm.evalkit.metrics import first_relevant_rank, hit_at_k
from pingfarm.evalkit.run_eval import load_sets
from pingfarm.rag.chunking import chunk_all
from pingfarm.rag.documents import load_documents
from pingfarm.rag.embed import Embedder
from pingfarm.rag.search import Retriever

docs = load_documents()
emb = Embedder()
sets = load_sets()
setups = {
    "fixed-200+header": ("fixed", {"size": 200, "overlap": 40, "header": True}),
    "section-200+header": ("section", {"max_words": 200, "header": True}),
}
rows = []
for cname, (strategy, kw) in setups.items():
    retr = Retriever(chunk_all(docs, strategy, **kw), emb)
    for mode in ("dense", "hybrid"):
        for rerank in (False, True):
            for balance in (False, True):
                for sname, qs in sets.items():
                    ranks = {"all": [], "en": [], "hi": [], "te": []}
                    for q in qs:
                        hits = retr.search(q.text, mode=mode, k=5, rerank=rerank, balance_languages=balance)
                        r = first_relevant_rank(hits, q)
                        ranks["all"].append(r)
                        ranks[q.lang].append(r)
                    row = {"chunking": cname, "mode": mode, "rerank": rerank, "balance": balance, "set": sname}
                    for key, rs in ranks.items():
                        row[f"hit@3_{key}"] = round(hit_at_k(rs, 3), 3) if rs else None
                    rows.append(row)
        print("done", cname, mode, flush=True)
df = pd.DataFrame(rows)
df.to_csv(ROOT / "results" / "language_balance.csv", index=False)
pd.set_option("display.width", 250)
print(df.to_string(index=False))
