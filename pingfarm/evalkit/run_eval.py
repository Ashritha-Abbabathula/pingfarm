"""Run the retrieval experiments and print a results table.

    python -m pingfarm.evalkit.run_eval                # all configurations
    python -m pingfarm.evalkit.run_eval --quick        # skip the slow reranker

Every row = one configuration: (chunking, retriever, reranker). For each we report
Hit@1, Hit@3, Hit@5 and MRR, overall and per question language.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from pingfarm.evalkit.metrics import Question, first_relevant_rank, hit_at_k, mrr
from pingfarm.rag.chunking import chunk_all
from pingfarm.rag.documents import load_documents
from pingfarm.rag.embed import DEFAULT_MODEL, Embedder
from pingfarm.rag.search import Retriever

ROOT = Path(__file__).resolve().parents[2]


def load_questions(path: Path = ROOT / "data" / "eval" / "questions.jsonl") -> list[Question]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [Question(r["id"], r["lang"], r["text"], r["doc_ids"], r["evidence"]) for r in rows]


def evaluate(retriever: Retriever, questions, mode: str, rerank: bool, k: int = 5) -> pd.DataFrame:
    """One row per question: the rank at which the first relevant chunk appeared."""
    out = []
    for q in questions:
        hits = retriever.search(q.text, mode=mode, k=k, rerank=rerank)
        out.append({"id": q.id, "lang": q.lang, "rank": first_relevant_rank(hits, q)})
    return pd.DataFrame(out)


def summarize(per_question: pd.DataFrame) -> dict:
    ranks = [None if pd.isna(r) else int(r) for r in per_question["rank"]]
    return {
        "hit@1": hit_at_k(ranks, 1),
        "hit@3": hit_at_k(ranks, 3),
        "hit@5": hit_at_k(ranks, 5),
        "mrr": mrr(ranks),
    }


def load_sets() -> dict[str, list[Question]]:
    """starter: 39 hand-written questions (en/hi/te), evidence-phrase labels.
    real: farmer questions from the Kisan Call Centre that our knowledge base can answer, document-level labels."""
    sets = {"starter": load_questions()}
    real_path = ROOT / "data" / "eval" / "real_questions.jsonl"
    if real_path.exists():
        rows = [json.loads(l) for l in real_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        sets["real"] = [Question(r["id"], r["lang"], r["text"], r["doc_ids"], r["evidence"]) for r in rows if r["answerable"]]
    return sets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip reranker runs")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="retrieval_experiments.csv")
    args = ap.parse_args()

    docs = load_documents()
    sets = load_sets()
    embedder = Embedder(args.model)
    print(f"{len(docs)} documents; " + ", ".join(f"{k}: {len(v)} questions" for k, v in sets.items()) + f"; embedder={args.model}\n")

    chunkings = {
        "fixed-200": ("fixed", {"size": 200, "overlap": 40}),
        "fixed-200+header": ("fixed", {"size": 200, "overlap": 40, "header": True}),
        "section-200+header": ("section", {"max_words": 200, "header": True}),
    }
    rows = []
    for cname, (strategy, kw) in chunkings.items():
        chunks = chunk_all(docs, strategy, **kw)
        retriever = Retriever(chunks, embedder)
        for mode in ("bm25", "dense", "hybrid"):
            for rerank in ([False] if args.quick else [False, True]):
                for set_name, questions in sets.items():
                    per_q = evaluate(retriever, questions, mode, rerank)
                    row = {"question_set": set_name, "n_questions": len(questions), "chunking": cname,
                           "n_chunks": len(chunks), "retriever": mode, "rerank": rerank, **summarize(per_q)}
                    for lang in ("en", "hi", "te"):
                        sub = per_q[per_q["lang"] == lang]
                        row[f"hit@3_{lang}"] = summarize(sub)["hit@3"] if len(sub) else None
                    rows.append(row)
                # save after EVERY configuration: a long run that is interrupted must not lose its finished work
                (ROOT / "results").mkdir(exist_ok=True)
                pd.DataFrame(rows).round(3).to_csv(ROOT / "results" / args.out, index=False)
                print(f"done {cname:20s} {mode:7s} rerank={rerank}", flush=True)

    df = pd.DataFrame(rows).round(3)
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results" / args.out, index=False)
    pd.set_option("display.width", 220)
    print("\n", df.to_string(index=False))


if __name__ == "__main__":
    main()
