"""Train and evaluate the learned topic gate, and compare it with the old similarity gate.

    python scripts/train_topic_gate.py

Writes models/topic_gate.json and results/topic_gate.json.

Data (all English training data; Hindi/Telugu are held out to test cross-language transfer):
  positives : real KCC tomato/potato questions the knowledge base can answer  + the starter questions (English)
  negatives : (a) real KCC tomato/potato questions on topics we do NOT cover (pests we have no page for,
                  fertiliser, weeds, storage, vague ones), (b) real KCC questions about other crops,
              (c) 30 hand-written off-topic questions (OOS), English part
Held out    : starter Hindi/Telugu questions (positives) and OOS Hindi/Telugu (negatives).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_real_questions import RULES, norm  # noqa: E402
from calibrate_abstain import OOS  # noqa: E402

from pingfarm.evalkit.run_eval import load_questions  # noqa: E402
from pingfarm.rag.chunking import chunk_all  # noqa: E402
from pingfarm.rag.documents import load_documents  # noqa: E402
from pingfarm.rag.embed import Embedder  # noqa: E402
from pingfarm.rag.search import Retriever  # noqa: E402
from pingfarm.topic_gate import TopicGate  # noqa: E402


def matched(low: str, crop: str) -> bool:
    return any(c == crop and re.search(rx, low) for rx, c, _, _ in RULES)


def main():
    emb = Embedder()
    kcc = pd.read_csv(ROOT / "data" / "kcc" / "tomato_potato.csv")
    kcc["q"] = kcc["QueryText"].map(norm)
    kcc = kcc[kcc["q"].str.len().between(12, 160)].drop_duplicates("q")
    kcc["crop2"] = kcc["Crop"].str.lower().map(lambda c: "potato" if "potato" in c else "tomato")
    kcc["covered"] = [matched(q.lower(), c) for q, c in zip(kcc["q"], kcc["crop2"])]
    other = pd.read_csv(ROOT / "data" / "kcc" / "other_crops.csv")
    other["q"] = other["QueryText"].map(norm)
    other = other[other["q"].str.len().between(12, 160)].drop_duplicates("q")
    # cap templated duplicates: at most 40 covered questions per rule would be nicer, but a simple cap is enough
    pos_kcc = kcc[kcc.covered].sample(min(300, int(kcc.covered.sum())), random_state=0)
    neg_in_crop = kcc[~kcc.covered]
    neg_other = other.sample(min(400, len(other)), random_state=0)

    starter = load_questions()
    oos_en = [en for _, en, _, _ in OOS]

    rows = []  # text, label, group, lang
    for q in pos_kcc["q"]: rows.append((q, 1, "kcc_covered", "en"))
    for q in starter:
        if q.lang == "en": rows.append((q.text, 1, "starter", "en"))
    for q in neg_in_crop["q"]: rows.append((q, 0, "kcc_tomato_potato_uncovered", "en"))
    for q in neg_other["q"]: rows.append((q, 0, "kcc_other_crop", "en"))
    for q in oos_en: rows.append((q, 0, "oos", "en"))
    # short KCC-style questions in all three languages, so the gate learns Hindi/Telugu phrasing too
    trans = json.loads((ROOT / "data" / "eval" / "gate_translations.json").read_text(encoding="utf-8"))["items"]
    for it in trans:
        for lang in ("en", "hi", "te"):
            rows.append((it[lang], it["y"], "translated_" + ("pos" if it["y"] else "neg"), lang))
    df = pd.DataFrame(rows, columns=["text", "y", "group", "lang"])
    print(df.groupby(["y", "group"]).size().to_string(), flush=True)

    X = emb.embed_queries(df["text"].tolist())
    y = df["y"].values

    # cross-validated predictions (so every English example is scored by a model that never saw it)
    oof = np.zeros(len(df))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        clf = LogisticRegression(C=2.0, class_weight="balanced", max_iter=2000).fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y, oof)
    # threshold: keep 90% of answerable questions
    is_en = (df.lang == "en").values
    thr = float(np.percentile(oof[(y == 1) & is_en], 10))
    # Hindi/Telugu scores sit lower than English ones, so they get their own cut-off (same rule: keep 90% of answerable)
    thr_indic = float(np.percentile(oof[(y == 1) & ~is_en], 10))
    print(f"\nEnglish cross-validated AUC {auc:.3f}; threshold {thr:.3f} keeps 90% of answerable questions", flush=True)
    per_group = {g: round(float((oof[(df.group == g).values] < thr).mean()), 3) for g in df.group.unique() if g not in ("kcc_covered", "starter", "translated_pos")}
    print("share REFUSED per negative group:", per_group, flush=True)

    # final model on all data
    final = LogisticRegression(C=2.0, class_weight="balanced", max_iter=2000).fit(X, y)
    gate = TopicGate(final.coef_[0].astype("float32"), float(final.intercept_[0]), thr, thr_indic)
    gate.save()

    # ---- cross-language transfer: Hindi/Telugu held-out (never trained on)
    ho_pos = [(q.text, q.lang) for q in starter if q.lang != "en"]
    ho_neg = [(t, lang) for _, _, hi, te in OOS for t, lang in ((hi, "hi"), (te, "te"))]
    Xh = emb.embed_queries([t for t, _ in ho_pos + ho_neg])
    yh = np.r_[np.ones(len(ho_pos)), np.zeros(len(ho_neg))]
    ph = np.array([gate.score(v) for v in Xh])
    auc_h = roc_auc_score(yh, ph)
    langs = np.array([l for _, l in ho_pos + ho_neg])
    hold = {"auc_hi_te": round(float(auc_h), 3),
            "threshold_indic": round(thr_indic, 4),
            "answerable_kept": round(float((ph[yh == 1] >= thr_indic).mean()), 3),
            "off_topic_refused": round(float((ph[yh == 0] < thr_indic).mean()), 3),
            "off_topic_refused_hi": round(float((ph[(yh == 0) & (langs == "hi")] < thr_indic).mean()), 3),
            "off_topic_refused_te": round(float((ph[(yh == 0) & (langs == "te")] < thr_indic).mean()), 3)}
    print("Hindi/Telugu held-out:", hold, flush=True)

    # ---- the OLD gate (closest-passage similarity) on exactly the same Hindi/Telugu + English-OOS items
    retr = Retriever(chunk_all(load_documents(), "fixed", header=True, size=200, overlap=40), emb)
    sim = np.array([retr.top_similarity(t) for t, _ in ho_pos + ho_neg])
    old_auc = roc_auc_score(yh, sim)
    old_thr = float(np.percentile(sim[yh == 1], 10))
    old = {"auc_hi_te": round(float(old_auc), 3), "off_topic_refused": round(float((sim[yh == 0] < old_thr).mean()), 3)}
    print("OLD similarity gate on the same held-out items:", old, flush=True)

    # ---- the hard case: tomato/potato questions the KB cannot answer (pests we have no page for, fertiliser...)
    res = {"n_train": int(len(df)), "english_cv_auc": round(float(auc), 3), "threshold": round(thr, 4), "threshold_indic": round(thr_indic, 4),
           "refused_per_negative_group": per_group, "held_out_hindi_telugu": hold, "old_similarity_gate": old}
    out = ROOT / "results" / "topic_gate.json"
    out.write_text(json.dumps(res, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
