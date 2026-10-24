"""Train a small intent classifier (logistic regression on the multilingual question embeddings) and compare it
with the nearest-centroid detector on held-out sets.

Intents: advice | spray_timing | price | outlook | other (anything that is not about tomato/potato farming help).

Training data:
  * the hand-written example sentences in pingfarm/intents.py (English, Hindi, Telugu, romanised)
  * real Kisan Call Centre queries:  Market Information -> price;  Weather -> outlook;
        Plant Protection about TOMATO/POTATO -> advice;  any question about OTHER crops -> other
  * gate_translations.json: off-topic and other-crop items -> other, covered tomato/potato items -> advice
Held out (never trained on): romanised 24, off-topic OOS 30, outlook 10, and 20% of the KCC pool.

    python scripts/train_intents.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_real_questions import norm  # noqa: E402
from calibrate_abstain import OOS  # noqa: E402
from eval_intents import OUTLOOK, ROMANISED  # noqa: E402

from pingfarm.evalkit.run_eval import load_questions  # noqa: E402
from pingfarm.intents import EXAMPLES_V2, IntentClassifier, IntentDetector  # noqa: E402
from pingfarm.rag.embed import Embedder  # noqa: E402


W_HAND, W_TRANS = 6.0, 3.0


def kcc_pool():
    rows = []
    for f, ours in (("tomato_potato.csv", True), ("other_crops.csv", False)):
        d = pd.read_csv(ROOT / "data" / "kcc" / f)
        d["q"] = d["QueryText"].map(norm)
        d["qt"] = d["QueryType"].astype(str).str.strip()
        d = d[d["q"].str.len().between(8, 120)].drop_duplicates("q")
        for q, qt in zip(d["q"], d["qt"]):
            if qt == "Market Information":
                rows.append((q, "price"))
            elif qt == "Weather":
                rows.append((q, "outlook"))
            elif ours and qt == "Plant Protection":
                rows.append((q, "advice"))
            elif not ours:
                rows.append((q, "other"))
    df = pd.DataFrame(rows, columns=["q", "y"]).drop_duplicates("q")
    # cap each class so no single class (other-crop questions) dominates
    return pd.concat([g.sample(min(len(g), 350), random_state=0) for _, g in df.groupby("y")])


def main():
    emb = Embedder()
    kcc = kcc_pool()
    kcc_tr, kcc_te = train_test_split(kcc, test_size=0.2, stratify=kcc["y"], random_state=0)
    trans = json.loads((ROOT / "data" / "eval" / "gate_translations.json").read_text(encoding="utf-8"))["items"]
    texts, labels, weights = [], [], []
    # hand-written examples (3 languages + romanised) are few but are the style we care about: weight them up
    # so the many short English KCC queries do not drown them out
    for intent, ex in EXAMPLES_V2.items():
        texts += ex
        labels += [intent] * len(ex)
        weights += [W_HAND] * len(ex)
    for q, y in zip(kcc_tr["q"], kcc_tr["y"]):
        texts.append(q); labels.append(y); weights.append(1.0)
    for it in trans:
        for lang in ("en", "hi", "te"):
            if it["y"] == 1:
                texts.append(it[lang]); labels.append("advice"); weights.append(W_TRANS)
            elif "tomato" not in it["en"].lower() and "potato" not in it["en"].lower():
                texts.append(it[lang]); labels.append("other"); weights.append(W_TRANS)
    print("training examples per intent:", pd.Series(labels).value_counts().to_dict(), flush=True)

    X = emb.embed_queries(texts)
    clf = LogisticRegression(C=4.0, class_weight="balanced", max_iter=3000).fit(X, labels, sample_weight=np.array(weights))
    model = IntentClassifier(list(clf.classes_), clf.coef_.astype("float32"), clf.intercept_.astype("float32"))
    model.save()

    centroid = IntentDetector(emb)  # the old approach, with the same example sentences
    trained = lambda t: model.detect(emb.embed_queries([t])[0])

    def acc(detect, items, accept=None):
        ok = 0
        for t, y in items:
            got = detect(t)
            ok += (got == y) if accept is None else (got in accept.get(y, {y}))
        return round(ok / len(items), 3)

    tests = {
        "romanised (24)": [(t, i) for i, ts in ROMANISED.items() for t in ts],
        "outlook (10)": [(t, "outlook") for t in OUTLOOK],
        "off_topic OOS (30)": [(t, "other") for _, en, hi, te in OOS for t in (en, hi, te)],
        "starter advice en/hi/te (39)": [(q.text, "advice") for q in load_questions()],
        "KCC held-out (n=%d)" % len(kcc_te): list(zip(kcc_te["q"], kcc_te["y"])),
    }
    # spray_timing and outlook both answer weather questions: a KCC "Weather" query may go to either
    accept = {"outlook": {"outlook", "spray_timing"}}
    out = {}
    for name, items in tests.items():
        out[name] = {"centroid": acc(centroid.detect, items, accept if "KCC" in name else None),
                     "trained": acc(trained, items, accept if "KCC" in name else None)}
    # per-class accuracy on the KCC held-out set for the trained model
    per = {}
    for (t, y) in tests[[k for k in tests if k.startswith("KCC")][0]]:
        per.setdefault(y, []).append(trained(t) in accept.get(y, {y}))
    out["KCC held-out per class (trained)"] = {k: round(sum(v) / len(v), 3) for k, v in per.items()}
    print(json.dumps(out, indent=2))
    (ROOT / "results" / "intents_v3.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
