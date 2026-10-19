"""Measure the "I don't know" cut-off instead of guessing it.

Idea: for every question we look at the similarity of the closest passage. Real tomato/potato
questions should sit higher than off-topic ones. We pick the cut-off so that at least 90% of real
questions are still answered, then report how many off-topic questions get (correctly) refused.

    python scripts/calibrate_abstain.py
Writes data/eval/calibration.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pingfarm.evalkit.run_eval import load_questions  # noqa: E402
from pingfarm.rag.chunking import chunk_all  # noqa: E402
from pingfarm.rag.documents import load_documents  # noqa: E402
from pingfarm.rag.embed import Embedder  # noqa: E402
from pingfarm.rag.search import Retriever  # noqa: E402

OOS = [
    ("cricket", "Who won the last cricket world cup?", "आखिरी क्रिकेट वर्ल्ड कप कौन जीता?", "చివరి క్రికెట్ ప్రపంచ కప్ ఎవరు గెలిచారు?"),
    ("capital", "What is the capital of France?", "फ्रांस की राजधानी क्या है?", "ఫ్రాన్స్ రాజధాని ఏది?"),
    ("biryani", "How do I make chicken biryani?", "चिकन बिरयानी कैसे बनाते हैं?", "చికెన్ బిర్యానీ ఎలా చేయాలి?"),
    ("gold", "What is the price of gold today?", "आज सोने का भाव क्या है?", "ఈ రోజు బంగారం ధర ఎంత?"),
    ("phone", "How can I fix a cracked mobile phone screen?", "टूटी हुई मोबाइल स्क्रीन कैसे ठीक करें?", "పగిలిన మొబైల్ స్క్రీన్ ఎలా బాగు చేయాలి?"),
    ("cattle", "My cow has a fever. What medicine should I give?", "मेरी गाय को बुखार है। कौन सी दवा दूं?", "నా ఆవుకు జ్వరం వచ్చింది. ఏ మందు ఇవ్వాలి?"),
    # near misses: farming, but not in our knowledge base (the hard cases)
    ("rice", "How do I grow rice in the monsoon season?", "मानसून में धान की खेती कैसे करें?", "వర్షాకాలంలో వరి సాగు ఎలా చేయాలి?"),
    ("cotton", "How do I control pink bollworm in cotton?", "कपास में गुलाबी सुंडी का नियंत्रण कैसे करें?", "పత్తిలో గులాబీ రంగు పురుగును ఎలా నియంత్రించాలి?"),
    ("sugarcane", "How do I control stem borer in sugarcane?", "गन्ने में तना छेदक कीट को कैसे रोकें?", "చెరకులో కాండం తొలుచు పురుగును ఎలా అరికట్టాలి?"),
    ("loan", "How do I apply for a farm loan?", "खेती के लिए कर्ज के लिए आवेदन कैसे करें?", "వ్యవసాయ రుణానికి ఎలా దరఖాస్తు చేయాలి?"),
]


def main():
    r = Retriever(chunk_all(load_documents(), "fixed", header=True, size=200, overlap=40), Embedder())
    inscope = [(q.text, q.lang) for q in load_questions()]
    out_rows = [(t, lang) for _, en, hi, te in OOS for t, lang in ((en, "en"), (hi, "hi"), (te, "te"))]
    sin = np.array([r.top_similarity(t) for t, _ in inscope])
    sout = np.array([r.top_similarity(t) for t, _ in out_rows])
    y = np.r_[np.ones(len(sin)), np.zeros(len(sout))]
    auc = roc_auc_score(y, np.r_[sin, sout])
    threshold = float(np.percentile(sin, 10))          # keeps ~90% of real questions
    result = {
        "abstain_threshold": round(threshold, 4),
        "auc_in_vs_out_of_scope": round(float(auc), 3),
        "in_scope_answered": round(float((sin >= threshold).mean()), 3),
        "out_of_scope_refused": round(float((sout < threshold).mean()), 3),
        "n_in_scope": len(sin), "n_out_of_scope": len(sout),
        "in_scope_sim_mean": round(float(sin.mean()), 3), "out_of_scope_sim_mean": round(float(sout.mean()), 3),
        "per_language_refused": {
            lang: round(float(np.mean([s < threshold for s, (_, l) in zip(sout, out_rows) if l == lang])), 3)
            for lang in ("en", "hi", "te")
        },
    }
    out = ROOT / "data" / "eval" / "calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
