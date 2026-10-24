"""Measure intent detection before (V1 examples) and after (V2 examples).

Test sets (none of these sentences are in the example lists):
  romanised : 24 hand-written sentences, Hindi and Telugu typed in English letters, 8 per intent
  kcc       : real English Kisan Call Centre queries. QueryType "Market Information" -> price;
              "Weather" -> spray_timing (the closest of our three intents); "Plant Protection" -> advice
  starter   : the 39 starter questions (advice) in English/Hindi/Telugu

    python scripts/eval_intents.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_real_questions import norm  # noqa: E402
from calibrate_abstain import OOS  # noqa: E402

from pingfarm.evalkit.run_eval import load_questions  # noqa: E402
from pingfarm.intents import EXAMPLES_V1, EXAMPLES_V2, IntentDetector  # noqa: E402
from pingfarm.rag.embed import Embedder  # noqa: E402

ROMANISED = {
    "spray_timing": [
        "kal dawa chhidakna theek rahega kya", "baarish se pehle spray kar du kya", "aaj mausam kaisa rahega, dawai dalni hai",
        "rain padithe mandu kottocha", "repati varsham unda, mandu kottala", "spray ki ee vaatavaranam sarigga undi",
        "kya aaj spray ke liye mausam sahi hai", "varsham raaka mundu pichikari cheyyala",
    ],
    "price": [
        "tamatar ka aaj ka bhav batao", "aloo mandi mein kya rate hai", "mere aloo ka kitna paisa milega",
        "tomato rate ivala entha undi", "bangaladumpa ki market dhara cheppandi", "tamata kilo entha ki ammali",
        "tamatar mandi ka daam", "aloo ka bhav aaj kal kya chal raha hai",
    ],
    "advice": [
        "tamatar ki fasal mein bimari lag gayi hai", "aloo ke patte peele pad rahe hain kya karein", "tamatar mein keede lag gaye",
        "tomato mokkalu endipotunnayi emi cheyali", "bangaladumpa aakulu nallaga maaripoyayi", "tomato lo purugu mandu ela vadali",
        "aloo mein jhulsa rog aa gaya hai", "tomato pantalo ela rogam vachindi",
    ],
}


OUTLOOK = [
    "what should I do for my tomato crop this week with the weather forecast",
    "will the weather this week cause blight in my potato",
    "should I plant tomato or potato now",
    "is it a good time to grow potato in my soil",
    "इस हफ्ते के मौसम में मेरी आलू की फसल को क्या खतरा है",
    "अभी कौन सी फसल लगाना सही रहेगा",
    "ఈ వారం వాతావరణంలో నా టమాటా పంటకు ఏదైనా ప్రమాదం ఉందా",
    "ఇప్పుడు ఏ పంట వేయడం మంచిది",
    "is hafte ke mausam mein meri fasal ko kya khatra hai",
    "ee vaaram vaatavaranam tho naa pantaku ibbandi untunda",
]


def kcc_sets():
    out = []
    for f in ("tomato_potato.csv", "other_crops.csv"):
        d = pd.read_csv(ROOT / "data" / "kcc" / f)
        d["q"] = d["QueryText"].map(norm)
        d["qt"] = d["QueryType"].astype(str).str.strip()
        d = d[d["q"].str.len().between(8, 120)].drop_duplicates("q")
        for qt, intent, n in (("Market Information", "price", 60), ("Weather", "spray_timing", 60), ("Plant Protection", "advice", 60)):
            sub = d[d["qt"] == qt]
            out += [(q, intent) for q in sub.sample(min(n, len(sub)), random_state=0)["q"]]
    return out


def main():
    emb = Embedder()
    tests = {
        "romanised": [(t, i) for i, ts in ROMANISED.items() for t in ts],
        "kcc_real_english": kcc_sets(),
        "starter_en_hi_te": [(q.text, "advice") for q in load_questions()],
        "outlook (V2 only)": [(t, "outlook") for t in OUTLOOK],
        "off_topic_30 (V2 only)": [(t, "other") for _, en, hi, te in OOS for t in (en, hi, te)],
    }
    res = {}
    for name, ex in (("V1 (before)", EXAMPLES_V1), ("V2 (after)", EXAMPLES_V2)):
        det = IntentDetector(emb, ex)
        res[name] = {}
        for tname, items in tests.items():
            if "V2 only" in tname and ex is EXAMPLES_V1:
                continue
            ok = [det.detect(t) == i for t, i in items]
            res[name][tname] = {"n": len(items), "accuracy": round(sum(ok) / len(ok), 3)}
            if tname == "romanised":
                per = {}
                for (t, i), o in zip(items, ok):
                    per.setdefault(i, []).append(o)
                res[name][tname]["per_intent"] = {k: round(sum(v) / len(v), 2) for k, v in per.items()}
            if tname == "kcc_real_english":
                per = {}
                for (t, i), o in zip(items, ok):
                    per.setdefault(i, []).append(o)
                res[name][tname]["per_intent"] = {k: round(sum(v) / len(v), 2) for k, v in per.items()}
    print(json.dumps(res, indent=2))
    (ROOT / "results" / "intents.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
