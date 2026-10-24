"""Build the REAL-question evaluation set from the Kisan Call Centre sample (data/kcc/tomato_potato.csv).

Each real query is mapped, by simple keyword rules, to the knowledge-base documents that should answer it.
Queries with no matching rule are marked answerable=false (the knowledge base has no page for them, e.g.
"fertilizer dose in tomato"): the system should say "not sure" for those.

Honest limits (also in the study guide):
  * QueryText is the call-centre OPERATOR's English summary of what the farmer said, not the farmer's words.
  * Labels come from keyword rules plus my reading, not from an agronomist. Relevance here is document-level:
    a retrieved chunk counts if it comes from an accepted document (no evidence phrase), so it is a looser
    test than the starter set.

Run:  python scripts/build_real_questions.py          -> data/eval/real_questions.jsonl
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "kcc" / "tomato_potato.csv"
OUT = ROOT / "data" / "eval" / "real_questions.jsonl"

# (regex on lower-cased query, crop or None, topic name, accepted document ids)
T = "tnau_tomato_"
P = "tnau_potato_"
RULES = [
    (r"spotted wilt", "tomato", "spotted-wilt", [T + "07"]),
    (r"early blight", "tomato", "early-blight", [T + "02"]),
    (r"late blight", "tomato", "late-blight", [T + "08"]),
    (r"blight", "tomato", "blight", [T + "02", T + "08"]),
    (r"leaf curl|curling|\bcurl\b|\bcurl", "tomato", "leaf-curl", [T + "06"]),
    (r"mosaic", "tomato", "mosaic", [T + "12", T + "07"]),
    (r"powdery mildew|mildew", "tomato", "powdery-mildew", [T + "10"]),
    (r"leaf spot|leaf-spot|spots", "tomato", "leaf-spot", [T + "04", T + "11"]),
    (r"wilt", "tomato", "wilt", [T + "03", T + "05"]),
    (r"damping", "tomato", "damping-off", [T + "01"]),
    (r"canker", "tomato", "canker", [T + "09"]),
    (r"fruit borer|fruit-borer|fruit & shoot|shoot borer|borer|helicoverpa|american bollworm", "tomato", "fruit-borer", [T + "pest_1"]),
    (r"leaf miner|leafminer|leaf minor", "tomato", "leaf-miner", [T + "pest_2"]),
    (r"caterpillar|defoliator|leaf eating", "tomato", "caterpillar", [T + "pest_3"]),
    (r"white ?fly", "tomato", "whitefly", [T + "pest_4"]),
    (r"thrips|thirps", "tomato", "thrips", [T + "pest_5"]),
    (r"mealy ?bug", "tomato", "mealybug", [T + "pest_6"]),
    (r"\bmite", "tomato", "mites", [T + "pest_7"]),
    (r"pinworm", "tomato", "pinworm", [T + "pest_8"]),
    (r"sucking pest", "tomato", "sucking-pests", [T + "pest_4", T + "pest_5", T + "pest_6", T + "pest_7"]),
    (r"variet|hybrid", "tomato", "varieties", [T + "season_varieties"]),
    (r"season|sowing time|time of sowing|when to (?:sow|plant)", "tomato", "season", [T + "season_varieties", T + "soil"]),
    (r"\bsoil\b|\bph\b", "tomato", "soil", [T + "soil"]),
    (r"irrigat|water management", "tomato", "irrigation", [T + "irrigation"]),
    (r"seed rate|nursery|seedling", "tomato", "seed-nursery", [T + "seed"]),
    (r"early blight", "potato", "early-blight", [P + "02", P + "cultivation"]),
    (r"late blight", "potato", "late-blight", [P + "01", "icar_indfarm_potato_2019", "icar_indfarm_review", P + "cultivation"]),
    (r"blight", "potato", "blight", [P + "01", P + "02", P + "cultivation"]),
    (r"scab", "potato", "scab", [P + "05"]),
    (r"scurf", "potato", "scurf", [P + "04"]),
    (r"soft rot|rot", "potato", "rot", [P + "03", P + "07"]),
    (r"wilt|brown rot", "potato", "wilt", [P + "06"]),
    (r"variet", "potato", "varieties", [P + "cultivation"]),
    (r"seed rate|sowing|planting|season|soil", "potato", "cultivation", [P + "cultivation"]),
]


def norm(q: str) -> str:
    q = re.sub(r"[\r\n\t]+", " ", str(q)).strip().strip("?:. ")
    return re.sub(r"\s+", " ", q)


def main():
    df = pd.read_csv(SRC)
    df["q"] = df["QueryText"].map(norm)
    df = df[df["q"].str.len().between(12, 160)]
    rows, seen_text, per_topic = [], set(), {}
    for _, r in df.sample(frac=1, random_state=0).iterrows():
        q, low = r["q"], r["q"].lower()
        crop = "potato" if "potato" in str(r["Crop"]).lower() else "tomato"
        key = re.sub(r"[^a-z]", "", low)
        if key in seen_text:
            continue
        seen_text.add(key)
        topic, docs = None, []
        for rx, c, name, d in RULES:
            if c == crop and re.search(rx, low):
                topic, docs = name, d
                break
        # cap per topic so templated operator sentences (e.g. Madhya Pradesh "sucking pest") do not dominate
        tag = (crop, topic or "none")
        per_topic[tag] = per_topic.get(tag, 0) + 1
        if per_topic[tag] > (6 if topic else 12):
            continue
        rows.append({
            "id": f"kcc{len(rows):04d}", "lang": "en", "text": q, "crop": crop, "state": r["StateName"].title(),
            "query_type": str(r["QueryType"]).strip(), "topic": topic or "uncovered",
            "answerable": bool(docs), "doc_ids": docs, "evidence": [],
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    out = pd.DataFrame(rows)
    print(len(out), "real questions;", int(out.answerable.sum()), "answerable by the knowledge base,", int((~out.answerable).sum()), "uncovered")
    print(out.groupby(["crop", "topic"]).size().to_string())


if __name__ == "__main__":
    main()
