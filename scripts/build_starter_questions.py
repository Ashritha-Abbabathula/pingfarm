"""Builds data/eval/questions.jsonl: the STARTER evaluation set.

Each base question is written once in English, Hindi and Telugu, and points at the
document(s) that answer it plus an `evidence` phrase that must appear in the passage.

IMPORTANT (YOU DO THIS): the Hindi and Telugu sentences are machine-drafted. Have a
native speaker read them, fix awkward phrasing, and add questions written the way
real farmers actually talk (spelling mistakes, mixed English words, short fragments).
A starter set of ~40 questions is enough to build the machinery; the real goal is
100 to 200 questions checked against agronomy sources.

Run:  python scripts/build_starter_questions.py
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "eval" / "questions.jsonl"

# (key, doc_ids, evidence phrases, English, Hindi, Telugu)
BASE = [
    ("leafcurl-control", ["tnau_tomato_06"], ["yellow sticky traps", "barrier crop"],
     "How can I stop leaf curl virus from spreading in my tomato crop?",
     "मेरी टमाटर की फसल में पत्ती मोड़क (लीफ कर्ल) वायरस को फैलने से कैसे रोकें?",
     "నా టమాటా పంటలో ఆకు ముడత వైరస్ వ్యాపించకుండా ఎలా ఆపాలి?"),
    ("tomato-lateblight-weather", ["tnau_tomato_08"], ["cool nights", "18 - 22"],
     "In what weather does late blight of tomato spread?",
     "टमाटर का पछेती झुलसा (लेट ब्लाइट) किस मौसम में फैलता है?",
     "టమాటా ఆలస్య తెగులు (లేట్ బ్లైట్) ఏ వాతావరణంలో వ్యాపిస్తుంది?"),
    ("tomato-bacterial-wilt-symptom", ["tnau_tomato_05"], ["white bacterial ooze"],
     "My tomato plants suddenly wilt and white liquid comes out of the cut stem. What is this?",
     "मेरे टमाटर के पौधे अचानक मुरझा जाते हैं और कटे तने से सफेद रस निकलता है। यह कौन सा रोग है?",
     "నా టమాటా మొక్కలు అకస్మాత్తుగా వాడిపోతున్నాయి, కోసిన కాండం నుండి తెల్లని ద్రవం వస్తోంది. ఇది ఏ తెగులు?"),
    ("tomato-fusarium-conditions", ["tnau_tomato_03"], ["low soil moisture"],
     "Which conditions favour Fusarium wilt in tomato?",
     "टमाटर में फ्यूजेरियम उकठा रोग किन परिस्थितियों में होता है?",
     "టమాటాలో ఫ్యూజేరియం వాడు తెగులు ఏ పరిస్థితుల్లో వస్తుంది?"),
    ("potato-resistant-varieties", ["tnau_potato_01", "icar_indfarm_potato_2019"],
     ["kufri naveen", "kufri giridhari"],
     "Which potato varieties resist late blight?",
     "आलू की कौन सी किस्में पछेती झुलसा रोग के प्रति प्रतिरोधी हैं?",
     "బంగాళదుంపలో ఆలస్య తెగులును తట్టుకునే రకాలు ఏవి?"),
    ("potato-scab-fertilizer", ["tnau_potato_05"], ["calcium ammonium nitrate"],
     "Which fertilizer should I avoid to reduce common scab in potato?",
     "आलू में साधारण स्कैब रोग कम करने के लिए कौन सा उर्वरक नहीं डालना चाहिए?",
     "బంగాళదుంపలో కామన్ స్కాబ్ తగ్గించడానికి ఏ ఎరువును వాడకూడదు?"),
    ("potato-blackscurf-seed", ["tnau_potato_04"], ["disease free seed tubers"],
     "How do I prevent black scurf in potato?",
     "आलू में ब्लैक स्कर्फ रोग से बचाव के लिए क्या करें?",
     "బంగాళదుంపలో బ్లాక్ స్కర్ఫ్ నివారణకు ఏమి చేయాలి?"),
    ("potato-brownrot-symptom", ["tnau_potato_06"], ["foul odour"],
     "Potato plants wilt at tuber formation and tubers ooze with a bad smell. What is it?",
     "आलू के पौधे कंद बनते समय मुरझा रहे हैं और कंदों से बदबूदार रस निकल रहा है। यह कौन सा रोग है?",
     "బంగాళదుంప మొక్కలు దుంపలు ఏర్పడే సమయంలో వాడిపోతున్నాయి, దుంపల నుండి దుర్వాసనతో ద్రవం వస్తోంది. ఇది ఏ తెగులు?"),
    ("tomato-powdery-mildew-spray", ["tnau_tomato_10"], ["propiconazole"],
     "What can I spray for powdery mildew on tomato?",
     "टमाटर में पाउडरी मिल्ड्यू के लिए क्या छिड़काव करें?",
     "టమాటాలో బూడిద తెగులుకు (పౌడరీ మిల్డ్యూ) ఏమి పిచికారీ చేయాలి?"),
    ("tomato-earlyblight-seed", ["tnau_tomato_02"], ["trichoderma asperellum @ 4g/kg"],
     "How should I treat tomato seeds against early blight?",
     "टमाटर के बीजों को अगेती झुलसा रोग से बचाने के लिए कैसे उपचारित करें?",
     "టమాటా విత్తనాలను ముందస్తు తెగులు (ఎర్లీ బ్లైట్) నుండి కాపాడడానికి ఎలా శుద్ధి చేయాలి?"),
    ("potato-first-spray-timing", ["icar_indfarm_potato_2019"], ["canopy closure"],
     "When should the first preventive spray against late blight be applied on potato?",
     "आलू में पछेती झुलसा के लिए पहला रोकथाम वाला छिड़काव कब करना चाहिए?",
     "బంగాళదుంపలో ఆలస్య తెగులు నివారణకు మొదటి పిచికారీ ఎప్పుడు చేయాలి?"),
    ("potato-metalaxyl-resistance", ["icar_indfarm_potato_2019"],
     ["developed resistance against metalaxyl"],
     "Why is metalaxyl no longer reliable against potato late blight?",
     "आलू के पछेती झुलसा में मेटालैक्सिल दवा अब भरोसेमंद क्यों नहीं है?",
     "బంగాళదుంప ఆలస్య తెగులుకు మెటాలాక్సిల్ మందు ఇప్పుడు ఎందుకు నమ్మదగినది కాదు?"),
    ("potato-disease-forecasting", ["icar_indfarm_review"], ["disease forecasting allows"],
     "What is disease forecasting and how does it help potato farmers?",
     "रोग पूर्वानुमान क्या है और यह आलू किसानों की कैसे मदद करता है?",
     "వ్యాధి సూచన (డిసీజ్ ఫోర్కాస్టింగ్) అంటే ఏమిటి, ఇది బంగాళదుంప రైతులకు ఎలా సహాయపడుతుంది?"),
]


# Extra accepted documents/evidence added when the knowledge base grew to Hindi/Telugu pages and the
# TNAU cultivation and pest pages. A question may now be answered by a source in ANY language, so the
# evidence list mixes English, Hindi and Telugu phrases (all verified to exist in the downloaded text).
EXTRA = {
    "leafcurl-control": (["tnau_tomato_pest_4", "vik_te_tomato", "vik_hi_tomato_cultivation"],
                         ["అడ్డుపంట", "सफेद मक्खी"]),
    "tomato-bacterial-wilt-symptom": (["vik_te_tomato"], ["బాక్టీరియల్ విల్డ్"]),
    "potato-resistant-varieties": (["tnau_potato_cultivation", "vik_hi_potato_late_blight"],
                                   ["late blight resistant varieties", "कुफरी नवीन"]),
    "tomato-powdery-mildew-spray": (["vik_te_summer_vegetables"], ["బూడిద తెగులు"]),
    "potato-first-spray-timing": (["tnau_potato_cultivation", "vik_hi_potato_late_blight"],
                                  ["45, 60 and 75 days after planting", "लक्षण दिखाई देते ही"]),
    "potato-disease-forecasting": (["vik_hi_potato_late_blight"], ["पूर्वानुमान"]),
}


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, docs, evidence, en, hi, te in BASE:
        extra_docs, extra_ev = EXTRA.get(key, ([], []))
        docs, evidence = docs + extra_docs, evidence + extra_ev
        for lang, text in (("en", en), ("hi", hi), ("te", te)):
            rows.append(
                {"id": f"{key}.{lang}", "lang": lang, "text": text, "doc_ids": docs, "evidence": evidence}
            )
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} questions to {OUT}")


if __name__ == "__main__":
    main()
