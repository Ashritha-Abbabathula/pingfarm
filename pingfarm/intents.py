"""What does the farmer want? Nearest-centroid intent detection on the multilingual question embeddings.

For each intent we keep a handful of example questions (English, Hindi, Telugu, and the same two languages typed in
ENGLISH LETTERS, which is how many people really type on a phone). An example is embedded; the average of an
intent's examples is its centroid; a new question goes to the nearest centroid.

Fixing a wrong answer = adding one example sentence. No retraining.
"""
import numpy as np

# version 1 (before romanised text and real KCC phrasing were added); kept so the improvement can be measured
EXAMPLES_V1 = {
    "spray_timing": [
        "Can I spray fungicide tomorrow? Will it rain?",
        "Is it a good day to spray my crop?",
        "कल दवा का छिड़काव कर सकता हूं? बारिश होगी क्या?",
        "आज स्प्रे करना ठीक रहेगा?",
        "రేపు మందు పిచికారీ చేయవచ్చా? వర్షం పడుతుందా?",
        "ఈ రోజు పిచికారీ చేయడం మంచిదేనా?",
    ],
    "price": [
        "What is the mandi price of tomato today?",
        "What rate is potato selling at in the market?",
        "आज टमाटर का मंडी भाव क्या है?",
        "आलू का रेट कितना चल रहा है?",
        "ఈ రోజు టమాటా మార్కెట్ ధర ఎంత?",
        "బంగాళదుంప రేటు ఎంత ఉంది?",
    ],
    "advice": [
        "My tomato leaves have brown spots. What disease is this?",
        "How do I control late blight in potato?",
        "मेरे टमाटर के पत्तों पर धब्बे हैं, क्या करूं?",
        "आलू में झुलसा रोग का इलाज क्या है?",
        "నా టమాటా ఆకులపై మచ్చలు ఉన్నాయి, ఏమి చేయాలి?",
        "బంగాళదుంపలో తెగులును ఎలా నివారించాలి?",
    ],
}

# version 2 = version 1 plus romanised Hindi/Telugu and short real Kisan-Call-Centre style phrasing
EXAMPLES_V2 = {
    "spray_timing": EXAMPLES_V1["spray_timing"] + [
        "kal dawai chhidak sakta hoon kya, baarish hogi?",
        "aaj spray karna theek rahega kya",
        "barish aane wali hai kya, dawa dalni chahiye",
        "repu mandu spray cheyyavacha, varsham padutunda",
        "ee roju spray cheyadam manchida",
        "rain forecast for spraying",
        "weather forecast for next few days",
    ],
    "price": EXAMPLES_V1["price"] + [
        "aaj tamatar ka mandi bhav kya hai",
        "aloo ka rate kitna chal raha hai",
        "tamatar kitne rupaye kilo bik raha hai",
        "ee roju tomato dhara entha",
        "bangaladumpa rate ela undi",
        "market lo tamata ela ammutunnaru",
        "market information tomato rate",
        "price of potato in mandi",
        "how much will I get for my tomatoes",
    ],
    "advice": EXAMPLES_V1["advice"] + [
        "mere tamatar ke patton par daag hain, kya karun",
        "aloo mein jhulsa rog ka ilaj kya hai",
        "naa tomato aakulu meeda machalu unnayi, emi cheyali",
        "bangaladumpa lo teguluni ela apali",
        "leaf curl control in tomato",
        "how to control blight in potato",
        "tomato wilt management",
        # dose and quantity questions are advice, not price ("how much" must not mean "how much does it cost")
        "How much mancozeb should I mix per litre of water?",
        "What is the dose of fungicide for potato late blight?",
        "How much spray should I use per acre for whitefly in tomato?",
        "टमाटर में कितनी मात्रा में दवा डालनी चाहिए?",
        "आलू के झुलसा रोग में प्रति लीटर पानी में कितना मैन्कोजेब मिलाएं?",
        "టమాటాలో ఎంత మోతాదులో మందు వాడాలి?",
        "bimari ke liye dawai kitni matra mein dalni chahiye",
        "tamata ki dawa ka dose kitna hai",
    ],
}


EXAMPLES_V2["outlook"] = [
    "What should I do this week for my crop given the weather?",
    "Will the coming weather cause disease in my tomato?",
    "Which crop should I grow now, tomato or potato?",
    "Is this a good time to plant potato in my soil?",
    "Weather forecast and disease risk for my field",
    "इस हफ्ते मौसम के हिसाब से मेरी फसल के लिए क्या करना चाहिए?",
    "क्या अभी आलू लगाना ठीक रहेगा?",
    "आने वाले मौसम में बीमारी का खतरा है क्या?",
    "ఈ వారం వాతావరణం ప్రకారం నా పంటకు ఏమి చేయాలి?",
    "ఇప్పుడు టమాటా వేయాలా బంగాళదుంప వేయాలా?",
    "ravaali mausam ke hisaab se meri fasal ke liye kya karna chahiye",
    "ee vaaram vatavaranam batti naa pantaku emi cheyali",
]


# Without an explicit "none of the above" class, nearest-centroid routing must pick ONE of the real intents even for
# "what is the capital of France?". We found this the hard way (see the study guide), so off-topic and small-talk
# examples form their own intent.
EXAMPLES_V2["other"] = [
    "Who is the prime minister of India?",
    "Tell me a joke",
    "What is two plus two?",
    "Recipe for dosa at home",
    "Who won the football match yesterday?",
    "hello how are you",
    "thank you",
    "latest movie news",
    "How do I get a driving licence?",
    "आज की ताज़ा खबरें क्या हैं",
    "आपका नाम क्या है",
    "नमस्ते कैसे हो",
    "మీ పేరు ఏమిటి",
    "ఈ రోజు సినిమా వార్తలు",
    "మీరు ఎలా ఉన్నారు",
    "namaste kya haal hai",
    "ela unnaru",
    "aap kaun ho",
]


class IntentDetector:
    def __init__(self, embedder, examples: dict[str, list[str]] = EXAMPLES_V2):
        self.embedder = embedder
        self.names = list(examples)
        cents = np.vstack([embedder.embed_queries(examples[n]).mean(axis=0) for n in self.names])
        self.vecs = cents / np.linalg.norm(cents, axis=1, keepdims=True)

    def scores(self, text: str) -> dict[str, float]:
        v = self.embedder.embed_queries([text])[0]
        return dict(zip(self.names, (self.vecs @ v).tolist()))

    def detect(self, text: str) -> str:
        s = self.scores(text)
        return max(s, key=s.get)


class IntentClassifier:
    """The trained replacement for nearest-centroid: multinomial logistic regression on the same question embeddings.

    Why: with five intents including an open-ended "other", averaging examples into centroids was too blurry
    (see results/intents.json vs results/intents_v3.json). A trained classifier learns where the boundaries are.
    """
    PATH = None

    def __init__(self, names, W, b):
        self.names, self.W, self.b = list(names), np.asarray(W, dtype="float32"), np.asarray(b, dtype="float32")

    def scores(self, vec) -> dict[str, float]:
        z = self.W @ vec + self.b
        e = np.exp(z - z.max())
        return dict(zip(self.names, (e / e.sum()).tolist()))

    def detect(self, vec) -> str:
        s = self.scores(vec)
        return max(s, key=s.get)

    def save(self, path=None):
        import json
        from pathlib import Path

        path = Path(path or Path(__file__).resolve().parents[1] / "models" / "intents.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"names": self.names, "W": self.W.tolist(), "b": self.b.tolist()}))

    @classmethod
    def load(cls, path=None):
        import json
        from pathlib import Path

        path = Path(path or Path(__file__).resolve().parents[1] / "models" / "intents.json")
        if not path.exists():
            return None
        d = json.loads(path.read_text())
        return cls(d["names"], d["W"], d["b"])
