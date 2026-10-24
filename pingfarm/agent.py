"""The agent: decide what the farmer wants, call the right tools, answer, stay safe.

A deliberately small loop (no agent framework) so every step can be read and explained:

    1. location message?      -> remember it
    2. photo?                 -> classifier says which disease; low confidence -> ask for a better photo
    3. text?                  -> work out the intent (advice / spray timing / price)
    4. gather facts           -> retrieval (RAG), weather, prices
    5. "do I know this?"      -> if the closest passage is too far away, say "not sure" (abstain)
    6. write the answer       -> language model or extractive, then the dose safety guard
    7. reply with sources

Intent detection uses the SAME multilingual embedder as retrieval: we embed a few example
questions per intent (English, Hindi, Telugu) and pick the nearest. No extra model, easy to explain.
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .generation import Answer, sources_from
from .intents import EXAMPLES_V2, IntentClassifier, IntentDetector
from .rag.embed import Embedder
from .rag.search import Retriever
from . import profile as profile_tool
from .state import State
from .tools import markets as markets_tool
from .tools import prices as price_tool
from .tools import risk as risk_tool
from .tools import suitability as suit_tool
from .tools import weather as weather_tool
from .topic_gate import TopicGate

INTENT_EXAMPLES = EXAMPLES_V2  # kept for the notebooks; the examples live in pingfarm/intents.py

CROP_WORDS = {
    "tomato": ["tomato", "टमाटर", "టమాటా", "టమోటా"],
    "potato": ["potato", "आलू", "బంగాళదుంప", "ఆలు"],
}

ABSTAIN_TEXT = ("I am not sure about this one, and I do not want to guess. Please ask your local agriculture "
                "officer or Krishi Vigyan Kendra. You can also send a clear photo of the leaf.")
SCOPE_TEXT = ("I can help with tomato and potato: disease advice (send a leaf photo or describe the problem), whether it is a good day "
              "to spray, market prices near you, and the weather outlook for your crop. For anything else, please ask your local "
              "agriculture officer or Krishi Vigyan Kendra.")
ASK_LOCATION ="Please share your location (WhatsApp: attach, then Location) so I can check the weather."
ASK_BETTER_PHOTO = ("I could not tell clearly from this photo. Please send a close, well-lit photo of one affected "
                    "leaf (top side, and underside if possible).")

# ---- how the classifier's label maps to a question the knowledge base understands
LABEL_TO_QUERY = {
    "tomato__bacterial_spot": "Tomato bacterial leaf spot Xanthomonas management",
    "tomato__early_blight": "Tomato early blight Alternaria solani management",
    "tomato__late_blight": "Tomato late blight Phytophthora infestans management",
    "tomato__leaf_mold": "Tomato leaf mold management",
    "tomato__septoria_leaf_spot": "Tomato Septoria leaf spot management",
    "tomato__yellow_leaf_curl": "Tomato leaf curl virus whitefly management",
    "tomato__mosaic_virus": "Tomato mosaic virus management",
    "tomato__healthy": None,
    "potato__early_blight": "Potato early blight Alternaria solani management",
    "potato__late_blight": "Potato late blight Phytophthora infestans management",
}


@dataclass
class Reply:
    text: str
    sources: list[dict] = field(default_factory=list)
    abstained: bool = False
    trace: list[str] = field(default_factory=list)  # human-readable steps, for debugging and demos


def detect_crop(text: str) -> str | None:
    low = text.lower()
    for crop, words in CROP_WORDS.items():
        if any(w.lower() in low for w in words):
            return crop
    return None


class Agent:
    def __init__(self, retriever: Retriever, answerer, embedder: Embedder, state: State,
                 classifier=None, abstain_threshold: float | None = None,
                 weather_fn=weather_tool.fetch_forecast, price_fn=price_tool.fetch_prices,
                 calibration_path: Path | None = None, rerank: bool = False, use_gate: bool = True):
        self.retriever, self.answerer, self.state = retriever, answerer, state
        self.classifier, self.rerank = classifier, rerank
        self.weather_fn, self.price_fn = weather_fn, price_fn
        self.embedder = embedder
        self.abstain_threshold = abstain_threshold if abstain_threshold is not None else _load_threshold(calibration_path)
        self.intents = IntentDetector(embedder)                      # nearest-centroid (fallback)
        self.intent_clf = IntentClassifier.load() if use_gate else None  # trained classifier (preferred)
        self._intent_names, self._intent_vecs = self.intents.names, self.intents.vecs  # used by the notebooks
        # the learned "can our documents answer this?" check; falls back to the similarity gate if it is missing
        self.gate = TopicGate.load() if use_gate else None

    # ---------------------------------------------------------------- helpers
    def detect_intent(self, text: str) -> str:
        if self.intent_clf is not None:
            vec = self.embedder.embed_queries([text])[0]
            intent = self.intent_clf.detect(vec)
            # safety net: if the router says "off-topic" but the topic gate is confident the knowledge base can
            # answer it, trust the gate (an in-scope Hindi/Telugu question must not get the generic scope message)
            if intent == "other" and self.gate is not None and self.gate.answerable(vec, text):
                return "advice"
            return intent
        return self.intents.detect(text)

    def _rag(self, query: str, trace: list[str], k: int = 3, gate_text: str | None = None):
        """Retrieve for `query`. The "do I know this?" gate looks at `gate_text` (the farmer's own words):
        we must NOT judge it on `query`, because adding a crop name ("tomato ...") drags ANY question,
        even about France, closer to our tomato documents and past the cut-off."""
        hits = self.retriever.search(query, mode="dense", k=k, rerank=self.rerank)
        sim = self.retriever.top_similarity(gate_text if gate_text is not None else query)
        trace.append(f"retrieved {len(hits)} passages; closest similarity {sim:.3f} (abstain below {self.abstain_threshold:.3f})")
        return hits, sim

    # ---------------------------------------------------------------- main entry point
    def handle(self, user_id: str, text: str | None = None, image: bytes | None = None,
               location: tuple[float, float] | None = None) -> Reply:
        trace: list[str] = []
        self.state.touch_inbound(user_id)  # remember when they last wrote (WhatsApp's 24-hour messaging window)
        if text and profile_tool.wants_deletion(text):
            self.state.delete_user(user_id)
            return Reply("Done. I deleted your profile, location and message history.", trace=["deleted user data"])
        if location:
            self.state.update_user(user_id, lat=location[0], lon=location[1], state=None, district=None)  # re-derive place later
            return Reply("Thank you, I saved your location. You can now ask if it is a good day to spray, "
                         "or ask for this week's weather outlook for your crop.", trace=["saved location"])

        user = self.state.get_user(user_id)
        crop = detect_crop(text or "") or user.crop
        if crop and crop != user.crop:
            self.state.update_user(user_id, crop=crop)

        # ---- photo path
        if image is not None:
            if self.classifier is None:
                return Reply("Photo diagnosis is not switched on yet. Please describe the problem in words.", trace=["no classifier"])
            label, prob = self.classifier.predict(image)
            trace.append(f"classifier: {label} (confidence {prob:.2f})")
            if label is None:  # classifier abstained
                return Reply(ASK_BETTER_PHOTO, abstained=True, trace=trace)
            query = LABEL_TO_QUERY.get(label)
            if query is None:
                return Reply("The leaf looks healthy to me. If you still see a problem, send another photo or describe it.", trace=trace)
            hits, sim = self._rag(query, trace)
            ans = self.answerer.answer(query, hits, context=f"The photo looks like: {label.replace('__', ' ').replace('_', ' ')} (confidence {prob:.0%}).")
            self.state.add_history(user_id, "user", "[photo]")
            self.state.add_history(user_id, "bot", ans.text)
            return Reply(ans.text, ans.sources, trace=trace + ([ "dose guard removed text"] if ans.guard_triggered else []))

        text = (text or "").strip()
        if not text:
            return Reply("Please send a question or a photo of the leaf.", trace=["empty"])

        # ---- things the farmer tells us about themselves: soil, planting date, alert preference
        notes = []
        soil = profile_tool.parse_soil(text)
        if soil:
            self.state.update_user(user_id, soil=soil)
            notes.append(f"soil: {soil}")
        planted = profile_tool.parse_planting_date(text)
        if planted:
            self.state.update_user(user_id, planted_on=planted)
            notes.append(f"planted on {planted}")
        alerts_on = profile_tool.parse_alert_command(text)
        if alerts_on is not None:
            if alerts_on and (user.lat is None or not crop):
                return Reply("To send alerts I need your location (share a pin) and your crop (tomato or potato). "
                             "Then say 'alerts on' again.", trace=["alerts need location and crop"])
            self.state.update_user(user_id, alerts=alerts_on)
            msg = ("Alerts are ON. I will message you only when the forecast is high-risk for your crop, at most once a day. "
                   "Say 'alerts off' to stop." if alerts_on else "Alerts are OFF.")
            return Reply(msg, trace=["alerts " + ("on" if alerts_on else "off")])
        if notes and len(text) < 70 and "?" not in text:
            return Reply("Noted (" + ", ".join(notes) + "). You can now ask me for the weather outlook for your crop.",
                         trace=["profile updated: " + ", ".join(notes)])
        user = self.state.get_user(user_id)

        intent = self.detect_intent(text)
        trace.append(f"intent: {intent}; crop: {crop}")
        self.state.add_history(user_id, "user", text)

        # ---- not about our crops at all: say what we CAN do instead of guessing
        if intent == "other":
            return Reply(SCOPE_TEXT, abstained=True, trace=trace + ["off-topic or small talk"])

        # ---- spray timing: weather tool
        if intent == "spray_timing":
            if user.lat is None:
                return Reply(ASK_LOCATION, trace=trace)
            try:
                advice = weather_tool.spray_advice(self.weather_fn(user.lat, user.lon))
            except Exception as exc:
                return Reply("I could not reach the weather service right now. Please try again later.", trace=trace + [f"weather error: {type(exc).__name__}"])
            trace.append(f"weather: ok_to_spray={advice.ok_to_spray}")
            self.state.add_history(user_id, "bot", advice.reason)
            return Reply(advice.reason, [{"title": "Open-Meteo forecast", "url": "https://open-meteo.com", "doc_id": "open-meteo"}], trace=trace)

        # ---- outlook: weather-based disease risk, soil and season fit, which crop suits the weather
        if intent == "outlook":
            if user.lat is None:
                return Reply(ASK_LOCATION, trace=trace)
            try:
                forecast = self.weather_fn(user.lat, user.lon)
            except Exception as exc:
                return Reply("I could not reach the weather service right now.", trace=trace + [f"weather error: {type(exc).__name__}"])
            choosing = re.search(r"which crop|what (?:can|should) i (?:grow|plant)|tomato or potato|grow now|plant (?:potato|tomato)|आलू लगाना|लगाना ठीक|టమాటా వేయాలా", text.lower())
            if choosing or not crop:
                body = suit_tool.compare_crops(user.soil, forecast)
                if crop:
                    body += "\n\n" + suit_tool.outlook(crop, user.soil, forecast)
            else:
                stage = profile_tool.crop_stage(crop, user.planted_on)
                head = f"Your {crop} is at: {stage}.\n" if stage else ""
                body = head + risk_tool.summarize(crop, risk_tool.assess(crop, forecast)) + "\n\n" + suit_tool.outlook(crop, user.soil, forecast)
            self.state.add_history(user_id, "bot", body)
            sources = [{"title": "Open-Meteo forecast", "url": "https://open-meteo.com", "doc_id": "open-meteo"},
                       {"title": "TNAU Agritech Portal", "url": "https://agritech.tnau.ac.in", "doc_id": "tnau"}]
            return Reply(body, sources, trace=trace + ["outlook from forecast"])

        # ---- price: mandi tool (nearby markets when we know where the farmer is)
        if intent == "price":
            if not crop:
                return Reply("Which crop? Tomato or potato?", trace=trace)
            try:
                if user.lat is not None and self.price_fn is price_tool.fetch_prices:
                    msg, err = markets_tool.markets_for(user, crop, self.state)
                    if err == "no_key":
                        msg = price_tool.describe_prices(None, crop)
                else:
                    msg = price_tool.describe_prices(self.price_fn(crop.title(), user.state), crop)
            except Exception as exc:
                return Reply("I could not reach the market-price service right now.", trace=trace + [f"price error: {type(exc).__name__}"])
            return Reply(msg, [{"title": "Agmarknet via data.gov.in", "url": "https://data.gov.in", "doc_id": "agmarknet"}], trace=trace)

        # ---- advice: retrieval (RAG)
        # a Hindi/Telugu question may not name the crop in English, so we add it for the retriever
        query = text if (not crop or crop in text.lower()) else f"{crop} {text}"
        hits, sim = self._rag(query, trace, gate_text=text)
        if self.gate is not None:
            vec = self.embedder.embed_queries([text])[0]
            p = self.gate.score(vec)
            trace.append(f"topic gate: p(answerable)={p:.2f}")
            if not self.gate.answerable(vec, text):
                return Reply(ABSTAIN_TEXT, abstained=True, trace=trace)
        elif sim < self.abstain_threshold:
            return Reply(ABSTAIN_TEXT, abstained=True, trace=trace)
        ans = self.answerer.answer(text, hits)
        self.state.add_history(user_id, "bot", ans.text)
        return Reply(ans.text, ans.sources, trace=trace + (["dose guard removed text"] if ans.guard_triggered else []))


def _load_threshold(path: Path | None) -> float:
    """The "I don't know" cut-off is MEASURED, not guessed: scripts/calibrate_abstain.py writes it."""
    path = path or Path(__file__).resolve().parents[1] / "data" / "eval" / "calibration.json"
    if path.exists():
        return float(json.loads(path.read_text())["abstain_threshold"])
    return 0.80  # conservative default until calibrated
