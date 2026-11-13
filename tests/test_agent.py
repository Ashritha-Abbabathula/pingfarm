"""End-to-end agent tests. They load the real embedding model (cached locally), so they take a few
seconds. Skipped automatically if the corpus has not been downloaded."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytestmark = pytest.mark.skipif(not (ROOT / "data" / "raw").exists(), reason="run scripts/fetch_corpus.py first")


class FakeClassifier:
    def __init__(self, label, prob):
        self.label, self.prob = label, prob

    def predict(self, image_bytes):
        return self.label, self.prob


@pytest.fixture(scope="module")
def parts():
    from pingfarm.generation import ExtractiveAnswerer
    from pingfarm.rag.chunking import chunk_all
    from pingfarm.rag.documents import load_documents
    from pingfarm.rag.embed import Embedder
    from pingfarm.rag.search import Retriever

    emb = Embedder()
    retr = Retriever(chunk_all(load_documents(), "fixed", header=True, size=200, overlap=40), emb)
    return emb, retr, ExtractiveAnswerer()


def make_agent(parts, classifier=None, forecast=None):
    from pingfarm.agent import Agent
    from pingfarm.state import State
    from pingfarm.tools.weather import DayForecast

    emb, retr, ans = parts
    fc = forecast or [DayForecast("d1", 0, 0, 20, 30), DayForecast("d2", 0, 0, 20, 30)]
    return Agent(retr, ans, emb, State(), classifier=classifier, abstain_threshold=0.79,
                 weather_fn=lambda lat, lon: fc, price_fn=lambda crop, state: None)


@pytest.mark.parametrize("text,intent", [
    ("Can I spray tomorrow or will it rain?", "spray_timing"),
    ("कल स्प्रे कर सकता हूं क्या बारिश होगी?", "spray_timing"),
    ("టమాటా ధర ఈ రోజు ఎంత?", "price"),
    ("What is the potato rate in the mandi?", "price"),
    ("My potato leaves have dark patches, what should I do?", "advice"),
])
def test_intent_detection(parts, text, intent):
    assert make_agent(parts).detect_intent(text) == intent


def test_advice_has_sources(parts):
    r = make_agent(parts).handle("u1", text="How do I control leaf curl virus in tomato?")
    assert not r.abstained and r.sources


def test_off_topic_question_is_refused(parts):
    r = make_agent(parts).handle("u2", text="What is the capital of France?")
    assert r.abstained


def test_remembered_crop_does_not_unlock_off_topic_questions(parts):
    """Regression: a stored crop used to be prefixed to the query, which pushed an off-topic question
    ('capital of France') above the abstain threshold and returned tomato late-blight text."""
    a = make_agent(parts)
    a.handle("u7", text="How do I control leaf curl virus in tomato?")       # now the crop 'tomato' is remembered
    assert a.handle("u7", text="What is the capital of France?").abstained


def test_spray_needs_location_then_uses_weather(parts):
    a = make_agent(parts)
    assert "location" in a.handle("u3", text="Can I spray tomorrow?").text.lower()
    a.handle("u3", location=(17.4, 78.5))
    assert "rain" in a.handle("u3", text="Can I spray tomorrow?").text.lower()


def test_rainy_forecast_blocks_spraying(parts):
    from pingfarm.tools.weather import DayForecast

    rainy = [DayForecast("d1", 0, 10, 20, 30), DayForecast("d2", 9, 90, 20, 30)]
    a = make_agent(parts, forecast=rainy)
    a.handle("u4", location=(17.4, 78.5))
    assert "wash off" in a.handle("u4", text="Is it ok to spray tomorrow?").text


def test_photo_path_uses_classifier_and_retrieves(parts):
    a = make_agent(parts, classifier=FakeClassifier("tomato__late_blight", 0.93))
    r = a.handle("u5", image=b"fake")
    assert r.sources and any("tomato" in s["doc_id"] for s in r.sources)


def test_unsure_photo_asks_for_better_photo(parts):
    a = make_agent(parts, classifier=FakeClassifier(None, 0.3))
    r = a.handle("u6", image=b"fake")
    assert r.abstained and "photo" in r.text.lower()


def test_webhook_dedupes_and_replies(parts, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("PINGFARM_ALLOW_UNSIGNED", "1")  # this test posts without a Meta signature

    from pingfarm.app import create_app
    from pingfarm.whatsapp import WhatsAppClient

    wa = WhatsAppClient()  # dry run
    client = TestClient(create_app(make_agent(parts), wa))
    payload = {"entry": [{"changes": [{"value": {"messages": [
        {"id": "wamid.1", "from": "9199", "type": "text", "text": {"body": "How do I control leaf curl virus in tomato?"}}]}}]}]}
    for _ in range(3):  # Meta retries the same message
        assert client.post("/webhook", json=payload).status_code == 200
    assert len(wa.sent) == 1


def test_twilio_webhook_checks_signature_dedupes_and_replies(parts, monkeypatch):
    import base64
    import hashlib
    import hmac

    from fastapi.testclient import TestClient

    from pingfarm.app import create_app
    from pingfarm.twilio_channel import TwilioClient
    from pingfarm.whatsapp import WhatsAppClient

    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "tok")
    tw = TwilioClient(dry_run=True)
    client = TestClient(create_app(make_agent(parts), WhatsAppClient(), tw))
    form = {"From": "whatsapp:+919999999999", "MessageSid": "SMabc", "NumMedia": "0",
            "Body": "How do I control leaf curl virus in tomato?"}
    url = "http://testserver/twilio"  # what the test client calls; Twilio signs the public address it called
    sig = base64.b64encode(hmac.new(b"tok", (url + "".join(k + form[k] for k in sorted(form))).encode(), hashlib.sha1).digest()).decode()

    assert client.post("/twilio", data=form, headers={"X-Twilio-Signature": "wrong"}).status_code == 403
    assert client.post("/twilio", data=form).status_code == 403
    for _ in range(3):  # Twilio can retry the same message
        r = client.post("/twilio", data=form, headers={"X-Twilio-Signature": sig})
        assert r.status_code == 200 and "<Response" in r.text
    assert len(tw.sent) == 1 and tw.sent[0][0] == "+919999999999"


# ---------------------------------------------------------------- profile, outlook, alerts, deletion
def _risky():
    from pingfarm.tools.weather import DayForecast
    return [DayForecast(f"2026-10-2{i}", 6, 80, 12, 22, 93) for i in range(1, 5)]


def test_soil_is_remembered_and_outlook_uses_it(parts):
    a = make_agent(parts, forecast=_risky())
    a.handle("p1", location=(17.4, 78.5))
    a.handle("p1", text="I grow potato in my field")
    r = a.handle("p1", text="my soil is black soil")
    assert "soil: black" in r.text
    r = a.handle("p1", text="What should I do this week for my potato given the weather?")
    assert "late blight" in r.text.lower() and "Soil (black)" in r.text


def test_outlook_asks_for_location_first(parts):
    a = make_agent(parts)
    assert "location" in a.handle("p2", text="Which crop should I grow now, tomato or potato?").text.lower()


def test_alerts_need_location_and_crop_then_turn_on(parts):
    a = make_agent(parts)
    assert "location" in a.handle("p3", text="alerts on").text.lower()
    a.handle("p3", location=(17.4, 78.5))
    a.handle("p3", text="I grow tomato in my field")
    assert "ON" in a.handle("p3", text="alerts on").text
    assert a.state.get_user("p3").alerts is True
    assert "OFF" in a.handle("p3", text="alerts off").text


def test_delete_my_data(parts):
    a = make_agent(parts)
    a.handle("p4", location=(17.4, 78.5))
    assert "deleted" in a.handle("p4", text="delete my data").text.lower()
    assert a.state.get_user("p4").lat is None


def test_romanised_price_question_is_a_price_intent(parts):
    assert make_agent(parts).detect_intent("aaj tamatar ka mandi bhav kya hai") == "price"


def test_a_place_typed_as_words_works_like_a_location_pin(parts):
    """Telegram (and plain web chat) users can type where they are; no location pin is needed."""
    a = make_agent(parts)
    a.geocode_fn = lambda name, state, cache: (19.99, 73.79) if name.lower() == "nashik" else None
    first = a.handle("tg1", text="Can I spray tomorrow or will it rain?")
    assert "village or town" in first.text and "WhatsApp" not in first.text
    saved = a.handle("tg1", text="Nashik, Maharashtra")                      # a plain reply right after we asked
    assert "Nashik, Maharashtra" in saved.text
    assert a.state.get_user("tg1").lat == 19.99
    again = a.handle("tg1", text="Can I spray tomorrow or will it rain?")
    assert again.sources and again.sources[0]["doc_id"] == "open-meteo"      # now answered from the forecast


def test_unknown_place_is_asked_again_and_ordinary_questions_are_not_taken_for_places(parts):
    a = make_agent(parts)
    a.geocode_fn = lambda name, state, cache: None
    a.handle("tg2", text="Can I spray tomorrow or will it rain?")
    r = a.handle("tg2", text="I live in Nowhereville")
    assert "could not find" in r.text and a.state.get_user("tg2").lat is None
    b = make_agent(parts)
    b.geocode_fn = lambda *args: (_ for _ in ()).throw(AssertionError("must not geocode a question"))
    assert b.handle("tg3", text="How do I control leaf curl virus in tomato?").sources  # never treated as a place


def test_parse_place_phrases():
    from pingfarm.profile import parse_place

    assert parse_place("I live in Nashik, Maharashtra") == ("Nashik", "Maharashtra")
    assert parse_place("my village is Kothur") == ("Kothur", None)
    assert parse_place("Nashik", awaiting=False) is None                      # a bare word is not a place unless we asked
    assert parse_place("Nashik", awaiting=True) == ("Nashik", None)
    assert parse_place("how much per litre?", awaiting=True) is None          # a question is never a place
