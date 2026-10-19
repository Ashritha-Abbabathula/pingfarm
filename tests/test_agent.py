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


def test_webhook_dedupes_and_replies(parts):
    from fastapi.testclient import TestClient

    from pingfarm.app import create_app
    from pingfarm.whatsapp import WhatsAppClient

    wa = WhatsAppClient()  # dry run
    client = TestClient(create_app(make_agent(parts), wa))
    payload = {"entry": [{"changes": [{"value": {"messages": [
        {"id": "wamid.1", "from": "9199", "type": "text", "text": {"body": "How do I control leaf curl virus in tomato?"}}]}}]}]}
    for _ in range(3):  # Meta retries the same message
        assert client.post("/webhook", json=payload).status_code == 200
    assert len(wa.sent) == 1
