"""Dose questions are advice, not price. Uses the real embedding model and the saved router (offline)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pingfarm.intents import IntentClassifier
from pingfarm.rag.embed import Embedder


@pytest.fixture(scope="module")
def route():
    emb, clf = Embedder(), IntentClassifier.load()
    return lambda text: clf.detect(emb.embed_queries([text])[0])


@pytest.mark.parametrize("text", [
    "How much mancozeb should I mix per litre for potato late blight?",
    "how much dimethoate per acre for tomato whitefly",
    "What quantity of fungicide should I spray on potato?",
    "టమాటాకు ఎంత మందు వాడాలి",
])
def test_how_much_medicine_is_advice(route, text):
    assert route(text) == "advice"


@pytest.mark.parametrize("text", [
    "How much does tomato cost today?",
    "What is the price of potato in the mandi?",
    "आज आलू का भाव क्या है?",
])
def test_real_price_questions_still_go_to_price(route, text):
    assert route(text) == "price"
