import httpx

from pingfarm.generation import ExtractiveAnswerer, GeminiAnswerer, default_answerer
from pingfarm.rag.documents import Chunk
from pingfarm.rag.search import Hit


def _hit(text):
    return Hit(Chunk("d1", 0, text, "Early blight", "http://x"), 1.0)


def _answerer(reply_text):
    def handler(request):
        assert request.headers["x-goog-api-key"] == "test-key"
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": reply_text}]}}]})

    return GeminiAnswerer(client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_gemini_answer_is_dose_guarded(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    hit = _hit("Spray mancozeb at 2.5 g per litre of water.")
    ans = _answerer("Spray mancozeb at 2.5 g per litre [1]. Add 900 ml of copper per acre.").answer("q", [hit])
    assert "2.5 g" in ans.text and "900" not in ans.text
    assert ans.guard_triggered


def test_default_answerer_prefers_gemini_then_falls_back(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert isinstance(default_answerer(), ExtractiveAnswerer)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    chosen = default_answerer()
    assert type(chosen).__name__ == "ResilientAnswerer" and isinstance(chosen.primary, GeminiAnswerer)


def _failing(status):
    def handler(request):
        return httpx.Response(status, json={"error": {"message": "x"}})

    return GeminiAnswerer(client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_outage_falls_back_to_quoting_the_sources_after_retries():
    from pingfarm.generation import ResilientAnswerer

    naps = []
    answerer = ResilientAnswerer(_failing(503), retries=2, sleep=naps.append)
    ans = answerer.answer("q", [_hit("Spray mancozeb at 2.5 g per litre.")])
    assert "trusted sources say" in ans.text and ans.sources  # quoted the source instead of failing
    assert len(naps) == 2  # tried 3 times in all, waiting between tries


def test_a_wrong_key_is_not_retried():
    from pingfarm.generation import ResilientAnswerer

    naps = []
    ans = ResilientAnswerer(_failing(403), retries=2, sleep=naps.append).answer("q", [_hit("Tomato text.")])
    assert naps == [] and "trusted sources say" in ans.text


def test_a_working_model_is_used_as_is(monkeypatch):
    from pingfarm.generation import ResilientAnswerer

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    ans = ResilientAnswerer(_answerer("Spray at 2.5 g per litre [1]."), sleep=lambda s: None).answer(
        "q", [_hit("Spray mancozeb at 2.5 g per litre.")])
    assert ans.text.startswith("Spray at 2.5 g")


def test_next_model_is_tried_when_the_first_is_overloaded(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    seen = []

    def handler(request):
        seen.append(request.url.path)
        if "model-a" in request.url.path:
            return httpx.Response(503, json={"error": {"message": "high demand"}})
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "Fine [1]."}]}}]})

    answerer = GeminiAnswerer(model="model-a, model-b", client=httpx.Client(transport=httpx.MockTransport(handler)))
    ans = answerer.answer("q", [_hit("Tomato text.")])
    assert ans.text.startswith("Fine") and len(seen) == 2 and "model-a" in seen[0] and "model-b" in seen[1]
