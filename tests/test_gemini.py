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
    assert isinstance(default_answerer(), GeminiAnswerer)
