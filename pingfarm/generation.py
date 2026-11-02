"""Writing the final answer, and making sure it is safe.

Answerers share one interface (GeminiAnswerer is the same idea as ClaudeAnswerer, with Google's API):
  ClaudeAnswerer      : a language model writes a short reply, in the farmer's language,
                        using ONLY the retrieved passages.
  ExtractiveAnswerer  : no AI at all. Quotes the most relevant passages with their source.
                        Used when there is no API key, in tests, and as a fallback.

Whichever one writes the answer, `dose_guard` then checks it. This is Decision D9:
NEVER state a pesticide dose that is not written in a retrieved source.
"""
import os
import re
from dataclasses import dataclass

from .rag.search import Hit

REFUSAL = ("I am not sure about the exact amount to use. Please check the label on the product or ask your "
           "local agriculture officer (Krishi Vigyan Kendra) before spraying.")

SYSTEM_PROMPT = """You are PingFarm, a careful assistant for small farmers in India.

Rules you must follow:
1. Answer ONLY from the numbered passages provided. If they do not answer the question, say you are not sure
   and suggest asking a local agriculture officer. Never guess.
2. NEVER state a pesticide, fungicide or fertiliser amount, dose, or concentration unless it is written in
   the passages. If a passage gives one, copy it exactly with its unit.
3. Reply in the same language as the farmer's question (English, Hindi or Telugu). Keep it short and in plain
   words a farmer would use. Use short steps.
4. After each fact, cite its passage like [1]. Do not cite passages you did not use.
5. Never recommend banned products or mixing chemicals. When unsure, recommend asking an expert."""


@dataclass
class Answer:
    text: str
    sources: list[dict]
    guard_triggered: bool = False


def sources_from(hits: list[Hit]) -> list[dict]:
    seen, out = set(), []
    for h in hits:
        if h.chunk.doc_id in seen:
            continue
        seen.add(h.chunk.doc_id)
        out.append({"title": h.chunk.title, "url": h.chunk.url, "doc_id": h.chunk.doc_id})
    return out


# --------------------------------------------------------------------- safety guard
# amount + unit, e.g. "200 ml/ac", "0.2 %", "4g/kg", "2.5 kg per hectare", "500 ppm".
# Hindi and Telugu numerals and unit words are recognised too, so a reply in the farmer's language
# cannot smuggle in an invented dose.
_INDIC_DIGITS = str.maketrans("०१२३४५६७८९౦౧౨౩౪౫౬౭౮౯", "01234567890123456789")
_UNIT_CANON = {
    "ml": "ml", "मिली": "ml", "मि.ली.": "ml", "मि.ली": "ml", "మి.లీ.": "ml", "మి.లీ": "ml", "మిల్లీ": "ml",
    "l": "l", "litre": "l", "litres": "l", "liter": "l", "liters": "l", "लीटर": "l", "లీటర్": "l", "లీటర్లు": "l",
    "kg": "kg", "किलो": "kg", "కిలో": "kg",
    "g": "g", "gm": "g", "ग्राम": "g", "గ్రాము": "g", "గ్రాములు": "g",
    "mg": "mg", "ppm": "ppm", "%": "%", "percent": "%", "प्रतिशत": "%", "శాతం": "%",
}
_UNITS = sorted(_UNIT_CANON, key=len, reverse=True)  # longest first so "ml" is not read as "m"
_DOSE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(" + "|".join(re.escape(u) for u in _UNITS) + r")(?![A-Za-z])", re.IGNORECASE
)


def _doses_in(text: str) -> set[str]:
    text = text.translate(_INDIC_DIGITS)
    return {f"{n.replace(',', '.')}{_UNIT_CANON[u.lower()]}" for n, u in _DOSE.findall(text)}


def dose_guard(answer: str, passages: list[str]) -> tuple[str, bool]:
    """Remove any sentence that states a dose not present in the passages.

    Returns (safe_answer, triggered). If every sentence is removed, returns the refusal message.
    Known limit: it recognises Latin digits and common units only (not Devanagari or Telugu numerals).
    """
    allowed = set()
    for p in passages:
        allowed |= _doses_in(p)
    kept, triggered = [], False
    for sentence in re.split(r"(?<=[.!?।])\s+|\n+", answer):
        if not sentence.strip():
            continue
        if _doses_in(sentence) - allowed:
            triggered = True
            continue
        kept.append(sentence.strip())
    if not kept:
        return REFUSAL, True
    safe = " ".join(kept)
    if triggered:
        safe += " " + REFUSAL
    return safe, triggered


# --------------------------------------------------------------------- answerers
class ExtractiveAnswerer:
    """Quote, do not paraphrase. Honest and fully traceable."""

    def __init__(self, n_passages: int = 2, max_chars: int = 600):
        self.n, self.max_chars = n_passages, max_chars

    def answer(self, question: str, hits: list[Hit], context: str = "") -> Answer:
        chosen = hits[: self.n]
        lines = []
        for i, h in enumerate(chosen, 1):
            text = " ".join(h.chunk.text.split())
            lines.append(f"[{i}] {text[: self.max_chars].rstrip()}...")
        body = "Here is what trusted sources say:\n" + "\n".join(lines)
        if context:
            body = context + "\n\n" + body
        safe, triggered = dose_guard(body, [h.chunk.text for h in chosen])
        return Answer(safe, sources_from(chosen), triggered)


class ClaudeAnswerer:
    def __init__(self, model: str | None = None, client=None, max_tokens: int = 500):
        import anthropic

        self.client = client or anthropic.Anthropic()
        self.model = model or os.environ.get("PINGFARM_MODEL", "claude-haiku-5-5")
        self.max_tokens = max_tokens

    def answer(self, question: str, hits: list[Hit], context: str = "") -> Answer:
        passages = "\n\n".join(f"[{i}] {' '.join(h.chunk.text.split())}" for i, h in enumerate(hits, 1))
        user = f"Passages:\n{passages}\n\n"
        if context:
            user += f"Extra facts (weather/prices):\n{context}\n\n"
        user += f"Farmer's question: {question}"
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            # the long, unchanging rules are cached so repeat questions cost less
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        safe, triggered = dose_guard(text, [h.chunk.text for h in hits] + [context])
        return Answer(safe, sources_from(hits), triggered)


class GeminiAnswerer:
    """Same job as ClaudeAnswerer, using Google's Gemini API (has a free tier). Plain HTTP, no extra package."""

    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, model: str | None = None, client=None, max_tokens: int = 1500):  # headroom: newer models count their thinking in this limit
        import httpx

        self.client = client or httpx.Client(timeout=60)
        self.model = model or os.environ.get("PINGFARM_GEMINI_MODEL", "gemini-3.8-flash")
        self.max_tokens = max_tokens

    def answer(self, question: str, hits: list[Hit], context: str = "") -> Answer:
        passages = "\n\n".join(f"[{i}] {' '.join(h.chunk.text.split())}" for i, h in enumerate(hits, 1))
        user = f"Passages:\n{passages}\n\n"
        if context:
            user += f"Extra facts (weather/prices):\n{context}\n\n"
        user += f"Farmer's question: {question}"
        resp = self.client.post(
            self.URL.format(model=self.model),
            headers={"x-goog-api-key": os.environ.get("GEMINI_API_KEY", "")},
            json={
                "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"maxOutputTokens": self.max_tokens},
            },
        )
        resp.raise_for_status()
        parts = resp.json()["candidates"][0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts)
        safe, triggered = dose_guard(text, [h.chunk.text for h in hits] + [context])
        return Answer(safe, sources_from(hits), triggered)


def default_answerer():
    """Gemini if its key is set, else Claude if its key is set, otherwise the extractive fallback."""
    if os.environ.get("GEMINI_API_KEY"):
        return GeminiAnswerer()
    if os.environ.get("ANTHROPIC_API_KEY"):
        return ClaudeAnswerer()
    return ExtractiveAnswerer()
