"""The web service.

    uvicorn pingfarm.app:app --reload

Endpoints
    GET  /                 a simple chat page (demo without WhatsApp)
    GET  /health           is it alive, and which parts are switched on
    POST /ask              JSON {user_id, text, lat, lon}  -> reply   (for testing)
    POST /ask/photo        multipart photo upload          -> reply
    GET  /webhook          Meta's verification handshake
    POST /webhook          incoming WhatsApp messages
"""
import os
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel

from .agent import Agent
from .generation import default_answerer
from .rag.chunking import chunk_all
from .rag.documents import load_documents
from .rag.embed import Embedder
from .rag.search import Retriever
from .state import State
from .whatsapp import WhatsAppClient, parse_incoming, verify_signature

ROOT = Path(__file__).resolve().parents[1]

# Secrets live in a local .env file (git-ignored), never in the code. Real environment variables win.
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env", override=False)


def build_agent(state: State | None = None, answerer=None, classifier=None) -> Agent:
    """Wire everything together. Config from our experiments: fixed-200 chunks + header, dense search + reranker
    (hit@3 0.77 on the starter set). The abstain threshold was calibrated on this same retriever."""
    docs = load_documents()
    chunks = chunk_all(docs, "fixed", header=True, size=200, overlap=40)
    embedder = Embedder()
    retriever = Retriever(chunks, embedder)
    return Agent(retriever, answerer or default_answerer(), embedder, state or State(ROOT / "data" / "pingfarm.sqlite"),
                 classifier=classifier, rerank=True)


def _load_classifier():
    path = ROOT / "models" / "leaf_classifier.pt"
    if not path.exists():
        return None
    from .vision.classifier import LeafClassifier

    return LeafClassifier.load(path)


class AskBody(BaseModel):
    user_id: str = "demo"
    text: str | None = None
    lat: float | None = None
    lon: float | None = None


def create_app(agent: Agent | None = None, wa: WhatsAppClient | None = None) -> FastAPI:
    app = FastAPI(title="PingFarm")
    app.state.agent = agent
    app.state.wa = wa or WhatsAppClient()

    def get_agent() -> Agent:
        if app.state.agent is None:  # built on first use so importing the module stays fast
            app.state.agent = build_agent(classifier=_load_classifier())
        return app.state.agent

    @app.get("/health")
    def health():
        a = app.state.agent
        return {"ok": True, "agent_loaded": a is not None,
                "photo_diagnosis": bool(a and a.classifier),
                "llm": (getattr(a.answerer, "label", None) or type(a.answerer).__name__) if a else None,
                "whatsapp_dry_run": app.state.wa.dry_run}

    @app.post("/ask")
    def ask(body: AskBody):
        loc = (body.lat, body.lon) if body.lat is not None and body.lon is not None else None
        r = get_agent().handle(body.user_id, text=body.text, location=loc)
        return {"text": r.text, "sources": r.sources, "abstained": r.abstained, "trace": r.trace}

    @app.post("/ask/photo")
    async def ask_photo(user_id: str = Form("demo"), photo: UploadFile = File(...)):
        data = await photo.read()
        if len(data) > 8_000_000:
            raise HTTPException(413, "photo too large")
        r = get_agent().handle(user_id, image=data)
        return {"text": r.text, "sources": r.sources, "abstained": r.abstained, "trace": r.trace}

    @app.get("/webhook")
    def verify(mode: str = Query("", alias="hub.mode"), token: str = Query("", alias="hub.verify_token"),
               challenge: str = Query("", alias="hub.challenge")):
        expected = os.environ.get("WHATSAPP_VERIFY_TOKEN")
        if mode == "subscribe" and expected and token == expected:
            return PlainTextResponse(challenge)
        raise HTTPException(403, "verification failed")

    @app.post("/webhook")
    async def webhook(request: Request, background: BackgroundTasks):
        raw = await request.body()
        if not verify_signature(raw, request.headers.get("X-Hub-Signature-256"), os.environ.get("WHATSAPP_APP_SECRET")):
            raise HTTPException(403, "bad signature")
        # Answer Meta immediately (it retries if we are slow); do the real work in the background.
        for msg in parse_incoming(await request.json()):
            background.add_task(process_message, app, msg)
        return {"status": "received"}

    @app.get("/", response_class=HTMLResponse)
    def home():
        return (ROOT / "pingfarm" / "static" / "chat.html").read_text(encoding="utf-8")

    return app


def process_message(app: FastAPI, msg) -> None:
    agent, wa = app.state.agent or build_agent(classifier=_load_classifier()), app.state.wa
    app.state.agent = agent
    if not agent.state.first_time_seeing(msg.msg_id):
        return  # a retry of a message we already handled
    try:
        if msg.kind == "audio":
            wa.send_text(msg.sender, "Voice notes are coming soon. For now please type your question or send a photo.")
            return
        if msg.kind == "image":
            reply = agent.handle(msg.sender, image=wa.download_media(msg.media_id))
        elif msg.kind == "location":
            reply = agent.handle(msg.sender, location=msg.location)
        elif msg.kind == "text":
            reply = agent.handle(msg.sender, text=msg.text)
        else:
            return
        body = reply.text
        if reply.sources:
            body += "\n\nSources: " + "; ".join(s["title"] for s in reply.sources[:2])
        wa.send_text(msg.sender, body)
    except Exception:  # never leave a farmer without any answer
        wa.send_text(msg.sender, "Sorry, something went wrong on my side. Please try again in a little while.")
        raise


app = create_app()
