"""Telegram as a third way in (next to Meta's WhatsApp Cloud API and Twilio's WhatsApp sandbox).

Why: Telegram bots are free and need no business verification, no payment and no public web address. The bot asks
Telegram for new messages ("long polling") instead of waiting for Telegram to call us, so it works from a laptop
with nothing else running.

    TELEGRAM_BOT_TOKEN   from @BotFather in Telegram (looks like 123456:ABC...). It is a secret: only in .env.

Messages become the same Incoming object the other channels produce, so the agent does not know the difference.
"""
import logging
import os
import threading

import httpx

from .whatsapp import Incoming

API = "https://api.telegram.org"
MAX_BODY = 4000  # Telegram allows 4096 characters per message
log = logging.getLogger("uvicorn.error")

# httpx logs every request address at INFO level, and Telegram's address contains the bot token.
logging.getLogger("httpx").setLevel(logging.WARNING)


def parse_update(update: dict) -> list[Incoming]:
    """One Telegram update -> zero or one Incoming. Ids are prefixed so they never clash with other channels."""
    m = update.get("message")
    if not m or "chat" not in m:
        return []
    msg_id = f"tg:{update.get('update_id')}"
    inc = Incoming(msg_id=msg_id, sender=str(m["chat"]["id"]), kind="other")
    if m.get("location"):
        inc.kind, inc.location = "location", (float(m["location"]["latitude"]), float(m["location"]["longitude"]))
    elif m.get("photo"):
        inc.kind, inc.media_id, inc.text = "image", m["photo"][-1]["file_id"], m.get("caption")  # last = biggest size
    elif m.get("voice") or m.get("audio"):
        inc.kind = "audio"
    elif m.get("text"):
        text = m["text"].strip()
        if text.startswith("/start"):
            text = "hello"  # Telegram sends /start when someone opens the bot: treat it as a greeting
        inc.kind, inc.text = "text", text
    return [inc] if inc.kind != "other" else []


class TelegramClient:
    """Same two methods the app uses on the other clients (send_text, download_media), plus get_updates."""

    def __init__(self, token: str | None = None, dry_run: bool | None = None):
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self.dry_run = dry_run if dry_run is not None else not self.token
        self.sent: list[tuple[str, str]] = []

    def _url(self, method: str) -> str:
        return f"{API}/bot{self.token}/{method}"

    def send_text(self, to: str, body: str):
        self.sent.append((to, body))
        if self.dry_run:
            return
        for i in range(0, len(body), MAX_BODY) or [0]:
            httpx.post(self._url("sendMessage"), json={"chat_id": to, "text": body[i:i + MAX_BODY]},
                       timeout=20).raise_for_status()

    def download_media(self, file_id: str) -> bytes:
        info = httpx.get(self._url("getFile"), params={"file_id": file_id}, timeout=20).raise_for_status().json()
        path = info["result"]["file_path"]
        return httpx.get(f"{API}/file/bot{self.token}/{path}", timeout=60).raise_for_status().content

    def get_updates(self, offset: int | None, wait: int = 25) -> list[dict]:
        params = {"timeout": wait, "allowed_updates": '["message"]'}
        if offset is not None:
            params["offset"] = offset
        r = httpx.get(self._url("getUpdates"), params=params, timeout=wait + 10).raise_for_status().json()
        return r.get("result", [])


def poll_forever(app, client: TelegramClient, stop: threading.Event, handler) -> None:
    """Fetch new messages and hand each to `handler(app, msg, client)`. Survives network errors."""
    offset = None
    while not stop.is_set():
        try:
            for update in client.get_updates(offset):
                offset = update["update_id"] + 1
                for msg in parse_update(update):
                    try:
                        handler(app, msg, client)
                    except Exception:
                        log.exception("telegram message failed")
        except Exception as e:  # network blip, Telegram hiccup: wait and try again
            log.warning("telegram poll error: %s", type(e).__name__)  # never log the exception text: it holds the token
            stop.wait(5)
