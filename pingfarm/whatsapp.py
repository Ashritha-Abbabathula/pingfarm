"""WhatsApp Cloud API plumbing (Meta). All secrets come from environment variables.

    WHATSAPP_TOKEN         access token
    WHATSAPP_PHONE_ID      the sending phone-number id
    WHATSAPP_VERIFY_TOKEN  a string YOU choose; Meta echoes it when you register the webhook
    WHATSAPP_APP_SECRET    required: lets us verify each request really came from Meta (no secret = every request rejected)

Gotchas this module handles:
  * Meta retries slow webhooks -> the caller de-duplicates by message id (state.first_time_seeing)
  * photos/voice notes arrive as an id; you must call the API twice, with your token, to get the bytes
  * every webhook request is signed; we reject unsigned or wrongly signed ones (and all of them if no app secret is set)
"""
import hashlib
import hmac
import os
from dataclasses import dataclass

import httpx

GRAPH = "https://graph.facebook.com/v20.0"


@dataclass
class Incoming:
    msg_id: str
    sender: str
    kind: str                      # text | image | location | audio | other
    text: str | None = None
    media_id: str | None = None
    location: tuple[float, float] | None = None


def verify_signature(raw_body: bytes, header: str | None, app_secret: str | None) -> bool:
    """True only if the request is signed correctly.

    With no app secret configured every request is REJECTED, because anyone who finds a public webhook address
    could otherwise post fake messages. For local development only, set PINGFARM_ALLOW_UNSIGNED=1."""
    if not app_secret:
        return os.environ.get("PINGFARM_ALLOW_UNSIGNED") == "1"
    if not header or not header.startswith("sha256="):
        return False
    expected = hmac.new(app_secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header.removeprefix("sha256="))


def parse_statuses(payload: dict) -> list[tuple[str, list[tuple[int | None, str]]]]:
    """Delivery reports Meta sends about messages WE sent: (status, [(error code, error title)]).
    Phone numbers are deliberately left out so they never reach a log file."""
    out = []
    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            for st in change.get("value", {}).get("statuses", []):
                errors = [(e.get("code"), e.get("title") or e.get("message", "")) for e in st.get("errors", [])]
                out.append((st.get("status", "?"), errors))
    return out


def parse_incoming(payload: dict) -> list[Incoming]:
    """Pull the user messages out of Meta's nested webhook JSON."""
    out = []
    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            for m in change.get("value", {}).get("messages", []) or []:
                kind = m.get("type", "other")
                inc = Incoming(msg_id=m["id"], sender=m["from"], kind=kind)
                if kind == "text":
                    inc.text = m["text"]["body"]
                elif kind == "image":
                    inc.media_id = m["image"]["id"]
                    inc.text = m["image"].get("caption")
                elif kind == "audio":
                    inc.media_id = m["audio"]["id"]
                elif kind == "location":
                    inc.location = (m["location"]["latitude"], m["location"]["longitude"])
                else:
                    inc.kind = "other"
                out.append(inc)
    return out


class WhatsAppClient:
    def __init__(self, token: str | None = None, phone_id: str | None = None, dry_run: bool | None = None):
        self.token = token or os.environ.get("WHATSAPP_TOKEN")
        self.phone_id = phone_id or os.environ.get("WHATSAPP_PHONE_ID")
        # With no credentials we only log what we WOULD send. Safe for demos and tests.
        self.dry_run = dry_run if dry_run is not None else not (self.token and self.phone_id)
        self.sent: list[tuple[str, str]] = []

    def send_text(self, to: str, body: str):
        self.sent.append((to, body))
        if self.dry_run:
            return
        httpx.post(
            f"{GRAPH}/{self.phone_id}/messages",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"messaging_product": "whatsapp", "to": to, "type": "text", "text": {"body": body[:4000]}},
            timeout=20,
        ).raise_for_status()

    def send_template(self, to: str, name: str, params: list[str], language: str = "en"):
        """Send a pre-approved template message (the only kind allowed outside the 24-hour window)."""
        self.sent.append((to, f"[template:{name}] " + " | ".join(params)))
        if self.dry_run:
            return
        httpx.post(
            f"{GRAPH}/{self.phone_id}/messages",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"messaging_product": "whatsapp", "to": to, "type": "template",
                  "template": {"name": name, "language": {"code": language},
                               "components": [{"type": "body", "parameters": [{"type": "text", "text": p} for p in params]}]}},
            timeout=20,
        ).raise_for_status()

    def download_media(self, media_id: str) -> bytes:
        headers = {"Authorization": f"Bearer {self.token}"}
        meta = httpx.get(f"{GRAPH}/{media_id}", headers=headers, timeout=20).raise_for_status().json()
        return httpx.get(meta["url"], headers=headers, timeout=60).raise_for_status().content
