"""WhatsApp through Twilio's WhatsApp sandbox (a second way in, next to Meta's own Cloud API).

Why: Meta only delivers a bot's replies once the business behind it is verified. Twilio's sandbox needs no business
verification, so it is the quickest way to show the bot working on a real phone.

    TWILIO_ACCOUNT_SID     starts with AC...
    TWILIO_AUTH_TOKEN      also signs every webhook request we receive (no token = every request rejected)
    TWILIO_WHATSAPP_FROM   the sandbox number, default whatsapp:+14155238886

Differences from Meta: Twilio posts a plain web form (not JSON), signs it with HMAC-SHA1 over the full URL plus the
sorted form fields, and a photo arrives as a web address we must fetch with our account credentials.
"""
import base64
import hashlib
import hmac
import os

import httpx

from .whatsapp import Incoming

API = "https://api.twilio.com/2010-04-01"
SANDBOX_NUMBER = "whatsapp:+14155238886"
MAX_BODY = 1500  # WhatsApp through Twilio allows 1600 characters


def verify_twilio_signature(url: str, params: dict[str, str], signature: str | None, auth_token: str | None) -> bool:
    """True only if X-Twilio-Signature matches. No auth token configured means every request is rejected
    (PINGFARM_ALLOW_UNSIGNED=1 switches that off for local development only)."""
    if not auth_token:
        return os.environ.get("PINGFARM_ALLOW_UNSIGNED") == "1"
    if not signature:
        return False
    data = url + "".join(k + params[k] for k in sorted(params))
    expected = base64.b64encode(hmac.new(auth_token.encode(), data.encode("utf-8"), hashlib.sha1).digest()).decode()
    return hmac.compare_digest(expected, signature)


def parse_twilio(form: dict[str, str]) -> list[Incoming]:
    """Turn Twilio's form fields into the same Incoming message the Meta channel produces."""
    sender = form.get("From", "").removeprefix("whatsapp:")
    msg_id = form.get("MessageSid") or form.get("SmsMessageSid") or ""
    if not sender or not msg_id:
        return []
    inc = Incoming(msg_id=msg_id, sender=sender, kind="other")
    if form.get("Latitude") and form.get("Longitude"):
        try:
            inc.kind, inc.location = "location", (float(form["Latitude"]), float(form["Longitude"]))
        except ValueError:
            pass
    elif int(form.get("NumMedia", "0") or 0) > 0:
        kind = form.get("MediaContentType0", "")
        inc.media_id = form.get("MediaUrl0")
        inc.text = form.get("Body") or None
        inc.kind = "image" if kind.startswith("image/") else "audio" if kind.startswith("audio/") else "other"
    elif form.get("Body"):
        inc.kind, inc.text = "text", form["Body"]
    return [inc] if inc.kind != "other" or inc.media_id else []


class TwilioClient:
    """Same two methods the app uses on the Meta client: send_text and download_media."""

    def __init__(self, sid: str | None = None, token: str | None = None, from_number: str | None = None,
                 dry_run: bool | None = None):
        self.sid = sid or os.environ.get("TWILIO_ACCOUNT_SID")
        self.token = token or os.environ.get("TWILIO_AUTH_TOKEN")
        self.from_number = from_number or os.environ.get("TWILIO_WHATSAPP_FROM") or SANDBOX_NUMBER
        self.dry_run = dry_run if dry_run is not None else not (self.sid and self.token)
        self.sent: list[tuple[str, str]] = []

    def send_text(self, to: str, body: str):
        self.sent.append((to, body))
        if self.dry_run:
            return
        number = to if to.startswith("+") else "+" + to
        httpx.post(
            f"{API}/Accounts/{self.sid}/Messages.json",
            auth=(self.sid, self.token),
            data={"From": self.from_number, "To": "whatsapp:" + number, "Body": body[:MAX_BODY]},
            timeout=20,
        ).raise_for_status()

    def download_media(self, url: str) -> bytes:
        return httpx.get(url, auth=(self.sid, self.token), follow_redirects=True, timeout=60).raise_for_status().content
