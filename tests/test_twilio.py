"""Twilio WhatsApp sandbox channel: signature check, message parsing and sending (no network)."""
import base64
import hashlib
import hmac
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pingfarm.twilio_channel import TwilioClient, parse_twilio, verify_twilio_signature

URL = "https://example.trycloudflare.com/twilio"
PARAMS = {"From": "whatsapp:+919999999999", "Body": "tomato leaf curl", "MessageSid": "SM1", "NumMedia": "0"}


def sign(url, params, token):
    """Twilio's documented recipe, written out independently of the code under test."""
    data = url + "".join(k + params[k] for k in sorted(params))
    return base64.b64encode(hmac.new(token.encode(), data.encode(), hashlib.sha1).digest()).decode()


def test_good_signature_passes_and_tampering_fails():
    sig = sign(URL, PARAMS, "tok")
    assert verify_twilio_signature(URL, PARAMS, sig, "tok")
    assert not verify_twilio_signature(URL, {**PARAMS, "Body": "something else"}, sig, "tok")
    assert not verify_twilio_signature(URL, PARAMS, sig, "other-token")
    assert not verify_twilio_signature("https://elsewhere.example/twilio", PARAMS, sig, "tok")
    assert not verify_twilio_signature(URL, PARAMS, None, "tok")


def test_no_token_rejects_everything_unless_local_dev_flag(monkeypatch):
    assert not verify_twilio_signature(URL, PARAMS, "anything", None)
    monkeypatch.setenv("PINGFARM_ALLOW_UNSIGNED", "1")
    assert verify_twilio_signature(URL, PARAMS, None, None)
    assert not verify_twilio_signature(URL, PARAMS, None, "tok")  # a configured token is always enforced


def test_parse_text_location_image_audio_and_empty():
    text = parse_twilio(PARAMS)[0]
    assert (text.kind, text.text, text.sender, text.msg_id) == ("text", "tomato leaf curl", "+919999999999", "SM1")

    loc = parse_twilio({"From": "whatsapp:+91", "MessageSid": "SM2", "Latitude": "17.4", "Longitude": "78.5"})[0]
    assert loc.kind == "location" and loc.location == (17.4, 78.5)

    img = parse_twilio({"From": "whatsapp:+91", "MessageSid": "SM3", "NumMedia": "1", "MediaContentType0": "image/jpeg",
                        "MediaUrl0": "https://api.twilio.com/media/1", "Body": ""})[0]
    assert img.kind == "image" and img.media_id == "https://api.twilio.com/media/1"

    audio = parse_twilio({"From": "whatsapp:+91", "MessageSid": "SM4", "NumMedia": "1", "MediaContentType0": "audio/ogg",
                          "MediaUrl0": "https://api.twilio.com/media/2"})[0]
    assert audio.kind == "audio"

    assert parse_twilio({"From": "whatsapp:+91", "MessageSid": "SM5"}) == []  # nothing in it
    assert parse_twilio({}) == []


def test_dry_run_client_records_and_never_calls_the_network(monkeypatch):
    for name in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    client = TwilioClient()
    assert client.dry_run and client.from_number == "whatsapp:+14155238886"
    client.send_text("+919999999999", "hello")
    assert client.sent == [("+919999999999", "hello")]
