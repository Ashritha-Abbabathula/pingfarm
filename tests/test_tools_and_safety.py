"""Tests for the tools, the safety guard, the WhatsApp plumbing and state. No network, no models."""
import hashlib
import hmac
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pingfarm.generation import REFUSAL, ExtractiveAnswerer, dose_guard
from pingfarm.rag.documents import Chunk
from pingfarm.rag.search import Hit
from pingfarm.state import State
from pingfarm.tools import prices, weather
from pingfarm.whatsapp import WhatsAppClient, parse_incoming, verify_signature


# ---------------------------------------------------------------- weather
def _forecast_client(rain_mm, chance):
    def handler(request):
        return httpx.Response(200, json={"daily": {
            "time": ["2026-10-19", "2026-10-20", "2026-10-21"],
            "precipitation_sum": [0, rain_mm, 0],
            "precipitation_probability_max": [10, chance, 5],
            "temperature_2m_max": [30, 29, 31], "temperature_2m_min": [20, 21, 20]}})
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_weather_says_wait_when_rain_expected():
    fc = weather.fetch_forecast(17.4, 78.5, client=_forecast_client(rain_mm=8, chance=80))
    advice = weather.spray_advice(fc)
    assert not advice.ok_to_spray and "2026-10-20" in advice.reason


def test_weather_ok_when_dry():
    fc = weather.fetch_forecast(17.4, 78.5, client=_forecast_client(rain_mm=0, chance=10))
    assert weather.spray_advice(fc).ok_to_spray


# ---------------------------------------------------------------- prices
def test_prices_without_key_does_not_invent_numbers(monkeypatch):
    monkeypatch.delenv("DATA_GOV_API_KEY", raising=False)
    recs = prices.fetch_prices("Tomato")
    assert recs is None
    assert "not set up" in prices.describe_prices(recs, "tomato")


def test_prices_parses_and_states_market_and_date():
    rows = {"records": [
        {"state": "Telangana", "district": "Hyderabad", "market": "Bowenpally", "commodity": "Tomato",
         "arrival_date": "18/10/2026", "min_price": "1500", "max_price": "2500", "modal_price": "2000"},
        {"state": "Telangana", "market": "Broken row without prices", "commodity": "Tomato"},
    ]}
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=rows)))
    recs = prices.fetch_prices("Tomato", "Telangana", api_key="x", client=client)
    assert len(recs) == 1                       # the incomplete row is skipped, not guessed
    text = prices.describe_prices(recs, "tomato")
    assert "Bowenpally" in text and "18/10/2026" in text and "2000" in text


# ---------------------------------------------------------------- safety guard (decision D9)
def test_guard_removes_invented_dose():
    passages = ["Spray Mancozeb 35% SC @ 1000 g/ac every 10 days."]
    answer = "Spray mancozeb at 1000 g/ac. Also add 5 ml of soap per litre."
    safe, triggered = dose_guard(answer, passages)
    assert triggered
    assert "1000" in safe and "5 ml" not in safe
    assert REFUSAL in safe


def test_guard_allows_doses_found_in_sources():
    passages = ["Seed treatment: Trichoderma asperellum @ 4g/kg seeds."]
    safe, triggered = dose_guard("Treat seed with Trichoderma at 4g/kg seeds.", passages)
    assert not triggered and "4g/kg" in safe


def test_guard_refuses_when_everything_is_invented():
    safe, triggered = dose_guard("Use 50 ml per tank.", ["No numbers here."])
    assert triggered and safe == REFUSAL


def test_extractive_answerer_cites_sources():
    chunk = Chunk("d1", 0, "Brown spots with rings. Spray Zineb 75% WP @ 800 g/ac.", "TNAU page", "http://x")
    ans = ExtractiveAnswerer().answer("q", [Hit(chunk, 1.0)])
    assert ans.sources[0]["title"] == "TNAU page" and "800 g/ac" in ans.text


# ---------------------------------------------------------------- whatsapp plumbing
PAYLOAD = {"entry": [{"changes": [{"value": {"messages": [
    {"id": "m1", "from": "9199", "type": "text", "text": {"body": "hello"}},
    {"id": "m2", "from": "9199", "type": "image", "image": {"id": "img9", "caption": "leaf"}},
    {"id": "m3", "from": "9199", "type": "location", "location": {"latitude": 17.4, "longitude": 78.5}},
    {"id": "m4", "from": "9199", "type": "sticker"},
]}}]}]}


def test_parse_incoming_all_kinds():
    msgs = parse_incoming(PAYLOAD)
    assert [m.kind for m in msgs] == ["text", "image", "location", "other"]
    assert msgs[1].media_id == "img9" and msgs[2].location == (17.4, 78.5)


def test_signature_check():
    body, secret = b'{"a":1}', "s3cret"
    good = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    assert verify_signature(body, good, secret)
    assert not verify_signature(body, "sha256=deadbeef", secret)
    assert not verify_signature(body, None, secret)
    assert not verify_signature(body, None, None)  # no secret configured: fail closed


def test_unsigned_requests_only_allowed_when_explicitly_enabled(monkeypatch):
    monkeypatch.delenv("PINGFARM_ALLOW_UNSIGNED", raising=False)
    assert not verify_signature(b"{}", None, None)
    monkeypatch.setenv("PINGFARM_ALLOW_UNSIGNED", "1")
    assert verify_signature(b"{}", None, None)
    assert not verify_signature(b"{}", None, "s3cret")  # a configured secret is always enforced


def test_dry_run_client_never_calls_network():
    wa = WhatsAppClient(token=None, phone_id=None)
    wa.send_text("9199", "hi")
    assert wa.dry_run and wa.sent == [("9199", "hi")]


# ---------------------------------------------------------------- state
def test_dedupe_and_profile():
    st = State()
    assert st.first_time_seeing("m1") is True
    assert st.first_time_seeing("m1") is False          # Meta retry
    st.update_user("u", lat=1.0, lon=2.0)
    st.update_user("u", crop="tomato")                  # does not erase the location
    u = st.get_user("u")
    assert (u.lat, u.lon, u.crop) == (1.0, 2.0, "tomato")
    st.add_history("u", "user", "hi"); st.add_history("u", "bot", "hello")
    assert st.recent_history("u") == [("user", "hi"), ("bot", "hello")]


def test_guard_catches_hindi_and_telugu_doses():
    passages = ["Spray Mancozeb @ 1000 g/ac."]
    safe, triggered = dose_guard("प्रति एकड़ २०० मि.ली. दवा डालें।", passages)      # Hindi numerals + unit
    assert triggered and "२००" not in safe
    safe, triggered = dose_guard("ఎకరానికి ౫౦౦ గ్రాములు వేయండి.", passages)           # Telugu numerals + unit
    assert triggered
    # the SAME dose written in another script/unit as the source is allowed
    safe, triggered = dose_guard("प्रति एकड़ १००० ग्राम छिड़कें।", passages)
    assert not triggered


def test_privacy_page_is_public_and_shows_no_personal_contact_by_default(monkeypatch):
    from fastapi.testclient import TestClient

    from pingfarm.app import create_app

    monkeypatch.delenv("PINGFARM_CONTACT_EMAIL", raising=False)
    client = TestClient(create_app(agent=object(), wa=WhatsAppClient()))
    page = client.get("/privacy")
    assert page.status_code == 200 and "delete my data" in page.text and "@" not in page.text
    monkeypatch.setenv("PINGFARM_CONTACT_EMAIL", "me@example.org")
    assert "me@example.org" in client.get("/privacy").text


def test_delivery_reports_are_parsed_without_phone_numbers():
    from pingfarm.whatsapp import parse_statuses

    payload = {"entry": [{"changes": [{"value": {"statuses": [
        {"status": "delivered", "recipient_id": "919999999999"},
        {"status": "failed", "recipient_id": "919999999999", "errors": [{"code": 131047, "title": "Re-engagement message"}]},
    ]}}]}]}
    assert parse_statuses(payload) == [("delivered", []), ("failed", [(131047, "Re-engagement message")])]
    assert "9199" not in str(parse_statuses(payload))
    assert parse_statuses({"entry": [{"changes": [{"value": {"messages": []}}]}]}) == []
