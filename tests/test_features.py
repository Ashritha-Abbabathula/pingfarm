"""Tests for profile parsing, disease-risk rules, suitability, markets, proactive alerts and data deletion."""
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pingfarm import profile as P
from pingfarm.alerts import run_alerts
from pingfarm.state import State
from pingfarm.tools import markets, risk, suitability
from pingfarm.tools.prices import PriceRecord
from pingfarm.tools.weather import DayForecast
from pingfarm.whatsapp import WhatsAppClient


def D(date_, rain=0.0, tmin=20.0, tmax=30.0, rh=None):
    return DayForecast(date_, rain, 0, tmin, tmax, rh)


# ---------------------------------------------------------------- profile parsing
def test_soil_in_three_languages_and_romanised():
    assert P.parse_soil("my field has black soil") == "black"
    assert P.parse_soil("मेरी काली मिट्टी है") == "black"
    assert P.parse_soil("మాది ఎర్ర నేల") == "red"
    assert P.parse_soil("nalla regadi nela") == "black"
    assert P.parse_soil("hello") is None


def test_planting_date_relative_and_absolute():
    today = date(2026, 10, 20)
    assert P.parse_planting_date("I planted 10 days ago", today) == "2026-10-10"
    assert P.parse_planting_date("sown 2 weeks back", today) == "2026-10-06"
    assert P.parse_planting_date("I planted on 5 oct", today) == "2026-10-05"
    assert P.parse_planting_date("how many days for harvest", today) is None


def test_crop_stage():
    assert "flowering" in P.crop_stage("tomato", "2026-08-20", date(2026, 10, 20))
    assert "tuber" in P.crop_stage("potato", "2026-09-01", date(2026, 10, 20))
    assert P.crop_stage("tomato", None) is None


def test_alert_and_delete_commands():
    assert P.parse_alert_command("please send alerts") is True
    assert P.parse_alert_command("alerts off") is False
    assert P.parse_alert_command("how are you") is None
    assert P.wants_deletion("delete my data")
    assert not P.wants_deletion("my data is good")


# ---------------------------------------------------------------- disease risk rules
def test_potato_damp_cool_spell_is_high_risk():
    fc = [D(f"2026-10-2{i}", rain=6, tmin=12, tmax=22, rh=93) for i in range(1, 5)]
    r = risk.assess("potato", fc)
    assert r[0].level == "high" and "late blight" in r[0].disease
    assert "TNAU" in r[0].source


def test_potato_dry_warm_is_low_risk():
    fc = [D(f"2026-10-2{i}", rain=0, tmin=22, tmax=35, rh=40) for i in range(1, 5)]
    assert risk.assess("potato", fc)[0].level == "low"


def test_tomato_hot_dry_flags_heat_stress_and_never_states_doses():
    fc = [D(f"2026-10-2{i}", rain=0, tmin=24, tmax=34, rh=35) for i in range(1, 5)]
    rs = risk.assess("tomato", fc)
    assert any("heat" in r.disease for r in rs)
    text = risk.summarize("tomato", rs)
    assert "ml" not in text.lower().replace("humid", "") or "spray amounts" in text


def test_tomato_wet_cool_nights_is_high_late_blight():
    fc = [D(f"2026-10-2{i}", rain=5, tmin=17, tmax=26, rh=92) for i in range(1, 4)]
    assert any(r.level == "high" and "late blight" in r.disease for r in risk.assess("tomato", fc))


# ---------------------------------------------------------------- suitability
def test_suitability_quotes_sources_and_marks_general_knowledge():
    fc = [D(f"2026-10-2{i}", tmin=22, tmax=34) for i in range(1, 5)]
    text = suitability.outlook("tomato", "black", fc)
    assert "6.5-7.5" not in text or "pH" in text   # pH facts are only quoted with the soil-test caveat
    assert "general knowledge" in text and "soil test" in text
    assert "Hotter than ideal" in text


def test_compare_crops_hot_vs_cool():
    hot = [D("d", tmin=26, tmax=36)] * 3
    cool = [D("d", tmin=8, tmax=20)] * 3
    assert "hot for both" in suitability.compare_crops(None, hot).lower()
    assert "potato" in suitability.compare_crops(None, cool).lower()


# ---------------------------------------------------------------- markets
def _rec(market, district, price, date_):
    return PriceRecord(market, district, "Telangana", "Tomato", date_, price, price - 100, price + 100)


def test_markets_nearest_first_with_trend_and_honest_wording():
    st = State()
    recs = [_rec("A-Mandi", "Hyderabad", 2000, "18/10/2026"), _rec("A-Mandi", "Hyderabad", 1600, "15/10/2026"),
            _rec("B-Mandi", "Warangal", 2600, "18/10/2026")]
    places = {"A-Mandi": (17.4, 78.5), "B-Mandi": (18.0, 79.6)}
    ms = markets.build_markets(recs, 17.38, 78.48, st, geocode_fn=lambda name, state, cache: places.get(name))
    assert [m.name for m in ms] == ["A-Mandi", "B-Mandi"]          # nearest first
    assert ms[0].trend_pct and ms[0].trend_pct > 20                  # 2000 vs earlier 1600
    text = markets.describe_markets("tomato", ms)
    assert "A-Mandi" in text and "18/10/2026" in text and "not always more profit" in text


def test_markets_without_api_key_returns_error_not_numbers():
    class U:  # a profile-like object
        lat, lon, state, district = 17.4, 78.5, "Telangana", "Hyderabad"
    text, err = markets.markets_for(U(), "tomato", State(), price_fn=lambda c, s, limit=500: None, place_fn=lambda a, b: (None, None))
    assert text is None and err == "no_key"


# ---------------------------------------------------------------- alerts
RISKY = [D(f"2026-10-2{i}", rain=6, tmin=12, tmax=22, rh=93) for i in range(1, 5)]
CALM = [D(f"2026-10-2{i}", rain=0, tmin=22, tmax=35, rh=40) for i in range(1, 5)]


def _opted_in_user(st, uid="u1", crop="potato"):
    st.update_user(uid, lat=17.4, lon=78.5, crop=crop, alerts=True)
    st.touch_inbound(uid)


def test_alert_sent_once_per_day_and_only_when_high_risk():
    st, wa = State(), WhatsAppClient()
    _opted_in_user(st)
    today = date(2026, 10, 20)
    assert run_alerts(st, wa, forecast_fn=lambda a, b: RISKY, today=today)[0].status == "sent"
    assert run_alerts(st, wa, forecast_fn=lambda a, b: RISKY, today=today)[0].status == "already_sent"
    assert len(wa.sent) == 1 and "alerts off" in wa.sent[0][1].lower()
    assert run_alerts(st, wa, forecast_fn=lambda a, b: CALM, today=date(2026, 10, 21))[0].status == "no_risk"


def test_alert_not_sent_without_opt_in():
    st, wa = State(), WhatsAppClient()
    st.update_user("u2", lat=17.4, lon=78.5, crop="potato", alerts=False)
    assert run_alerts(st, wa, forecast_fn=lambda a, b: RISKY) == []


def test_alert_outside_24h_window_needs_template():
    st, wa = State(), WhatsAppClient()
    _opted_in_user(st)
    long_ago = time.time() + 3 * 24 * 3600   # "now" is 3 days after the farmer last wrote
    res = run_alerts(st, wa, forecast_fn=lambda a, b: RISKY, now=long_ago, template=None)
    assert res[0].status == "blocked_window" and wa.sent == []
    res = run_alerts(st, wa, forecast_fn=lambda a, b: RISKY, now=long_ago, template="weather_alert")
    assert res[0].status == "sent" and wa.sent[0][1].startswith("[template:weather_alert]")


def test_one_farmers_failure_does_not_stop_the_rest():
    st, wa = State(), WhatsAppClient()
    _opted_in_user(st, "bad")
    _opted_in_user(st, "good")
    calls = {"n": 0}

    def flaky(lat, lon):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("forecast service down")
        return RISKY

    statuses = sorted(r.status for r in run_alerts(st, wa, forecast_fn=flaky))
    assert statuses == ["error", "sent"]


# ---------------------------------------------------------------- deletion and migration
def test_delete_user_removes_everything():
    st = State()
    _opted_in_user(st, "gone")
    st.add_history("gone", "user", "hello")
    st.mark_alerted("gone", "2026-10-20", "x")
    st.delete_user("gone")
    u = st.get_user("gone")
    assert u.lat is None and u.crop is None and not u.alerts
    assert st.recent_history("gone") == [] and not st.already_alerted("gone", "2026-10-20", "x")


def test_existing_old_database_is_upgraded(tmp_path):
    import sqlite3

    path = tmp_path / "old.sqlite"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE users(user_id TEXT PRIMARY KEY, lat REAL, lon REAL, crop TEXT, state TEXT, updated REAL)")
    db.execute("INSERT INTO users VALUES ('old', 1.0, 2.0, 'tomato', 'X', 0)")
    db.commit(); db.close()
    st = State(path)
    assert st.get_user("old").crop == "tomato"
    st.update_user("old", soil="red")
    assert st.get_user("old").soil == "red"
