"""Proactive alerts: message the farmer first when the forecast is risky for their crop.

Rules of the road (also in the study guide):
  * OPT-IN ONLY. A farmer must say "alerts on". "alerts off" or "delete my data" stops everything.
  * At most one alert of each kind per farmer per day (stored in SQLite).
  * Only HIGH risks are sent. Medium and low stay in the on-demand "outlook" answer.
  * WhatsApp rule: a business may send free-form text only within 24 hours of the farmer's last message. Outside that
    window it must use a pre-approved TEMPLATE message (needs Meta approval and a verified business account). Without
    one we do not send; we log the alert as 'blocked_window' so nothing is lost silently.
  * Alerts never contain a spray amount.

Run it on a schedule (Windows Task Scheduler or cron), for example every morning:
    python scripts/run_alerts.py
"""
import os
import time
from dataclasses import dataclass
from datetime import date

from .state import State
from .tools import risk as risk_tool
from .tools import weather as weather_tool
from .whatsapp import WhatsAppClient

WINDOW_SECONDS = 24 * 3600


@dataclass
class AlertResult:
    user_id: str
    status: str   # sent | no_risk | already_sent | blocked_window | error
    detail: str = ""


def alert_text(crop: str, risks) -> str:
    high = [r for r in risks if r.level == "high"]
    lines = [f"PingFarm alert for your {crop}:"]
    for r in high:
        lines.append(f"- {r.disease}: {r.why}.")
    lines.append("Please check your plants. Reply with the disease name for advice. I do not give spray amounts unless a source states them. "
                 "Reply 'alerts off' to stop.")
    return "\n".join(lines)


def run_alerts(state: State, wa: WhatsAppClient, forecast_fn=weather_tool.fetch_forecast,
               today: date | None = None, now: float | None = None, template: str | None = None) -> list[AlertResult]:
    today_s = (today or date.today()).isoformat()
    now = now or time.time()
    template = template or os.environ.get("WHATSAPP_ALERT_TEMPLATE")
    out: list[AlertResult] = []
    for u in state.alert_users():
        try:
            forecast = forecast_fn(u.lat, u.lon)
            risks = risk_tool.assess(u.crop, forecast)
            high = [r for r in risks if r.level == "high"]
            if not high:
                out.append(AlertResult(u.user_id, "no_risk"))
                continue
            kind = "|".join(sorted(r.disease for r in high))
            if state.already_alerted(u.user_id, today_s, kind):
                out.append(AlertResult(u.user_id, "already_sent"))
                continue
            text = alert_text(u.crop, risks)
            last = state.last_inbound(u.user_id)
            inside_window = last is not None and (now - last) <= WINDOW_SECONDS
            if inside_window:
                wa.send_text(u.user_id, text)
                status = "sent"
            elif template and hasattr(wa, "send_template"):
                wa.send_template(u.user_id, template, [u.crop, ", ".join(r.disease for r in high)])
                status = "sent"
            else:
                status = "blocked_window"   # would need an approved template message
            if status == "sent":
                state.mark_alerted(u.user_id, today_s, kind)
            out.append(AlertResult(u.user_id, status, kind))
        except Exception as exc:  # one farmer's failure must never stop the rest
            out.append(AlertResult(u.user_id, "error", f"{type(exc).__name__}"))
    return out
