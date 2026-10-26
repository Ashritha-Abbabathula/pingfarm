"""Check that your keys work, WITHOUT printing them.

    python scripts/check_keys.py

Reads the same .env file as the app. Each check prints OK, SKIPPED (key not set) or FAILED with the
reason. The Anthropic check sends one tiny request (a fraction of a cent).
"""
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def report(name, ok, detail=""):
    print(f"{name:12s} {'OK' if ok else 'FAILED'}  {detail}")
    return ok


def check_anthropic():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return print("anthropic    SKIPPED (ANTHROPIC_API_KEY not set)") or None
    try:
        import anthropic

        model = os.environ.get("PINGFARM_MODEL", "claude-haiku-5-5")
        r = anthropic.Anthropic().messages.create(
            model=model, max_tokens=20, messages=[{"role": "user", "content": "Reply with the word ready."}]
        )
        return report("anthropic", True, f"model {model} answered: {r.content[0].text.strip()[:30]!r}")
    except Exception as exc:
        return report("anthropic", False, f"{type(exc).__name__}: {str(exc)[:160]}")


def check_gemini():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return print("gemini       SKIPPED (GEMINI_API_KEY not set)") or None
    model = os.environ.get("PINGFARM_GEMINI_MODEL", "gemini-2.5-flash")
    try:
        r = httpx.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": key},
            json={"contents": [{"parts": [{"text": "Reply with the word ready."}]}]},
            timeout=30,
        )
        if r.status_code == 200:
            return report("gemini", True, f"model {model} answered")
        return report("gemini", False, f"HTTP {r.status_code}: {r.json().get('error', {}).get('message', '')[:160]}")
    except Exception as exc:
        return report("gemini", False, f"{type(exc).__name__}: {str(exc)[:160]}")


def check_data_gov():
    key = os.environ.get("DATA_GOV_API_KEY")
    if not key:
        return print("data.gov.in  SKIPPED (DATA_GOV_API_KEY not set)") or None
    try:
        sys.path.insert(0, str(ROOT))
        from pingfarm.tools.prices import describe_prices, fetch_prices

        recs = fetch_prices("Tomato", api_key=key)
        if not recs:
            return report("data.gov.in", False, "key accepted but no Tomato rows came back (field names or filters may need adjusting)")
        return report("data.gov.in", True, f"{len(recs)} rows. Example: {describe_prices(recs, 'tomato')[:140]}")
    except Exception as exc:
        return report("data.gov.in", False, f"{type(exc).__name__}: {str(exc)[:160]}")


def check_whatsapp():
    token, phone_id = os.environ.get("WHATSAPP_TOKEN"), os.environ.get("WHATSAPP_PHONE_ID")
    if not (token and phone_id):
        return print("whatsapp     SKIPPED (WHATSAPP_TOKEN / WHATSAPP_PHONE_ID not set)") or None
    try:
        r = httpx.get(f"https://graph.facebook.com/v20.0/{phone_id}", headers={"Authorization": f"Bearer {token}"}, timeout=20)
        if r.status_code == 200:
            return report("whatsapp", True, f"phone number: {r.json().get('display_phone_number', '?')}")
        err = r.json().get("error", {})
        return report("whatsapp", False, f"HTTP {r.status_code}: {err.get('message', '')[:160]}")
    except Exception as exc:
        return report("whatsapp", False, f"{type(exc).__name__}: {str(exc)[:160]}")


if __name__ == "__main__":
    for fn in (check_gemini, check_anthropic, check_data_gov, check_whatsapp):
        fn()
    for name in ("WHATSAPP_VERIFY_TOKEN", "WHATSAPP_APP_SECRET"):
        print(f"{name:22s} {'set' if os.environ.get(name) else 'NOT SET'}")
