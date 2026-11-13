"""Tests must never see real keys, even when a real .env exists next to the code.

pingfarm.app loads .env when it is imported, so without this a test could pick up a real WhatsApp token and try to
send a real message, or a real app secret and reject its own unsigned test requests."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pingfarm.app  # noqa: E402,F401  (loads .env once, here, so the fixture below can clear the keys before every test)

REAL_KEYS = (
    "WHATSAPP_TOKEN", "WHATSAPP_PHONE_ID", "WHATSAPP_APP_SECRET", "WHATSAPP_VERIFY_TOKEN",
    "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "DATA_GOV_API_KEY", "PINGFARM_ALLOW_UNSIGNED",
)


@pytest.fixture(autouse=True)
def no_real_keys(monkeypatch):
    for name in REAL_KEYS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PINGFARM_PRELOAD", "0")  # never load the big models in the background during a test
