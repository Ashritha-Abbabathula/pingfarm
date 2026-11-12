"""Mandi (wholesale market) prices from the Government of India open-data API (data.gov.in).

Needs a free API key in the environment variable DATA_GOV_API_KEY. Without a key the tool
reports "unavailable" instead of inventing a number. Mandi data is patchy and often a few
days old, so every answer states the market AND the date.

NOTE: the response fields below follow the published resource schema; this module has been
tested against recorded sample data only, not the live API (that needs your key).
"""
import os
from dataclasses import dataclass

import httpx

RESOURCE = "9ef84268-d588-465a-a308-a864a43d0070"  # daily mandi prices
URL = f"https://api.data.gov.in/resource/{RESOURCE}"


@dataclass
class PriceRecord:
    market: str
    district: str
    state: str
    commodity: str
    date: str
    modal_price: float  # rupees per quintal (100 kg)
    min_price: float
    max_price: float


def fetch_prices(commodity: str, state: str | None = None, api_key: str | None = None,
                 client: httpx.Client | None = None, limit: int = 50) -> list[PriceRecord] | None:
    """Return records, or None when no API key is configured."""
    api_key = api_key or os.environ.get("DATA_GOV_API_KEY")
    if not api_key:
        return None
    params = {"api-key": api_key, "format": "json", "limit": limit, "filters[commodity]": commodity}
    if state:
        params["filters[state]"] = state
    own = client is None
    client = client or httpx.Client(timeout=20)  # a farmer is waiting: give up on a slow API quickly
    try:
        rows = client.get(URL, params=params).raise_for_status().json().get("records", [])
    finally:
        if own:
            client.close()
    out = []
    for r in rows:
        try:
            out.append(PriceRecord(
                market=r["market"], district=r.get("district", ""), state=r.get("state", ""),
                commodity=r["commodity"], date=r["arrival_date"],
                modal_price=float(r["modal_price"]), min_price=float(r["min_price"]),
                max_price=float(r["max_price"]),
            ))
        except (KeyError, ValueError, TypeError):
            continue  # skip incomplete rows rather than guess
    return out


def describe_prices(records: list[PriceRecord] | None, commodity: str) -> str:
    if records is None:
        return "Market price lookup is not set up yet (no data.gov.in key)."
    if not records:
        return f"I could not find a recent mandi price for {commodity}."
    latest = max(records, key=lambda r: r.date[-4:] + r.date[3:5] + r.date[:2])  # dd/mm/yyyy
    return (f"{latest.commodity} at {latest.market} mandi ({latest.district}, {latest.state}) on {latest.date}: "
            f"Rs {latest.modal_price:.0f} per quintal (range Rs {latest.min_price:.0f} to Rs {latest.max_price:.0f}). "
            "Prices change daily; confirm with the mandi before selling.")
