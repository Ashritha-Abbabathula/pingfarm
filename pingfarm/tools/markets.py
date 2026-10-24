"""Markets near the farmer, and how prices are moving. Built on the government mandi price data (data.gov.in).

What this can and cannot say (be honest about it to farmers and in interviews):
  * It reports PRICES and their TREND at markets near the farmer. It does not know costs, transport, quality or
    how many farmers are selling, so "higher price" is not the same as "better profit".
  * Distances are estimated by looking up each market's town name on a free geocoder; some will be missing.
  * Mandi data is patchy and often a few days old. Every line states the market and the date.
  * Needs DATA_GOV_API_KEY. Without it the module returns nothing rather than inventing numbers.
"""
import math
from dataclasses import dataclass

import httpx

from .prices import PriceRecord, fetch_prices

NOMINATIM = "https://nominatim.openstreetmap.org/reverse"
GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
UA = {"User-Agent": "PingFarm student project (check4ashritha@gmail.com)"}


@dataclass
class Market:
    name: str
    district: str
    state: str
    price: float          # modal price, Rs per quintal
    date: str
    km: float | None      # distance from the farmer if we could place the market
    trend_pct: float | None  # latest vs the market's earlier prices in the data, percent


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def reverse_place(lat: float, lon: float, client: httpx.Client | None = None) -> tuple[str | None, str | None]:
    """(state, district) for a coordinate, from OpenStreetMap's free service. Be gentle: one call per farmer, then cache."""
    own = client is None
    client = client or httpx.Client(timeout=15, headers=UA)
    try:
        j = client.get(NOMINATIM, params={"lat": lat, "lon": lon, "format": "json", "zoom": 10}).raise_for_status().json()
    finally:
        if own:
            client.close()
    a = j.get("address", {})
    return a.get("state"), a.get("state_district") or a.get("county") or a.get("city")


def geocode(name: str, state: str | None, cache, client: httpx.Client | None = None) -> tuple[float, float] | None:
    key = f"{name}|{state or ''}".lower()
    hit = cache.db.execute("SELECT lat, lon FROM geocache WHERE name=?", (key,)).fetchone()
    if hit:
        return (hit[0], hit[1]) if hit[0] is not None else None
    own = client is None
    client = client or httpx.Client(timeout=15)
    try:
        res = client.get(GEOCODE, params={"name": name, "count": 5, "country": "IN", "language": "en"}).raise_for_status().json().get("results") or []
    except Exception:
        return None
    finally:
        if own:
            client.close()
    pick = next((r for r in res if state and state.lower() in (r.get("admin1") or "").lower()), res[0] if res else None)
    cache.db.execute("INSERT OR REPLACE INTO geocache(name, lat, lon) VALUES (?,?,?)",
                     (key, pick["latitude"] if pick else None, pick["longitude"] if pick else None))
    cache.db.commit()
    return (pick["latitude"], pick["longitude"]) if pick else None


def _date_key(d: str) -> str:  # dd/mm/yyyy -> yyyymmdd for sorting
    return d[-4:] + d[3:5] + d[:2]


def build_markets(records: list[PriceRecord], lat: float | None, lon: float | None, cache,
                  geocode_fn=geocode, max_geocode: int = 12) -> list[Market]:
    """One row per market: latest price, trend, and distance if we can place it. Nearest first (else cheapest data order)."""
    by_market: dict[tuple[str, str, str], list[PriceRecord]] = {}
    for r in records:
        by_market.setdefault((r.market, r.district, r.state), []).append(r)
    out: list[Market] = []
    for (name, district, state), rs in by_market.items():
        rs = sorted(rs, key=lambda r: _date_key(r.date))
        last = rs[-1]
        earlier = [r.modal_price for r in rs[:-1]]
        trend = ((last.modal_price - sum(earlier) / len(earlier)) / (sum(earlier) / len(earlier)) * 100) if earlier and sum(earlier) else None
        out.append(Market(name, district, state, last.modal_price, last.date, None, trend))
    if lat is not None and lon is not None:
        for m in sorted(out, key=lambda m: _date_key(m.date), reverse=True)[:max_geocode]:
            pos = geocode_fn(f"{m.name}", m.state, cache)
            if pos:
                m.km = round(haversine_km(lat, lon, pos[0], pos[1]), 0)
    placed = sorted([m for m in out if m.km is not None], key=lambda m: m.km)
    rest = [m for m in out if m.km is None]
    return placed + rest


def describe_markets(crop: str, markets: list[Market], limit: int = 4) -> str:
    if not markets:
        return f"I could not find recent mandi prices for {crop} near you."
    near = markets[:limit]
    lines = []
    for m in near:
        where = f"{m.km:.0f} km away" if m.km is not None else "distance unknown"
        trend = ""
        if m.trend_pct is not None and abs(m.trend_pct) >= 3:
            trend = f", {'up' if m.trend_pct > 0 else 'down'} {abs(m.trend_pct):.0f}% vs earlier days in the data"
        lines.append(f"- {m.name} ({m.district}), {where}: Rs {m.price:.0f} per quintal on {m.date}{trend}")
    best = max(markets, key=lambda m: m.price)
    best_line = ""
    if best not in near:
        km = f", {best.km:.0f} km away" if best.km is not None else ""
        best_line = f"\nHighest price in this data: {best.name} ({best.district}){km}: Rs {best.price:.0f} on {best.date}."
    return (f"{crop.title()} prices at markets near you:\n" + "\n".join(lines) + best_line +
            "\nA higher price is not always more profit: think about transport cost and freshness. Confirm at the mandi before you go.")


def markets_for(profile, crop: str, cache, price_fn=fetch_prices, place_fn=reverse_place, geocode_fn=geocode):
    """Everything in one call. Returns (text, error) where error is a short reason or None."""
    state, district = profile.state, profile.district
    if (state is None or district is None) and profile.lat is not None:
        try:
            state2, district2 = place_fn(profile.lat, profile.lon)
            state, district = state or state2, district or district2
        except Exception:
            pass
    recs = price_fn(crop.title(), state, limit=500) if price_fn is fetch_prices else price_fn(crop.title(), state)
    if recs is None:
        return None, "no_key"
    return describe_markets(crop, build_markets(recs, profile.lat, profile.lon, cache, geocode_fn)), None
