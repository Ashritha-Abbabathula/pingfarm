"""Weather-aware spraying advice, from the free Open-Meteo forecast API (no key needed)."""
from dataclasses import dataclass

import httpx

OPEN_METEO = "https://api.open-meteo.com/v1/forecast"

# Thresholds are deliberately simple and explainable. A spray washed off by rain is wasted
# money and leaves the crop unprotected, so we are cautious.
RAIN_MM_LIMIT = 2.0        # this much rain in a day washes most sprays off
RAIN_CHANCE_LIMIT = 60     # percent probability of rain


@dataclass
class DayForecast:
    date: str
    rain_mm: float
    rain_chance: int
    t_min: float
    t_max: float


@dataclass
class SprayAdvice:
    ok_to_spray: bool
    reason: str


def fetch_forecast(lat: float, lon: float, days: int = 3, client: httpx.Client | None = None) -> list[DayForecast]:
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "precipitation_sum,precipitation_probability_max,temperature_2m_max,temperature_2m_min",
        "forecast_days": days,
        "timezone": "auto",
    }
    own = client is None
    client = client or httpx.Client(timeout=15)
    try:
        data = client.get(OPEN_METEO, params=params).raise_for_status().json()["daily"]
    finally:
        if own:
            client.close()
    return [
        DayForecast(
            date=data["time"][i],
            rain_mm=float(data["precipitation_sum"][i] or 0),
            rain_chance=int(data["precipitation_probability_max"][i] or 0),
            t_min=float(data["temperature_2m_min"][i]),
            t_max=float(data["temperature_2m_max"][i]),
        )
        for i in range(len(data["time"]))
    ]


def spray_advice(forecast: list[DayForecast]) -> SprayAdvice:
    """Look at today and tomorrow. Rain soon after spraying wastes the spray."""
    window = forecast[:2]
    for day in window:
        if day.rain_mm >= RAIN_MM_LIMIT or day.rain_chance >= RAIN_CHANCE_LIMIT:
            return SprayAdvice(
                False,
                f"Rain is likely on {day.date} ({day.rain_mm:.1f} mm, {day.rain_chance}% chance). "
                "Spraying now may wash off. Wait for a dry spell.",
            )
    return SprayAdvice(True, "No significant rain is forecast for today and tomorrow, so spraying should hold.")
