"""Disease-risk outlook from the weather forecast, using the favourable conditions written in our sources.

Every rule below is taken from a document in the knowledge base and says which one. These are SIMPLE screening
rules (is the weather the kind that favours this disease?), not a disease forecast model. They tell a farmer to
look at their crop and to ask for advice; they never state a spray dose.

Sources (in data/sources.json):
  tomato late blight   TNAU tomato page 8: cool nights, warm days, extended wet weather from rain and fog; best sporulation 18-22 C
  potato late blight   TNAU potato page 1: RH > 90 %, 10-25 C, night temperature 10 C, cloudiness before rain, rain >= 0.1 mm;
                       Vikaspedia (Hindi): fungus grows at 16-20 C, spores germinate at 10-20 C
  tomato early blight  TNAU tomato page 2: warm (24-29 C), humid, rainy weather
  tomato powdery mildew TNAU tomato page 10: warm and dry weather
  tomato fusarium wilt TNAU tomato page 3: dry weather, high soil temperature, low soil moisture
  tomato bacterial wilt TNAU tomato page 5: high soil moisture
  tomato leaf curl     TNAU tomato page 6: spread by whiteflies (see whitefly page: warm dry spells favour them)
  heat stress          Vikaspedia (Hindi) tomato page: fruit colour stops above ~30 C; hot dry wind makes flowers drop
"""
from dataclasses import dataclass

from .weather import DayForecast


@dataclass
class Risk:
    disease: str
    level: str          # low | medium | high
    days: list[str]     # dates the risk applies to
    why: str
    source: str


def _fmt_days(days: list[DayForecast]) -> str:
    return ", ".join(d.date[5:] for d in days)


def assess(crop: str, forecast: list[DayForecast], humidity: dict[str, float] | None = None) -> list[Risk]:
    """Screen the next days. `humidity` maps date -> mean relative humidity (%), if known."""
    hum = humidity or {}
    risks: list[Risk] = []

    def rh(d):
        return hum.get(d.date, d.rh)

    if crop == "potato":
        hot = [d for d in forecast if 10 <= (d.t_min + d.t_max) / 2 <= 25 and (d.rain_mm >= 0.1 or (rh(d) or 0) >= 90)]
        strong = [d for d in hot if d.rain_mm >= 2 or (rh(d) or 0) >= 90]
        if len(strong) >= 2:
            risks.append(Risk("potato late blight", "high", [d.date for d in strong],
                              f"cool, damp weather on {_fmt_days(strong)} (avg 10-25 C with rain or high humidity), the kind that favours late blight",
                              "TNAU potato page 1; Vikaspedia (Hindi) potato late blight"))
        elif hot:
            risks.append(Risk("potato late blight", "medium", [d.date for d in hot],
                              f"some damp, cool weather on {_fmt_days(hot)}", "TNAU potato page 1"))
        else:
            risks.append(Risk("potato late blight", "low", [], "no cool, damp spell in the forecast", "TNAU potato page 1"))
        return risks

    # ---- tomato
    late = [d for d in forecast if d.t_min <= 20 and 22 <= d.t_max <= 30 and (d.rain_mm >= 1 or (rh(d) or 0) >= 90)]
    if len(late) >= 2:
        risks.append(Risk("tomato late blight", "high", [d.date for d in late],
                          f"cool nights with warm days and wet weather on {_fmt_days(late)}", "TNAU tomato page 8"))
    elif late:
        risks.append(Risk("tomato late blight", "medium", [d.date for d in late], f"some cool, wet weather on {_fmt_days(late)}", "TNAU tomato page 8"))

    early = [d for d in forecast if 24 <= (d.t_min + d.t_max) / 2 <= 29 and (d.rain_mm >= 1 or (rh(d) or 0) >= 80)]
    if len(early) >= 3:
        risks.append(Risk("tomato early blight", "high", [d.date for d in early],
                          f"warm, humid, rainy weather (24-29 C) on {_fmt_days(early)}", "TNAU tomato page 2"))
    elif early:
        risks.append(Risk("tomato early blight", "medium", [d.date for d in early], f"warm, damp weather on {_fmt_days(early)}", "TNAU tomato page 2"))

    dry_hot = [d for d in forecast if d.rain_mm < 0.5 and d.t_max >= 30]
    if len(dry_hot) >= 3:
        risks.append(Risk("heat stress and dry-weather problems", "medium", [d.date for d in dry_hot],
                          f"hot, dry days (30 C and above) on {_fmt_days(dry_hot)}: flowers can drop, fruit colour suffers, "
                          "and dry weather with warm soil favours Fusarium wilt and powdery mildew",
                          "Vikaspedia (Hindi) tomato page; TNAU tomato pages 3 and 10"))

    wet = [d for d in forecast if d.rain_mm >= 10]
    if wet:
        risks.append(Risk("waterlogging and bacterial wilt", "medium", [d.date for d in wet],
                          f"heavy rain on {_fmt_days(wet)}: keep drainage open; high soil moisture favours bacterial wilt", "TNAU tomato page 5"))
    if not risks:
        risks.append(Risk("tomato diseases", "low", [], "no disease-favouring spell in the forecast", "TNAU tomato pages"))
    return risks


def summarize(crop: str, risks: list[Risk]) -> str:
    order = {"high": 0, "medium": 1, "low": 2}
    lines = []
    for r in sorted(risks, key=lambda x: order[x.level]):
        icon = {"high": "HIGH", "medium": "MEDIUM", "low": "LOW"}[r.level]
        lines.append(f"- {icon}: {r.disease}. {r.why.capitalize()}.")
    head = f"Weather outlook for your {crop} (next days):"
    tail = ("Check your leaves, especially the undersides. Ask me about the disease for what the sources recommend. "
            "I do not give spray amounts unless a source states them.")
    return head + "\n" + "\n".join(lines) + "\n" + tail
