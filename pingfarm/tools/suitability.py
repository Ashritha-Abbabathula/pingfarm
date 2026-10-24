"""Does the soil and coming weather suit tomato or potato? (Only these two crops: that is what our sources cover.)

Facts come from our documents; where a line is general knowledge rather than a document statement it is marked.
This is a screening note, not a recommendation to plant: soil pH, water, market and local season matter, so it always
points the farmer to a soil test at the nearest Krishi Vigyan Kendra (KVK).

Sources:
  tomato soil     TNAU tomato soil page: well-drained loamy soils rich in organic matter, pH 6.5-7.5; sowing May-June and Nov-Dec (Tamil Nadu)
  tomato climate  Vikaspedia (Hindi) tomato page: best average temperature 18-27 C; below 10 C or above 30 C colour fails;
                  hot dry wind makes flowers drop; any soil works but light-acid to loam is best, with organic matter and drainage
  potato          TNAU potato page: friable, porous, well-drained soil, pH 4.8-5.4, prefers cool weather; plains planting Oct-Nov
"""
from .weather import DayForecast

SOIL_NOTES = {
    "loamy": ("good", "Loamy soil is what the sources recommend for tomato, and potato needs friable, well-drained soil."),
    "red": ("ok", "Red soils are usually light and drain well (general knowledge): fine for both if you add organic matter; they may need more frequent watering."),
    "sandy": ("ok", "Sandy soils drain fast (general knowledge): they need more frequent light watering and organic matter; fine for potato tubers if kept moist."),
    "black": ("careful", "Black soils hold water (general knowledge): drainage is the worry. Both crops need well-drained soil, and wet soil favours bacterial wilt in tomato."),
    "clay": ("careful", "Clay soils hold water (general knowledge): raise beds or ridges for drainage, because both crops need well-drained soil."),
    "laterite": ("ok", "Laterite soils are usually acidic and well drained (general knowledge): potato's preferred pH is on the acidic side; tomato prefers 6.5-7.5, so a soil test is worth it."),
    "alluvial": ("good", "Alluvial soils are generally fertile and loamy (general knowledge): suitable for both if drainage is good."),
}


def climate_note(crop: str, forecast: list[DayForecast]) -> str:
    if not forecast:
        return ""
    avg_max = sum(d.t_max for d in forecast) / len(forecast)
    avg_min = sum(d.t_min for d in forecast) / len(forecast)
    avg = (avg_max + avg_min) / 2
    if crop == "tomato":
        if avg_max > 30:
            fit = f"Hotter than ideal: sources give 18-27 C as best and fruit colour and flower set suffer above about 30 C (forecast daytime highs average {avg_max:.0f} C)."
        elif avg < 18:
            fit = f"Cooler than ideal: sources give 18-27 C as best (forecast average {avg:.0f} C)."
        else:
            fit = f"Within the range the sources give as best (18-27 C; forecast average {avg:.0f} C)."
    else:
        if avg > 25:
            fit = f"Warm for potato: it prefers cool weather and the late-blight rules favour 10-25 C damp spells; forecast average {avg:.0f} C, highs {avg_max:.0f} C."
        else:
            fit = f"Cool enough for potato (forecast average {avg:.0f} C)."
    return fit


def outlook(crop: str, soil: str | None, forecast: list[DayForecast]) -> str:
    lines = [f"Soil and weather fit for {crop} (screening only):"]
    if soil and soil in SOIL_NOTES:
        lines.append(f"- Soil ({soil}): {SOIL_NOTES[soil][1]}")
    else:
        lines.append("- Soil: tell me your soil in simple words (red, black, sandy, loamy, clay) and I will comment.")
    c = climate_note(crop, forecast)
    if c:
        lines.append(f"- Weather: {c}")
    if crop == "tomato":
        lines.append("- Season (Tamil Nadu guidance): sowing in May-June and November-December. Your region may differ.")
    else:
        lines.append("- Season: plains planting October-November (TNAU); hills have other windows.")
    lines.append("I cannot judge soil pH or nutrients from here: please get a soil test at your nearest KVK before choosing or planting.")
    return "\n".join(lines)


def compare_crops(soil: str | None, forecast: list[DayForecast]) -> str:
    """Which of tomato or potato do the coming temperatures favour? A plain, hedged comparison."""
    if not forecast:
        return "I need your location to look at the weather."
    avg_max = sum(d.t_max for d in forecast) / len(forecast)
    avg_min = sum(d.t_min for d in forecast) / len(forecast)
    avg = (avg_max + avg_min) / 2
    if avg_max <= 27 and avg >= 15:
        verdict = "Temperatures suit both tomato (best 18-27 C) and potato (cool weather) right now."
    elif avg_max > 30:
        verdict = "It is hot for both: tomato struggles above about 30 C and potato prefers cool weather. Waiting for cooler weather looks wiser."
    elif avg < 15:
        verdict = "It is cool: better for potato than for tomato (tomato prefers 18-27 C)."
    else:
        verdict = "Mixed: tomato is fine up to about 27 C; potato prefers cooler."
    return (f"Forecast averages: highs {avg_max:.0f} C, lows {avg_min:.0f} C. {verdict} "
            "This only looks at temperature. Water, soil, season, seed and the market price also decide what pays; ask your KVK.")
