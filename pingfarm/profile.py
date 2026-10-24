"""Understanding what a farmer tells us about themselves: soil, planting date, alert preferences.

Plain keyword rules, on purpose: the set of things a farmer might say is small, and a rule is easy to read,
test and fix. Hindi, Telugu and English-letter (romanised) spellings are included.
"""
import re
from datetime import date, timedelta

SOIL_WORDS = {
    "red": ["red soil", "red earth", "लाल मिट्टी", "लाल मिट्टी", "ఎర్ర నేల", "ఎర్రమట్టి", "erra nela", "lal mitti"],
    "black": ["black soil", "black cotton", "काली मिट्टी", "నల్ల నేల", "నల్ల రేగడి", "nalla regadi", "nalla nela", "kali mitti"],
    "sandy": ["sandy", "बलुई", "रेतीली", "ఇసుక నేల", "ఇసుక", "isuka nela", "baluyi"],
    "loamy": ["loam", "दोमट", "ఒండ్రు", "చెల్క", "domat"],
    "clay": ["clay", "चिकनी मिट्टी", "బంకమట్టి", "banka matti", "chikni mitti"],
    "laterite": ["laterite", "लैटेराइट", "గరప నేల", "garapa nela"],
    "alluvial": ["alluvial", "जलोढ़", "ఒండ్రు మట్టి", "jalodh"],
}

ALERT_ON = ["alerts on", "alert on", "send alerts", "notify me", "alerts chalu", "अलर्ट चालू", "अलर्ट भेजो", "అలర్ట్ ఆన్", "alerts pettandi"]
ALERT_OFF = ["alerts off", "alert off", "stop alerts", "no alerts", "alerts band", "अलर्ट बंद", "అలర్ట్ ఆఫ్", "alerts apandi"]
DELETE = ["delete my data", "forget me", "mera data delete", "मेरा डेटा हटाओ", "నా డేటా తొలగించండి", "naa data tholaginchandi"]


def parse_soil(text: str) -> str | None:
    low = text.lower()
    for soil, words in SOIL_WORDS.items():
        if any(w.lower() in low for w in words):
            return soil
    return None


def parse_planting_date(text: str, today: date | None = None) -> str | None:
    """'planted 10 days ago', '3 weeks back', 'sown on 5 October' -> ISO date. Returns None if unsure."""
    today = today or date.today()
    low = text.lower()
    m = re.search(r"(\d{1,3})\s*(day|days|din|రోజుల|రోజు)\b", low)
    if m and any(w in low for w in ("ago", "back", "pehle", "క్రితం", "plant", "sow", "बोया", "लगाया", "నాటాను", "విత్తాను")):
        return (today - timedelta(days=int(m.group(1)))).isoformat()
    m = re.search(r"(\d{1,2})\s*(week|weeks|hafte|hafta|వారాల)\b", low)
    if m and any(w in low for w in ("ago", "back", "pehle", "క్రితం", "plant", "sow", "बोया", "लगाया", "నాటాను", "విత్తాను")):
        return (today - timedelta(weeks=int(m.group(1)))).isoformat()
    months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    m = re.search(r"(\d{1,2})(?:st|nd|rd|th)?\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*", low)
    if m and any(w in low for w in ("plant", "sow", "planted", "sown", "बोया", "लगाया", "నాటాను", "విత్తాను")):
        month = months.index(m.group(2)) + 1
        d = date(today.year, month, min(int(m.group(1)), 28 if month == 2 else 30))
        if d > today:
            d = date(today.year - 1, month, d.day)
        return d.isoformat()
    return None


def parse_alert_command(text: str) -> bool | None:
    low = text.lower()
    if any(w.lower() in low for w in ALERT_OFF):
        return False
    if any(w.lower() in low for w in ALERT_ON):
        return True
    return None


def wants_deletion(text: str) -> bool:
    low = text.lower()
    return any(w.lower() in low for w in DELETE)


def crop_stage(crop: str, planted_on: str | None, today: date | None = None) -> str | None:
    """A rough stage from days since planting. Stage boundaries are generic, not variety-specific."""
    if not planted_on:
        return None
    days = ((today or date.today()) - date.fromisoformat(planted_on)).days
    if days < 0:
        return None
    if crop == "tomato":
        bounds = [(25, "nursery/establishment"), (55, "vegetative growth"), (85, "flowering and fruit set"), (999, "fruiting and harvest")]
    else:
        bounds = [(25, "emergence and early growth"), (60, "tuber initiation"), (90, "tuber bulking"), (999, "maturity")]
    for limit, name in bounds:
        if days <= limit:
            return f"{name} (about {days} days after planting)"
    return None
