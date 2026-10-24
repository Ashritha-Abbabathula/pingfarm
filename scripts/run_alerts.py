"""Send proactive weather-risk alerts to farmers who opted in. Run daily (Task Scheduler / cron).

    python scripts/run_alerts.py            # real sending only if WhatsApp credentials are set; otherwise a dry run
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from pingfarm.alerts import run_alerts  # noqa: E402
from pingfarm.state import State  # noqa: E402
from pingfarm.whatsapp import WhatsAppClient  # noqa: E402

if __name__ == "__main__":
    wa = WhatsAppClient()
    results = run_alerts(State(ROOT / "data" / "pingfarm.sqlite"), wa)
    for r in results:
        print(r.user_id[-4:].rjust(6, "*"), r.status, r.detail)
    print(f"{len(results)} farmers checked; dry_run={wa.dry_run}")
