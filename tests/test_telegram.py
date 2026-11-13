"""Telegram channel: message parsing, sending and the polling loop (no network)."""
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pingfarm.telegram_channel import TelegramClient, parse_update, poll_forever


def upd(uid, **message):
    return {"update_id": uid, "message": {"chat": {"id": 555}, **message}}


def test_parse_text_start_location_photo_voice_and_junk():
    t = parse_update(upd(1, text="  leaf curl in tomato "))[0]
    assert (t.kind, t.text, t.sender, t.msg_id) == ("text", "leaf curl in tomato", "555", "tg:1")

    assert parse_update(upd(2, text="/start"))[0].text == "hello"

    loc = parse_update(upd(3, location={"latitude": 17.4, "longitude": 78.5}))[0]
    assert loc.kind == "location" and loc.location == (17.4, 78.5)

    img = parse_update(upd(4, photo=[{"file_id": "small"}, {"file_id": "big"}], caption="spots"))[0]
    assert img.kind == "image" and img.media_id == "big" and img.text == "spots"

    assert parse_update(upd(5, voice={"file_id": "v"}))[0].kind == "audio"

    assert parse_update(upd(6, sticker={"file_id": "s"})) == []  # nothing we understand
    assert parse_update({"update_id": 7}) == []  # not a message
    assert parse_update({}) == []


def test_dry_run_client_records_and_never_calls_the_network(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    client = TelegramClient()
    assert client.dry_run
    client.send_text("555", "hello")
    assert client.sent == [("555", "hello")]


def test_polling_hands_each_message_over_advances_the_offset_and_survives_errors():
    stop = threading.Event()
    calls = []

    class Fake(TelegramClient):
        def __init__(self):
            super().__init__(token="x")
            self.rounds = [
                RuntimeError("network down with secret-token-text"),
                [upd(10, text="one"), upd(11, text="two")],
            ]

        def get_updates(self, offset, wait=25):
            calls.append(offset)
            item = self.rounds.pop(0) if self.rounds else None
            if item is None:
                stop.set()
                return []
            if isinstance(item, Exception):
                raise item
            return item

    seen = []
    stop.wait = lambda t=None: False  # do not really sleep in the test
    poll_forever(None, Fake(), stop, lambda app, msg, client: seen.append(msg.text))
    assert seen == ["one", "two"]
    assert calls[0] is None and calls[-1] == 12  # after update 11 it asks for 12 onwards


def test_format_reply_splits_steps_drops_citation_numbers_and_lists_sources():
    from pingfarm.telegram_channel import format_reply

    raw = "Control steps: 1. Use clean seed [2]. 2. Remove weeds [1, 3]. 3. Spray 200 ml/ac [2]."
    out = format_reply(raw, [{"title": "TNAU page", "url": "https://x.example/a"}, {"title": "B", "url": ""},
                             {"title": "third is cut", "url": "https://x.example/c"}])
    assert "[" not in out.split("Sources:")[0]  # no citation numbers left in the answer
    assert "\n2. Remove weeds" in out and "\n3. Spray 200 ml/ac" in out  # one step per line, dose untouched
    assert out.endswith("- TNAU page\n  https://x.example/a\n- B")  # two sources only, link only when there is one
    assert format_reply("A sentence with 2.5 kg per acre.", []) == "A sentence with 2.5 kg per acre."  # not a list


def test_app_uses_the_channels_formatter_and_typing_indicator():
    from pingfarm.agent import Reply
    from pingfarm.app import process_message
    from pingfarm.whatsapp import Incoming

    events = []

    class FakeState:
        def first_time_seeing(self, _):
            return True

    class FakeAgent:
        state = FakeState()

        def handle(self, sender, text=None, **_):
            events.append("handled")
            return Reply("Do this [1].", [{"title": "Src", "url": "https://u.example"}])

    class App:
        class state:  # noqa: N801
            get_agent = staticmethod(lambda: FakeAgent())
            wa = None

    client = TelegramClient(token="x", dry_run=True)
    process_message(App, Incoming(msg_id="tg:1", sender="9", kind="text", text="hi"), client)
    assert client.sent == [("9", "Do this.\n\nSources:\n- Src\n  https://u.example")]
