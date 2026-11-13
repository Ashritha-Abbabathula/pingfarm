"""The privacy policy page (Meta asks for a public URL before an app can be published). Plain words, no tracking."""
import html
import os

POLICY = """
<h1>PingFarm privacy policy</h1>
<p>PingFarm is a free farm advisory assistant for tomato and potato. A farmer sends a question, a leaf photo or a
place name or location in a chat app (WhatsApp or Telegram) and gets advice taken from public agriculture documents, with the source shown.
It is a learning project and not a commercial service.</p>

<h2>What we collect</h2>
<ul>
  <li>Your chat account number or ID (WhatsApp number or Telegram ID) and the messages you send to PingFarm (questions, leaf photos, a place name or location pin).</li>
  <li>Things you choose to tell the assistant: your crop, soil type, planting date, and whether you want weather alerts.</li>
</ul>

<h2>Why</h2>
<p>Only to answer your question and, if you ask for them, to send weather alerts for your crop.
We do not sell your data, share it for advertising, or use it to identify you in any other way.</p>

<h2>Who else sees it</h2>
<p>The chat app you use (WhatsApp by Meta, or Telegram) carries the messages. To write a reply, the text of your question and the matching passages from
public documents may be sent to a language-model service (Google Gemini). Weather forecasts are fetched for a
location you share. Leaf photos are analysed on our own server.</p>

<h2>How long we keep it</h2>
<p>Your profile and message history stay until you delete them.</p>

<h2>Deleting your data</h2>
<p>Send the message <b>delete my data</b> (or <b>alerts off</b> to stop alerts). Your profile, location and message
history are removed straight away.</p>

<h2>Advice limits</h2>
<p>PingFarm gives general information from published sources. It does not replace a local agriculture officer
(Krishi Vigyan Kendra). It never states a pesticide amount that is not written in a source.</p>
{contact}
"""

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>PingFarm privacy policy</title>
<style>body{{font-family:system-ui,sans-serif;max-width:42rem;margin:2rem auto;padding:0 1rem;line-height:1.55;color:#1a1a1a}}
h1{{font-size:1.6rem}}h2{{font-size:1.15rem;margin-top:1.6rem}}</style></head>
<body>{body}</body></html>"""


def privacy_html() -> str:
    """The page. The contact line shows only if PINGFARM_CONTACT_EMAIL is set (nothing personal is published by default)."""
    email = os.environ.get("PINGFARM_CONTACT_EMAIL", "").strip()
    contact = f"<h2>Contact</h2><p>Questions: {html.escape(email)}</p>" if email else ""
    return PAGE.format(body=POLICY.format(contact=contact))
