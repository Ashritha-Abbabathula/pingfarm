# Run and connect PingFarm

## 1. Run the web app locally (no keys needed)

```bash
python -m pingfarm.evalkit.run_eval --quick   # optional: reproduce the retrieval table
uvicorn pingfarm.app:app --reload
```

Open http://127.0.0.1:8000 for the chat page. Try a question in English, Hindi or Telugu, share a
location (browser prompt), and upload a leaf photo once a model exists in `models/`.
Without API keys the reply quotes the sources directly (the "extractive" answerer) and WhatsApp runs
in dry-run mode (messages are recorded, not sent).

## 2. Turn on the language model

Set `ANTHROPIC_API_KEY` (see `.env.example`). Replies are then written in the farmer's language using
only the retrieved passages, and still pass through the dose guard.

## 3. Connect WhatsApp (Meta Cloud API)

1. Create a Meta developer app, add the WhatsApp product, and note the test phone-number id and token.
2. Choose any string as `WHATSAPP_VERIFY_TOKEN`; set `WHATSAPP_TOKEN`, `WHATSAPP_PHONE_ID`, `WHATSAPP_APP_SECRET`.
3. Make your laptop reachable from the internet with a tunnel (for example `cloudflared tunnel --url http://localhost:8000`
   or `ngrok http 8000`), or deploy to a free host such as Render.
4. In the Meta console set the webhook callback URL to `https://<your-host>/webhook` and the verify token
   to your string, then subscribe to the `messages` field.
5. Send a message from a number added as a test recipient.

Gotchas (also in `pingfarm/whatsapp.py`): Meta retries slow webhooks (handled by de-duplication);
photos and voice notes arrive as ids that must be downloaded with your token; a public number needs a
verified business account, but the test number is enough for a demo and a small pilot.

## 4. Prices

Get a free key from data.gov.in and set `DATA_GOV_API_KEY`. The price tool has been tested against
recorded sample data only, not the live service.
