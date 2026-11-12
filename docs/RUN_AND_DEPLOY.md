# Run PingFarm and connect your keys

PingFarm works with **no keys** (replies quote the sources, WhatsApp is a dry run). Each key below switches on
one more capability. You can add them one at a time, in this order.

**Golden rules:** keys go only in the file `.env` (git-ignored). Never paste a key into chat, a notebook,
a commit, or a screenshot. If one leaks, delete it in the provider's console and make a new one.

## 0. Set up `.env`

```bash
copy .env.example .env      # Windows (use cp on Mac/Linux)
```

Open `.env` in a text editor and fill values in as you get them (no quotes, no spaces around `=`).
After each step, check it:

```bash
python scripts/check_keys.py
```

It prints OK / SKIPPED / FAILED for each key and never prints the key itself.

## 1. data.gov.in key (free, about 5 minutes): market prices

1. Go to https://data.gov.in and click **Register** (or Login). Confirm your email.
2. After login open **My Account** (your name, top right). Find **API Key** / **Get API Key** and generate it.
3. Put it in `.env` as `DATA_GOV_API_KEY=...`.
4. Run `python scripts/check_keys.py`. You want `data.gov.in OK` and an example price line.

If it says *"no Tomato rows came back"*: the key is fine but the price tool's field names or filters need adjusting
to the live data. The tool was built from the published schema and tested only on sample data, so this is the first
place a small fix may be needed. Send me the message and I will fix it.

## 2a. Gemini key (free tier): Gemini writes the replies (recommended, no payment needed)

1. Go to https://aistudio.google.com/apikey and sign in with a Google account.
2. Click **Create API key** and copy it.
3. Put it in `.env` as `GEMINI_API_KEY=...` (never paste it anywhere else). Leave `PINGFARM_GEMINI_MODEL` as it is; if the check says the model is not found, look up the current free Flash model name on the same site and put it there.
4. Run `python scripts/check_keys.py`. You want `gemini OK`.
5. Start the app (section 4) and check `/health`: `"llm"` should say `GeminiAnswerer`.

The free tier has daily request limits (see the AI Studio page). If both Gemini and Anthropic keys are set, Gemini is used.

## 2b. Anthropic key (paid, optional): Claude writes the replies

1. Go to https://console.anthropic.com and sign in or create an account.
2. Add a small amount of credit under **Billing** (for example $5). Each reply costs a fraction of a cent with the default small model.
3. Open **API Keys**, **Create Key**, name it `pingfarm`, copy it once (it is not shown again).
4. Put it in `.env` as `ANTHROPIC_API_KEY=...`. Leave `PINGFARM_MODEL` as it is.
5. Run `python scripts/check_keys.py`. You want `anthropic OK`.
6. Start the app (section 4) and check `/health`: `"llm"` should now say `ClaudeAnswerer`.

Cost control: set a monthly spend limit in the console. Replies are short and the long rules are cached.

## 3. Meta WhatsApp Cloud API (free test number): real WhatsApp messages

You need a normal Facebook account. The **test number** is free and enough for a demo and a small pilot.

**A. Create the app**
1. Go to https://developers.facebook.com, log in, accept the developer terms.
2. **My Apps**, **Create App**. Choose the **Business** type (menu wording changes over time; pick the option that adds business messaging). Name it `PingFarm`.
3. On the app dashboard find **WhatsApp** and click **Set up**. Meta creates a test business account and a **test phone number** for you.

**B. Get the values** (WhatsApp, **API Setup** page)
1. **Phone number ID**, shown under the test number. Put in `.env` as `WHATSAPP_PHONE_ID=...`.
2. **Temporary access token**. It expires in about 24 hours, which is fine for a first test. Put in `.env` as `WHATSAPP_TOKEN=...`.
3. Under **To**, add your own phone as a recipient and enter the code WhatsApp sends you. The test number can message only recipients you add here (a handful).
4. **App secret**: **App settings**, **Basic**, **Show** next to App Secret. Put in `.env` as `WHATSAPP_APP_SECRET=...`.
5. **Change `WHATSAPP_VERIFY_TOKEN` from the template's placeholder** and choose any random string for it (for example 20 random letters). You will type the same string into Meta in step D.

Run `python scripts/check_keys.py`: `whatsapp OK` shows the test number if the token and phone id are right.

**C. Make your laptop reachable from the internet** (Meta must be able to call your webhook)
- Start the app: `uvicorn pingfarm.app:app --port 8000`
- In a second terminal use a tunnel, for example Cloudflare's: `cloudflared tunnel --url http://localhost:8000`
  (or `ngrok http 8000`). It prints a public `https://...` address. Keep both running.

**D. Register the webhook** (WhatsApp, **Configuration**)
1. **Callback URL**: `https://<your-tunnel-address>/webhook`
2. **Verify token**: the exact string you chose as `WHATSAPP_VERIFY_TOKEN`.
3. Click **Verify and save**. Meta calls your server; it succeeds only if the app is running and the token matches.
4. Under **Webhook fields** subscribe to **messages**.

**E. Try it**
From the phone you added as a recipient, send a message to the test number: *"How do I control leaf curl virus in tomato?"*
You should get a reply with the source named. Send a leaf photo and a location pin too.

**Later (not needed for a demo)**
- The 24-hour token: for a pilot lasting days, create a **System User** in Meta Business Settings and generate a permanent token for your app.
- A public number anyone can message needs Meta **business verification**. Pricing and free allowances change, so read Meta's current WhatsApp pricing page before a real pilot.
- Voice notes get a polite "coming soon": speech-to-text is not built.

## 4. Run and check the app

```bash
uvicorn pingfarm.app:app --reload        # chat page at http://127.0.0.1:8000
```

`http://127.0.0.1:8000/health` shows which parts are switched on (`llm`, `photo_diagnosis`, `whatsapp_dry_run`).
`whatsapp_dry_run: false` means real sending is on.

## Troubleshooting
- **Webhook verification fails:** the app is not running, the tunnel address changed (free tunnels change on restart), or the verify token differs by one character.
- **No reply on WhatsApp:** check you added your phone as a recipient, the token has not expired (24 h), and the terminal running `uvicorn` for errors.
- **Reply arrives twice:** should not happen (message ids are de-duplicated); tell me if it does.
- **A model download fails with a certificate error:** wait and retry; the network is intermittently interfering. Do not turn certificate checking off.
