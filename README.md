# PingFarm

A farmer asks a crop question in English, Hindi or Telugu, or sends a photo of a sick leaf, and gets
sourced advice from trusted agriculture documents (ICAR, TNAU, Vikaspedia). WhatsApp-ready FastAPI service.

Scope: **tomato and potato** (diseases, pests, cultivation); questions in **English, Hindi, Telugu**.

> Status (version 2, branch `v2-wip`): retrieval, evaluation on real farmer questions, leaf-photo classifier,
> learned topic gate and intent router, farmer profile, weather-risk outlook, soil/weather fit, nearby markets,
> opt-in alerts, agent, safety guard and WhatsApp/web service are built and tested (60 tests passing).
> **Not done:** live tests of the language-model answerers (Gemini or Claude), live WhatsApp and price-API
> connections (need your credentials), speech-to-text, and the farmer pilot.
> See [DECISIONS.md](DECISIONS.md) and [docs/RUN_AND_DEPLOY.md](docs/RUN_AND_DEPLOY.md).

```
 text / photo / location -> agent -> intent: advice | spray timing | price
   photo -> classifier -> disease name -> query     advice -> retrieval (+reranker) -> answer -> dose guard
   spray timing -> weather forecast                 price -> mandi API      "not sure" if nothing relevant
```

## Results (honest numbers, small test sets)

**Retrieval** (39 starter questions = 13 topics x 3 languages; the Hindi and Telugu wording is machine-drafted)

| Configuration | 21 English pages (v1) | 41 pages (v2) | v2 English | v2 Hindi | v2 Telugu |
|---|---|---|---|---|---|
| Dense, fixed-200 chunks | 0.54 | 0.31 | 0.69 | 0.08 | 0.15 |
| + contextual chunk headers | 0.67 | 0.41 | 0.92 | 0.15 | 0.15 |
| + cross-encoder reranker | 0.77 | 0.49 | 0.77 | 0.31 | 0.38 |
| + language-balanced search (used in the app) | not built | **0.77** | 0.85 | 0.69 | 0.77 |

**This got worse when the knowledge base grew from 21 to 41 documents, mostly in Hindi and Telugu.** The cause was a same-language bias: on 22 of the 26 Hindi/Telugu questions that missed (dense + header), all top-3 results were Hindi or Telugu pages, although the answers mostly live in the English pages. The fix fuses a second, English-only ranking with the normal one (`balance_languages`); the last row is the result, back at the old 21-document level (full table: `results/language_balance.csv`). Caveat: only 26 Hindi/Telugu questions, and the fix was chosen and measured on the same sets, so the gain is probably a little optimistic. It helps dense search; hybrid search was mixed (the app uses dense). Old numbers: `results/retrieval_experiments_v1_21docs.csv`.

**Retrieval on 153 real farmer questions** (Kisan Call Centre open data, English only, document-level labels, a
looser test than the starter set; the text is the call-centre operator's summary, not the farmer's own words)

| Configuration | Hit@3 | Hit@1 |
|---|---|---|
| Dense, fixed-200 chunks | 0.69 | 0.46 |
| + contextual chunk headers | 0.79 | 0.60 |
| + cross-encoder reranker | 0.74 | 0.56 |
| + language-balanced search (used in the app) | 0.75 | not measured |
| Section chunks + headers, dense, no reranker (best of 18) | 0.86 | 0.69 |

The reranker lowers real-question Hit@3 slightly for the header setups (0.79 to 0.74, or 0.75 with the language fix), but with the language fix it lifts Hindi and Telugu a lot, so it stays on (D36).
Weakest topics: blight (0.08, partly a labelling problem) and irrigation (0.20, a retrieval failure: the page exists).
Keyword (BM25) search scores near 0 on Hindi and Telugu (no shared words). Hybrid vs dense is inconclusive.
One starter question moves Hit@k by 2.6 points (7.7 within one language): treat differences as directional.
Full table: `results/retrieval_experiments.csv`.

**Leaf-photo classifier** (MobileNetV3-small, 10 tomato/potato classes, trained on a CPU)

| Trained on | Lab-photo test (n=1425) | Field-photo test (n=85) |
|---|---|---|
| Lab photos only | 97.9% | **24.7%** |
| Lab + 748 field photos | 98.1% | **56.5%** (95% margin about +-10 points) |

The lab-to-field gap (distribution shift) is the main finding. The "not sure" cut-off (tau 0.85, chosen on a
validation split) answers about 1 in 8 field photos at roughly 80% accuracy: safe, but conservative.

**"I don't know" gate for text:** the first gate (closest-passage similarity, AUC 0.84) refused only about half
of off-topic questions. The learned gate (logistic regression on question embeddings, trained on real KCC
questions) reaches English cross-validated AUC 0.925 and refuses about 91% of other-crop questions, 70% of
off-topic ones and 62% of tomato/potato questions we do not cover. **On Hindi and Telugu it is only about as good
as the old gate** (AUC 0.825 vs 0.84, tiny held-out set of 26 valid questions) and it still lets some uncovered
Telugu questions through. The dose guard and visible sources are the backstops.

**Intent router:** a trained classifier replaced nearest-centroid after two new intents made the centroid router
fall from 88% to 63% on real questions. Trained: about 90% on real questions and off-topic detection;
slightly lower on small hand-written romanised Hindi/Telugu sets (83% to 79%).

**Tried and rejected:** temperature scaling for the photo classifier (calibration error 0.098 to 0.130).

## What is built

| Piece | Where |
|---|---|
| Retrieval: chunking, multilingual embeddings, FAISS, BM25, RRF, reranker | `pingfarm/rag/` |
| Evaluation (Hit@k, MRR on evidence text) | `pingfarm/evalkit/` |
| Leaf classifier and "not sure" | `pingfarm/vision/`, `scripts/train_classifier.py`, `scripts/recalibrate_tau.py` |
| Agent, intent detection, abstaining | `pingfarm/agent.py`, `scripts/calibrate_abstain.py` |
| Answer writers and pesticide-dose safety guard | `pingfarm/generation.py` |
| Weather and mandi-price tools | `pingfarm/tools/` |
| Learned topic gate and trained intent router | `pingfarm/topic_gate.py`, `pingfarm/intents.py`, `scripts/train_topic_gate.py`, `scripts/train_intents.py` |
| Farmer profile (soil, planting date, alerts, delete data) | `pingfarm/profile.py` |
| Weather risk outlook, soil/weather fit, nearby markets | `pingfarm/tools/risk.py`, `suitability.py`, `markets.py` |
| Opt-in alerts (24-hour WhatsApp rule) | `pingfarm/alerts.py`, `scripts/run_alerts.py` |
| Real farmer questions (KCC open data) | `scripts/fetch_kcc_sample.py`, `scripts/build_real_questions.py` |
| SQLite state, message de-duplication | `pingfarm/state.py` |
| WhatsApp webhook, signature check, web chat page | `pingfarm/whatsapp.py`, `pingfarm/app.py`, `pingfarm/static/` |

## Safety rules (design constraints, enforced in code)
- Never state a pesticide dose that is not in a retrieved source (`dose_guard`; also Hindi/Telugu numerals).
- Always show the source; say "not sure" and point to a local agriculture officer when unsure.
- A diagnosis from a photo is never the answer: it only becomes a query to the cited documents.
- The corpus is not redistributed: only the source list is committed.

## Run it

```bash
python -m venv <somewhere>/pingfarm-venv   # then activate it
pip install -r requirements.txt
python scripts/fetch_corpus.py              # downloads the source documents
python scripts/build_starter_questions.py   # writes data/eval/questions.jsonl
python -m pytest -q tests                   # 60 tests (about 2 minutes)
python -m pingfarm.evalkit.run_eval         # every retrieval experiment (--quick skips the reranker)
python scripts/prepare_classifier_data.py   # downloads and prepares leaf photos (about 1 GB, slow)
python scripts/train_classifier.py          # trains both classifier experiments (~35 min on CPU)
python scripts/calibrate_abstain.py         # measures the text "I don't know" cut-off
python scripts/check_keys.py              # checks your optional keys (.env), never prints them
uvicorn pingfarm.app:app --reload           # chat page at http://127.0.0.1:8000
```

## Known limits
- Small test sets: 39 text questions (machine-drafted Hindi/Telugu; check the wording with a native speaker), 153 real
  questions (English only, operator summaries) and 85 field photos.
- The knowledge base has 41 documents but only 6 are Hindi or Telugu. Starter-set Hindi/Telugu retrieval fell after
  it grew; a language-balanced search recovered it, but only 26 Hindi/Telugu questions measure that.
  Of the 38 real-question misses (of 153), the right page was in the top 20 every time, so they are ranking failures;
  blight is the largest group and 10 had the wrong crop's page on top (`results/error_analysis.csv`).
- The weather-risk, soil-fit and market features are screening rules from the documents, not validated forecasts;
  prices are not profit; there is no soil measurement.
- PlantVillage may contain several photos of one leaf; its train/test split was not independently checked,
  so the lab score is an upper bound.
- The reranker lowers English real-question Hit@3 a little (see D36).
- Romanised Hindi/Telugu ("mera tamatar ka bhav") intent detection is only partly handled (about 79% on a small set).
- The price and markets tools were tested on recorded sample data only; WhatsApp and the Gemini/Claude answerers were
  not exercised live. WhatsApp alert templates still need Meta's approval.
- The Hindi/Telugu topic gate is only about as good as the old gate.

## How this was built
PingFarm was built as a learning project with Claude (an AI assistant) writing most of the code, tests, notebooks
and study guide with me, step by step. I review the results, and the numbers above come from runs saved in `results/`.
Hindi and Telugu wording was machine-drafted and is marked for a native speaker to check.
