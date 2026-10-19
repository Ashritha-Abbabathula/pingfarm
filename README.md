# PingFarm

A farmer asks a crop question in English, Hindi or Telugu, or sends a photo of a sick leaf, and gets
sourced advice from trusted agriculture documents (ICAR, TNAU). WhatsApp-ready FastAPI service.

Scope: **tomato and potato diseases**; questions in **English, Hindi, Telugu**.

> Status: retrieval, evaluation, leaf-photo classifier, agent, safety guard and WhatsApp/web service are
> built and tested (31 tests). **Not done:** a language model writing the replies (the plug-in exists, but
> needs a key and is untested), live WhatsApp and price-API connections (need your credentials),
> speech-to-text, and the farmer pilot. See [DECISIONS.md](DECISIONS.md) and
> [docs/RUN_AND_DEPLOY.md](docs/RUN_AND_DEPLOY.md).

```
 text / photo / location -> agent -> intent: advice | spray timing | price
   photo -> classifier -> disease name -> query     advice -> retrieval (+reranker) -> answer -> dose guard
   spray timing -> weather forecast                 price -> mandi API      "not sure" if nothing relevant
```

## Results (honest numbers, small test sets)

**Retrieval** (39 starter questions = 13 topics x 3 languages; the Hindi and Telugu wording is machine-drafted)

| Configuration | Hit@3 | Hit@3 English | Hit@3 Hindi | Hit@3 Telugu |
|---|---|---|---|---|
| Dense, fixed-200 chunks | 0.54 | 0.69 | 0.54 | 0.38 |
| + contextual chunk headers | 0.67 | 0.92 | 0.69 | 0.38 |
| + cross-encoder reranker (used in the app) | 0.77 | 0.85 | 0.77 | 0.69 |

Keyword (BM25) search scores about 0 on Hindi and Telugu (no shared words). Hybrid vs dense is inconclusive.
One question moves overall Hit@k by 2.6 points (7.7 within one language): treat differences as directional.
Full table: `results/retrieval_experiments.csv`.

**Leaf-photo classifier** (MobileNetV3-small, 10 tomato/potato classes, trained on a CPU)

| Trained on | Lab-photo test (n=1425) | Field-photo test (n=85) |
|---|---|---|
| Lab photos only | 97.9% | **24.7%** |
| Lab + 748 field photos | 98.1% | **56.5%** (95% margin about +-10 points) |

The lab-to-field gap (distribution shift) is the main finding. The "not sure" cut-off (tau 0.85, chosen on a
validation split) answers about 1 in 8 field photos at roughly 80% accuracy: safe, but conservative.

**"I don't know" gate for text:** keeps 90% of real questions but refuses only about half of off-topic
ones (AUC 0.84). Near-miss farming questions (cotton, rice) leak through; the dose guard and visible
sources are the backstops.

## What is built

| Piece | Where |
|---|---|
| Retrieval: chunking, multilingual embeddings, FAISS, BM25, RRF, reranker | `pingfarm/rag/` |
| Evaluation (Hit@k, MRR on evidence text) | `pingfarm/evalkit/` |
| Leaf classifier and "not sure" | `pingfarm/vision/`, `scripts/train_classifier.py`, `scripts/recalibrate_tau.py` |
| Agent, intent detection, abstaining | `pingfarm/agent.py`, `scripts/calibrate_abstain.py` |
| Answer writers and pesticide-dose safety guard | `pingfarm/generation.py` |
| Weather and mandi-price tools | `pingfarm/tools/` |
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
python -m pytest -q tests                   # 31 tests
python -m pingfarm.evalkit.run_eval         # every retrieval experiment (--quick skips the reranker)
python scripts/prepare_classifier_data.py   # downloads and prepares leaf photos (about 1 GB, slow)
python scripts/train_classifier.py          # trains both classifier experiments (~35 min on CPU)
python scripts/calibrate_abstain.py         # measures the text "I don't know" cut-off
python scripts/check_keys.py              # checks your optional keys (.env), never prints them
uvicorn pingfarm.app:app --reload           # chat page at http://127.0.0.1:8000
```

## Known limits
- Small test sets: 39 text questions (machine-drafted Hindi/Telugu) and 85 field photos.
- The knowledge base is mostly English and mostly TNAU/ICAR; Hindi/Telugu source documents are still to be added.
- PlantVillage may contain several photos of one leaf; its train/test split was not independently checked,
  so the lab score is an upper bound.
- The reranker sees 20 candidates out of about 75 chunks, so its gain may shrink on a larger corpus.
- Romanised Hindi/Telugu ("mera tamatar ka bhav") is not handled by intent detection yet.
- The price tool was tested on recorded sample data only; WhatsApp and the language-model answerer were not exercised live.
