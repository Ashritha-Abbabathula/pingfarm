# PingFarm

A farmer asks a crop question in English, Hindi or Telugu and gets an answer grounded in trusted
agriculture sources (ICAR, TNAU), with the source shown. This repository holds the **retrieval
("RAG") layer and its evaluation**: the part that finds the right passage before any language model
writes a reply.

Scope for now: **tomato and potato diseases**, questions in **English, Hindi and Telugu**.

> Status: retrieval layer and evaluation harness working. Crop-disease photo classifier,
> advisory generation, WhatsApp wrapper and farmer pilot are next. See [DECISIONS.md](DECISIONS.md).

## What is built

| Piece | File | What it does |
|---|---|---|
| Document loader | `pingfarm/rag/documents.py` | Reads the downloaded knowledge base |
| Chunking | `pingfarm/rag/chunking.py` | Fixed-window and section-aware chunkers |
| Embeddings | `pingfarm/rag/embed.py` | Multilingual E5 vectors (Hindi/Telugu question finds English passage) |
| Search | `pingfarm/rag/search.py` | Dense (FAISS), BM25, hybrid (RRF), optional cross-encoder reranker |
| Evaluation | `pingfarm/evalkit/` | Hit@k and MRR, judged on text evidence so any chunking can be compared fairly |
| Data | `data/sources.json`, `scripts/` | Source list, downloader, starter question set |

## Results so far (starter set: 39 questions = 13 topics x 3 languages)

| Configuration (dense search, fixed 200-word chunks) | Hit@3 | Hit@3 English | Hit@3 Hindi | Hit@3 Telugu |
|---|---|---|---|---|
| Plain chunks | 0.54 | 0.69 | 0.54 | 0.38 |
| With contextual chunk headers | 0.67 | 0.92 | 0.69 | 0.38 |

- Dense (meaning-based) search beats keyword search by a wide margin; keyword search scores about 0 on Hindi and Telugu questions because no words are shared with English passages.
- Contextual chunk headers gave the largest single gain.
- Hybrid search versus dense is inconclusive at this test-set size.
- The reranker is implemented but **not yet measured** (its model download failed on the development network). Full table: `results/retrieval_experiments.csv`.
- One question moves overall Hit@k by 2.6 points (7.7 points within a single language), so small differences are noise. Treat the table as directional.

## Run it

```bash
python -m venv <somewhere>/pingfarm-venv && activate it
pip install -r requirements.txt
python scripts/fetch_corpus.py              # downloads the source documents
python scripts/build_starter_questions.py   # writes data/eval/questions.jsonl
python -m pytest -q tests                   # fast unit tests
python -m pingfarm.evalkit.run_eval         # runs every retrieval experiment
```

## Safety rules (design constraints)
- Never state a pesticide dose unless a retrieved source states it; otherwise say "not sure".
- Always show the source and its organisation. Say when the question is outside the knowledge base.
- The corpus is not redistributed: only the list of sources is committed.

## Known limits
- The starter evaluation set is small (39 questions, 13 topics x 3 languages) and the Hindi and
  Telugu wording is machine-drafted. Results are directional, not final.
- The knowledge base is mostly English and mostly TNAU/ICAR pages; Hindi/Telugu source documents
  (state agriculture universities) are still to be added.
