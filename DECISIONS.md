# PingFarm: Decision Log

Why each choice was made, what else was considered, and what would change it.
Newest decisions at the bottom.

| # | Decision | Options considered | Why this one | Would change if... |
|---|---|---|---|---|
| D1 | Build my own retrieval (RAG) layer, borrow the language model | Hosted RAG product; LangChain; fully own | Retrieval is where the interesting, measurable work is. Writing chunking, search and ranking myself means I can explain and tune every step. | Time ran out: then use a library for plumbing, keep my own eval. |
| D2 | Scope: tomato and potato, Hindi + Telugu questions | All crops; one region; rice/cotton | PlantVillage (photo data for layer 3) covers tomato and potato well, and the crops are common. Small scope = finishable and measurable. | I find good rice/cotton data and documents. |
| D3 | Knowledge base: public ICAR / TNAU pages and articles, listed in `data/sources.json` | Scrape everything; paid datasets | Trusted sources, citable. We publish only the source list, not the copied text. | A university shares an official dataset. |
| D4 | Embeddings: `intfloat/multilingual-e5-small` | BGE-M3 (stronger, 5x bigger); English-only MiniLM | Handles Hindi and Telugu questions against English passages, small enough for a laptop CPU. | Eval shows weak Telugu retrieval: try BGE-M3 (Experiment 4). |
| D5 | Vector search: FAISS flat index | Hosted vector DB; numpy | A few thousand chunks: exact search is instant. No server to run or pay for. | Corpus grows past ~1M chunks. |
| D6 | Hybrid search = dense + BM25 merged with RRF | Dense only; weighted score sum | RRF needs no score tuning; BM25 catches exact terms (fungicide names, disease names) that embeddings blur. | Eval shows no gain over dense only. |
| D7 | Reranker: multilingual MiniLM cross-encoder, **on** in the app | BGE-reranker-v2-m3 (bigger); none | Measured: dense+header hit@3 0.67 to 0.77, Telugu 0.38 to 0.69. Small enough for CPU. | Latency becomes a problem for live WhatsApp replies. |
| D8 | Evaluation matches text evidence, not chunk ids | Label chunk ids by hand | Chunk ids change whenever chunking changes. Text evidence lets the same questions judge every configuration fairly. | Never: this is a core design choice. |
| D9 | Safety rule: never state pesticide dosages unless a retrieved source says so | Let the model answer from memory | A wrong dose can destroy a crop or harm people. The system must refuse or say "not sure". | Never relaxed. |
| D10 | Python virtual environments kept outside OneDrive | Inside the project folder | OneDrive syncs every file; thousands of small package files make it crawl. | Project moves off OneDrive. |
| D11 | Prepend a contextual header ("Tomato disease: Late blight ...") to every chunk | Do nothing; rewrite the documents | Short TNAU pages never name their crop, so tomato questions pulled potato-article chunks. One line of context per chunk fixed most of it: dense fixed-200 hit@3 0.54 to 0.67 (English 0.69 to 0.92). | A larger test set shows the gain disappears. |
| D12 | Do not disable SSL certificate checking when a model download fails | `verify=False`, unofficial mirrors | The reranker download failed with a self-signed-certificate error on this network. Turning off verification would let anyone on that network tamper with downloaded model files. | I can fetch the model through a trusted path (another network, Kaggle, Colab). |
| D13 | Agent = plain Python loop, no framework | LangChain; LLM that picks tools | Readable, testable, explainable | Tool choice becomes too varied for rules |
| D14 | Intent = nearest centroid of example questions (en/hi/te) | Keywords; LLM; fine-tuned classifier | Reuses the multilingual embedder. Known gap: romanised Hindi ("mera tamatar ka bhav") | More than about 5 intents |
| D15 | Abstain on low closest-passage similarity; threshold calibrated from data; the gate runs on the farmer's own words | Fixed guess; always answer | Wrong confident answers are worse than "not sure". AUC 0.84: separates imperfectly, near-miss farming questions leak through. A crop-name prefix once pushed an off-topic question past the gate (fixed, regression test) | A reranker score or a topic classifier separates better |
| D16 | Dose guard in code (also Hindi/Telugu numerals and units), not only in the prompt | Trust the prompt | Code enforces, a prompt only requests. Does not catch doses in words or "double the amount" | Never |
| D17 | Extractive fallback answerer; Claude answerer plugs in via an env key | Require an API key | Works and is testable with no key or cost | Claude replies are proven better in the eval |
| D18 | SQLite state and message-id de-duplication | Redis / Postgres | One file; enough for a pilot | Many concurrent users |
| D19 | Reply 200 to Meta instantly, work in a background task; verify the HMAC signature | Work inside the request | Meta retries slow webhooks; signatures stop spoofing | Never |
| D20 | Voice notes: honest "coming soon" | Fake it | Speech-to-text for Telugu is a separate hard problem | A speech API (Sarvam / Whisper) is added |
| D21 | Transfer learning: ImageNet MobileNetV3-small | Train from scratch; bigger network | About 5k photos is too few for scratch; tiny model trains on a laptop CPU | Thousands of field photos and a GPU |
| D22 | Train on lab + field photos, test only on held-out field photos | Lab-only test | Deployment looks like field photos; lab-only testing flatters the model (97.9% lab vs 24.7% field) | Farmers' own phone photos become the test set |
| D24 | Choose the "not sure" cut-off on a validation split and record whether the target was reached | Tune on test; silent fallback | A silent fallback (0.5) once posed as calibrated. A 90% target was unreachable; 80% gives tau 0.85, answering about 1 in 8 field photos | More field data, temperature scaling |
| D26 | Tomato and potato only, no healthy-potato class | All crops | Matches the knowledge base; healthy potato exists only in the lab set (background shortcut) | More documents for more crops |
| D27 | Classifier output becomes a text query to retrieval, not a final answer | Let the classifier answer | Advice must come from cited documents | Never |
