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
| D7 | Reranker: multilingual MiniLM cross-encoder, optional | BGE-reranker-v2-m3 (bigger); none | Small enough for CPU. Kept optional so Experiment 3 can measure what it buys. | Latency becomes a problem for live WhatsApp replies. |
| D8 | Evaluation matches text evidence, not chunk ids | Label chunk ids by hand | Chunk ids change whenever chunking changes. Text evidence lets the same questions judge every configuration fairly. | Never: this is a core design choice. |
| D9 | Safety rule: never state pesticide dosages unless a retrieved source says so | Let the model answer from memory | A wrong dose can destroy a crop or harm people. The system must refuse or say "not sure". | Never relaxed. |
| D10 | Python virtual environments kept outside OneDrive | Inside the project folder | OneDrive syncs every file; thousands of small package files make it crawl. | Project moves off OneDrive. |
| D11 | Prepend a contextual header ("Tomato disease: Late blight ...") to every chunk | Do nothing; rewrite the documents | Short TNAU pages never name their crop, so tomato questions pulled potato-article chunks. One line of context per chunk fixed most of it: dense fixed-200 hit@3 0.54 to 0.67 (English 0.69 to 0.92). | A larger test set shows the gain disappears. |
| D12 | Do not disable SSL certificate checking when a model download fails | `verify=False`, unofficial mirrors | The reranker download failed with a self-signed-certificate error on this network. Turning off verification would let anyone on that network tamper with downloaded model files. | I can fetch the model through a trusted path (another network, Kaggle, Colab). |
