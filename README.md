# Study Assistant (RAG)

Upload lecture PDFs or web pages, ask questions, and get answers that **cite the file and page** they came from.
Includes a "quiz me" mode, follow-up-aware conversation memory, and a small evaluation harness that compares
chunk sizes, top-k and embedding models.

## Architecture

```
            ┌────────────┐   PDF / URL    ┌───────────────────────────── FastAPI ─────────────────────────────┐
            │ Streamlit  │ ─────────────▶ │ /ingest/pdf  /ingest/url                                          │
            │    UI      │                │   pypdf (1 Document per page) / BeautifulSoup                     │
            │            │                │   → LlamaIndex SentenceSplitter (size/overlap)                    │
            │            │                │   → bge-small embeddings (local, sentence-transformers)           │
            │            │                │   → Chroma (persistent, metadata: source + page)                  │
            │            │  question      │                                                                   │
            │            │ ─────────────▶ │ /ask   history ─▶ condense follow-up into standalone question     │
            │            │                │        ─▶ top-k retrieval ─▶ LLM answers from numbered passages   │
            │            │ ◀───────────── │        ─▶ answer + citations [{source, page, snippet}]            │
            │            │  /quiz         │ /quiz  sample or topic-retrieve passages ─▶ LLM writes Q&A        │
            └────────────┘                └───────────────────────────────────────────────────────────────────┘
```

- **RAG framework:** LlamaIndex. **Vector store:** Chroma (local, persisted in `storage/chroma`).
- **Embeddings:** `BAAI/bge-small-en-v1.5`, runs locally; no key needed.
- **LLM:** Anthropic or OpenAI (whichever API key is set), or a free local model via Ollama when no key is set. With no key the app falls back to an
  *extractive* answer (best-matching sentences, still cited), so retrieval, citations and evaluation all work
  offline. Quiz mode needs an LLM.
- **Citations:** each page of a PDF is loaded as its own document, so every chunk carries an exact page number.
  Web pages are cited by URL (page 1).
- **Memory:** per-session history; a follow-up like "and the decoder?" is rewritten into a standalone question
  before retrieval. (Sessions live in process memory and reset on server restart.)

## Setup

Requires Python 3.10+.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # large: pulls PyTorch for sentence-transformers
cp .env.example .env                     # optional: add ANTHROPIC_API_KEY or OPENAI_API_KEY

uvicorn app.main:app --reload            # API + docs at http://localhost:8000/docs
streamlit run ui/streamlit_app.py        # UI at http://localhost:8501  (second terminal)
```

The first run downloads the embedding model (~130 MB) and some NLTK data; once cached, the app runs offline (set `HF_HUB_OFFLINE=0` to force online, e.g. when changing `EMBED_MODEL`). A sample PDF is in
[data/sample](data/sample) (Vaswani et al., *Attention Is All You Need*).

Quick API check:

```bash
curl -F file=@data/sample/attention_is_all_you_need.pdf localhost:8000/ingest/pdf
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
     -d '{"question": "Why are dot products scaled by sqrt(d_k)?"}'
```

Run the smoke test: `pytest -q tests` (no API key needed).

### No API key? Use a free local model

```bash
curl -fsSL https://ollama.com/install.sh | sh     # installs Ollama (needs sudo)
ollama pull llama3.2:3b                           # ~2 GB download
```

Remove or blank any invalid `*_API_KEY` in `.env` (and `unset` it in your shell), then restart uvicorn. With Ollama
running, answers and quizzes are generated locally; no account needed. Answers are slower and weaker than a hosted
model; try `OLLAMA_MODEL=qwen2.5:7b` if you have the RAM.

## Evaluation

[eval/questions.json](eval/questions.json) holds 28 hand-written questions over the sample paper, each with a
reference answer, the page(s) it lives on, and "evidence" strings that must appear in a chunk for it to count as
containing the answer. Metrics (all computed without an LLM):

- **page@k**: an expected page is among the top-k retrieved chunks
- **evid@k**: some top-k chunk contains *all* evidence strings for the question
- **mrr**: mean reciprocal rank of the first chunk containing the evidence

Reproduce with `python -m eval.run_eval` (add `--judge` and an API key to also grade generated answers with an LLM).
Results, 28 questions, 15-page PDF:

| config | chunks | page@1 | page@2 | page@4 | page@8 | evid@1 | evid@2 | evid@4 | evid@8 | mrr |
|---|---|---|---|---|---|---|---|---|---|---|
| chunk=128/16, bge-small | 110 | 0.89 | 0.96 | 1.00 | 1.00 | 0.46 | 0.64 | 0.71 | 0.79 | 0.58 |
| chunk=256/32, bge-small | 56 | 0.82 | 0.96 | 1.00 | 1.00 | 0.46 | 0.71 | 0.75 | 0.79 | 0.61 |
| **chunk=512/64, bge-small** | 29 | 0.64 | 0.96 | 0.96 | 1.00 | **0.57** | **0.75** | **0.79** | **0.86** | **0.68** |
| chunk=1024/128, bge-small | 15 | 0.64 | 0.75 | 0.82 | 1.00 | 0.54 | 0.64 | 0.71 | 0.86 | 0.63 |
| chunk=512/64, MiniLM-L6 | 29 | 0.61 | 0.82 | 0.93 | 0.96 | 0.39 | 0.61 | 0.71 | 0.75 | 0.54 |

**Takeaways**

- 512/64 gives the best answer retrieval (evid@4 0.79, MRR 0.68), so it is the default. Smaller chunks (128) win
  on page@1 because they are more focused, but they often split the answer away from its surrounding text.
- Raising top-k from 4 to 8 helps (evid 0.79 → 0.86) but pulls in more context for the LLM; k=4 is the default
  trade-off. Pages are found almost always by k=4 (page@4 ≥ 0.96 for chunks ≤ 512).
- `bge-small` beat `all-MiniLM-L6-v2` at the same chunking on every metric.

**Caveats:** one document and 28 questions, so differences of a few points are within noise. Evidence matching
requires all strings in one chunk, which penalizes small chunks. End-to-end answer quality (`--judge`) has not
been run because no API key was available while building this.

## Layout

```
app/        config, ingestion, RAG (retrieve/ask/quiz), LLM wrapper, FastAPI routes
ui/         Streamlit front end
eval/       labelled questions + run_eval.py
tests/      end-to-end smoke test
```

## Demo

_Add a screenshot or GIF of the Streamlit UI here (`streamlit run ui/streamlit_app.py`)._
