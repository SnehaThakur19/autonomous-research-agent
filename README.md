# 🔬 Autonomous Research Agent

An autonomous AI research agent built with **LangGraph** that answers complex research
questions by decomposing them into research tasks, retrieving evidence from multiple
sources, verifying every claim against its evidence, self-correcting when evidence is
insufficient, and producing fully traceable, citation-grounded reports — with a polished
**multi-step wizard web UI** (Streamlit).

> **🚀 Try it live:** [autonomous-research-agent-4htnpcgk7ukhzwf8mrmluj.streamlit.app](https://autonomous-research-agent-4htnpcgk7ukhzwf8mrmluj.streamlit.app/)

---

## 💡 Project Idea

Research questions are rarely answerable from a single paragraph. A human researcher
breaks a big question into sub-questions, looks things up in the paper, checks external
literature, doubts a claim, searches again, and only then writes an answer — with
citations. This project gives an LLM agent exactly that workflow:

1. **Decompose** the question into focused research subtasks.
2. **Retrieve** evidence from the local PDF (embedding search + CrossEncoder reranking).
3. **Extract claims** and **verify each claim** against the retrieved evidence.
4. **Validate citations** programmatically — every cited chunk/page must actually exist.
5. **Critique** evidence sufficiency. If gaps remain, **self-correct**: rewrite the query
   and switch tools (re-search the PDF → academic paper search on arXiv → web search).
6. **Report** — an evidence-grounded answer where every finding carries a citation, and
   unsupported claims are explicitly flagged instead of being invented.

The result: an answer you can *trust and audit*, because every sentence maps back to a
source (PDF page + chunk ID, or a URL).

---

## 🏗️ Architecture

```
                        ┌──────────────────────────────┐
                        │  Plan: decompose query into  │
                        │  research subtasks (LLM)     │
                        └──────────────┬───────────────┘
                                       ▼
                        ┌──────────────────────────────┐
                        │  Retrieve: per-subtask PDF   │
                        │  embedding search → merge →  │
                        │  CrossEncoder reranking      │
                        └──────────────┬───────────────┘
                                       ▼
                        ┌──────────────────────────────┐
                        │  Extract claims (LLM)        │
                        └──────────────┬───────────────┘
                                       ▼
                        ┌──────────────────────────────┐
                        │  Verify claims vs evidence   │
                        │  SUPPORTED / PARTIAL / NOT   │
                        └──────────────┬───────────────┘
                                       ▼
                        ┌──────────────────────────────┐
                        │  Citation validation         │
                        │  (programmatic, regex-based) │
                        └──────────────┬───────────────┘
                                       ▼
                        ┌──────────────────────────────┐         ┌────────────────┐
                        │  Critic: evidence-sufficiency│──GOOD──▶│  Final report  │
                        │  check + tool recommendation│         │  + citations   │
                        └──────┬───────────────┬───────┘         └────────────────┘
                           NEEDS_RESEARCH      │
                               ▼               │
                        ┌──────────────────────┴───────┐
                        │  Self-correction loop        │
                        │  (bounded iterations):       │
                        │  • rewrite query             │
                        │  • pick tool: PDF / arXiv /  │
                        │    Web                       │
                        │  • re-verify claims          │◀───┐
                        └──────────────────────────────┘    │
                                loop until GOOD or cap ─────┘
```

The workflow is a **LangGraph `StateGraph`** — each box is a node; conditional edges
route between "keep researching" and "write the report".

---

## ✨ Features

| Feature | How it works |
|---|---|
| **Query decomposition** | Planner node splits complex questions into ≤3 focused subtasks; retrieval runs per subtask, results merged & deduped |
| **Hybrid RAG** | Local PDF retrieval (pypdf → chunking → MiniLM embeddings → ChromaDB) + **academic search (arXiv API, no key needed)** + **web search (Tavily)** |
| **Embedding retrieval + reranking** | `all-MiniLM-L6-v2` vector search → `ms-marco-MiniLM-L-6-v2` CrossEncoder reranking to top-5 evidence |
| **Self-correction** | Critic decides GOOD / NEEDS_RESEARCH and recommends the next tool (PDF re-retrieval, arXiv, or web); query is rewritten each attempt; loop bounded by max iterations |
| **Claim-to-evidence verification** | LLM verifies each extracted claim as SUPPORTED / PARTIALLY / NOT SUPPORTED with reasons |
| **Citation validation** | Programmatic parser checks every cited evidence number exists and maps it to source file + page + chunk ID; invalid citations flagged |
| **Traceable reports** | Report separates Key Findings (cited) from Unsupported Claims and Evidence gaps; includes Sources with URLs |
| **Multi-step wizard UI** | Upload → Question → Plan → Research (live event timeline) → Report; state persists across back/next; copy/download reports |

---

## 📁 Project Structure

```
autonomous-research-agent/
├── streamlit_app.py            # Streamlit Cloud entrypoint
├── requirements.txt            # Python dependencies (CPU torch)
├── .env                        # GROQ_API_KEY, TAVILY_API_KEY (local only, not in git)
├── .streamlit/config.toml      # Streamlit server config
├── data/                       # Research PDFs
├── chroma_db/                  # Persistent vector store (generated, not in git)
├── app/
│   ├── webapp.py               # ⭐ Multi-step wizard web UI (Streamlit)
│   ├── main.py                 # CLI entrypoint
│   ├── ingest.py               # PDF ingestion CLI (load→chunk→embed→index)
│   ├── report_generator.py     # Final evidence-grounded report (LLM)
│   ├── test_all_features.py    # 17-feature test suite
│   ├── graph/
│   │   ├── state.py            # ResearchState (TypedDict)
│   │   ├── workflow.py         # LangGraph StateGraph + conditional routing
│   │   ├── nodes.py            # All graph nodes (plan/retrieve/verify/...)
│   │   └── web_verification.py # External (web/arXiv) claim re-verification
│   ├── prompts/
│   │   └── prompts.py          # Central repository of all LLM prompts
│   ├── rag/
│   │   ├── loader.py           # PDF loading (pypdf)
│   │   ├── chunker.py          # Recursive chunking (1000 chars, 150 overlap)
│   │   ├── embeddings.py       # SentenceTransformer embeddings
│   │   ├── vectorstore.py      # ChromaDB persistent store
│   │   ├── retriever.py        # Vector similarity retrieval
│   │   └── reranker.py         # CrossEncoder reranking
│   ├── research/
│   │   ├── task_decomposer.py  # Query → research subtasks (LLM)
│   │   ├── query_generator.py  # Query rewriting for self-correction
│   │   ├── web_search.py       # Tavily web search
│   │   └── academic_search.py  # arXiv academic paper search
│   └── verification/
│       ├── claim_extractor.py  # Extract factual claims (LLM)
│       ├── verifier.py         # Verify claims vs evidence (LLM)
│       ├── critic.py           # Evidence-sufficiency critic + tool choice
│       └── citation_validator.py # Programmatic citation validation
```

---

## 🚀 Run Locally

```bash
git clone https://github.com/SnehaThakur19/autonomous-research-agent.git
cd autonomous-research-agent

python -m venv venv
venv\Scripts\activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Create .env with your keys:
#   GROQ_API_KEY=your_groq_key
#   TAVILY_API_KEY=your_tavily_key

# Ingest a PDF (from project root):
python -m app.ingest data/research_paper.pdf --rebuild

# Web UI (opens http://localhost:8501):
streamlit run streamlit_app.py

# Or CLI:
python -m app.main "What architecture is proposed and what results does it achieve?" --verbose

# Run the full test suite (17 features):
python -m app.test_all_features
```

---

## ☁️ Deploy on Streamlit Community Cloud

1. Fork/push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **Create app** → select this
   repo, branch `main`, main file path `streamlit_app.py`.
3. In the app's **Settings → Secrets**, add:

   ```toml
   GROQ_API_KEY = "your_groq_key"
   TAVILY_API_KEY = "your_tavily_key"
   ```

4. Deploy — the app builds with CPU-only torch and loads ML models on first use.

---

## 🔑 API Keys

| Key | Used for | Where to get |
|---|---|---|
| `GROQ_API_KEY` | All LLM calls (claim extraction, verification, critic, report) | [console.groq.com](https://console.groq.com) |
| `TAVILY_API_KEY` | Web search during self-correction | [tavily.com](https://tavily.com) |

arXiv academic search needs **no API key**.

---

## 🧪 Verification

The test suite (`app/test_all_features.py`) exercises every feature end-to-end:
PDF loading → chunking → embeddings → vector store → retrieval → reranking →
query decomposition → arXiv search → web search → claim extraction → verification →
citation validation → critic → query rewriting → web re-verification → **full graph run**.
All 17 features pass.

---

## 📄 License

MIT — feel free to use and modify.
