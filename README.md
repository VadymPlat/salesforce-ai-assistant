# Salesforce AI Assistant

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B?logo=streamlit&logoColor=white)
![Claude](https://img.shields.io/badge/Claude-Opus%204.6-8A2BE2?logo=anthropic&logoColor=white)
![RAG](https://img.shields.io/badge/Architecture-RAG-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A production-ready **Retrieval Augmented Generation (RAG)** application that answers questions about Salesforce by searching a curated knowledge base and generating intelligent, context-grounded responses using Anthropic's Claude Opus 4.6. Built as a portfolio project demonstrating end-to-end AI system design skills relevant to an AI/Cloud Architect role.

> Ask it anything about Salesforce — security models, Apex triggers, Flow automation, data architecture, SOQL, governor limits — and get accurate, cited answers in real time.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                               │
│                    Streamlit Chat (app.py)                          │
│         Real-time streaming · Source citations · Sidebar mgmt      │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ user question
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        RAG PIPELINE (rag.py)                        │
│                                                                     │
│   1. RETRIEVE ──► 2. AUGMENT PROMPT ──► 3. STREAM GENERATION       │
│                                                                     │
└──────────┬──────────────────────────────────────┬───────────────────┘
           │ semantic search                       │ augmented prompt
           ▼                                       ▼
┌──────────────────────┐              ┌────────────────────────────────┐
│   VECTOR STORE       │              │     CLAUDE OPUS 4.6 API        │
│  (vectorstore.py)    │              │     (Anthropic)                │
│                      │              │                                │
│  sentence-           │              │  · Adaptive thinking           │
│  transformers        │              │  · Streaming response          │
│  (all-MiniLM-L6-v2)  │              │  · Grounded in retrieved docs  │
│                      │              │                                │
│  numpy cosine sim    │              └────────────────────────────────┘
│  + disk persistence  │
└──────────┬───────────┘
           │ indexed on first run
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE (ingestion.py)                │
│                                                                     │
│   Load .txt files  ──►  Fetch URLs  ──►  Clean HTML  ──►  Chunk    │
│   (data/ folder)        (requests +      (BeautifulSoup)  (600 ch  │
│                          BeautifulSoup)                   overlap) │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ source documents
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        KNOWLEDGE BASE (data/)                       │
│                                                                     │
│   salesforce_basics.txt  ·  security_model.txt                     │
│   data_model.txt         ·  automation.txt                         │
│   + any custom .txt files or URLs you add via the sidebar          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## How RAG Works

Traditional LLMs can hallucinate or give outdated answers because they rely solely on their training data. RAG solves this by first **retrieving** the most relevant passages from a trusted knowledge base, then **augmenting** the model's prompt with that real content before generating a response. The model is never asked to "remember" Salesforce — it is given the relevant documentation and asked to synthesise an answer from it. This makes responses more accurate, grounded, and auditable, since the user can inspect the exact source passages used.

---

## Technology Stack

| Layer | Technology | Why This Choice |
|---|---|---|
| **LLM** | Claude Opus 4.6 (`anthropic`) | Best-in-class reasoning, 200K context window, adaptive thinking for complex questions, streaming API |
| **Embeddings** | `sentence-transformers` (`all-MiniLM-L6-v2`) | Runs entirely locally — no extra API key, low latency, 384-dim vectors balance quality and speed |
| **Vector Store** | `numpy` (cosine similarity + disk persistence) | No external service needed, transparent implementation, fully compatible with Python 3.14+ |
| **UI** | `streamlit` | Production-quality chat UI in minimal code, native streaming support via `st.write_stream` |
| **Ingestion** | `requests` + `beautifulsoup4` | Lightweight, zero-dependency HTML scraping for loading live Salesforce documentation |
| **Config** | `python-dotenv` | Industry-standard secret management, keeps API keys out of source control |

---

## Project Structure

```
salesforce-ai-assistant/
│
├── app.py                      # Streamlit chat UI — entry point
│                               # Handles: streaming, chat history, sidebar doc management
│
├── src/
│   ├── __init__.py
│   ├── ingestion.py            # Document loading pipeline
│   │                           # Fetches URLs, cleans HTML, chunks text with overlap
│   ├── vectorstore.py          # Embedding + semantic search engine
│   │                           # Encodes docs with sentence-transformers, persists to disk,
│   │                           # retrieves via cosine similarity at query time
│   └── rag.py                  # Core RAG orchestration
│                               # Ties retrieval and generation together, streams Claude's response
│
├── data/
│   ├── salesforce_basics.txt   # Core Salesforce concepts (objects, SOQL, Apex, APIs, LWC)
│   ├── security_model.txt      # Profiles, Permission Sets, OWD, Role Hierarchy, Sharing Rules
│   ├── data_model.txt          # Object design, relationships, field types, LDV patterns
│   └── automation.txt          # Flows, Apex Triggers, Order of Execution, Approval Processes
│
├── vectorstore/                # Auto-created — persisted embeddings and doc metadata
│   ├── embeddings.npy          # numpy array of shape (N, 384)
│   └── docs.json               # Chunk text + source metadata
│
├── requirements.txt            # Python dependencies
├── .env.example                # API key template (copy to .env)
└── .gitignore                  # Excludes .env and vectorstore/ from git
```

---

## Setup and Installation

### Prerequisites
- Python 3.9+
- An [Anthropic API key](https://console.anthropic.com)

### 1. Clone the repository

```bash
git clone https://github.com/VadymPlat/salesforce-ai-assistant.git
cd salesforce-ai-assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> On first run, `sentence-transformers` will download the embedding model (~80 MB). This only happens once.

### 4. Configure your API key

```bash
cp .env.example .env
```

Edit `.env` and add your Anthropic API key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

### 5. Launch the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`

### 6. Load the knowledge base

Click **"Load / Reload Salesforce Docs"** in the sidebar. The app will:
- Read all `.txt` files from the `data/` folder
- Embed each document chunk using the local model
- Save embeddings to disk (takes ~30 seconds the first time)

Subsequent launches reuse the saved embeddings instantly.

---

## Example Questions

The assistant can answer a wide range of Salesforce questions:

**Security & Access**
- _"What is the difference between a Profile and a Permission Set?"_
- _"How do Organisation-Wide Defaults interact with the Role Hierarchy?"_
- _"What is criteria-based sharing and when should I use it?"_

**Data Modelling**
- _"When should I use a Master-Detail relationship instead of a Lookup?"_
- _"How do I model a many-to-many relationship in Salesforce?"_
- _"What are External IDs and how do upsert operations work?"_

**Automation**
- _"What is the order of execution when a record is saved in Salesforce?"_
- _"When should I use Flow vs Apex Trigger?"_
- _"How do I prevent trigger recursion in Apex?"_

**Development**
- _"What is SOQL and how does it differ from SQL?"_
- _"How do Apex governor limits work and how do I avoid hitting them?"_
- _"What is a Lightning Web Component?"_

---

## Extending the Knowledge Base

**Add local documents:**
Drop any `.txt` file into `data/` and click "Reload Docs" in the sidebar.

**Add a URL:**
Paste any publicly accessible Salesforce documentation URL into the sidebar input and click "Ingest URL".

**Add default URLs:**
Edit `src/ingestion.py` and add entries to `DEFAULT_SALESFORCE_URLS`.

---

## Known Limitations

| Limitation | Detail |
|---|---|
| **Static knowledge base** | The app does not connect to Salesforce live. Answers are only as current as the last ingestion. |
| **In-memory search at scale** | The numpy vector store loads all embeddings into RAM. For >100K chunks, a dedicated vector database (Pinecone, Weaviate) would be more appropriate. |
| **No JS-rendered pages** | The URL scraper uses `requests` and cannot execute JavaScript. Some Salesforce documentation pages require a headless browser to render. |
| **English only** | The embedding model and prompts are optimised for English. |
| **No conversation memory** | Each question is answered independently. Multi-turn follow-up questions (e.g., "Tell me more about that") lose context between turns. |

---

## Future Improvements

- [ ] **Conversation memory** — pass recent chat history to Claude to support follow-up questions
- [ ] **Hybrid search** — combine semantic (vector) search with keyword (BM25) search for better precision on exact terms like API names
- [ ] **Automatic re-ingestion** — detect when `data/` changes and re-index without a manual button click
- [ ] **Headless browser ingestion** — use Playwright to scrape JavaScript-rendered Salesforce Help pages
- [ ] **Confidence scoring** — display a warning when retrieved chunks have low similarity scores, signalling the question is outside the knowledge base
- [ ] **Multi-org support** — allow switching between different Salesforce knowledge bases (Sales Cloud, Service Cloud, Marketing Cloud)
- [ ] **Authentication** — add a login layer for production deployment
- [ ] **Docker deployment** — containerise the app for one-command cloud deployment

---

## What I Learned Building This

**RAG architecture in practice:** Understanding the nuances of chunk size, overlap, and retrieval count required experimentation. Too-large chunks reduce precision; too-small chunks lose context. An overlap of ~13% of chunk size proved effective.

**Embedding model selection:** Choosing a local model (`all-MiniLM-L6-v2`) over a cloud API for embeddings reduces latency, eliminates a second API dependency, and makes the app fully self-contained for demos.

**Python 3.14 compatibility challenges:** ChromaDB's Pydantic v1 dependency broke on Python 3.14. Re-implementing the vector store with numpy was a valuable exercise — it demystified what a vector database actually does (store vectors, compute cosine similarity, return top-K results).

**Streaming UX matters:** Implementing token-by-token streaming via Python generators and `st.write_stream` makes the app feel responsive even when Claude takes 5–10 seconds to generate a full answer. This is a critical UX decision for AI applications.

**Prompt engineering for RAG:** Structuring the retrieved context with numbered excerpts and source labels helps Claude cite information accurately and avoid blending sources. The system prompt's instruction to "say so honestly" when context is insufficient significantly reduces hallucination.

---

## Author

**Vadym Platoshyn** — Salesforce Solution Architect transitioning to AI/Cloud Architecture

16+ years of enterprise architecture experience across Salesforce implementations. This project is part of an 18-month journey from Salesforce SA to AI/Cloud Architect.

[GitHub](https://github.com/VadymPlat) · [LinkedIn](https://www.linkedin.com/in/vadym-p-b2318111)
