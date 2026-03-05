# Salesforce AI Assistant

A RAG (Retrieval Augmented Generation) application that answers questions about Salesforce by searching through documentation and generating intelligent responses using Claude claude-opus-4-6.

## How it Works

```
User Question
     |
     v
[ Embedding Model ]  <-- encodes the question into a vector
     |
     v
[ ChromaDB Vector Store ]  <-- finds the most similar doc chunks
     |
     v
[ Claude claude-opus-4-6 (Anthropic) ]  <-- generates a grounded answer
     |
     v
Streamed Answer
```

1. **Ingestion** (`src/ingestion.py`): Salesforce documentation is fetched from URLs or loaded from local `.txt` files, then split into overlapping chunks.
2. **Embedding** (`src/vectorstore.py`): Each chunk is converted into a vector using `sentence-transformers` (runs locally, no extra API key needed) and stored in ChromaDB.
3. **Retrieval + Generation** (`src/rag.py`): When a question arrives, the most relevant chunks are retrieved and passed to Claude as context. Claude streams back a grounded answer.
4. **UI** (`app.py`): A Streamlit chat interface streams tokens to the screen in real time.

## Setup

### 1. Clone the project and navigate into it

```bash
cd salesforce-ai-assistant
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set your Anthropic API key

```bash
cp .env.example .env
# Edit .env and add your key from https://console.anthropic.com
```

### 4. Run the app

```bash
streamlit run app.py
```

### 5. Load documents

Click **"Load / Reload Salesforce Docs"** in the sidebar. This:
- Loads the sample doc from `data/salesforce_basics.txt`
- Fetches additional pages from Salesforce documentation URLs
- Embeds everything and stores it in ChromaDB (persisted to `./vectorstore/`)

Documents are only ingested once — subsequent runs reuse the saved vector store.

## Adding More Documentation

**Option A — Local text files:**
Drop any `.txt` file into the `data/` directory and click "Reload Docs".

**Option B — URLs via the sidebar:**
Paste any Salesforce documentation URL into the sidebar and click "Ingest URL".

**Option C — Edit the default URL list:**
Open `src/ingestion.py` and add URLs to `DEFAULT_SALESFORCE_URLS`.

## Project Structure

```
salesforce-ai-assistant/
├── app.py                  # Streamlit chat UI
├── requirements.txt        # Python dependencies
├── .env.example            # API key template
├── .gitignore
├── README.md
├── data/
│   └── salesforce_basics.txt   # Sample Salesforce documentation
├── src/
│   ├── __init__.py
│   ├── ingestion.py        # Fetch, clean, and chunk documents
│   ├── vectorstore.py      # ChromaDB + sentence-transformers
│   └── rag.py              # RAG pipeline (retrieve + generate)
└── vectorstore/            # Auto-created by ChromaDB (gitignored)
```

## Tech Stack

| Component | Library | Why |
|---|---|---|
| LLM | `anthropic` (Claude claude-opus-4-6) | Best-in-class reasoning and long context |
| Embeddings | `sentence-transformers` | Free, runs locally, no extra API key |
| Vector DB | `chromadb` | Simple, persistent, no server needed |
| UI | `streamlit` | Fast to build, great for demos |
| Scraping | `requests` + `beautifulsoup4` | Lightweight HTML parsing |

## Example Questions to Try

- What is SOQL and how does it differ from SQL?
- How do Apex governor limits work?
- What is the difference between a Profile and a Permission Set?
- How do I use the Salesforce REST API to create a record?
- What is a Lightning Web Component?
- When should I use a Flow vs. Apex?
