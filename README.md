---
title: Enterprise Knowledge Assistant
emoji: 🧠
sdk: docker
app_port: 7860
---

# 🧠 Enterprise Knowledge Assistant
### RAG-Powered Document Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-green)](https://langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-orange)](https://faiss.ai/)
[![Gemini](https://img.shields.io/badge/Gemini-1.5_Flash-purple?logo=google)](https://deepmind.google/technologies/gemini/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> Upload any PDF, DOCX, or TXT document and ask natural language questions about it.
> Get precise, cited answers powered by Google Gemini and semantic vector search.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 **Multi-format support** | PDF, DOCX, and TXT documents |
| 🔍 **Semantic search** | FAISS vector search with sentence-transformers |
| 📚 **Source citations** | Every answer cites the exact document + page number |
| 💬 **Conversation memory** | Multi-turn chat with full history |
| 📊 **Analytics dashboard** | Query stats, response times, token usage |
| 💾 **Persistent storage** | SQLite for chat history, FAISS for vector index |
| 🔒 **Production architecture** | Clean layered design, custom exceptions, structured logging |
| ⚡ **Fast** | Groq-level speed via Gemini Flash, batched embeddings |

---

## 🏗️ Architecture

```
User Query
    ↓
[Streamlit UI]          ← Presentation Layer
    ↓
[RAG Pipeline]          ← Application Layer (Orchestrator)
    ↓              ↘
[FAISS Search]    [Gemini LLM]   ← Infrastructure Layer
    ↓
[Top-K Chunks + Citations]
    ↓
[Answer + Sources → UI]
```

**Clean Architecture principles:**
- **Domain layer** (`core/`) — pure Python dataclasses, zero external dependencies
- **Application layer** (`services/`) — business logic, pipeline orchestration
- **Infrastructure layer** (`database/`) — SQLite, FAISS, external APIs
- **Presentation layer** (`ui/`) — Streamlit components only

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | Streamlit | Interactive web UI |
| Orchestration | LangChain | Text splitting, pipeline |
| LLM | Google Gemini 1.5 Flash | Answer generation |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Text → vectors |
| Vector DB | FAISS | Semantic similarity search |
| Metadata DB | SQLite | Chat history, analytics |
| PDF parsing | PyMuPDF (fitz) | Page-aware text extraction |
| DOCX parsing | python-docx | Word document support |
| Config | pydantic-settings | Env var validation |
| Logging | Python logging + RotatingFileHandler | Structured logs |
| Testing | pytest + pytest-cov | Unit tests with coverage |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Google Gemini API key (free at [aistudio.google.com](https://aistudio.google.com))

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/enterprise-knowledge-assistant.git
cd enterprise-knowledge-assistant

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 5. Run the app
streamlit run main.py
```

Open your browser at **http://localhost:8501** 🎉

---

## 📁 Project Structure

```
enterprise-knowledge-assistant/
├── app/
│   ├── core/                   # Domain layer (no external deps)
│   │   ├── models.py           # Document, Chunk, ChatMessage, Citation
│   │   ├── config.py           # Centralised configuration (pydantic)
│   │   └── exceptions.py       # Custom exception hierarchy
│   ├── services/               # Application layer (business logic)
│   │   ├── document_processor.py  # PDF/DOCX/TXT → Chunks
│   │   ├── embedding_service.py   # Text → 384-dim vectors
│   │   ├── vector_store.py        # FAISS index management
│   │   ├── llm_service.py         # Gemini API wrapper
│   │   └── rag_pipeline.py        # End-to-end orchestrator
│   ├── database/               # Infrastructure layer
│   │   ├── db_manager.py       # SQLite connection + schema
│   │   └── repositories.py     # CRUD operations
│   ├── ui/                     # Presentation layer
│   │   ├── components.py       # Reusable UI widgets
│   │   ├── sidebar.py          # Document management panel
│   │   ├── chat_interface.py   # Conversation window
│   │   └── analytics.py        # Dashboard
│   └── utils/
│       ├── logger.py           # Structured logging
│       └── helpers.py          # Pure utility functions
├── data/
│   ├── uploads/                # Persisted uploaded files
│   ├── faiss_index/            # Saved FAISS index
│   └── db/                     # SQLite database
├── tests/
│   └── test_document_processor.py
├── main.py                     # Streamlit entry point
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Configuration

All settings live in `.env` (copy from `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | **Required** | Your Google AI Studio API key |
| `GEMINI_MODEL` | `gemini-1.5-flash` | LLM model to use |
| `GEMINI_TEMPERATURE` | `0.1` | Response creativity (0=focused, 1=creative) |
| `CHUNK_SIZE` | `1000` | Characters per text chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RETRIEVAL_TOP_K` | `5` | Chunks to retrieve per query |
| `SIMILARITY_THRESHOLD` | `0.3` | Minimum relevance score |
| `MAX_FILE_SIZE_MB` | `50` | Upload size limit |

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=app --cov-report=html

# Run specific test file
pytest tests/test_document_processor.py -v
```

---

## 📊 How RAG Works

```
1. INGESTION (at upload time)
   PDF/DOCX/TXT
       ↓ extract text per page (PyMuPDF / python-docx)
       ↓ split into overlapping chunks (LangChain RecursiveCharacterTextSplitter)
       ↓ generate embeddings (sentence-transformers → 384-dim float32)
       ↓ store in FAISS index + metadata dict

2. RETRIEVAL (at query time)
   User question
       ↓ embed query (same model, same 384-dim space)
       ↓ FAISS inner product search → top-K similar chunks
       ↓ filter by similarity threshold (0.3)

3. GENERATION
   Top-K chunks + question
       ↓ build prompt with [Source N] tags
       ↓ Gemini generates answer with context
       ↓ return answer + citations (filename + page number)
```

---

## 🗺️ Development Roadmap

- [x] Phase 1 — Architecture & project setup
- [x] Phase 2 — Document parsing (PDF, DOCX, TXT)
- [x] Phase 3 — Embedding & FAISS indexing
- [x] Phase 4 — Gemini RAG pipeline
- [x] Phase 5 — Streamlit chat interface
- [x] Phase 6 — SQLite + analytics dashboard
- [ ] Phase 7 — Docker containerisation
- [ ] Phase 8 — HuggingFace Spaces deployment
- [ ] Phase 9 — Multi-user support with user authentication

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'feat: add amazing feature'`
4. Push: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) file.

---

## 👨‍💻 Author

Built by [Your Name](https://github.com/yourusername) as an industry-level capstone project
demonstrating end-to-end RAG system design with clean architecture principles.

**Connect:** [LinkedIn](https://linkedin.com/in/yourprofile) | [GitHub](https://github.com/yourusername)

---

## 🚀 Deployment

Set `GROQ_API_KEY` and/or `GEMINI_API_KEY` as **secrets / environment variables** on the host
(never commit `.env`). The included `Dockerfile` works on Hugging Face Spaces (Docker SDK),
Render, Railway, Fly.io, or any VPS: `docker build -t eka . && docker run -p 7860:7860 -e GROQ_API_KEY=... eka`.
Note: `data/` (FAISS index, SQLite, uploads) is lost on free hosts when the container restarts —
attach a persistent volume mounted at `/app/data` if you need it to persist.
