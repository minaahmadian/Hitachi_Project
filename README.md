# Hitachi AI Backend

AI-powered document processing and decision support system with vector search capabilities.

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Docker & Docker Compose (for vector database)
- Groq API Key (for LLM and embeddings)

### 2. Setup Environment

```bash
# Clone and setup
git clone https://github.com/ManInBlackout/hitachi-ai-backend.git
cd hitachi-ai-backend
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure

```bash
# Copy environment template and edit
cp .env.example .env
```

Add your API key to `.env`:
```env
GROQ_API_KEY=your_groq_key_here
```

## Running the Application

This project has two main components:

### A. LangGraph Agents (Decision Support)

Run the AI agents for safety-critical decision support:

```bash
python main.py
```

**What it does:**
- Loads test data from `data/`
- Runs Formal Auditor → Context Detective → Lead Safety Assessor
- Outputs GO/NO-GO decision and prints results to console

**Requires:** `GROQ_API_KEY` in `.env`

### B. Vector Store (Document RAG)

For document storage and retrieval:

1. Start Qdrant:
```bash
docker compose up -d
# or if using legacy docker-compose: docker-compose up -d
```

2. Create a test file and use the Vector Store:

```bash
# Create a test document
echo "This is a test document about project updates." > document.txt
```

```python
from dotenv import load_dotenv
load_dotenv(".env")  # Load env vars first

from backend import VectorStore

# Initialize
store = VectorStore()
store.connect()
store.setup_collection(dimension=768)

# Add documents
store.add_file("document.txt")
store.add_email(
    body="Meeting notes...",
    subject="Project Update",
    from_addr="team@example.com"
)

# Query
result = store.query("What are the project updates?", top_k=5)
print(result.context)
```

**Requires:** `GROQ_API_KEY` in `.env` + Docker running

## Architecture

### Core Components

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  Documents  │───▶│  Processing  │───▶│  VectorDB   │
│ (Files/API) │    │ Chunk+Embed  │    │  (Qdrant)   │
└─────────────┘    └──────────────┘    └─────────────┘
                                             │
┌─────────────┐    ┌──────────────┐          │
│   Response  │◀───│     RAG      │◀─────────┘
│    (LLM)    │    │ Query+Context│
└─────────────┘    └──────────────┘
```

### Modules

- **`agents/`** - LangGraph agents for decision support (Formal Auditor, Context Detective, Lead Assessor)
- **`vectordb/`** - Generic vector database interface with Qdrant provider
- **`documents/`** - Unified document model (files, emails, API content)
- **`processing/`** - Chunking and embedding pipeline
- **`rag/`** - Retrieval-augmented generation system
- **`backend/`** - High-level integration layer

## Environment Variables

| Variable            | Description                       | Default              |
| ------------------- | --------------------------------- | -------------------- |
| `GROQ_API_KEY`      | Groq API key for LLM & embeddings | required             |
| `QDRANT_HOST`       | Qdrant server host                | localhost            |
| `QDRANT_PORT`       | Qdrant server port                | 6333                 |
| `QDRANT_COLLECTION` | Default collection name           | documents            |
| `EMBEDDING_MODEL`   | Groq embedding model              | nomic-embed-text     |
| `CHUNK_SIZE`        | Text chunk size (characters)      | 500                  |
| `CHUNK_OVERLAP`     | Chunk overlap (characters)        | 50                   |

## Docker Commands

```bash
# Start Qdrant (modern Docker with compose plugin)
docker compose up -d
# Legacy docker-compose alternative: docker-compose up -d
```

## Supported Document Types

- **Text files**: `.txt`, `.md`
- **PDFs**: `.pdf` (requires `pip install PyPDF2`)
- **Emails**: `.eml` files or raw email strings
- **Direct input**: Any text content

## Contributing

1. Create a feature branch: `git checkout -b feature/name`
2. Make changes and commit: `git commit -m "description"`
3. Push and open a Pull Request
4. Wait for review before merging

Main branch is protected - no direct pushes.