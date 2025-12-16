# NSure-AI

An intelligent document analysis API that extracts precise answers from insurance policy PDFs using retrieval-augmented generation (RAG). Built with FastAPI and Google Gemini, designed for production deployment.

## Key Features

- **Hybrid Retrieval System**: Combines BM25 keyword matching with FAISS vector similarity search for superior document retrieval accuracy
- **Intelligent Caching**: PostgreSQL-backed caching layer eliminates redundant processing, reducing response times for repeated queries by 95%
- **Production-Ready Architecture**: Async-first design with connection pooling, automatic retry logic, and graceful error handling
- **Accurate Answer Extraction**: Fine-tuned prompts with Gemini 1.5 Pro ensure concise, policy-specific responses

## Technical Architecture

```
Request → FastAPI → Hybrid Retriever → Gemini LLM → Response
                         │
              ┌──────────┴──────────┐
              │                     │
         BM25 (Keyword)      FAISS (Semantic)
              │                     │
              └──────────┬──────────┘
                         │
                  Document Chunks
```

### Core Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| API Layer | FastAPI | Async request handling with automatic OpenAPI docs |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Local semantic embeddings (no API costs) |
| Vector Store | FAISS | In-memory similarity search |
| LLM | Google Gemini 1.5 Pro | Answer generation |
| Cache | PostgreSQL (asyncpg) | Persistent document and query caching |
| PDF Parser | PyMuPDF | Fast, accurate text extraction |

## API Reference

### POST /hackrx/run

Analyze an insurance document and answer questions.

**Headers**
```
Authorization: Bearer <your_api_token>
Content-Type: application/json
```

**Request Body**
```json
{
  "documents": "https://example.com/policy.pdf",
  "questions": [
    "What is the coverage limit?",
    "Are pre-existing conditions covered?"
  ]
}
```

**Response**
```json
{
  "answers": [
    "The coverage limit is Rs. 5,00,000 per policy year.",
    "Pre-existing conditions are covered after a 36-month continuous waiting period."
  ]
}
```

### GET /health

Health check endpoint for monitoring.

## Deployment

### Environment Variables

| Variable | Description |
|----------|-------------|
| `GOOGLE_API_KEY` | Google AI API key with Gemini access |
| `DATABASE_URL` | PostgreSQL connection string |
| `API_TOKEN` | Bearer token for API authentication |
| `CLEAR_CACHE_ON_RESTART` | Clear cache on startup (true/false) |

### Vercel Deployment

1. Connect your GitHub repository to Vercel
2. Set environment variables in Vercel dashboard
3. Deploy

### Local Development

```bash
git clone https://github.com/IndAlok/NSure-AI.git
cd NSure-AI
pip install -r requirements.txt
# Create .env file with required variables
uvicorn main:app --reload
```

## Performance Optimizations

- **Pre-loaded Models**: Embedding model loads at startup, not per-request
- **Connection Pooling**: Database connections reused across requests
- **LRU Caching**: Document text and chunks cached in memory
- **GZip Compression**: Responses compressed for faster transfer
- **Async Processing**: Concurrent question answering when possible

## Project Structure

```
NSure-AI/
├── main.py           # FastAPI application and endpoints
├── rag_core.py       # RAG pipeline and hybrid retrieval
├── database.py       # PostgreSQL cache implementation  
├── utils.py          # PDF extraction utilities
├── requirements.txt  # Python dependencies
├── vercel.json       # Vercel deployment config
└── README.md
```

## License

MIT
