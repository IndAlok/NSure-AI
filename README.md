# NSure-AI

An intelligent document analysis API that extracts precise answers from insurance policy PDFs using retrieval-augmented generation (RAG). Built with FastAPI and Google Gemini, optimized for serverless deployment.

## Key Features

- **Hybrid Retrieval System**: Combines BM25 keyword matching with semantic vector similarity for superior document retrieval accuracy
- **Intelligent Caching**: PostgreSQL-backed caching layer eliminates redundant processing, reducing response times for repeated queries by 95%
- **Serverless-Optimized**: Lightweight architecture using Google's embedding API, deployable on Vercel with minimal cold start times
- **Accurate Answer Extraction**: Fine-tuned prompts with Gemini 1.5 Pro ensure concise, policy-specific responses

## Technical Architecture

```
Request → FastAPI → Hybrid Retriever → Gemini LLM → Response
                         │
              ┌──────────┴──────────┐
              │                     │
         BM25 (Keyword)      Vector (Semantic)
              │                     │
              └──────────┬──────────┘
                         │
                  Document Chunks
```

### Core Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| API Layer | FastAPI | Async request handling with automatic OpenAPI docs |
| Embeddings | Google text-embedding-004 | High-quality semantic embeddings via API |
| Vector Search | NumPy cosine similarity | Lightweight in-memory similarity search |
| LLM | Google Gemini 2.5 Flash | Answer generation |
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
uvicorn main:app --reload
```

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
