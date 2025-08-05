---
title: NSure-AI
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: "4.36.2"
app_file: app.py
pinned: true
---

# NSure-AI 🛡️
*Smart Insurance Document Assistant*

A lightning-fast API that reads insurance PDFs and answers questions about them.

## What it does
- Takes any insurance PDF URL
- Answers specific questions about the policy
- Remembers documents to avoid reprocessing
- Works blazingly fast with smart caching

## Live Demo
🚀 **Try it now**: [https://indalok-nsure-ai.hf.space](https://indalok-nsure-ai.hf.space)

## Quick Start

### Test the API
```bash
curl -X POST "https://indalok-nsure-ai.hf.space/hackrx/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ee3aca9314e8c88b242c5f86bdb52d0bbb80293d95ced9beb6553a7fbb8cd1ce" \
  -d '{
    "documents": "https://your-pdf-url.com/policy.pdf",
    "questions": [
      "What is the coverage limit?",
      "Are pre-existing conditions covered?"
    ]
  }'
```

### Run Locally
```bash
git clone https://github.com/IndAlok/NSure-AI.git
cd NSure-AI
pip install -r requirements.txt
uvicorn main:app --reload
```

## How it works

1. **PDF Processing**: Downloads and extracts text from insurance documents
2. **Smart Chunking**: Breaks documents into meaningful sections
3. **Vector Search**: Finds relevant parts using AI embeddings
4. **Answer Generation**: Uses Google Gemini 2.5 Flash to generate precise answers
5. **Caching**: Remembers processed documents for instant responses

## Tech Stack

- **Backend**: FastAPI + Python
- **AI Models**: Google Gemini 1.5 Pro + Sentence Transformers
- **Vector DB**: FAISS (in-memory)
- **PDF Parser**: PyMuPDF
- **Deployment**: Docker + HuggingFace Spaces

## Project Structure
```
NSure-AI/
├── main.py           # FastAPI server & API endpoints
├── rag_core.py       # Core RAG logic & document processing
├── utils.py          # PDF parsing utilities
├── requirements.txt  # Python dependencies
├── Dockerfile       # Container setup
└── README.md        # This file
```

## API Reference

### POST /hackrx/run
Process a document and get answers to questions.

**Headers:**
- `Authorization: Bearer ee3aca9314e8c88b242c5f86bdb52d0bbb80293d95ced9beb6553a7fbb8cd1ce`
- `Content-Type: application/json`

**Body:**
```json
{
  "documents": "https://example.com/policy.pdf",
  "questions": ["Your question here"]
}
```

**Response:**
```json
{
  "answers": ["Detailed answer based on the document"]
}
```

## Development

### Environment Setup
```bash
# Create virtual environment
python -m venv env
source env/bin/activate  # Linux/Mac
# or
env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Add your Google API key
echo "GOOGLE_API_KEY=your-key-here" > .env
```

### Docker Build
```bash
docker build -t nsure-ai .
docker run -p 7860:7860 nsure-ai
```

## Why These Choices?

- **Gemini 1.5 Pro**: Google's fastest and most capable model with excellent accuracy
- **Local Embeddings**: No API costs for document processing
- **FAISS**: Fastest vector search without external dependencies
- **FastAPI**: Modern async framework with auto-documentation
- **Docker**: Consistent deployment across platforms

## Performance Features

- ⚡ **Pre-loaded Models**: All AI models load on startup, not per request
- 🧠 **Smart Caching**: Documents processed once, answers served instantly
- 💰 **Cost Efficient**: Local embeddings + efficient Gemini model
- 🔄 **Auto-retry**: Handles temporary failures gracefully
- 📊 **Memory Optimized**: Uses lightweight models for stable deployment

## Troubleshooting

**Common Issues:**
- *401 Unauthorized*: Check your Bearer token
- *500 Server Error*: Invalid PDF URL or Google API key issue
- *Timeout*: Large documents may take 30-60 seconds on first request

**Need Help?**
- Check the interactive docs: `/docs` endpoint
- Verify your PDF is publicly accessible
- Ensure Google API key has credits and Gemini API access enabled

---

*Built for HackRx 2025 | Made with ☕ and determination*
