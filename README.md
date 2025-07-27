# NSure-AI: Intelligent Query-Retrieval System

**A high-performance, LLM-powered API to process insurance documents and answer contextual questions. Built for the HackRx hackathon.**

---

## Table of Contents
- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Key Optimizations](#key-optimizations)
- [Tech Stack](#tech-stack)
- [API Documentation](#api-documentation)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)

## Project Overview

NSure-AI is a robust Retrieval-Augmented Generation (RAG) system designed to meet the hackathon's challenge. It takes a URL to a large document (PDF), processes it, and provides precise, context-aware answers to a list of user questions. The system is built to be fast, accurate, efficient, and highly explainable.

## System Architecture

The system follows a modular, cache-optimized architecture:

1.  **API Layer (FastAPI):** Exposes a secure endpoint (`/hackrx/run`) that handles incoming requests, authentication, and response formatting.
2.  **Caching Layer:** A smart in-memory cache ensures that a document is processed (downloaded, chunked, and embedded) only once. Subsequent queries against the same document are near-instantaneous.
3.  **RAG Core Engine (`rag_core.py`):**
    * **Document Loader (`utils.py`):** Fetches the PDF from the URL and extracts clean text using the efficient `PyMuPDF` library.
    * **Semantic Chunker:** Splits the text into meaningful, overlapping chunks using `RecursiveCharacterTextSplitter`.
    * **Local Embedder:** Converts text chunks into vector embeddings using the fast and effective `BAAI/bge-small-en-v1.5` model, which runs locally on the CPU.
    * **Vector Store (FAISS):** Indexes all document chunks in an in-memory `FAISS` database for ultra-fast similarity searches.
    * **Retriever:** For a given question, it retrieves the top 5 most relevant document chunks from the FAISS index.
    * **Generator (LLM):** An optimized prompt combines the user's question with the retrieved context and feeds it to `gpt-4o-mini` to generate a precise, factual answer.

## Key Optimizations

This solution was engineered to excel in all evaluation categories:

* **Latency:**
    * **In-Memory FAISS:** Blazing-fast retrieval without network overhead.
    * **Local Embeddings:** Eliminates API calls and network latency for the embedding step.
    * **Instant Startup (Pre-Bundled Model):** The embedding model is "baked in" to the application, eliminating any download delay on server startup. The first API call is just as fast as any other.
    * **Singleton Caching:** The document processing pipeline is cached in memory after the first request for a specific URL, making subsequent queries on the same document instantaneous.
* **Token Efficiency & Cost:**
    * **Local Embeddings:** Zero cost for embedding.
    * **Optimized Prompt:** The prompt is engineered to be concise, and the `stuff` chain method is highly efficient for the context sizes we are handling.
* **Accuracy & Explainability:**
    * **Semantic Chunking:** Provides better, more coherent context to the LLM.
    * **Strict Prompting:** The LLM is explicitly instructed to answer *only* from the provided context, preventing hallucinations and making every answer directly traceable to a source chunk.
* **Reusability & Modularity:**
    * The code is logically separated into modules (`main.py`, `rag_core.py`, `utils.py`), making it easy to understand, maintain, and extend.

## Tech Stack

| Component         | Technology                  | Reason                                                          |
| ----------------- | --------------------------- | --------------------------------------------------------------- |
| **Web Framework** | `FastAPI`                   | High performance, automatic docs, Pydantic support.             |
| **LLM** | `OpenAI GPT-4o mini`        | Superior speed and cost-effectiveness with top-tier intelligence. |
| **Vector DB** | `FAISS (in-memory)`         | Extreme speed for similarity search, no setup needed.           |
| **Embeddings** | `BAAI/bge-small-en-v1.5`    | Top-tier performance for speed and retrieval accuracy, runs locally. |
| **PDF Parsing** | `PyMuPDF`                   | Superior speed and accuracy over alternatives.                  |
| **Orchestration** | `LangChain`                 | Simplifies the RAG pipeline construction.                       |

## API Documentation

The API is self-documenting. Once the server is running, visit `http://localhost:8000/docs` for a full, interactive Swagger UI.

### Endpoint: `POST /hackrx/run`

* **Description:** Processes a document and answers questions.
* **Authentication:** `Authorization: Bearer <your_token>`
* **Request Body:**
    ```json
    {
    "documents": "https://path/to/your/policy.pdf",
    "questions": [
        "What is the grace period for premium payment?",
        "Does this policy cover maternity expenses?"
    ]
    }
    ```
* **Success Response (200 OK):**
    ```json
    {
      "answers": [
        "A grace period of thirty days is provided for premium payment...",
        "Information not found in the provided policy document."
      ]
    }
    ```

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/NSure-AI.git
    cd NSure-AI
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    pip install huggingface-hub # Needed only for the one-time model download
    ```

4.  **Download the Embedding Model:**
    Run the helper script to download the model files into your project. This ensures instant startup times.
    ```bash
    python download_model.py
    ```

5.  **Create a `.env` file** in the root directory and add your OpenAI API key:
    ```
    OPENAI_API_KEY="sk-..."
    ```

## How to Run

1.  Ensure your virtual environment is activated.
2.  Start the FastAPI server using Uvicorn:
    ```bash
    uvicorn main:app --reload
    ```
3.  The API will be live at `http://localhost:8000`.
