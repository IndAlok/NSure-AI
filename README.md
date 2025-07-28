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

The system uses a production-grade architecture optimized for performance and stability on cloud platforms.

1.  **FastAPI with Lifespan Events:** The API server pre-loads all heavy AI models (`LLM` and `Embedder`) into memory on startup. This "pre-warming" process eliminates cold-start timeouts and ensures the API is instantly ready to handle requests.
2.  **RAG Core Engine (`rag_core.py`):**
    * **Document Loader (`utils.py`):** Fetches and parses PDF text efficiently using `PyMuPDF`.
    * **Local Embedder:** Uses the lightweight and highly-efficient **`sentence-transformers/all-MiniLM-L6-v2`** model, which is bundled with the application to ensure zero runtime downloads and low memory usage.
    * **Vector Store (FAISS):** Indexes document chunks in an in-memory `FAISS` database for microsecond-level similarity searches.
    * **Generator (LLM):** Feeds the retrieved context to **`gpt-4o-mini`** to generate precise, factual answers.
3.  **Caching Layer:** An in-memory cache stores the processed RAG pipeline for each unique document URL, making subsequent queries on the same document instantaneous.

## Key Optimizations

This solution was engineered to excel in all evaluation categories:

* **Latency & Stability:**
    * **Lifespan Model Pre-loading:** All heavy models are loaded on server startup, not during a request. This completely solves the "cold start" 502 timeout errors common on cloud platforms.
    * **Hyper-Efficient Local Embeddings:** Switched to the `all-MiniLM-L6-v2` model for its excellent balance of speed, accuracy, and critically, low memory footprint, ensuring stability on free-tier hosting.
    * **In-Memory FAISS:** Blazing-fast retrieval without network overhead.
* **Token Efficiency & Cost:**
    * **Local Embeddings:** Zero API cost for creating text embeddings.
    * **`gpt-4o-mini`:** Uses one of the most cost-effective and fastest models from OpenAI for final answer generation.
* **Reusability & Modularity:**
    * **Production Dockerfile:** A multi-stage `Dockerfile` creates a small, secure, and efficient production image.
    * **Clean Code Separation:** Logic is cleanly separated into modules (`main.py`, `rag_core.py`, `utils.py`).

## Tech Stack

| Component         | Technology                           | Reason                                                              |
| ----------------- | ------------------------------------ | ------------------------------------------------------------------- |
| **Web Framework** | `FastAPI`                            | High performance, modern async features, and lifespan events.       |
| **LLM** | `OpenAI GPT-4o mini`                 | Superior speed and cost-effectiveness with top-tier intelligence.   |
| **Vector DB** | `FAISS (in-memory)`                  | Extreme speed for similarity search, no setup needed.               |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Excellent performance with a very low memory footprint for stability. |
| **PDF Parsing** | `PyMuPDF`                            | Superior speed and accuracy over alternatives.                      |
| **Deployment** | `Docker` / `Render`                  | Containerization for consistency and reliable cloud deployment.     |

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
