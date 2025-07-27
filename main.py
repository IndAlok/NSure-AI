# main.py
"""
This script sets up the FastAPI application, defines the API endpoints,
handles authentication, and orchestrates the RAG pipeline.
"""
from fastapi import FastAPI, Depends, HTTPException, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Dict

from rag_core import RAGCore

# --- Application Metadata ---
app_metadata = {
    "title": "NSure-AI: Intelligent Query-Retrieval System",
    "description": "A high-performance, LLM-powered API to process insurance documents and answer contextual questions. Built for the HackRx hackathon.",
    "version": "1.0.0",
}

app = FastAPI(**app_metadata)

# --- Configuration & Global State ---

# The required bearer token for API authentication, as specified in the problem statement.
# In a real-world app, this would come from a secure vault.
REQUIRED_BEARER_TOKEN = "ee3aca9314e8c88b242c5f86bdb52d0bbb80293d95ced9beb6553a7fbb8cd1ce"

# A simple in-memory cache to store initialized RAGCore instances.
# The key is the document URL, and the value is the RAGCore object.
# This is a critical optimization to avoid reprocessing the same document.
pipeline_cache: Dict[str, RAGCore] = {}

# --- Security & Authentication ---

# Define the security scheme for Bearer Token authentication
bearer_scheme = HTTPBearer(
    description="Your unique API key for authentication."
)

def validate_token(credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)):
    """
    A dependency that validates the provided bearer token against the required token.
    This is injected into the endpoint to protect it.
    """
    if credentials.scheme != "Bearer" or credentials.credentials != REQUIRED_BEARER_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# --- Pydantic Models for API Data Structure ---

class QueryRequest(BaseModel):
    documents: str = Field(..., description="A public URL to the PDF document to be processed.")
    questions: List[str] = Field(..., min_length=1, description="A list of questions to ask about the document.")

class QueryResponse(BaseModel):
    answers: List[str] = Field(..., description="A list of answers corresponding to the input questions.")

# --- API Endpoints ---

@app.get("/", tags=["Status"])
def read_root():
    """A simple health check endpoint to confirm the API is running."""
    return {"status": "NSure-AI API is running. Head to /docs for API documentation."}


@app.post("/hackrx/run", response_model=QueryResponse, tags=["Core Functionality"])
async def run_submission(
    request: QueryRequest, 
    token: str = Depends(validate_token)
):
    """
    This endpoint processes a document from a URL and answers a list of questions.

    It performs the following steps:
    1.  Validates the Bearer token.
    2.  Checks if a RAG pipeline for the document URL is already cached.
    3.  If not cached, it initializes a new `RAGCore` instance (which downloads, chunks, and embeds the doc) and caches it.
    4.  Iterates through the list of questions and generates an answer for each using the RAG pipeline.
    5.  Returns a structured JSON response with the list of answers.
    """
    doc_url = request.documents
    
    # Check cache first to see if this document has already been processed.
    if doc_url in pipeline_cache:
        rag_pipeline = pipeline_cache[doc_url]
        print(f"Cache HIT: Using existing RAG pipeline for URL: {doc_url}")
    else:
        # If not in cache, this is the first time we're seeing this URL.
        # Initialize the full RAG pipeline and store it in the cache.
        print(f"Cache MISS: Initializing new RAG pipeline for URL: {doc_url}")
        try:
            rag_pipeline = RAGCore(document_url=doc_url)
            pipeline_cache[doc_url] = rag_pipeline
        except Exception as e:
            # If initialization fails (e.g., bad URL, parsing error), return a server error.
            raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

    answers = []
    # Process each question sequentially using the initialized pipeline.
    for i, question in enumerate(request.questions):
        print(f"--- Answering question {i+1}/{len(request.questions)} ---")
        try:
            answer = rag_pipeline.answer_question(question)
            answers.append(answer)
        except Exception as e:
            # If a single question fails, append an error message and continue.
            print(f"Error answering question '{question}': {e}")
            answers.append("An error occurred while processing this question.")
            
    return QueryResponse(answers=answers)
