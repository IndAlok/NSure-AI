# main.py
"""
This script sets up the FastAPI application, defines the API endpoints,
handles authentication, and orchestrates the RAG pipeline.

It uses a lifespan event to pre-load heavy models on startup, preventing
request timeouts and solving the "cold start" problem.
"""
import warnings
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Dict

# Suppress the specific FutureWarning from torch
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.modules.module")

# --- Global State & Model Cache ---
# We will store our pre-loaded models here.
# Using a dictionary makes it easy to manage.
model_cache = {}
# Cache for initialized RAGCore instances for each document
pipeline_cache: Dict[str, "RAGCore"] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for the FastAPI application.
    This function runs on application startup and shutdown.
    """
    # --- Startup ---
    print("--- Application Startup: Loading models... ---")
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI
    from dotenv import load_dotenv

    load_dotenv() # Load OPENAI_API_KEY from .env

    # Pre-load the embedding model into the cache
    model_cache["embedding_model"] = HuggingFaceEmbeddings(
        model_name="./models/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'}
    )
    print("   -> Embedding model loaded.")

    # Pre-load the LLM into the cache
    model_cache["llm"] = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    print("   -> LLM loaded.")
    print("--- Model loading complete. Application is ready. ---")

    yield # The application runs here

    # --- Shutdown ---
    print("--- Application Shutdown ---")
    model_cache.clear()
    pipeline_cache.clear()


# --- Application Setup ---
app = FastAPI(
    title="NSure-AI: Intelligent Query-Retrieval System",
    lifespan=lifespan # Attach the lifespan event handler
)

# --- Dynamic Import for RAGCore ---
# We do this to avoid circular imports and ensure models are ready.
from rag_core import RAGCore

# --- Security & Authentication ---
REQUIRED_BEARER_TOKEN = "ee3aca9314e8c88b242c5f86bdb52d0bbb80293d95ced9beb6553a7fbb8cd1ce"
bearer_scheme = HTTPBearer()

def validate_token(credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)):
    if credentials.scheme != "Bearer" or credentials.credentials != REQUIRED_BEARER_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication token"
        )
    return credentials.credentials

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    documents: str = Field(..., description="A public URL to the PDF document.")
    questions: List[str] = Field(..., min_length=1, description="List of questions.")

class QueryResponse(BaseModel):
    answers: List[str]

# --- API Endpoints ---
@app.get("/", tags=["Status"])
def read_root():
    return {"status": "NSure-AI API is running."}

@app.post("/hackrx/run", response_model=QueryResponse, tags=["Core Functionality"])
async def run_submission(request: QueryRequest, token: str = Depends(validate_token)):
    doc_url = request.documents

    if doc_url in pipeline_cache:
        rag_pipeline = pipeline_cache[doc_url]
        print(f"Cache HIT: Using existing RAG pipeline for URL: {doc_url}")
    else:
        print(f"Cache MISS: Initializing new RAG pipeline for URL: {doc_url}")
        try:
            # Initialize RAGCore, passing in the pre-loaded models from our cache
            rag_pipeline = RAGCore(
                document_url=doc_url,
                embedding_model=model_cache["embedding_model"],
                llm=model_cache["llm"]
            )
            pipeline_cache[doc_url] = rag_pipeline
        except Exception as e:
            print(f"Error during RAGCore initialization: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

    answers = []
    for question in request.questions:
        answer = rag_pipeline.answer_question(question)
        answers.append(answer)

    return QueryResponse(answers=answers)
