import warnings
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Dict

warnings.filterwarnings("ignore", category=FutureWarning)

os.environ["HF_HOME"] = "/tmp/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface_cache/transformers"
os.environ["HF_HUB_CACHE"] = "/tmp/huggingface_cache/hub"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/tmp/huggingface_cache/sentence_transformers"

model_cache = {}
pipeline_cache = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- Loading AI models... ---")
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_google_genai import ChatGoogleGenerativeAI
    from dotenv import load_dotenv
    import tempfile

    load_dotenv()

    temp_cache_dir = tempfile.mkdtemp(prefix="hf_cache_")
    print(f"Using cache: {temp_cache_dir}")

    try:
        model_cache["embedding_model"] = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            cache_folder=temp_cache_dir
        )
        print("✅ Embeddings loaded")

        model_cache["llm"] = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro", 
            temperature=0.1,  # Low temperature for accuracy
            convert_system_message_to_human=True
        )
        print("✅ LLM loaded (Gemini 1.5 Flash)")
        print("🚀 Ready to serve requests!")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        try:
            model_cache["embedding_model"] = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            print("✅ Embeddings loaded (fallback)")
            
            # Fallback LLM with basic settings
            model_cache["llm"] = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.1
            )
            print("✅ LLM loaded (fallback - Gemini 1.5 Flash)")
        except Exception as e2:
            print(f"❌ Complete failure: {e2}")
            raise e2

    yield

    print("🔄 Shutting down...")
    model_cache.clear()
    pipeline_cache.clear()

app = FastAPI(
    title="NSure-AI: Smart Insurance Assistant",
    description="Upload insurance PDFs and get instant answers to your questions",
    version="1.0.0",
    lifespan=lifespan
)

from rag_core import RAGCore

API_TOKEN = "ee3aca9314e8c88b242c5f86bdb52d0bbb80293d95ced9beb6553a7fbb8cd1ce"
bearer_auth = HTTPBearer()

def check_auth(credentials: HTTPAuthorizationCredentials = Security(bearer_auth)):
    if credentials.scheme != "Bearer" or credentials.credentials != API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    return credentials.credentials

class QueryRequest(BaseModel):
    documents: str = Field(..., description="URL to PDF document")
    questions: List[str] = Field(..., min_length=1, description="Questions to answer")

class QueryResponse(BaseModel):
    answers: List[str]

@app.get("/", include_in_schema=False)
def home():
    return {"message": "NSure-AI is running! Check /docs for API info."}

@app.post("/hackrx/run", response_model=QueryResponse, tags=["Main"])
async def process_document(request: QueryRequest, token: str = Depends(check_auth)):
    doc_url = request.documents

    if doc_url in pipeline_cache:
        rag = pipeline_cache[doc_url]
        print(f"📋 Using cached pipeline for: {doc_url}")
    else:
        print(f"🔄 Creating new pipeline for: {doc_url}")
        try:
            rag = RAGCore(
                document_url=doc_url,
                embedding_model=model_cache["embedding_model"],
                llm=model_cache["llm"]
            )
            pipeline_cache[doc_url] = rag
        except Exception as e:
            print(f"❌ Pipeline creation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

    # Always use batch processing for better performance
    try:
        answers = rag.batch_answer_questions(request.questions)
    except Exception as e:
        print(f"❌ Batch processing failed, falling back to sequential: {e}")
        # Fallback to sequential processing if batch fails
        answers = []
        for q in request.questions:
            try:
                answer = rag.answer_question(q)
                answers.append(answer)
            except Exception as single_e:
                print(f"❌ Single question failed: {single_e}")
                answers.append(f"Error processing question: {str(single_e)}")

    return QueryResponse(answers=answers)