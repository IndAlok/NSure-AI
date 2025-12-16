import warnings
import os
import asyncio
import time
import functools
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from typing import List

warnings.filterwarnings("ignore")

model_cache = {}

def timed_cache(maxsize=128, ttl=3600):
    def decorator(func):
        cache = {}
        cache_times = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(sorted(kwargs.items()))
            current_time = time.time()
            
            if key in cache and (current_time - cache_times[key]) < ttl:
                return cache[key]
            
            result = func(*args, **kwargs)
            
            if len(cache) >= maxsize:
                oldest_key = min(cache_times.keys(), key=cache_times.get)
                del cache[oldest_key]
                del cache_times[oldest_key]
            
            cache[key] = result
            cache_times[key] = current_time
            return result
        return wrapper
    return decorator

@asynccontextmanager
async def lifespan(app: FastAPI):
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from dotenv import load_dotenv
    from database import db_cache
    
    load_dotenv()
    
    try:
        database_url = os.getenv("DATABASE_URL")
        if database_url:
            await db_cache.init_pool(database_url)

        await db_cache.initialize_and_clear_cache()

        google_api_key = os.getenv("GOOGLE_API_KEY")
        
        model_cache["embedding_model"] = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=google_api_key
        )

        model_cache["llm"] = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro", 
            temperature=0.1,
            max_tokens=80,
            timeout=25,
            max_retries=2,
            google_api_key=google_api_key
        )
        
        asyncio.create_task(periodic_cleanup())
        
    except Exception as e:
        raise e

    yield
    
    model_cache.clear()

async def periodic_cleanup():
    while True:
        await asyncio.sleep(21600)
        try:
            from database import db_cache
            await db_cache.cleanup_old_cache(3)
        except Exception:
            pass

app = FastAPI(
    title="NSure-AI",
    description="Intelligent Insurance Document Analysis API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(GZipMiddleware, minimum_size=500)

from rag_core import OptimizedRAGCore

bearer_auth = HTTPBearer()

def check_auth(credentials: HTTPAuthorizationCredentials = Security(bearer_auth)):
    api_token = os.getenv("API_TOKEN")
    if not api_token:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API token not configured"
        )
    if credentials.scheme != "Bearer" or credentials.credentials != api_token:
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
    return {"status": "running", "service": "NSure-AI"}

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    from fastapi.responses import Response
    return Response(status_code=204)

@app.get("/favicon.png", include_in_schema=False)
def favicon_png():
    from fastapi.responses import Response
    return Response(status_code=204)

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/hackrx/run", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    token: str = Depends(check_auth)
):
    try:
        rag = OptimizedRAGCore(
            embedding_model=model_cache["embedding_model"],
            llm=model_cache["llm"]
        )
        
        answers = await rag.process_queries(request.documents, request.questions)
        return QueryResponse(answers=answers)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))