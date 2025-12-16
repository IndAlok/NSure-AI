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
from fastapi.responses import HTMLResponse

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

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NSure-AI - Insurance Document Analysis API</title>
    <link rel="icon" type="image/svg+xml" href="/favicon.svg">
    <link rel="shortcut icon" href="/favicon.ico">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
        }
        .container {
            text-align: center;
            max-width: 600px;
            padding: 2rem;
        }
        h1 {
            font-size: 3rem;
            margin-bottom: 1rem;
            font-weight: 700;
        }
        p {
            font-size: 1.25rem;
            opacity: 0.9;
            margin-bottom: 2rem;
        }
        .status {
            display: inline-block;
            background: rgba(255, 255, 255, 0.2);
            padding: 0.5rem 1.5rem;
            border-radius: 50px;
            font-weight: 600;
            margin-bottom: 2rem;
        }
        .links {
            display: flex;
            gap: 1rem;
            justify-content: center;
            flex-wrap: wrap;
        }
        a {
            background: white;
            color: #1e3a8a;
            padding: 0.75rem 2rem;
            border-radius: 8px;
            text-decoration: none;
            font-weight: 600;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        a:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.2);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="status">🟢 API Running</div>
        <h1>NSure-AI</h1>
        <p>Intelligent Insurance Document Analysis API</p>
        <div class="links">
            <a href="/docs" target="_blank">API Documentation</a>
            <a href="/health" target="_blank">Health Check</a>
        </div>
    </div>
</body>
</html>"""

@app.get("/favicon.svg", include_in_schema=False)
def favicon_svg():
    from fastapi.responses import Response
    svg_content = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
<defs>
<linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="100%">
<stop offset="0%" style="stop-color:#1e3a8a"/>
<stop offset="100%" style="stop-color:#3b82f6"/>
</linearGradient>
</defs>
<path d="M50 10 L80 25 L80 60 Q80 80 50 90 Q20 80 20 60 L20 25 Z" fill="url(#g)" stroke="white" stroke-width="2"/>
<path d="M50 30 L65 37 L65 60 Q65 70 50 75 Q35 70 35 60 L35 37 Z" fill="white" opacity="0.3"/>
</svg>"""
    return Response(content=svg_content, media_type="image/svg+xml")

@app.get("/favicon.ico", include_in_schema=False)
def favicon_ico():
    from fastapi.responses import Response
    svg_content = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
<defs>
<linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="100%">
<stop offset="0%" style="stop-color:#1e3a8a"/>
<stop offset="100%" style="stop-color:#3b82f6"/>
</linearGradient>
</defs>
<path d="M50 10 L80 25 L80 60 Q80 80 50 90 Q20 80 20 60 L20 25 Z" fill="url(#g)" stroke="white" stroke-width="2"/>
<path d="M50 30 L65 37 L65 60 Q65 70 50 75 Q35 70 35 60 L35 37 Z" fill="white" opacity="0.3"/>
</svg>"""
    return Response(
        content=svg_content, 
        media_type="image/svg+xml",
        headers={
            "Cache-Control": "public, max-age=86400",
            "X-Content-Type-Options": "nosniff"
        }
    )

@app.get("/favicon.png", include_in_schema=False)
def favicon_png():
    from fastapi.responses import Response
    svg_content = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
<defs>
<linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="100%">
<stop offset="0%" style="stop-color:#1e3a8a"/>
<stop offset="100%" style="stop-color:#3b82f6"/>
</linearGradient>
</defs>
<path d="M50 10 L80 25 L80 60 Q80 80 50 90 Q20 80 20 60 L20 25 Z" fill="url(#g)" stroke="white" stroke-width="2"/>
<path d="M50 30 L65 37 L65 60 Q65 70 50 75 Q35 70 35 60 L35 37 Z" fill="white" opacity="0.3"/>
</svg>"""
    return Response(
        content=svg_content, 
        media_type="image/svg+xml",
        headers={
            "Cache-Control": "public, max-age=86400",
            "X-Content-Type-Options": "nosniff"
        }
    )

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