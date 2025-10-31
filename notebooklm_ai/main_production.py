from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import logging
import sys
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
import uvicorn
from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge
import structlog
from structlog import get_logger
import redis
import psutil
import os
from datetime import datetime, timedelta
import json
import hashlib
import jwt
from functools import wraps
    from core.entities import (
    from infrastructure.ai_engines import (
    from shared.config import NotebookLMConfig
    from presentation.api import NotebookLMRouter
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
NotebookLM AI - Production Main Application
Advanced document intelligence system with production-ready features.
"""


# Import our NotebookLM components
try:
        Document, Notebook, Source, Citation, Query, Response as AIResponse, 
        Conversation, User, DocumentType, SourceType, QueryType
    )
        AdvancedLLMEngine, DocumentProcessor, CitationGenerator, 
        ResponseOptimizer, MultiModalProcessor, AIEngineConfig
    )
except ImportError as e:
    print(f"Warning: Some components not available: {e}")
    # Create mock classes for production
    class MockEngine:
        def __init__(self, *args, **kwargs) -> Any:
            pass
        async def generate_response(self, *args, **kwargs) -> Any:
            return "Production mock response"
    
    AdvancedLLMEngine = MockEngine
    DocumentProcessor = MockEngine
    CitationGenerator = MockEngine
    ResponseOptimizer = MockEngine
    MultiModalProcessor = MockEngine

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter('notebooklm_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('notebooklm_request_duration_seconds', 'Request latency')
ACTIVE_CONNECTIONS = Gauge('notebooklm_active_connections', 'Active connections')
MEMORY_USAGE = Gauge('notebooklm_memory_bytes', 'Memory usage in bytes')
CPU_USAGE = Gauge('notebooklm_cpu_percent', 'CPU usage percentage')

# Security
security = HTTPBearer()
SECRET_KEY = os.getenv("NOTEBOOKLM_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("NOTEBOOKLM_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Redis for caching and rate limiting
redis_client = None
try:
    redis_url = os.getenv("NOTEBOOKLM_REDIS_URL", "redis://localhost:6379")
    redis_client = redis.from_url(redis_url, decode_responses=True)
    logger.info("Redis connected successfully")
except Exception as e:
    logger.warning(f"Redis connection failed: {e}")

# Rate limiting
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX_REQUESTS = 100  # requests per window

class RateLimiter:
    """Rate limiter using Redis."""
    
    def __init__(self, redis_client: redis.Redis):
        
    """__init__ function."""
self.redis = redis_client
        self.window = RATE_LIMIT_WINDOW
        self.max_requests = RATE_LIMIT_MAX_REQUESTS
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed."""
        if not self.redis:
            return True
        
        key = f"rate_limit:{client_id}"
        current = self.redis.get(key)
        
        if current is None:
            self.redis.setex(key, self.window, 1)
            return True
        
        current_count = int(current)
        if current_count >= self.max_requests:
            return False
        
        self.redis.incr(key)
        return True

rate_limiter = RateLimiter(redis_client)

class CacheManager:
    """Multi-level cache manager."""
    
    def __init__(self, redis_client: redis.Redis):
        
    """__init__ function."""
self.redis = redis_client
        self.memory_cache = {}
        self.cache_ttl = 3600  # 1 hour
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        # Try memory cache first
        if key in self.memory_cache:
            value, timestamp = self.memory_cache[key]
            if time.time() - timestamp < 300:  # 5 minutes TTL for memory
                return value
        
        # Try Redis cache
        if self.redis:
            try:
                value = self.redis.get(key)
                if value:
                    parsed_value = json.loads(value)
                    # Update memory cache
                    self.memory_cache[key] = (parsed_value, time.time())
                    return parsed_value
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
        
        return None
    
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache."""
        ttl = ttl or self.cache_ttl
        
        # Update memory cache
        self.memory_cache[key] = (value, time.time())
        
        # Update Redis cache
        if self.redis:
            try:
                self.redis.setex(key, ttl, json.dumps(value))
                return True
            except Exception as e:
                logger.warning(f"Redis cache set error: {e}")
        
        return False
    
    async def invalidate(self, pattern: str) -> bool:
        """Invalidate cache entries matching pattern."""
        if self.redis:
            try:
                keys = self.redis.keys(pattern)
                if keys:
                    self.redis.delete(*keys)
                return True
            except Exception as e:
                logger.warning(f"Cache invalidation error: {e}")
        
        return False

cache_manager = CacheManager(redis_client)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Verify JWT token."""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware."""
    client_id = request.client.host
    
    if not rate_limiter.is_allowed(client_id):
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded", "retry_after": RATE_LIMIT_WINDOW}
        )
    
    return call_next(request)

def metrics_middleware(request: Request, call_next):
    """Metrics collection middleware."""
    start_time = time.time()
    
    # Update active connections
    ACTIVE_CONNECTIONS.inc()
    
    # Update system metrics
    MEMORY_USAGE.set(psutil.virtual_memory().used)
    CPU_USAGE.set(psutil.cpu_percent())
    
    response = call_next(request)
    
    # Record metrics
    duration = time.time() - start_time
    REQUEST_LATENCY.observe(duration)
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    # Decrease active connections
    ACTIVE_CONNECTIONS.dec()
    
    return response

def security_middleware(request: Request, call_next):
    """Security middleware."""
    # Add security headers
    response = call_next(request)
    
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    
    return response

def logging_middleware(request: Request, call_next):
    """Request logging middleware."""
    start_time = time.time()
    
    # Log request
    logger.info(
        "Request started",
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host,
        user_agent=request.headers.get("user-agent")
    )
    
    response = call_next(request)
    
    # Log response
    duration = time.time() - start_time
    logger.info(
        "Request completed",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        duration=duration
    )
    
    return response

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting NotebookLM AI Production Server...")
    
    # Initialize AI engines
    try:
        config = AIEngineConfig(
            model_name=os.getenv("NOTEBOOKLM_MODEL_NAME", "microsoft/DialoGPT-medium"),
            max_length=int(os.getenv("NOTEBOOKLM_MAX_LENGTH", "2048")),
            temperature=float(os.getenv("NOTEBOOKLM_TEMPERATURE", "0.7")),
            use_quantization=os.getenv("NOTEBOOKLM_USE_QUANTIZATION", "true").lower() == "true",
            device=os.getenv("NOTEBOOKLM_DEVICE", "auto")
        )
        
        app.state.llm_engine = AdvancedLLMEngine(config)
        app.state.document_processor = DocumentProcessor()
        app.state.citation_generator = CitationGenerator()
        app.state.response_optimizer = ResponseOptimizer()
        app.state.multimodal_processor = MultiModalProcessor()
        
        logger.info("AI engines initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize AI engines: {e}")
    
    # Health check
    app.state.startup_time = datetime.utcnow()
    app.state.health_status = "healthy"
    
    yield
    
    # Shutdown
    logger.info("Shutting down NotebookLM AI Production Server...")
    
    # Cleanup
    if hasattr(app.state, 'llm_engine'):
        del app.state.llm_engine
    if hasattr(app.state, 'document_processor'):
        del app.state.document_processor

# Create FastAPI app
app = FastAPI(
    title="NotebookLM AI Production API",
    description="Advanced document intelligence system with production-ready features",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.middleware("http")(rate_limit_middleware)
app.middleware("http")(metrics_middleware)
app.middleware("http")(security_middleware)
app.middleware("http")(logging_middleware)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check system resources
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent()
        disk = psutil.disk_usage('/')
        
        # Check Redis connection
        redis_status = "healthy" if redis_client and redis_client.ping() else "unhealthy"
        
        # Check AI engines
        ai_engines_status = "healthy" if hasattr(app.state, 'llm_engine') else "unhealthy"
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "uptime": str(datetime.utcnow() - app.state.startup_time),
            "system": {
                "memory_usage_percent": memory.percent,
                "cpu_usage_percent": cpu,
                "disk_usage_percent": (disk.used / disk.total) * 100
            },
            "services": {
                "redis": redis_status,
                "ai_engines": ai_engines_status
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )

# Metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        prometheus_client.generate_latest(),
        media_type="text/plain"
    )

# Main API endpoints
@app.post("/api/v1/notebooks")
async def create_notebook(
    title: str,
    description: str = "",
    user_token: Dict[str, Any] = Depends(verify_token)
):
    """Create a new notebook."""
    try:
        # Generate notebook ID
        notebook_id = f"notebook_{int(time.time())}"
        
        # Create notebook
        notebook = {
            "id": notebook_id,
            "title": title,
            "description": description,
            "user_id": user_token.get("user_id"),
            "created_at": datetime.utcnow().isoformat(),
            "documents": [],
            "sources": [],
            "conversations": []
        }
        
        # Cache notebook
        cache_key = f"notebook:{notebook_id}"
        await cache_manager.set(cache_key, notebook)
        
        logger.info(f"Notebook created: {notebook_id}")
        return {"notebook": notebook}
    
    except Exception as e:
        logger.error(f"Failed to create notebook: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/notebooks/{notebook_id}/documents")
async def add_document(
    notebook_id: str,
    title: str,
    content: str,
    document_type: str = "txt",
    user_token: Dict[str, Any] = Depends(verify_token)
):
    """Add document to notebook."""
    try:
        # Process document
        if hasattr(app.state, 'document_processor'):
            analysis = app.state.document_processor.process_document(content, title)
        else:
            analysis = {
                "word_count": len(content.split()),
                "summary": content[:200] + "..." if len(content) > 200 else content,
                "key_points": [content[:100]],
                "sentiment": {"compound": 0.0}
            }
        
        # Create document
        document = {
            "id": f"doc_{int(time.time())}",
            "title": title,
            "content": content,
            "document_type": document_type,
            "analysis": analysis,
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Update notebook
        cache_key = f"notebook:{notebook_id}"
        notebook = await cache_manager.get(cache_key)
        
        if not notebook:
            raise HTTPException(status_code=404, detail="Notebook not found")
        
        notebook["documents"].append(document)
        await cache_manager.set(cache_key, notebook)
        
        logger.info(f"Document added to notebook {notebook_id}")
        return {"document": document}
    
    except Exception as e:
        logger.error(f"Failed to add document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/notebooks/{notebook_id}/query")
async def query_notebook(
    notebook_id: str,
    query: str,
    user_token: Dict[str, Any] = Depends(verify_token)
):
    """Query notebook with AI."""
    try:
        # Get notebook
        cache_key = f"notebook:{notebook_id}"
        notebook = await cache_manager.get(cache_key)
        
        if not notebook:
            raise HTTPException(status_code=404, detail="Notebook not found")
        
        # Prepare context from documents
        context = ""
        for doc in notebook["documents"][:3]:  # Use first 3 documents as context
            context += f"{doc['title']}: {doc['content'][:500]}...\n\n"
        
        # Generate response
        if hasattr(app.state, 'llm_engine'):
            response = await app.state.llm_engine.generate_response(query, context)
        else:
            response = f"Based on the documents in your notebook, here's what I found about: {query}"
        
        # Create conversation entry
        conversation = {
            "id": f"conv_{int(time.time())}",
            "query": query,
            "response": response,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Update notebook
        notebook["conversations"].append(conversation)
        await cache_manager.set(cache_key, notebook)
        
        logger.info(f"Query processed for notebook {notebook_id}")
        return {"conversation": conversation}
    
    except Exception as e:
        logger.error(f"Failed to query notebook: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/notebooks/{notebook_id}")
async def get_notebook(
    notebook_id: str,
    user_token: Dict[str, Any] = Depends(verify_token)
):
    """Get notebook details."""
    try:
        cache_key = f"notebook:{notebook_id}"
        notebook = await cache_manager.get(cache_key)
        
        if not notebook:
            raise HTTPException(status_code=404, detail="Notebook not found")
        
        return {"notebook": notebook}
    
    except Exception as e:
        logger.error(f"Failed to get notebook: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/notebooks")
async def list_notebooks(
    user_token: Dict[str, Any] = Depends(verify_token),
    limit: int = 10,
    offset: int = 0
):
    """List user notebooks."""
    try:
        # In a real implementation, this would query a database
        # For now, return mock data
        notebooks = [
            {
                "id": f"notebook_{i}",
                "title": f"Notebook {i}",
                "description": f"Description for notebook {i}",
                "created_at": datetime.utcnow().isoformat(),
                "document_count": i * 2,
                "conversation_count": i
            }
            for i in range(1, min(limit + 1, 6))
        ]
        
        return {"notebooks": notebooks, "total": len(notebooks)}
    
    except Exception as e:
        logger.error(f"Failed to list notebooks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Streaming response endpoint
@app.post("/api/v1/stream/query")
async def stream_query(
    query: str,
    context: str = "",
    user_token: Dict[str, Any] = Depends(verify_token)
):
    """Stream AI response."""
    async def generate_response():
        
    """generate_response function."""
try:
            if hasattr(app.state, 'llm_engine'):
                # Generate response in chunks
                response = await app.state.llm_engine.generate_response(query, context)
                chunks = response.split()
                
                for i, chunk in enumerate(chunks):
                    yield f"data: {json.dumps({'chunk': chunk, 'index': i})}\n\n"
                    await asyncio.sleep(0.1)  # Simulate streaming
                
                yield f"data: {json.dumps({'done': True})}\n\n"
            else:
                yield f"data: {json.dumps({'chunk': 'Production mock response', 'index': 0})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
        
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_response(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={"error": "Resource not found", "path": str(request.url)}
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: HTTPException):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

if __name__ == "__main__":
    # Production server configuration
    host = os.getenv("NOTEBOOKLM_HOST", "0.0.0.0")
    port = int(os.getenv("NOTEBOOKLM_PORT", "8000"))
    workers = int(os.getenv("NOTEBOOKLM_WORKERS", "4"))
    
    logger.info(f"Starting production server on {host}:{port} with {workers} workers")
    
    uvicorn.run(
        "main_production:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
        access_log=True,
        reload=False
    ) 