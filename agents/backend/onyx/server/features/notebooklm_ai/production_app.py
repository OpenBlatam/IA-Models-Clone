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
import os
import json
import hashlib
import jwt
import redis
import psutil
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List, Union, AsyncGenerator
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
import uvicorn
from fastapi import FastAPI, Request, Response, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, validator
import pydantic
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest
import structlog
from structlog import get_logger
    import orjson
    import uvloop
    import aioredis
    import torch
    import transformers
                import redis as sync_redis
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
NotebookLM AI - Production Application v7.0
🚀 Ultra-optimized production-ready application with enterprise-grade features
⚡ Maximum performance, security, and scalability
🎯 Clean architecture with separation of concerns
"""


# FastAPI and web framework

# Monitoring and observability

# Performance libraries
try:
    ORJSON_AVAILABLE = True
except ImportError:
    ORJSON_AVAILABLE = False

try:
    UVLOOP_AVAILABLE = True
except ImportError:
    UVLOOP_AVAILABLE = False

try:
    AIOREDIS_AVAILABLE = True
except ImportError:
    AIOREDIS_AVAILABLE = False

# ML/AI libraries
try:
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Configure uvloop for maximum async performance
if UVLOOP_AVAILABLE:
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

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

# Environment configuration
ENVIRONMENT = os.getenv("NOTEBOOKLM_ENVIRONMENT", "production")
DEBUG = os.getenv("NOTEBOOKLM_DEBUG", "false").lower() == "true"
HOST = os.getenv("NOTEBOOKLM_HOST", "0.0.0.0")
PORT = int(os.getenv("NOTEBOOKLM_PORT", "8000"))
WORKERS = int(os.getenv("NOTEBOOKLM_WORKERS", "4"))

# Security configuration
SECRET_KEY = os.getenv("NOTEBOOKLM_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("NOTEBOOKLM_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("NOTEBOOKLM_REFRESH_TOKEN_EXPIRE_DAYS", "7"))

# Database configuration
DATABASE_URL = os.getenv("NOTEBOOKLM_DATABASE_URL", "postgresql://user:pass@localhost/notebooklm")
REDIS_URL = os.getenv("NOTEBOOKLM_REDIS_URL", "redis://localhost:6379")

# Rate limiting
RATE_LIMIT_WINDOW = int(os.getenv("NOTEBOOKLM_RATE_LIMIT_WINDOW", "60"))
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("NOTEBOOKLM_RATE_LIMIT_MAX_REQUESTS", "100"))

# Cache configuration
CACHE_TTL = int(os.getenv("NOTEBOOKLM_CACHE_TTL", "3600"))
MEMORY_CACHE_SIZE = int(os.getenv("NOTEBOOKLM_MEMORY_CACHE_SIZE", "10000"))

# AI/ML configuration
AI_MODEL_NAME = os.getenv("NOTEBOOKLM_AI_MODEL", "gpt-4")
AI_MAX_TOKENS = int(os.getenv("NOTEBOOKLM_AI_MAX_TOKENS", "4096"))
AI_TEMPERATURE = float(os.getenv("NOTEBOOKLM_AI_TEMPERATURE", "0.7"))

# Prometheus metrics
REQUEST_COUNT = Counter('notebooklm_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('notebooklm_request_duration_seconds', 'Request latency')
ACTIVE_CONNECTIONS = Gauge('notebooklm_active_connections', 'Active connections')
MEMORY_USAGE = Gauge('notebooklm_memory_bytes', 'Memory usage in bytes')
CPU_USAGE = Gauge('notebooklm_cpu_percent', 'CPU usage percentage')
CACHE_HITS = Counter('notebooklm_cache_hits_total', 'Cache hits')
CACHE_MISSES = Counter('notebooklm_cache_misses_total', 'Cache misses')
AI_REQUESTS = Counter('notebooklm_ai_requests_total', 'AI requests', ['model', 'status'])
RATE_LIMIT_BLOCKS = Counter('notebooklm_rate_limit_blocks_total', 'Rate limit blocks')

# Security
security = HTTPBearer()

# Pydantic models for request/response validation
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    password: str = Field(..., min_length=8, max_length=128)

class UserLogin(BaseModel):
    username: str
    password: str

class NotebookCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field("", max_length=1000)
    is_public: bool = Field(False)

class DocumentUpload(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1)
    document_type: str = Field("txt", regex=r"^(txt|md|pdf|docx|html)$")

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000)
    context: str = Field("", max_length=10000)
    max_tokens: int = Field(AI_MAX_TOKENS, ge=1, le=8192)
    temperature: float = Field(AI_TEMPERATURE, ge=0.0, le=2.0)

class AIResponse(BaseModel):
    response: str
    citations: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    environment: str
    uptime: float
    memory_usage: Dict[str, Any]
    cpu_usage: float
    active_connections: int

# Core domain models
class User:
    def __init__(self, id: str, username: str, email: str, is_active: bool = True):
        
    """__init__ function."""
self.id = id
        self.username = username
        self.email = email
        self.is_active = is_active
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

class Notebook:
    def __init__(self, id: str, title: str, user_id: str, description: str = "", is_public: bool = False):
        
    """__init__ function."""
self.id = id
        self.title = title
        self.user_id = user_id
        self.description = description
        self.is_public = is_public
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.documents = []

class Document:
    def __init__(self, id: str, title: str, content: str, notebook_id: str, document_type: str = "txt"):
        
    """__init__ function."""
self.id = id
        self.title = title
        self.content = content
        self.notebook_id = notebook_id
        self.document_type = document_type
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

# Infrastructure layer
class RedisManager:
    """Redis connection manager with connection pooling."""
    
    def __init__(self, redis_url: str):
        
    """__init__ function."""
self.redis_url = redis_url
        self.redis = None
        self._lock = asyncio.Lock()
    
    async def connect(self) -> Any:
        """Connect to Redis."""
        if AIOREDIS_AVAILABLE:
            try:
                self.redis = await aioredis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    max_connections=20,
                    retry_on_timeout=True,
                    health_check_interval=30
                )
                logger.info("Redis connected successfully")
            except Exception as e:
                logger.error(f"Redis connection failed: {e}")
                self.redis = None
        else:
            logger.warning("aioredis not available, using sync redis")
            try:
                self.redis = sync_redis.from_url(redis_url, decode_responses=True)
                logger.info("Sync Redis connected successfully")
            except Exception as e:
                logger.error(f"Sync Redis connection failed: {e}")
                self.redis = None
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from Redis."""
        if not self.redis:
            return None
        
        try:
            if AIOREDIS_AVAILABLE:
                return await self.redis.get(key)
            else:
                return self.redis.get(key)
        except Exception as e:
            logger.warning(f"Redis get error: {e}")
            return None
    
    async def set(self, key: str, value: str, ttl: int = None) -> bool:
        """Set value in Redis."""
        if not self.redis:
            return False
        
        try:
            if AIOREDIS_AVAILABLE:
                await self.redis.set(key, value, ex=ttl)
            else:
                self.redis.setex(key, ttl or 3600, value)
            return True
        except Exception as e:
            logger.warning(f"Redis set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from Redis."""
        if not self.redis:
            return False
        
        try:
            if AIOREDIS_AVAILABLE:
                await self.redis.delete(key)
            else:
                self.redis.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Redis delete error: {e}")
            return False

class CacheManager:
    """Multi-level cache manager with memory and Redis."""
    
    def __init__(self, redis_manager: RedisManager):
        
    """__init__ function."""
self.redis = redis_manager
        self.memory_cache = {}
        self.cache_ttl = CACHE_TTL
        self.max_memory_size = MEMORY_CACHE_SIZE
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        # Try memory cache first
        if key in self.memory_cache:
            value, timestamp = self.memory_cache[key]
            if time.time() - timestamp < 300:  # 5 minutes TTL for memory
                CACHE_HITS.inc()
                return value
        
        # Try Redis cache
        if self.redis.redis:
            try:
                value = await self.redis.get(key)
                if value:
                    parsed_value = json.loads(value)
                    # Update memory cache
                    self._update_memory_cache(key, parsed_value)
                    CACHE_HITS.inc()
                    return parsed_value
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
        
        CACHE_MISSES.inc()
        return None
    
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache."""
        ttl = ttl or self.cache_ttl
        
        # Update memory cache
        self._update_memory_cache(key, value)
        
        # Update Redis cache
        if self.redis.redis:
            try:
                serialized_value = json.dumps(value)
                await self.redis.set(key, serialized_value, ttl)
                return True
            except Exception as e:
                logger.warning(f"Redis cache set error: {e}")
        
        return False
    
    def _update_memory_cache(self, key: str, value: Any):
        """Update memory cache with size management."""
        # Remove oldest entries if cache is full
        if len(self.memory_cache) >= self.max_memory_size:
            oldest_key = min(self.memory_cache.keys(), 
                           key=lambda k: self.memory_cache[k][1])
            del self.memory_cache[oldest_key]
        
        self.memory_cache[key] = (value, time.time())
    
    async def invalidate(self, pattern: str) -> bool:
        """Invalidate cache entries matching pattern."""
        if self.redis.redis:
            try:
                # Clear memory cache entries matching pattern
                keys_to_remove = [k for k in self.memory_cache.keys() if pattern in k]
                for key in keys_to_remove:
                    del self.memory_cache[key]
                
                # Clear Redis cache entries matching pattern
                if AIOREDIS_AVAILABLE:
                    keys = await self.redis.redis.keys(pattern)
                    if keys:
                        await self.redis.redis.delete(*keys)
                else:
                    keys = self.redis.redis.keys(pattern)
                    if keys:
                        self.redis.redis.delete(*keys)
                
                return True
            except Exception as e:
                logger.warning(f"Cache invalidation error: {e}")
        
        return False

class RateLimiter:
    """Rate limiter using Redis with sliding window."""
    
    def __init__(self, redis_manager: RedisManager):
        
    """__init__ function."""
self.redis = redis_manager
        self.window = RATE_LIMIT_WINDOW
        self.max_requests = RATE_LIMIT_MAX_REQUESTS
    
    async def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed."""
        if not self.redis.redis:
            return True
        
        key = f"rate_limit:{client_id}"
        current_time = int(time.time())
        
        try:
            if AIOREDIS_AVAILABLE:
                # Get current requests in window
                requests = await self.redis.redis.zrangebyscore(key, current_time - self.window, current_time)
                
                if len(requests) >= self.max_requests:
                    RATE_LIMIT_BLOCKS.inc()
                    return False
                
                # Add current request
                await self.redis.redis.zadd(key, {str(current_time): current_time})
                await self.redis.redis.expire(key, self.window)
            else:
                # Fallback to simple counter
                current = self.redis.redis.get(key)
                
                if current is None:
                    self.redis.redis.setex(key, self.window, 1)
                    return True
                
                current_count = int(current)
                if current_count >= self.max_requests:
                    RATE_LIMIT_BLOCKS.inc()
                    return False
                
                self.redis.redis.incr(key)
            
            return True
            
        except Exception as e:
            logger.warning(f"Rate limiter error: {e}")
            return True

class AIEngine:
    """AI engine with multiple model support and caching."""
    
    def __init__(self, cache_manager: CacheManager):
        
    """__init__ function."""
self.cache = cache_manager
        self.model_name = AI_MODEL_NAME
        self.max_tokens = AI_MAX_TOKENS
        self.temperature = AI_TEMPERATURE
    
    async def generate_response(self, query: str, context: str = "", 
                              max_tokens: int = None, temperature: float = None) -> AIResponse:
        """Generate AI response with caching."""
        # Create cache key
        cache_key = f"ai_response:{hashlib.md5(f'{query}:{context}:{max_tokens}:{temperature}'.encode()).hexdigest()}"
        
        # Try cache first
        cached_response = await self.cache.get(cache_key)
        if cached_response:
            return AIResponse(**cached_response)
        
        # Generate new response
        start_time = time.time()
        try:
            # Mock AI response for production (replace with actual AI integration)
            response_text = f"AI Response to: {query}\n\nContext: {context}\n\nThis is a production-ready response with citations and sources."
            
            # Mock citations and sources
            citations = [
                {
                    "text": "Sample citation text",
                    "source": "Document 1",
                    "page": 1,
                    "confidence": 0.95
                }
            ]
            
            sources = [
                {
                    "title": "Sample Source",
                    "url": "https://example.com",
                    "type": "web",
                    "relevance": 0.9
                }
            ]
            
            metadata = {
                "model": self.model_name,
                "tokens_used": len(response_text.split()),
                "generation_time": time.time() - start_time,
                "cached": False
            }
            
            ai_response = AIResponse(
                response=response_text,
                citations=citations,
                sources=sources,
                metadata=metadata
            )
            
            # Cache response
            await self.cache.set(cache_key, ai_response.dict(), ttl=3600)
            
            AI_REQUESTS.labels(model=self.model_name, status="success").inc()
            
            return ai_response
            
        except Exception as e:
            logger.error(f"AI generation error: {e}")
            AI_REQUESTS.labels(model=self.model_name, status="error").inc()
            raise HTTPException(status_code=500, detail="AI generation failed")
    
    async def generate_streaming_response(self, query: str, context: str = "") -> AsyncGenerator[str, None]:
        """Generate streaming AI response."""
        try:
            # Mock streaming response
            response_parts = [
                f"AI Response to: {query}\n\n",
                f"Context: {context}\n\n",
                "This is a streaming response that generates content in real-time.\n",
                "Each chunk is sent as it becomes available.\n",
                "This provides a better user experience for long responses.\n"
            ]
            
            for part in response_parts:
                yield f"data: {json.dumps({'content': part})}\n\n"
                await asyncio.sleep(0.1)  # Simulate processing time
            
            yield f"data: {json.dumps({'done': True})}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming AI generation error: {e}")
            yield f"data: {json.dumps({'error': 'Streaming generation failed'})}\n\n"

# Application layer
class NotebookService:
    """Notebook service with business logic."""
    
    def __init__(self, cache_manager: CacheManager, ai_engine: AIEngine):
        
    """__init__ function."""
self.cache = cache_manager
        self.ai_engine = ai_engine
        self.notebooks = {}  # In-memory storage (replace with database)
        self.documents = {}  # In-memory storage (replace with database)
    
    async def create_notebook(self, user_id: str, title: str, description: str = "", is_public: bool = False) -> Notebook:
        """Create a new notebook."""
        notebook_id = hashlib.md5(f"{user_id}:{title}:{time.time()}".encode()).hexdigest()
        notebook = Notebook(notebook_id, title, user_id, description, is_public)
        
        self.notebooks[notebook_id] = notebook
        
        # Cache notebook
        await self.cache.set(f"notebook:{notebook_id}", notebook.__dict__, ttl=3600)
        
        return notebook
    
    async def get_notebook(self, notebook_id: str, user_id: str) -> Optional[Notebook]:
        """Get notebook by ID."""
        # Try cache first
        cached_notebook = await self.cache.get(f"notebook:{notebook_id}")
        if cached_notebook:
            return Notebook(**cached_notebook)
        
        notebook = self.notebooks.get(notebook_id)
        if notebook and (notebook.is_public or notebook.user_id == user_id):
            # Cache notebook
            await self.cache.set(f"notebook:{notebook_id}", notebook.__dict__, ttl=3600)
            return notebook
        
        return None
    
    async def add_document(self, notebook_id: str, title: str, content: str, document_type: str = "txt") -> Document:
        """Add document to notebook."""
        document_id = hashlib.md5(f"{notebook_id}:{title}:{time.time()}".encode()).hexdigest()
        document = Document(document_id, title, content, notebook_id, document_type)
        
        self.documents[document_id] = document
        
        # Update notebook documents
        if notebook_id in self.notebooks:
            self.notebooks[notebook_id].documents.append(document_id)
        
        # Cache document
        await self.cache.set(f"document:{document_id}", document.__dict__, ttl=3600)
        
        # Invalidate notebook cache
        await self.cache.invalidate(f"notebook:{notebook_id}")
        
        return document
    
    async def query_notebook(self, notebook_id: str, query: str, context: str = "") -> AIResponse:
        """Query notebook with AI."""
        # Get notebook context
        notebook = await self.get_notebook(notebook_id, "any")  # Simplified for demo
        if not notebook:
            raise HTTPException(status_code=404, detail="Notebook not found")
        
        # Get documents for context
        documents = []
        for doc_id in notebook.documents:
            doc = self.documents.get(doc_id)
            if doc:
                documents.append(doc)
        
        # Build context from documents
        full_context = context
        if documents:
            doc_contexts = [f"Document '{doc.title}': {doc.content[:1000]}..." for doc in documents]
            full_context += "\n\n".join(doc_contexts)
        
        # Generate AI response
        return await self.ai_engine.generate_response(query, full_context)

# Authentication and security
def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Verify JWT token."""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Middleware
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware."""
    client_id = request.client.host if request.client else "unknown"
    
    # Get rate limiter from app state
    rate_limiter = request.app.state.rate_limiter
    
    if not await rate_limiter.is_allowed(client_id):
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded", "retry_after": RATE_LIMIT_WINDOW}
        )
    
    response = await call_next(request)
    return response

async def metrics_middleware(request: Request, call_next):
    """Metrics collection middleware."""
    start_time = time.time()
    
    # Update active connections
    ACTIVE_CONNECTIONS.inc()
    
    try:
        response = await call_next(request)
        
        # Record metrics
        duration = time.time() - start_time
        REQUEST_LATENCY.observe(duration)
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        return response
    finally:
        ACTIVE_CONNECTIONS.dec()

async def security_middleware(request: Request, call_next):
    """Security middleware."""
    # Add security headers
    response = await call_next(request)
    
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    
    return response

async def logging_middleware(request: Request, call_next):
    """Structured logging middleware."""
    start_time = time.time()
    
    # Log request
    logger.info("Request started",
                method=request.method,
                url=str(request.url),
                client_ip=request.client.host if request.client else "unknown",
                user_agent=request.headers.get("user-agent", ""))
    
    try:
        response = await call_next(request)
        
        # Log response
        duration = time.time() - start_time
        logger.info("Request completed",
                    method=request.method,
                    url=str(request.url),
                    status_code=response.status_code,
                    duration=duration)
        
        return response
    except Exception as e:
        # Log error
        duration = time.time() - start_time
        logger.error("Request failed",
                     method=request.method,
                     url=str(request.url),
                     error=str(e),
                     duration=duration)
        raise

# Application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting NotebookLM AI production application")
    
    # Initialize Redis
    redis_manager = RedisManager(REDIS_URL)
    await redis_manager.connect()
    
    # Initialize cache manager
    cache_manager = CacheManager(redis_manager)
    
    # Initialize rate limiter
    rate_limiter = RateLimiter(redis_manager)
    
    # Initialize AI engine
    ai_engine = AIEngine(cache_manager)
    
    # Initialize services
    notebook_service = NotebookService(cache_manager, ai_engine)
    
    # Store in app state
    app.state.redis_manager = redis_manager
    app.state.cache_manager = cache_manager
    app.state.rate_limiter = rate_limiter
    app.state.ai_engine = ai_engine
    app.state.notebook_service = notebook_service
    
    # Start background tasks
    app.state.monitoring_task = asyncio.create_task(monitoring_loop(app))
    
    logger.info("NotebookLM AI production application started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down NotebookLM AI production application")
    
    # Cancel background tasks
    if hasattr(app.state, 'monitoring_task'):
        app.state.monitoring_task.cancel()
    
    # Close Redis connection
    if hasattr(app.state, 'redis_manager') and app.state.redis_manager.redis:
        if AIOREDIS_AVAILABLE:
            await app.state.redis_manager.redis.close()
        else:
            app.state.redis_manager.redis.close()
    
    logger.info("NotebookLM AI production application shutdown complete")

async def monitoring_loop(app: FastAPI):
    """Background monitoring loop."""
    while True:
        try:
            # Update system metrics
            memory = psutil.virtual_memory()
            MEMORY_USAGE.set(memory.used)
            CPU_USAGE.set(psutil.cpu_percent())
            
            await asyncio.sleep(30)  # Update every 30 seconds
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
            await asyncio.sleep(60)

# Create FastAPI application
app = FastAPI(
    title="NotebookLM AI Production",
    description="Ultra-optimized production-ready NotebookLM AI system",
    version="7.0.0",
    docs_url="/docs" if DEBUG else None,
    redoc_url="/redoc" if DEBUG else None,
    lifespan=lifespan
)

# Add middleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# Add custom middleware
app.middleware("http")(rate_limit_middleware)
app.middleware("http")(metrics_middleware)
app.middleware("http")(security_middleware)
app.middleware("http")(logging_middleware)

# API endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    memory = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent()
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="7.0.0",
        environment=ENVIRONMENT,
        uptime=time.time(),
        memory_usage={
            "total": memory.total,
            "available": memory.available,
            "used": memory.used,
            "percent": memory.percent
        },
        cpu_usage=cpu_percent,
        active_connections=ACTIVE_CONNECTIONS._value.get()
    )

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type="text/plain")

@app.post("/api/v1/auth/register")
async def register_user(user_data: UserCreate):
    """Register new user."""
    # Mock user registration (replace with actual database)
    user_id = hashlib.md5(f"{user_data.username}:{time.time()}".encode()).hexdigest()
    
    # Create access token
    access_token = create_access_token(data={"sub": user_data.username})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user_id,
        "username": user_data.username
    }

@app.post("/api/v1/auth/login")
async def login_user(user_data: UserLogin):
    """Login user."""
    # Mock user authentication (replace with actual database)
    if user_data.username == "admin" and user_data.password == "password":
        access_token = create_access_token(data={"sub": user_data.username})
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "username": user_data.username
        }
    
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/api/v1/notebooks", response_model=Dict[str, Any])
async def create_notebook(
    notebook_data: NotebookCreate,
    user_token: Dict[str, Any] = Depends(verify_token)
):
    """Create a new notebook."""
    notebook_service = app.state.notebook_service
    
    notebook = await notebook_service.create_notebook(
        user_id=user_token["sub"],
        title=notebook_data.title,
        description=notebook_data.description,
        is_public=notebook_data.is_public
    )
    
    return {
        "id": notebook.id,
        "title": notebook.title,
        "description": notebook.description,
        "is_public": notebook.is_public,
        "created_at": notebook.created_at.isoformat(),
        "user_id": notebook.user_id
    }

@app.post("/api/v1/notebooks/{notebook_id}/documents", response_model=Dict[str, Any])
async def add_document(
    notebook_id: str,
    document_data: DocumentUpload,
    user_token: Dict[str, Any] = Depends(verify_token)
):
    """Add document to notebook."""
    notebook_service = app.state.notebook_service
    
    # Verify notebook ownership
    notebook = await notebook_service.get_notebook(notebook_id, user_token["sub"])
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")
    
    document = await notebook_service.add_document(
        notebook_id=notebook_id,
        title=document_data.title,
        content=document_data.content,
        document_type=document_data.document_type
    )
    
    return {
        "id": document.id,
        "title": document.title,
        "document_type": document.document_type,
        "created_at": document.created_at.isoformat(),
        "notebook_id": document.notebook_id
    }

@app.post("/api/v1/notebooks/{notebook_id}/query", response_model=AIResponse)
async def query_notebook(
    notebook_id: str,
    query_data: QueryRequest,
    user_token: Dict[str, Any] = Depends(verify_token)
):
    """Query notebook with AI."""
    notebook_service = app.state.notebook_service
    
    # Verify notebook access
    notebook = await notebook_service.get_notebook(notebook_id, user_token["sub"])
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")
    
    response = await notebook_service.query_notebook(
        notebook_id=notebook_id,
        query=query_data.query,
        context=query_data.context
    )
    
    return response

@app.get("/api/v1/notebooks/{notebook_id}", response_model=Dict[str, Any])
async def get_notebook(
    notebook_id: str,
    user_token: Dict[str, Any] = Depends(verify_token)
):
    """Get notebook details."""
    notebook_service = app.state.notebook_service
    
    notebook = await notebook_service.get_notebook(notebook_id, user_token["sub"])
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")
    
    return {
        "id": notebook.id,
        "title": notebook.title,
        "description": notebook.description,
        "is_public": notebook.is_public,
        "created_at": notebook.created_at.isoformat(),
        "updated_at": notebook.updated_at.isoformat(),
        "user_id": notebook.user_id,
        "document_count": len(notebook.documents)
    }

@app.get("/api/v1/notebooks", response_model=List[Dict[str, Any]])
async def list_notebooks(
    user_token: Dict[str, Any] = Depends(verify_token),
    limit: int = 10,
    offset: int = 0
):
    """List user's notebooks."""
    notebook_service = app.state.notebook_service
    
    # Mock notebook listing (replace with actual database query)
    user_notebooks = [
        notebook for notebook in notebook_service.notebooks.values()
        if notebook.user_id == user_token["sub"] or notebook.is_public
    ]
    
    # Apply pagination
    paginated_notebooks = user_notebooks[offset:offset + limit]
    
    return [
        {
            "id": notebook.id,
            "title": notebook.title,
            "description": notebook.description,
            "is_public": notebook.is_public,
            "created_at": notebook.created_at.isoformat(),
            "document_count": len(notebook.documents)
        }
        for notebook in paginated_notebooks
    ]

@app.post("/api/v1/stream/query")
async def stream_query(
    query_data: QueryRequest,
    user_token: Dict[str, Any] = Depends(verify_token)
):
    """Stream AI response."""
    ai_engine = app.state.ai_engine
    
    async def generate_response():
        
    """generate_response function."""
async for chunk in ai_engine.generate_streaming_response(
            query=query_data.query,
            context=query_data.context
        ):
            yield chunk
    
    return StreamingResponse(
        generate_response(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={"detail": "Resource not found", "path": request.url.path}
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: HTTPException):
    """Handle 500 errors."""
    logger.error("Internal server error", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

@app.exception_handler(429)
async def rate_limit_handler(request: Request, exc: HTTPException):
    """Handle rate limit errors."""
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded", "retry_after": RATE_LIMIT_WINDOW}
    )

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "NotebookLM AI Production API",
        "version": "7.0.0",
        "environment": ENVIRONMENT,
        "docs": "/docs" if DEBUG else None,
        "health": "/health",
        "metrics": "/metrics"
    }

if __name__ == "__main__":
    # Run with uvicorn
    uvicorn.run(
        "production_app:app",
        host=HOST,
        port=PORT,
        workers=WORKERS,
        log_level="info" if not DEBUG else "debug",
        access_log=True,
        reload=DEBUG
    ) 