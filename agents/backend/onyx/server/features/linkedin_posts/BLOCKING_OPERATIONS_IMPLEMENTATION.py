from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import logging
import time
import json
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse, StreamingResponse
from fastapi.encoders import jsonable_encoder
import uvicorn
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
import httpx
import aiohttp
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, String, Text, DateTime, Integer, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
import asyncpg
import aioredis
from aiocache import Cache, cached
from aiocache.serializers import PickleSerializer
import aiofiles
from aiofiles.os import wrap
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import structlog
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from circuitbreaker import circuit
import uvloop
import orjson
import ujson
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
LinkedIn Posts - Non-Blocking Operations Implementation
======================================================

Comprehensive implementation demonstrating how to avoid blocking operations
in FastAPI routes using async patterns, background tasks, thread pools,
and performance optimizations.
"""


# FastAPI and async imports

# Async HTTP client

# Database and caching

# File operations

# Monitoring and metrics

# Rate limiting and circuit breaker

# Performance optimization

# Configure uvloop for maximum performance
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

logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter('linkedin_posts_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('linkedin_posts_request_duration_seconds', 'Request latency')
BLOCKING_OPERATIONS = Counter('blocking_operations_total', 'Blocking operations detected')
BACKGROUND_TASKS = Counter('background_tasks_total', 'Background tasks executed')
THREAD_POOL_OPERATIONS = Counter('thread_pool_operations_total', 'Thread pool operations')
CACHE_HITS = Counter('cache_hits_total', 'Cache hits')
CACHE_MISSES = Counter('cache_misses_total', 'Cache misses')

# Global thread pool for CPU-bound operations
thread_pool = ThreadPoolExecutor(
    max_workers=min(32, (multiprocessing.cpu_count() + 4) * 2),
    thread_name_prefix="linkedin_posts"
)

# Global process pool for heavy CPU operations
process_pool = ProcessPoolExecutor(
    max_workers=multiprocessing.cpu_count(),
    mp_context=multiprocessing.get_context('spawn')
)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Pydantic models
class LinkedInPostRequest(BaseModel):
    content: str = Field(..., min_length=10, max_length=3000)
    post_type: str = Field(default="educational", regex="^(educational|promotional|personal|industry)$")
    tone: str = Field(default="professional", regex="^(professional|casual|enthusiastic|thoughtful)$")
    target_audience: str = Field(default="general", regex="^(general|executives|developers|marketers)$")
    include_hashtags: bool = Field(default=True)
    include_call_to_action: bool = Field(default=True)
    
    @validator('content')
    def validate_content(cls, v) -> bool:
        if not v.strip():
            raise ValueError('Content cannot be empty')
        return v.strip()

class LinkedInPostResponse(BaseModel):
    id: str
    content: str
    optimized_content: str
    hashtags: List[str]
    call_to_action: str
    sentiment_score: float
    readability_score: float
    engagement_prediction: float
    generated_image_url: Optional[str] = None
    created_at: str
    status: str

class PostOptimizationRequest(BaseModel):
    post_id: str
    optimization_type: str = Field(default="engagement", regex="^(engagement|clarity|professionalism|viral)$")

# Settings
class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://user:pass@localhost/linkedin_posts"
    REDIS_URL: str = "redis://localhost:6379"
    
    # AI Models
    OPENAI_API_KEY: str = ""
    HUGGINGFACE_TOKEN: str = ""
    MODEL_CACHE_DIR: str = "./model_cache"
    
    # Performance
    MAX_WORKERS: int = multiprocessing.cpu_count()
    CACHE_TTL: int = 3600
    RATE_LIMIT_PER_MINUTE: int = 100
    REQUEST_TIMEOUT: float = 30.0
    BACKGROUND_TASK_TIMEOUT: float = 300.0
    
    # Monitoring
    SENTRY_DSN: str = ""
    PROMETHEUS_PORT: int = 9090
    
    class Config:
        env_file = ".env"

# Database models
Base = declarative_base()

class LinkedInPost(Base):
    __tablename__ = "linkedin_posts"
    
    id = Column(String, primary_key=True)
    content = Column(Text, nullable=False)
    optimized_content = Column(Text)
    post_type = Column(String, nullable=False)
    tone = Column(String, nullable=False)
    target_audience = Column(String, nullable=False)
    hashtags = Column(JSON)
    call_to_action = Column(Text)
    sentiment_score = Column(Integer)
    readability_score = Column(Integer)
    engagement_prediction = Column(Integer)
    generated_image_url = Column(String)
    status = Column(String, default="draft")
    created_at = Column(DateTime)
    updated_at = Column(DateTime)

# Async Database Repository
class AsyncLinkedInPostRepository:
    def __init__(self, settings: Settings):
        
    """__init__ function."""
self.settings = settings
        self.engine = None
        self.session_factory = None
        self.redis = None
    
    async def initialize(self) -> Any:
        """Initialize database connections with connection pooling"""
        # Database engine with connection pooling
        self.engine = create_async_engine(
            self.settings.DATABASE_URL,
            pool_size=20,
            max_overflow=30,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False
        )
        
        # Session factory
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Redis connection with connection pooling
        self.redis = aioredis.from_url(
            self.settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            max_connections=20
        )
        
        # Create tables
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def create_post(self, post_data: Dict[str, Any]) -> str:
        """Create a new post asynchronously"""
        post_id = str(uuid.uuid4())
        
        async with self.session_factory() as session:
            post = LinkedInPost(
                id=post_id,
                **post_data
            )
            session.add(post)
            await session.commit()
            
            # Cache the post
            await self.redis.setex(
                f"post:{post_id}",
                self.settings.CACHE_TTL,
                json.dumps(post_data)
            )
            
            return post_id
    
    async def get_post(self, post_id: str) -> Optional[Dict[str, Any]]:
        """Get a post with caching"""
        # Try cache first
        cached_post = await self.redis.get(f"post:{post_id}")
        if cached_post:
            CACHE_HITS.inc()
            return json.loads(cached_post)
        
        CACHE_MISSES.inc()
        
        # Fallback to database
        async with self.session_factory() as session:
            result = await session.execute(
                f"SELECT * FROM linkedin_posts WHERE id = '{post_id}'"
            )
            post = result.fetchone()
            
            if post:
                post_dict = dict(post._mapping)
                # Cache the result
                await self.redis.setex(
                    f"post:{post_id}",
                    self.settings.CACHE_TTL,
                    json.dumps(post_dict)
                )
                return post_dict
            
            return None
    
    async def update_post(self, post_id: str, updates: Dict[str, Any]) -> bool:
        """Update a post asynchronously"""
        async with self.session_factory() as session:
            result = await session.execute(
                f"UPDATE linkedin_posts SET {', '.join([f'{k} = %s' for k in updates.keys()])} WHERE id = %s",
                list(updates.values()) + [post_id]
            )
            await session.commit()
            
            # Invalidate cache
            await self.redis.delete(f"post:{post_id}")
            
            return result.rowcount > 0

# Async File Operations
class AsyncFileHandler:
    def __init__(self, upload_dir: str = "./uploads"):
        
    """__init__ function."""
self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(exist_ok=True)
    
    async def save_file(self, file: UploadFile) -> str:
        """Save uploaded file asynchronously"""
        file_path = self.upload_dir / f"{uuid.uuid4()}_{file.filename}"
        
        async with aiofiles.open(file_path, 'wb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            # Read file in chunks to avoid memory issues
            chunk_size = 1024 * 1024  # 1MB chunks
            while chunk := await file.read(chunk_size):
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                await f.write(chunk)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        return str(file_path)
    
    async def read_file(self, file_path: str) -> str:
        """Read file content asynchronously"""
        async with aiofiles.open(file_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            return await f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
    
    async def delete_file(self, file_path: str) -> bool:
        """Delete file asynchronously"""
        try:
            await aiofiles.os.remove(file_path)
            return True
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {e}")
            return False

# CPU-Intensive Operations in Thread Pool
class CPUIntensiveProcessor:
    def __init__(self) -> Any:
        self.thread_pool = thread_pool
        self.process_pool = process_pool
    
    async def analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment using thread pool"""
        THREAD_POOL_OPERATIONS.inc()
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            self._analyze_sentiment_sync,
            text
        )
    
    def _analyze_sentiment_sync(self, text: str) -> float:
        """Synchronous sentiment analysis (CPU-intensive)"""
        # Simulate heavy NLP processing
        time.sleep(0.1)  # Simulate processing time
        
        # Simple sentiment calculation
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing']
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total_words = len(words)
        if total_words == 0:
            return 0.0
        
        return (positive_count - negative_count) / total_words
    
    async def calculate_readability(self, text: str) -> float:
        """Calculate readability score using thread pool"""
        THREAD_POOL_OPERATIONS.inc()
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            self._calculate_readability_sync,
            text
        )
    
    def _calculate_readability_sync(self, text: str) -> float:
        """Synchronous readability calculation (CPU-intensive)"""
        # Simulate heavy text analysis
        time.sleep(0.05)  # Simulate processing time
        
        sentences = text.split('.')
        words = text.split()
        syllables = sum(self._count_syllables(word) for word in words)
        
        if len(sentences) == 0 or len(words) == 0:
            return 0.0
        
        # Flesch Reading Ease formula
        return 206.835 - (1.015 * len(words) / len(sentences)) - (84.6 * syllables / len(words))
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word"""
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        on_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not on_vowel:
                count += 1
            on_vowel = is_vowel
        
        if word.endswith('e'):
            count -= 1
        if count == 0:
            count = 1
        return count

# Async External API Client
class AsyncExternalAPIClient:
    def __init__(self, timeout: float = 30.0):
        
    """__init__ function."""
self.timeout = timeout
        self.session = None
    
    async def __aenter__(self) -> Any:
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            connector=aiohttp.TCPConnector(
                limit=100,
                limit_per_host=20,
                ttl_dns_cache=300
            )
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> Any:
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    @circuit(failure_threshold=5, recovery_timeout=60)
    async async def call_external_api(self, url: str, method: str = "GET", data: Dict = None) -> Dict:
        """Call external API with circuit breaker pattern"""
        try:
            if method.upper() == "GET":
                async with self.session.get(url) as response:
                    return await response.json()
            elif method.upper() == "POST":
                async with self.session.post(url, json=data) as response:
                    return await response.json()
        except Exception as e:
            logger.error(f"External API call failed: {e}")
            raise HTTPException(status_code=503, detail="External service unavailable")

# Background Task Processor
class BackgroundTaskProcessor:
    def __init__(self, repository: AsyncLinkedInPostRepository, file_handler: AsyncFileHandler):
        
    """__init__ function."""
self.repository = repository
        self.file_handler = file_handler
        self.cpu_processor = CPUIntensiveProcessor()
    
    async def process_analytics(self, post_id: str):
        """Process analytics in background"""
        BACKGROUND_TASKS.inc()
        
        try:
            # Get post data
            post_data = await self.repository.get_post(post_id)
            if not post_data:
                logger.error(f"Post {post_id} not found for analytics")
                return
            
            # Process analytics asynchronously
            sentiment_score = await self.cpu_processor.analyze_sentiment(post_data['content'])
            readability_score = await self.cpu_processor.calculate_readability(post_data['content'])
            
            # Update post with analytics
            await self.repository.update_post(post_id, {
                'sentiment_score': sentiment_score,
                'readability_score': readability_score
            })
            
            logger.info(f"Analytics processed for post {post_id}")
            
        except Exception as e:
            logger.error(f"Error processing analytics for post {post_id}: {e}")
    
    async def send_notifications(self, post_id: str):
        """Send notifications in background"""
        BACKGROUND_TASKS.inc()
        
        try:
            # Simulate notification sending
            await asyncio.sleep(1)  # Simulate API call
            logger.info(f"Notifications sent for post {post_id}")
            
        except Exception as e:
            logger.error(f"Error sending notifications for post {post_id}: {e}")
    
    async def generate_image(self, content: str, post_id: str):
        """Generate image in background"""
        BACKGROUND_TASKS.inc()
        
        try:
            # Simulate image generation
            await asyncio.sleep(2)  # Simulate AI processing
            
            # Update post with generated image URL
            image_url = f"https://example.com/images/{post_id}.jpg"
            await self.repository.update_post(post_id, {
                'generated_image_url': image_url
            })
            
            logger.info(f"Image generated for post {post_id}")
            
        except Exception as e:
            logger.error(f"Error generating image for post {post_id}: {e}")

# Main FastAPI Application
class NonBlockingLinkedInPostsAPI:
    def __init__(self) -> Any:
        self.settings = Settings()
        self.app = FastAPI(
            title="LinkedIn Posts API - Non-Blocking",
            description="High-performance LinkedIn posts API with non-blocking operations",
            version="2.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Initialize components
        self.repository = AsyncLinkedInPostRepository(self.settings)
        self.file_handler = AsyncFileHandler()
        self.cpu_processor = CPUIntensiveProcessor()
        self.background_processor = BackgroundTaskProcessor(self.repository, self.file_handler)
        
        self._setup_middleware()
        self._setup_routes()
        self._setup_events()
    
    def _setup_middleware(self) -> Any:
        """Setup middleware for performance and monitoring"""
        
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Gzip compression
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Rate limiting
        self.app.state.limiter = limiter
        self.app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        
        # Performance monitoring middleware
        @self.app.middleware("http")
        async def performance_middleware(request: Request, call_next):
            
    """performance_middleware function."""
start_time = time.time()
            
            # Check for blocking operations
            if not asyncio.iscoroutinefunction(call_next):
                BLOCKING_OPERATIONS.inc()
            
            response = await call_next(request)
            
            duration = time.time() - start_time
            REQUEST_LATENCY.observe(duration)
            
            # Add performance headers
            response.headers["X-Response-Time"] = str(duration)
            
            return response
    
    def _setup_routes(self) -> Any:
        """Setup API routes with non-blocking operations"""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            try:
                # Check database connectivity
                await self.repository.redis.ping()
                return {"status": "healthy", "timestamp": time.time()}
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                raise HTTPException(status_code=503, detail="Service unhealthy")
        
        @self.app.get("/metrics")
        async def metrics():
            """Prometheus metrics endpoint"""
            return StreamingResponse(
                generate_latest(),
                media_type="text/plain"
            )
        
        @self.app.post("/api/v1/posts", response_model=LinkedInPostResponse)
        @limiter.limit("10/minute")
        async def create_post(
            request: Request,
            post_request: LinkedInPostRequest,
            background_tasks: BackgroundTasks
        ):
            """Create a new LinkedIn post with non-blocking operations"""
            REQUEST_COUNT.labels(method="POST", endpoint="/api/v1/posts").inc()
            
            try:
                # Set timeout for database operation
                post_data = post_request.dict()
                post_id = await asyncio.wait_for(
                    self.repository.create_post(post_data),
                    timeout=self.settings.REQUEST_TIMEOUT
                )
                
                # Add background tasks for heavy operations
                background_tasks.add_task(
                    self.background_processor.process_analytics,
                    post_id
                )
                background_tasks.add_task(
                    self.background_processor.send_notifications,
                    post_id
                )
                background_tasks.add_task(
                    self.background_processor.generate_image,
                    post_request.content,
                    post_id
                )
                
                return LinkedInPostResponse(
                    id=post_id,
                    content=post_request.content,
                    optimized_content=post_request.content,
                    hashtags=[],
                    call_to_action="",
                    sentiment_score=0.0,
                    readability_score=0.0,
                    engagement_prediction=0.0,
                    created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                    status="processing"
                )
                
            except asyncio.TimeoutError:
                raise HTTPException(status_code=504, detail="Operation timeout")
            except Exception as e:
                logger.error(f"Error creating post: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.get("/api/v1/posts/{post_id}", response_model=LinkedInPostResponse)
        @cached(ttl=300, serializer=PickleSerializer())
        async def get_post(post_id: str):
            """Get a LinkedIn post with caching"""
            REQUEST_COUNT.labels(method="GET", endpoint="/api/v1/posts/{post_id}").inc()
            
            try:
                post_data = await self.repository.get_post(post_id)
                if not post_data:
                    raise HTTPException(status_code=404, detail="Post not found")
                
                return LinkedInPostResponse(**post_data)
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error fetching post {post_id}: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.post("/api/v1/posts/{post_id}/optimize")
        async def optimize_post(
            post_id: str,
            optimization_request: PostOptimizationRequest
        ):
            """Optimize a post using thread pool for CPU-intensive operations"""
            REQUEST_COUNT.labels(method="POST", endpoint="/api/v1/posts/{post_id}/optimize").inc()
            
            try:
                # Get post data
                post_data = await self.repository.get_post(post_id)
                if not post_data:
                    raise HTTPException(status_code=404, detail="Post not found")
                
                # Use thread pool for CPU-intensive optimization
                loop = asyncio.get_event_loop()
                optimized_content = await loop.run_in_executor(
                    thread_pool,
                    self._optimize_content_sync,
                    post_data['content'],
                    optimization_request.optimization_type
                )
                
                # Update post
                await self.repository.update_post(post_id, {
                    'optimized_content': optimized_content
                })
                
                return {"status": "optimized", "optimized_content": optimized_content}
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error optimizing post {post_id}: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.post("/api/v1/upload")
        async def upload_file(file: UploadFile = File(...)):
            """Upload file asynchronously"""
            REQUEST_COUNT.labels(method="POST", endpoint="/api/v1/upload").inc()
            
            try:
                # Save file asynchronously
                file_path = await self.file_handler.save_file(file)
                
                return {
                    "filename": file.filename,
                    "file_path": file_path,
                    "size": file.size
                }
                
            except Exception as e:
                logger.error(f"Error uploading file: {e}")
                raise HTTPException(status_code=500, detail="Upload failed")
        
        @self.app.get("/api/v1/external-data")
        async def get_external_data():
            """Get external data with async HTTP client"""
            REQUEST_COUNT.labels(method="GET", endpoint="/api/v1/external-data").inc()
            
            try:
                async with AsyncExternalAPIClient() as client:
                    data = await client.call_external_api("https://jsonplaceholder.typicode.com/posts/1")
                    return data
                    
            except Exception as e:
                logger.error(f"Error fetching external data: {e}")
                raise HTTPException(status_code=503, detail="External service unavailable")
        
        @self.app.post("/api/v1/analyze")
        async def analyze_text(text: str):
            """Analyze text using thread pool for CPU-intensive operations"""
            REQUEST_COUNT.labels(method="POST", endpoint="/api/v1/analyze").inc()
            
            try:
                # Use thread pool for CPU-intensive analysis
                sentiment_score = await self.cpu_processor.analyze_sentiment(text)
                readability_score = await self.cpu_processor.calculate_readability(text)
                
                return {
                    "sentiment_score": sentiment_score,
                    "readability_score": readability_score,
                    "text_length": len(text)
                }
                
            except Exception as e:
                logger.error(f"Error analyzing text: {e}")
                raise HTTPException(status_code=500, detail="Analysis failed")
    
    def _optimize_content_sync(self, content: str, optimization_type: str) -> str:
        """Synchronous content optimization (CPU-intensive)"""
        # Simulate heavy content optimization
        time.sleep(0.5)  # Simulate processing time
        
        if optimization_type == "engagement":
            return f"🚀 {content} #engagement #growth"
        elif optimization_type == "clarity":
            return f"📝 {content} #clarity #communication"
        elif optimization_type == "professionalism":
            return f"💼 {content} #professional #business"
        elif optimization_type == "viral":
            return f"🔥 {content} #viral #trending"
        else:
            return content
    
    def _setup_events(self) -> Any:
        """Setup application events"""
        
        @self.app.on_event("startup")
        async def startup_event():
            """Initialize components on startup"""
            logger.info("Starting LinkedIn Posts API...")
            
            # Initialize database
            await self.repository.initialize()
            
            logger.info("LinkedIn Posts API started successfully")
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            """Cleanup on shutdown"""
            logger.info("Shutting down LinkedIn Posts API...")
            
            # Close thread pools
            thread_pool.shutdown(wait=True)
            process_pool.shutdown(wait=True)
            
            logger.info("LinkedIn Posts API shutdown complete")
    
    async def run_production_server(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the production server with optimized settings"""
        config = uvicorn.Config(
            self.app,
            host=host,
            port=port,
            workers=4,  # Multiple workers for CPU-bound tasks
            loop="uvloop",  # Faster event loop
            http="httptools",  # Faster HTTP parser
            access_log=False,  # Disable access logs for performance
            log_level="info"
        )
        
        server = uvicorn.Server(config)
        await server.serve()

# CLI interface
class LinkedInPostsCLI:
    def __init__(self) -> Any:
        self.api = NonBlockingLinkedInPostsAPI()
    
    async def create_post(self, content: str, post_type: str = "educational", tone: str = "professional"):
        """Create a post via CLI"""
        post_request = LinkedInPostRequest(
            content=content,
            post_type=post_type,
            tone=tone
        )
        
        # Simulate API call
        post_id = await self.api.repository.create_post(post_request.dict())
        print(f"Created post with ID: {post_id}")
        return post_id

# Main execution
async def main():
    """Main function"""
    api = NonBlockingLinkedInPostsAPI()
    
    # Run the server
    await api.run_production_server()

match __name__:
    case "__main__":
    asyncio.run(main()) 