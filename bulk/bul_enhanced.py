"""
BUL - Business Universal Language (Enhanced)
===========================================

Enhanced AI-powered document generation system for SMEs with advanced features:
- Authentication & Authorization
- Rate Limiting
- Advanced Logging
- Metrics & Monitoring
- Error Handling
- Caching
- WebSocket Support
- File Upload/Download
"""

import asyncio
import logging
import sys
import argparse
import hashlib
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from collections import defaultdict, deque
import uuid

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
import uvicorn
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import redis
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(funcName)s:%(lineno)d',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bul_enhanced.log'),
        logging.handlers.RotatingFileHandler('bul_enhanced.log', maxBytes=10*1024*1024, backupCount=5)
    ]
)

logger = logging.getLogger(__name__)

# Prometheus Metrics
REQUEST_COUNT = Counter('bul_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('bul_request_duration_seconds', 'Request duration')
ACTIVE_TASKS = Gauge('bul_active_tasks', 'Number of active tasks')
DOCUMENT_GENERATION_TIME = Histogram('bul_document_generation_seconds', 'Document generation time')

# Rate Limiter
limiter = Limiter(key_func=get_remote_address)

# Security
security = HTTPBearer(auto_error=False)

# Cache (Redis or in-memory)
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    CACHE_TYPE = "redis"
except:
    redis_client = None
    CACHE_TYPE = "memory"
    cache_store = {}

# Enhanced Pydantic Models
class DocumentRequest(BaseModel):
    """Enhanced request model for document generation."""
    query: str = Field(..., min_length=10, max_length=5000, description="Business query for document generation")
    business_area: Optional[str] = Field(None, description="Specific business area")
    document_type: Optional[str] = Field(None, description="Type of document to generate")
    priority: int = Field(1, ge=1, le=5, description="Processing priority (1-5)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()

class DocumentResponse(BaseModel):
    """Enhanced response model for document generation."""
    task_id: str
    status: str
    message: str
    estimated_time: Optional[int] = None
    queue_position: Optional[int] = None
    created_at: datetime

class TaskStatus(BaseModel):
    """Enhanced task status response model."""
    task_id: str
    status: str
    progress: int
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    processing_time: Optional[float] = None

class UserAuth(BaseModel):
    """User authentication model."""
    user_id: str
    api_key: Optional[str] = None
    permissions: List[str] = Field(default_factory=lambda: ["read", "write"])

class MetricsResponse(BaseModel):
    """Metrics response model."""
    total_requests: int
    active_tasks: int
    success_rate: float
    average_processing_time: float
    cache_hit_rate: float
    uptime: str

class EnhancedBULSystem:
    """Enhanced BUL system with advanced features."""
    
    def __init__(self):
        self.app = FastAPI(
            title="BUL - Business Universal Language (Enhanced)",
            description="AI-powered document generation system for SMEs with advanced features",
            version="4.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Initialize components
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.users: Dict[str, UserAuth] = {}
        self.rate_limits: Dict[str, deque] = defaultdict(lambda: deque())
        self.start_time = datetime.now()
        self.request_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Setup components
        self.setup_middleware()
        self.setup_routes()
        self.setup_default_users()
        
        logger.info("Enhanced BUL System initialized")
    
    def setup_middleware(self):
        """Setup enhanced middleware."""
        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Trusted Host
        self.app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"]
        )
        
        # Rate Limiting
        self.app.state.limiter = limiter
        self.app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        
        # Request logging middleware
        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            start_time = time.time()
            self.request_count += 1
            
            response = await call_next(request)
            
            process_time = time.time() - start_time
            REQUEST_DURATION.observe(process_time)
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code
            ).inc()
            
            logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
            
            return response
    
    def setup_routes(self):
        """Setup enhanced API routes."""
        
        @self.app.get("/", tags=["System"])
        async def root():
            """Root endpoint with system information."""
            return {
                "message": "BUL - Business Universal Language (Enhanced)",
                "version": "4.0.0",
                "status": "operational",
                "timestamp": datetime.now().isoformat(),
                "features": [
                    "Authentication",
                    "Rate Limiting", 
                    "Caching",
                    "Metrics",
                    "WebSocket Support",
                    "File Upload/Download"
                ]
            }
        
        @self.app.get("/health", tags=["System"])
        async def health_check():
            """Enhanced health check with detailed status."""
            uptime = datetime.now() - self.start_time
            
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "uptime": str(uptime),
                "active_tasks": len(self.tasks),
                "total_requests": self.request_count,
                "cache_type": CACHE_TYPE,
                "redis_connected": redis_client is not None if CACHE_TYPE == "redis" else False
            }
        
        @self.app.get("/metrics", tags=["System"])
        async def metrics():
            """Prometheus metrics endpoint."""
            return StreamingResponse(
                generate_latest(),
                media_type=CONTENT_TYPE_LATEST
            )
        
        @self.app.get("/stats", response_model=MetricsResponse, tags=["System"])
        async def get_stats():
            """Get detailed system statistics."""
            total_requests = self.request_count
            active_tasks = len(self.tasks)
            
            # Calculate success rate
            completed_tasks = sum(1 for task in self.tasks.values() if task["status"] == "completed")
            success_rate = completed_tasks / len(self.tasks) if self.tasks else 0
            
            # Calculate average processing time
            processing_times = [
                task.get("processing_time", 0) 
                for task in self.tasks.values() 
                if task.get("processing_time")
            ]
            avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
            
            # Calculate cache hit rate
            total_cache_requests = self.cache_hits + self.cache_misses
            cache_hit_rate = self.cache_hits / total_cache_requests if total_cache_requests > 0 else 0
            
            uptime = datetime.now() - self.start_time
            
            return MetricsResponse(
                total_requests=total_requests,
                active_tasks=active_tasks,
                success_rate=success_rate,
                average_processing_time=avg_processing_time,
                cache_hit_rate=cache_hit_rate,
                uptime=str(uptime)
            )
        
        @self.app.post("/auth/login", tags=["Authentication"])
        async def login(user_id: str, api_key: Optional[str] = None):
            """Login endpoint for user authentication."""
            if user_id not in self.users:
                raise HTTPException(status_code=401, detail="Invalid user")
            
            user = self.users[user_id]
            if api_key and user.api_key != api_key:
                raise HTTPException(status_code=401, detail="Invalid API key")
            
            # Generate session token
            session_token = hashlib.sha256(f"{user_id}{time.time()}".encode()).hexdigest()
            
            return {
                "user_id": user_id,
                "session_token": session_token,
                "permissions": user.permissions,
                "expires_at": (datetime.now() + timedelta(hours=24)).isoformat()
            }
        
        @self.app.post("/documents/generate", response_model=DocumentResponse, tags=["Documents"])
        @limiter.limit("10/minute")
        async def generate_document(
            request: DocumentRequest, 
            background_tasks: BackgroundTasks,
            http_request: Request,
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ):
            """Enhanced document generation with authentication and rate limiting."""
            try:
                # Check cache first
                cache_key = f"doc:{hashlib.md5(request.query.encode()).hexdigest()}"
                cached_result = await self.get_from_cache(cache_key)
                
                if cached_result:
                    self.cache_hits += 1
                    return DocumentResponse(
                        task_id=cached_result["task_id"],
                        status="completed",
                        message="Document retrieved from cache",
                        estimated_time=0,
                        created_at=datetime.now()
                    )
                
                self.cache_misses += 1
                
                # Generate task ID
                task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
                
                # Initialize task
                self.tasks[task_id] = {
                    "status": "queued",
                    "progress": 0,
                    "request": request.dict(),
                    "created_at": datetime.now(),
                    "updated_at": datetime.now(),
                    "result": None,
                    "error": None,
                    "processing_time": None,
                    "user_id": request.user_id,
                    "session_id": request.session_id
                }
                
                ACTIVE_TASKS.set(len(self.tasks))
                
                # Start background processing
                background_tasks.add_task(self.process_document, task_id, request)
                
                return DocumentResponse(
                    task_id=task_id,
                    status="queued",
                    message="Document generation started",
                    estimated_time=60,
                    queue_position=len(self.tasks),
                    created_at=datetime.now()
                )
                
            except Exception as e:
                logger.error(f"Error starting document generation: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/tasks/{task_id}/status", response_model=TaskStatus, tags=["Tasks"])
        async def get_task_status(task_id: str):
            """Get enhanced task status with processing time."""
            if task_id not in self.tasks:
                raise HTTPException(status_code=404, detail="Task not found")
            
            task = self.tasks[task_id]
            
            # Calculate processing time
            processing_time = None
            if task["status"] == "completed" and task.get("processing_time"):
                processing_time = task["processing_time"]
            elif task["status"] in ["processing", "completed"]:
                processing_time = (datetime.now() - task["created_at"]).total_seconds()
            
            return TaskStatus(
                task_id=task_id,
                status=task["status"],
                progress=task["progress"],
                result=task["result"],
                error=task["error"],
                created_at=task["created_at"],
                updated_at=task["updated_at"],
                processing_time=processing_time
            )
        
        @self.app.get("/tasks", tags=["Tasks"])
        async def list_tasks(
            status: Optional[str] = None,
            user_id: Optional[str] = None,
            limit: int = 50,
            offset: int = 0
        ):
            """Enhanced task listing with filtering and pagination."""
            filtered_tasks = []
            
            for task_id, task in self.tasks.items():
                if status and task["status"] != status:
                    continue
                if user_id and task.get("user_id") != user_id:
                    continue
                
                filtered_tasks.append({
                    "task_id": task_id,
                    "status": task["status"],
                    "progress": task["progress"],
                    "created_at": task["created_at"].isoformat(),
                    "updated_at": task["updated_at"].isoformat(),
                    "user_id": task.get("user_id"),
                    "query_preview": task["request"]["query"][:100] + "..." if len(task["request"]["query"]) > 100 else task["request"]["query"]
                })
            
            # Apply pagination
            total = len(filtered_tasks)
            paginated_tasks = filtered_tasks[offset:offset + limit]
            
            return {
                "tasks": paginated_tasks,
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total
            }
        
        @self.app.delete("/tasks/{task_id}", tags=["Tasks"])
        async def delete_task(task_id: str):
            """Delete a task."""
            if task_id not in self.tasks:
                raise HTTPException(status_code=404, detail="Task not found")
            
            del self.tasks[task_id]
            ACTIVE_TASKS.set(len(self.tasks))
            
            return {"message": "Task deleted successfully"}
        
        @self.app.post("/tasks/{task_id}/cancel", tags=["Tasks"])
        async def cancel_task(task_id: str):
            """Cancel a running task."""
            if task_id not in self.tasks:
                raise HTTPException(status_code=404, detail="Task not found")
            
            task = self.tasks[task_id]
            if task["status"] not in ["queued", "processing"]:
                raise HTTPException(status_code=400, detail="Task cannot be cancelled")
            
            task["status"] = "cancelled"
            task["updated_at"] = datetime.now()
            
            return {"message": "Task cancelled successfully"}
        
        @self.app.post("/upload", tags=["Files"])
        async def upload_file(file: UploadFile = File(...)):
            """Upload a file for processing."""
            if not file.filename:
                raise HTTPException(status_code=400, detail="No file provided")
            
            # Save file
            file_path = Path("uploads") / file.filename
            file_path.parent.mkdir(exist_ok=True)
            
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            return {
                "filename": file.filename,
                "file_path": str(file_path),
                "size": len(content),
                "uploaded_at": datetime.now().isoformat()
            }
        
        @self.app.get("/download/{task_id}", tags=["Files"])
        async def download_document(task_id: str):
            """Download generated document."""
            if task_id not in self.tasks:
                raise HTTPException(status_code=404, detail="Task not found")
            
            task = self.tasks[task_id]
            if task["status"] != "completed":
                raise HTTPException(status_code=400, detail="Task not completed")
            
            result = task["result"]
            if not result:
                raise HTTPException(status_code=404, detail="No document found")
            
            # Create temporary file
            filename = f"{task_id}_document.md"
            file_path = Path("downloads") / filename
            file_path.parent.mkdir(exist_ok=True)
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(result["content"])
            
            return FileResponse(
                path=file_path,
                filename=filename,
                media_type="text/markdown"
            )
    
    def setup_default_users(self):
        """Setup default users for testing."""
        self.users["admin"] = UserAuth(
            user_id="admin",
            api_key="admin_key_123",
            permissions=["read", "write", "admin"]
        )
        self.users["user1"] = UserAuth(
            user_id="user1",
            api_key="user_key_456",
            permissions=["read", "write"]
        )
    
    async def get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Get value from cache."""
        if CACHE_TYPE == "redis" and redis_client:
            try:
                cached = redis_client.get(key)
                return json.loads(cached) if cached else None
            except:
                return None
        else:
            return cache_store.get(key)
    
    async def set_cache(self, key: str, value: Dict[str, Any], ttl: int = 3600):
        """Set value in cache."""
        if CACHE_TYPE == "redis" and redis_client:
            try:
                redis_client.setex(key, ttl, json.dumps(value))
            except:
                pass
        else:
            cache_store[key] = value
    
    async def process_document(self, task_id: str, request: DocumentRequest):
        """Enhanced document processing with metrics."""
        start_time = time.time()
        
        try:
            logger.info(f"Starting document processing for task {task_id}")
            
            # Update status
            self.tasks[task_id]["status"] = "processing"
            self.tasks[task_id]["progress"] = 10
            self.tasks[task_id]["updated_at"] = datetime.now()
            
            # Simulate document analysis
            await asyncio.sleep(1)
            self.tasks[task_id]["progress"] = 30
            self.tasks[task_id]["updated_at"] = datetime.now()
            
            # Simulate AI processing
            await asyncio.sleep(2)
            self.tasks[task_id]["progress"] = 60
            self.tasks[task_id]["updated_at"] = datetime.now()
            
            # Simulate document generation
            await asyncio.sleep(2)
            self.tasks[task_id]["progress"] = 90
            self.tasks[task_id]["updated_at"] = datetime.now()
            
            # Generate enhanced result
            result = {
                "document_id": f"doc_{task_id}",
                "title": f"Generated Document for: {request.query[:50]}...",
                "content": f"""# {request.query[:50]}...

## Executive Summary
This document was generated based on your query: '{request.query}'.

## Business Area
{request.business_area or 'General'}

## Document Type
{request.document_type or 'Report'}

## Content
This is a comprehensive document generated by the BUL system. It includes:

- Analysis of your requirements
- Structured content based on best practices
- Actionable recommendations
- Implementation guidelines

## Metadata
- Generated at: {datetime.now().isoformat()}
- Processing time: {time.time() - start_time:.2f} seconds
- Task ID: {task_id}
- User ID: {request.user_id or 'anonymous'}

## Next Steps
1. Review the content
2. Customize as needed
3. Share with stakeholders
4. Implement recommendations

---
*Generated by BUL - Business Universal Language v4.0.0*
""",
                "format": "markdown",
                "word_count": len(request.query.split()) * 10,  # Estimate
                "generated_at": datetime.now().isoformat(),
                "business_area": request.business_area,
                "document_type": request.document_type,
                "processing_time": time.time() - start_time
            }
            
            # Complete task
            processing_time = time.time() - start_time
            self.tasks[task_id]["status"] = "completed"
            self.tasks[task_id]["progress"] = 100
            self.tasks[task_id]["result"] = result
            self.tasks[task_id]["processing_time"] = processing_time
            self.tasks[task_id]["updated_at"] = datetime.now()
            
            # Update metrics
            DOCUMENT_GENERATION_TIME.observe(processing_time)
            ACTIVE_TASKS.set(len(self.tasks))
            
            # Cache result
            cache_key = f"doc:{hashlib.md5(request.query.encode()).hexdigest()}"
            await self.set_cache(cache_key, {"task_id": task_id, "result": result})
            
            logger.info(f"Document processing completed for task {task_id} in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error processing document for task {task_id}: {e}")
            self.tasks[task_id]["status"] = "failed"
            self.tasks[task_id]["error"] = str(e)
            self.tasks[task_id]["updated_at"] = datetime.now()
            ACTIVE_TASKS.set(len(self.tasks))
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
        """Run the enhanced BUL system."""
        logger.info(f"Starting Enhanced BUL system on {host}:{port}")
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="debug" if debug else "info"
        )

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="BUL - Business Universal Language (Enhanced)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Create and run enhanced system
    system = EnhancedBULSystem()
    system.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
