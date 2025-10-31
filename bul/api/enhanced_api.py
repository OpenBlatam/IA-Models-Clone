"""
Enhanced BUL API
================

Modern, production-ready FastAPI application with advanced patterns,
comprehensive error handling, and optimal performance.
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, AsyncGenerator, Union
from enum import Enum

from fastapi import (
    FastAPI, 
    HTTPException, 
    BackgroundTasks, 
    Depends, 
    Request, 
    Response,
    WebSocket, 
    WebSocketDisconnect,
    status,
    Query,
    Path,
    Body
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field, validator, root_validator
from pydantic.types import PositiveInt, NonNegativeInt
import uvicorn

# Import BUL modules
from ..core import BULEngine, DocumentRequest, DocumentResponse, BusinessArea, DocumentType, get_global_bul_engine
from ..agents import SMEAgentManager, get_global_agent_manager
from ..utils import get_cache_manager, get_logger, log_api_call, monitor_performance
from ..security import get_rate_limiter, SecurityValidator, get_auth_manager
from ..monitoring.health_checker import get_health_checker
from ..config import get_config, is_production, is_development
from ..middleware import (
    RequestLoggingMiddleware,
    PerformanceMiddleware,
    SecurityMiddleware,
    ErrorHandlingMiddleware
)

# Configure logging
logger = get_logger(__name__)

# Enhanced Enums
class DocumentFormat(str, Enum):
    """Supported document formats"""
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"

class DocumentStyle(str, Enum):
    """Document styles"""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    TECHNICAL = "technical"
    CREATIVE = "creative"
    FORMAL = "formal"

class ProcessingPriority(str, Enum):
    """Processing priorities"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

class APIStatus(str, Enum):
    """API status values"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"

# Enhanced Pydantic Models with comprehensive validation
class BaseRequestModel(BaseModel):
    """Base request model with common fields"""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True
        validate_assignment = True
        extra = "forbid"

class DocumentRequestModel(BaseRequestModel):
    """Enhanced document generation request model"""
    query: str = Field(..., description="Business query or requirement", min_length=10, max_length=2000)
    business_area: Optional[str] = Field(None, description="Business area")
    document_type: Optional[str] = Field(None, description="Type of document to generate")
    company_name: Optional[str] = Field(None, description="Company name", max_length=100)
    industry: Optional[str] = Field(None, description="Industry sector", max_length=100)
    company_size: Optional[str] = Field(None, description="Company size")
    target_audience: Optional[str] = Field(None, description="Target audience", max_length=200)
    language: str = Field("es", description="Language for document generation")
    format: DocumentFormat = Field(DocumentFormat.MARKDOWN, description="Output format")
    style: DocumentStyle = Field(DocumentStyle.PROFESSIONAL, description="Document style")
    priority: ProcessingPriority = Field(ProcessingPriority.NORMAL, description="Processing priority")
    cache_ttl: int = Field(3600, description="Cache TTL in seconds", ge=60, le=86400)
    include_metadata: bool = Field(True, description="Include metadata in response")
    
    @validator('language')
    def validate_language(cls, v):
        allowed_languages = ['es', 'en', 'pt', 'fr', 'de', 'it', 'ru', 'zh', 'ja']
        if v not in allowed_languages:
            raise ValueError(f'Language must be one of: {allowed_languages}')
        return v
    
    @validator('business_area')
    def validate_business_area(cls, v):
        if v is not None:
            try:
                BusinessArea(v.lower())
            except ValueError:
                raise ValueError(f'Invalid business area: {v}')
        return v
    
    @validator('document_type')
    def validate_document_type(cls, v):
        if v is not None:
            try:
                DocumentType(v.lower())
            except ValueError:
                raise ValueError(f'Invalid document type: {v}')
        return v

class DocumentResponseModel(BaseModel):
    """Enhanced document generation response model"""
    id: str
    request_id: str
    content: str
    title: str
    summary: str
    business_area: str
    document_type: str
    word_count: int
    processing_time: float
    confidence_score: float
    created_at: datetime
    agent_used: Optional[str] = None
    format: str
    style: str
    metadata: Dict[str, Any]
    quality_score: Optional[float] = None
    readability_score: Optional[float] = None
    
    class Config:
        use_enum_values = True

class BatchDocumentRequestModel(BaseModel):
    """Batch document generation request model"""
    requests: List[DocumentRequestModel] = Field(..., max_items=10, description="List of document requests")
    parallel: bool = Field(True, description="Process requests in parallel")
    priority: ProcessingPriority = Field(ProcessingPriority.NORMAL, description="Overall batch priority")
    max_concurrent: int = Field(5, ge=1, le=10, description="Maximum concurrent requests")
    
    @validator('requests')
    def validate_requests(cls, v):
        if not v:
            raise ValueError('At least one request is required')
        return v

class SystemHealthModel(BaseModel):
    """Comprehensive system health model"""
    status: APIStatus
    timestamp: datetime
    version: str
    uptime: float
    components: Dict[str, Any]
    metrics: Dict[str, Any]
    performance: Dict[str, Any]
    dependencies: Dict[str, Any]

class AgentStatsModel(BaseModel):
    """Agent statistics model"""
    total_agents: int
    active_agents: int
    total_documents_generated: int
    average_success_rate: float
    agent_types: List[str]
    is_initialized: bool
    performance_metrics: Dict[str, Any]

class ErrorResponseModel(BaseModel):
    """Standardized error response model"""
    error: str
    detail: str
    timestamp: datetime
    request_id: Optional[str] = None
    status_code: int
    suggestions: Optional[List[str]] = None

# WebSocket connection manager with enhanced features
class ConnectionManager:
    """Enhanced WebSocket connection manager"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.user_connections: Dict[str, List[WebSocket]] = {}
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
        self.heartbeat_interval = 30  # seconds
        
    async def connect(self, websocket: WebSocket, user_id: str = None, metadata: Dict[str, Any] = None):
        """Connect a WebSocket with metadata tracking"""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        connection_meta = {
            "user_id": user_id,
            "connected_at": datetime.now(),
            "last_heartbeat": datetime.now(),
            "message_count": 0,
            "metadata": metadata or {}
        }
        self.connection_metadata[websocket] = connection_meta
        
        if user_id:
            if user_id not in self.user_connections:
                self.user_connections[user_id] = []
            self.user_connections[user_id].append(websocket)
        
        logger.info(f"WebSocket connected: {user_id or 'anonymous'}")
    
    def disconnect(self, websocket: WebSocket, user_id: str = None):
        """Disconnect a WebSocket"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        if websocket in self.connection_metadata:
            del self.connection_metadata[websocket]
        
        if user_id and user_id in self.user_connections:
            if websocket in self.user_connections[user_id]:
                self.user_connections[user_id].remove(websocket)
                if not self.user_connections[user_id]:
                    del self.user_connections[user_id]
        
        logger.info(f"WebSocket disconnected: {user_id or 'anonymous'}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific WebSocket"""
        try:
            await websocket.send_text(message)
            if websocket in self.connection_metadata:
                self.connection_metadata[websocket]["message_count"] += 1
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
    
    async def send_to_user(self, message: str, user_id: str):
        """Send message to all connections for a user"""
        if user_id in self.user_connections:
            for connection in self.user_connections[user_id]:
                await self.send_personal_message(message, connection)
    
    async def broadcast(self, message: str):
        """Broadcast message to all connections"""
        for connection in self.active_connections:
            await self.send_personal_message(message, connection)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            "total_connections": len(self.active_connections),
            "unique_users": len(self.user_connections),
            "connections_by_user": {user: len(connections) for user, connections in self.user_connections.items()}
        }

# Global instances
manager = ConnectionManager()
bul_engine: Optional[BULEngine] = None
agent_manager: Optional[SMEAgentManager] = None
cache_manager = None
rate_limiter = None
auth_manager = None
health_checker = None

# Performance tracking
request_count = 0
start_time = time.time()

# Enhanced dependency functions
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False))) -> Dict[str, Any]:
    """Get current user with enhanced authentication"""
    if not credentials:
        return {"user_id": "anonymous", "role": "guest", "permissions": ["read"]}
    
    # In production, validate JWT token
    try:
        # This would validate the actual JWT token
        user_data = await auth_manager.validate_token(credentials.credentials)
        return user_data
    except Exception as e:
        logger.warning(f"Authentication failed: {e}")
        return {"user_id": "anonymous", "role": "guest", "permissions": ["read"]}

async def get_engine_dependency() -> BULEngine:
    """Dependency to get the BUL engine with error handling"""
    global bul_engine
    if not bul_engine or not bul_engine.is_initialized:
        try:
            bul_engine = await get_global_bul_engine()
        except Exception as e:
            logger.error(f"Failed to initialize BUL engine: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Document generation service is temporarily unavailable"
            )
    return bul_engine

async def get_agent_manager_dependency() -> SMEAgentManager:
    """Dependency to get the agent manager with error handling"""
    global agent_manager
    if not agent_manager or not agent_manager.is_initialized:
        try:
            agent_manager = await get_global_agent_manager()
        except Exception as e:
            logger.error(f"Failed to initialize agent manager: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Agent management service is temporarily unavailable"
            )
    return agent_manager

async def get_cache_dependency():
    """Dependency to get cache manager"""
    global cache_manager
    if not cache_manager:
        cache_manager = get_cache_manager()
    return cache_manager

async def get_rate_limiter_dependency():
    """Dependency to get rate limiter"""
    global rate_limiter
    if not rate_limiter:
        rate_limiter = get_rate_limiter()
    return rate_limiter

# Application lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced application lifespan management"""
    global bul_engine, agent_manager, cache_manager, rate_limiter, auth_manager, health_checker
    
    # Startup
    logger.info("Starting BUL Enhanced API...")
    
    try:
        # Initialize core services
        cache_manager = get_cache_manager()
        rate_limiter = get_rate_limiter()
        auth_manager = get_auth_manager()
        health_checker = get_health_checker()
        
        # Initialize BUL engine
        bul_engine = await get_global_bul_engine()
        
        # Initialize agent manager
        agent_manager = await get_global_agent_manager()
        
        logger.info("BUL Enhanced API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start BUL Enhanced API: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down BUL Enhanced API...")
    
    try:
        if bul_engine:
            await bul_engine.close()
        if agent_manager:
            await agent_manager.close()
        if cache_manager:
            await cache_manager.close()
        
        logger.info("BUL Enhanced API shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# Create enhanced FastAPI app
app = FastAPI(
    title="BUL Enhanced API - Business Universal Language",
    description="Advanced document generation platform for SMEs with AI-powered agents, comprehensive monitoring, and enterprise-grade features",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Get configuration
config = get_config()

# Add enhanced middleware stack
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=config.server.allowed_hosts
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.server.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["*"],
    max_age=3600
)

if config.server.enable_compression:
    app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add custom middleware
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(PerformanceMiddleware)
app.add_middleware(SecurityMiddleware)
app.add_middleware(ErrorHandlingMiddleware)

# Enhanced OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="BUL Enhanced API",
        version="3.0.0",
        description="Advanced document generation platform for SMEs",
        routes=app.routes,
    )
    
    # Add custom schema information
    openapi_schema["info"]["x-logo"] = {
        "url": "https://bul-system.com/logo.png"
    }
    
    openapi_schema["info"]["contact"] = {
        "name": "BUL Support",
        "email": "support@bul-system.com",
        "url": "https://bul-system.com/support"
    }
    
    openapi_schema["info"]["license"] = {
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Enhanced root endpoint
@app.get("/", response_model=Dict[str, Any])
async def root():
    """Enhanced root endpoint with comprehensive system information"""
    return {
        "message": "BUL Enhanced API - Business Universal Language",
        "version": "3.0.0",
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc",
        "openapi": "/openapi.json",
        "features": [
            "AI-Powered Document Generation",
            "Intelligent Agent Management", 
            "Multi-format Export",
            "Real-time Processing",
            "Advanced Caching",
            "Comprehensive Monitoring",
            "Enterprise Security",
            "Rate Limiting",
            "WebSocket Support",
            "Batch Processing"
        ],
        "capabilities": {
            "languages": ["es", "en", "pt", "fr", "de", "it", "ru", "zh", "ja"],
            "formats": ["markdown", "html", "pdf", "docx", "txt"],
            "business_areas": [area.value for area in BusinessArea],
            "document_types": [doc_type.value for doc_type in DocumentType],
            "styles": [style.value for style in DocumentStyle],
            "priorities": [priority.value for priority in ProcessingPriority]
        },
        "performance": {
            "uptime": time.time() - start_time,
            "total_requests": request_count,
            "cache_enabled": config.cache.enabled,
            "compression_enabled": config.server.enable_compression
        }
    }

# Enhanced health check endpoint
@app.get("/health", response_model=SystemHealthModel)
async def health_check():
    """Comprehensive health check endpoint with detailed metrics"""
    global bul_engine, agent_manager, health_checker
    
    # Basic health check
    components = {
        "engine": {
            "status": "healthy" if bul_engine and bul_engine.is_initialized else "unhealthy",
            "initialized": bul_engine.is_initialized if bul_engine else False
        },
        "agent_manager": {
            "status": "healthy" if agent_manager and agent_manager.is_initialized else "unhealthy",
            "initialized": agent_manager.is_initialized if agent_manager else False
        },
        "cache": {
            "status": "healthy" if cache_manager else "unhealthy",
            "enabled": config.cache.enabled
        },
        "rate_limiter": {
            "status": "healthy" if rate_limiter else "unhealthy"
        }
    }
    
    # Calculate uptime
    uptime = time.time() - start_time
    
    # Get basic metrics
    metrics = {
        "total_requests": request_count,
        "uptime_seconds": uptime,
        "requests_per_minute": request_count / (uptime / 60) if uptime > 0 else 0,
        "memory_usage": "N/A"  # Could add psutil here
    }
    
    # Performance metrics
    performance = {
        "avg_response_time": 0.0,  # Would be calculated from actual metrics
        "cache_hit_rate": 0.0,
        "error_rate": 0.0
    }
    
    # Dependencies check
    dependencies = {
        "openrouter": "healthy",  # Would check actual API
        "database": "healthy",
        "cache_backend": "healthy"
    }
    
    # Run comprehensive health checks if available
    try:
        if health_checker:
            health_summary = health_checker.get_health_summary()
            components.update(health_summary.get("components", {}))
            metrics.update(health_summary.get("metrics", {}))
            performance.update(health_summary.get("performance", {}))
            dependencies.update(health_summary.get("dependencies", {}))
    except Exception as e:
        logger.warning(f"Health check failed: {e}")
        components["health_check"] = {"status": "error", "error": str(e)}
    
    # Calculate overall health
    healthy_components = sum(1 for c in components.values() if c.get("status") == "healthy")
    total_components = len(components)
    health_percentage = (healthy_components / total_components) * 100 if total_components > 0 else 0
    
    # Determine status
    if health_percentage >= 90:
        status = APIStatus.HEALTHY
    elif health_percentage >= 70:
        status = APIStatus.DEGRADED
    else:
        status = APIStatus.UNHEALTHY
    
    return SystemHealthModel(
        status=status,
        timestamp=datetime.now(),
        version="3.0.0",
        uptime=uptime,
        components=components,
        metrics=metrics,
        performance=performance,
        dependencies=dependencies
    )

# Enhanced document generation endpoint
@app.post("/generate", response_model=DocumentResponseModel)
async def generate_document(
    request: DocumentRequestModel,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    engine: BULEngine = Depends(get_engine_dependency),
    agent_mgr: SMEAgentManager = Depends(get_agent_manager_dependency),
    cache: Any = Depends(get_cache_dependency),
    rate_limiter: Any = Depends(get_rate_limiter_dependency)
):
    """Generate a business document with enhanced error handling and performance tracking"""
    global request_count
    request_count += 1
    
    # Rate limiting
    client_id = current_user.get('user_id', 'anonymous')
    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later."
        )
    
    # Log API call
    await log_api_call(
        endpoint="/generate",
        method="POST",
        user_id=client_id,
        request_data=request.dict()
    )
    
    try:
        # Convert API model to internal model
        business_area = None
        if request.business_area:
            try:
                business_area = BusinessArea(request.business_area.lower())
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid business area: {request.business_area}"
                )
        
        document_type = None
        if request.document_type:
            try:
                document_type = DocumentType(request.document_type.lower())
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid document type: {request.document_type}"
                )
        
        # Create document request
        doc_request = DocumentRequest(
            query=request.query,
            business_area=business_area or BusinessArea.STRATEGY,
            document_type=document_type or DocumentType.BUSINESS_PLAN,
            company_name=request.company_name or "",
            industry=request.industry or "",
            company_size=request.company_size or "",
            target_audience=request.target_audience or "",
            language=request.language,
            format=request.format.value
        )
        
        # Get best agent for the request
        best_agent = await agent_mgr.get_best_agent(doc_request)
        
        # Generate document with performance monitoring
        start_time = time.time()
        response = await engine.generate_document(doc_request)
        processing_time = time.time() - start_time
        
        # Add agent information to response
        agent_name = best_agent.name if best_agent else "Default Agent"
        
        # Calculate quality metrics
        quality_score = _calculate_quality_score(response.content)
        readability_score = _calculate_readability_score(response.content)
        
        # Convert to API response model
        api_response = DocumentResponseModel(
            id=response.id,
            request_id=response.request_id,
            content=response.content,
            title=response.title,
            summary=response.summary,
            business_area=response.business_area.value,
            document_type=response.document_type.value,
            word_count=response.word_count,
            processing_time=processing_time,
            confidence_score=response.confidence_score,
            created_at=response.created_at,
            agent_used=agent_name,
            format=request.format.value,
            style=request.style.value,
            metadata=response.metadata,
            quality_score=quality_score,
            readability_score=readability_score
        )
        
        # Background tasks
        background_tasks.add_task(
            _log_document_generation,
            doc_request,
            response,
            agent_name,
            processing_time
        )
        
        return api_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during document generation"
        )

# Enhanced batch document generation
@app.post("/generate/batch", response_model=List[DocumentResponseModel])
async def generate_documents_batch(
    request: BatchDocumentRequestModel,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    engine: BULEngine = Depends(get_engine_dependency),
    agent_mgr: SMEAgentManager = Depends(get_agent_manager_dependency)
):
    """Generate multiple documents in batch with enhanced error handling"""
    global request_count
    request_count += len(request.requests)
    
    if not engine or not agent_mgr:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="System not initialized"
        )
    
    try:
        responses = []
        
        if request.parallel:
            # Process in parallel with concurrency limit
            semaphore = asyncio.Semaphore(request.max_concurrent)
            
            async def process_single_document(doc_request: DocumentRequestModel):
                async with semaphore:
                    return await _generate_single_document(
                        doc_request, engine, agent_mgr
                    )
            
            tasks = [process_single_document(req) for req in request.requests]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    responses.append({
                        "error": str(result),
                        "request_index": i,
                        "success": False
                    })
                else:
                    responses.append(result)
        else:
            # Process sequentially
            for i, doc_request in enumerate(request.requests):
                try:
                    response = await _generate_single_document(
                        doc_request, engine, agent_mgr
                    )
                    responses.append(response)
                except Exception as e:
                    responses.append({
                        "error": str(e),
                        "request_index": i,
                        "success": False
                    })
        
        # Background task
        background_tasks.add_task(
            _log_batch_generation,
            request,
            responses,
            current_user
        )
        
        return responses
        
    except Exception as e:
        logger.error(f"Error generating batch documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during batch generation"
        )

# Helper functions
async def _generate_single_document(
    request: DocumentRequestModel,
    engine: BULEngine,
    agent_mgr: SMEAgentManager
) -> DocumentResponseModel:
    """Generate a single document with error handling"""
    # Convert request
    business_area = BusinessArea(request.business_area.lower()) if request.business_area else BusinessArea.STRATEGY
    document_type = DocumentType(request.document_type.lower()) if request.document_type else DocumentType.BUSINESS_PLAN
    
    doc_request = DocumentRequest(
        query=request.query,
        business_area=business_area,
        document_type=document_type,
        company_name=request.company_name or "",
        industry=request.industry or "",
        company_size=request.company_size or "",
        target_audience=request.target_audience or "",
        language=request.language,
        format=request.format.value
    )
    
    # Get best agent and generate
    best_agent = await agent_mgr.get_best_agent(doc_request)
    response = await engine.generate_document(doc_request)
    
    # Calculate quality metrics
    quality_score = _calculate_quality_score(response.content)
    readability_score = _calculate_readability_score(response.content)
    
    return DocumentResponseModel(
        id=response.id,
        request_id=response.request_id,
        content=response.content,
        title=response.title,
        summary=response.summary,
        business_area=response.business_area.value,
        document_type=response.document_type.value,
        word_count=response.word_count,
        processing_time=response.processing_time,
        confidence_score=response.confidence_score,
        created_at=response.created_at,
        agent_used=best_agent.name if best_agent else "Default Agent",
        format=request.format.value,
        style=request.style.value,
        metadata=response.metadata,
        quality_score=quality_score,
        readability_score=readability_score
    )

def _calculate_quality_score(content: str) -> float:
    """Calculate quality score for content"""
    # Simple quality metrics
    word_count = len(content.split())
    sentence_count = len([s for s in content.split('.') if s.strip()])
    
    if word_count == 0 or sentence_count == 0:
        return 0.0
    
    avg_sentence_length = word_count / sentence_count
    
    # Quality score based on content length and structure
    if avg_sentence_length > 20:
        return min(0.9, 0.5 + (word_count / 1000) * 0.4)
    else:
        return min(0.8, 0.3 + (word_count / 1000) * 0.5)

def _calculate_readability_score(content: str) -> float:
    """Calculate readability score for content"""
    # Simple readability metrics
    words = content.split()
    sentences = [s for s in content.split('.') if s.strip()]
    
    if not words or not sentences:
        return 0.0
    
    avg_words_per_sentence = len(words) / len(sentences)
    
    # Simple readability score (higher is better)
    if avg_words_per_sentence <= 15:
        return 0.9
    elif avg_words_per_sentence <= 20:
        return 0.7
    else:
        return 0.5

# Background task functions
async def _log_document_generation(
    request: DocumentRequest,
    response: DocumentResponse,
    agent_name: str,
    processing_time: float
):
    """Background task to log document generation"""
    logger.info(
        f"Document generated - ID: {response.id}, "
        f"Agent: {agent_name}, "
        f"Processing time: {processing_time:.2f}s, "
        f"Word count: {response.word_count}"
    )

async def _log_batch_generation(
    request: BatchDocumentRequestModel,
    results: List[Any],
    user: dict
):
    """Background task to log batch generation"""
    success_count = sum(1 for r in results if hasattr(r, 'id') or r.get("success", False))
    logger.info(
        f"Batch generation completed - "
        f"{success_count}/{len(results)} successful, "
        f"User: {user.get('user_id', 'anonymous')}"
    )

# Enhanced error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """Enhanced 404 handler"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "detail": f"The requested endpoint {request.url.path} was not found",
            "timestamp": datetime.now().isoformat(),
            "available_endpoints": [
                "/", "/health", "/generate", "/generate/batch",
                "/business-areas", "/document-types", "/agents",
                "/agents/stats", "/ws/{user_id}", "/docs"
            ]
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: HTTPException):
    """Enhanced 500 handler"""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat(),
            "request_id": getattr(request.state, 'request_id', None)
        }
    )

@app.exception_handler(429)
async def rate_limit_handler(request: Request, exc: HTTPException):
    """Enhanced rate limit handler"""
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "detail": "Too many requests. Please try again later.",
            "timestamp": datetime.now().isoformat(),
            "retry_after": 60
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)