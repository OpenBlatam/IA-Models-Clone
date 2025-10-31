"""
BUL Unified API
===============

Unified, modern API that consolidates all BUL functionality into a single,
well-structured, production-ready FastAPI application.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, AsyncGenerator
import asyncio
import logging
from datetime import datetime
import uuid
import json
import time
from enum import Enum

# Import BUL modules
from ..core import BULEngine, DocumentRequest, DocumentResponse, BusinessArea, DocumentType, get_global_bul_engine
from ..agents import SMEAgentManager, get_global_agent_manager
from ..utils import get_cache_manager, get_logger, log_api_call
from ..security import get_rate_limiter, SecurityValidator
from ..monitoring.health_checker import get_health_checker
from ..config import get_config

# Configure logging
logger = get_logger(__name__)

# Security
security = HTTPBearer(auto_error=False)

# Create unified FastAPI app
app = FastAPI(
    title="BUL API - Business Universal Language",
    description="Unified document generation platform for SMEs with AI-powered agents",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Get configuration
config = get_config()

# Add security middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=config.server.allowed_hosts
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.server.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    max_age=3600
)

# Enums for better type safety
class DocumentFormat(str, Enum):
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    DOCX = "docx"

class DocumentStyle(str, Enum):
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    TECHNICAL = "technical"
    CREATIVE = "creative"

class ProcessingPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

# Enhanced Pydantic models
class DocumentRequestModel(BaseModel):
    """Enhanced document generation request model"""
    query: str = Field(..., description="Business query or requirement", min_length=10, max_length=2000)
    business_area: Optional[str] = Field(None, description="Business area (marketing, sales, operations, etc.)")
    document_type: Optional[str] = Field(None, description="Type of document to generate")
    company_name: Optional[str] = Field(None, description="Company name", max_length=100)
    industry: Optional[str] = Field(None, description="Industry sector", max_length=100)
    company_size: Optional[str] = Field(None, description="Company size (small, medium, large)")
    target_audience: Optional[str] = Field(None, description="Target audience", max_length=200)
    language: str = Field("es", description="Language for document generation")
    format: DocumentFormat = Field(DocumentFormat.MARKDOWN, description="Output format")
    style: DocumentStyle = Field(DocumentStyle.PROFESSIONAL, description="Document style")
    priority: ProcessingPriority = Field(ProcessingPriority.NORMAL, description="Processing priority")
    cache_ttl: int = Field(3600, description="Cache TTL in seconds", ge=60, le=86400)
    
    @validator('language')
    def validate_language(cls, v):
        allowed_languages = ['es', 'en', 'pt', 'fr', 'de', 'it']
        if v not in allowed_languages:
            raise ValueError(f'Language must be one of: {allowed_languages}')
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

class BatchDocumentRequestModel(BaseModel):
    """Batch document generation request model"""
    requests: List[DocumentRequestModel] = Field(..., max_items=10, description="List of document requests")
    parallel: bool = Field(True, description="Process requests in parallel")
    priority: ProcessingPriority = Field(ProcessingPriority.NORMAL, description="Overall batch priority")

class SystemHealthModel(BaseModel):
    """System health model"""
    status: str
    timestamp: datetime
    version: str
    uptime: float
    components: Dict[str, Any]
    metrics: Dict[str, Any]

class AgentStatsModel(BaseModel):
    """Agent statistics model"""
    total_agents: int
    active_agents: int
    total_documents_generated: int
    average_success_rate: float
    agent_types: List[str]
    is_initialized: bool

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.user_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, user_id: str = None):
        await websocket.accept()
        self.active_connections.append(websocket)
        if user_id:
            if user_id not in self.user_connections:
                self.user_connections[user_id] = []
            self.user_connections[user_id].append(websocket)

    def disconnect(self, websocket: WebSocket, user_id: str = None):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if user_id and user_id in self.user_connections:
            if websocket in self.user_connections[user_id]:
                self.user_connections[user_id].remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def send_to_user(self, message: str, user_id: str):
        if user_id in self.user_connections:
            for connection in self.user_connections[user_id]:
                try:
                    await connection.send_text(message)
                except:
                    pass

manager = ConnectionManager()

# Global variables for engine and agent manager
bul_engine: Optional[BULEngine] = None
agent_manager: Optional[SMEAgentManager] = None

# Performance tracking
request_count = 0
start_time = time.time()

# Dependency functions
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user (simplified for demo)"""
    # In a real implementation, validate JWT token
    return {"user_id": "demo_user", "role": "user", "permissions": ["read", "write"]}

async def get_engine_dependency() -> BULEngine:
    """Dependency to get the BUL engine"""
    global bul_engine
    if not bul_engine:
        bul_engine = await get_global_bul_engine()
    return bul_engine

async def get_agent_manager_dependency() -> SMEAgentManager:
    """Dependency to get the agent manager"""
    global agent_manager
    if not agent_manager:
        agent_manager = await get_global_agent_manager()
    return agent_manager

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup"""
    global bul_engine, agent_manager
    
    try:
        bul_engine = await get_global_bul_engine()
        agent_manager = await get_global_agent_manager()
        logger.info("BUL Unified API started successfully")
    except Exception as e:
        logger.error(f"Failed to start BUL API: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global bul_engine
    if bul_engine:
        await bul_engine.close()
    logger.info("BUL Unified API shutdown")

# Core endpoints
@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with system information"""
    return {
        "message": "BUL API - Business Universal Language",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
        "features": [
            "AI-Powered Document Generation",
            "Intelligent Agent Management",
            "Multi-format Export",
            "Real-time Processing",
            "Advanced Caching",
            "Comprehensive Monitoring"
        ],
        "capabilities": {
            "languages": ["es", "en", "pt", "fr", "de", "it"],
            "formats": ["markdown", "html", "pdf", "docx"],
            "business_areas": [area.value for area in BusinessArea],
            "document_types": [doc_type.value for doc_type in DocumentType]
        }
    }

@app.get("/health", response_model=SystemHealthModel)
async def health_check():
    """Comprehensive health check endpoint"""
    global bul_engine, agent_manager
    
    # Basic health check
    components = {
        "engine": {
            "status": "healthy" if bul_engine and bul_engine.is_initialized else "unhealthy",
            "initialized": bul_engine.is_initialized if bul_engine else False
        },
        "agent_manager": {
            "status": "healthy" if agent_manager and agent_manager.is_initialized else "unhealthy",
            "initialized": agent_manager.is_initialized if agent_manager else False
        }
    }
    
    # Calculate uptime
    uptime = time.time() - start_time
    
    # Get basic metrics
    metrics = {
        "total_requests": request_count,
        "uptime_seconds": uptime,
        "memory_usage": "N/A"  # Could add psutil here
    }
    
    # Run comprehensive health checks if APIs are available
    try:
        import os
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        
        if openrouter_key:
            health_checker = get_health_checker()
            cache_manager = get_cache_manager()
            
            # Run health checks
            checks = await health_checker.run_all_checks(
                openrouter_key, 
                openai_key, 
                cache_manager
            )
            
            # Get comprehensive health summary
            health_summary = health_checker.get_health_summary()
            components.update(health_summary.get("components", {}))
            metrics.update(health_summary.get("metrics", {}))
        
    except Exception as e:
        logger.warning(f"Health check failed: {e}")
        components["health_check"] = {"status": "error", "error": str(e)}
    
    # Calculate overall health
    healthy_components = sum(1 for c in components.values() if c.get("status") == "healthy")
    total_components = len(components)
    health_percentage = (healthy_components / total_components) * 100 if total_components > 0 else 0
    
    return SystemHealthModel(
        status="healthy" if health_percentage >= 80 else "degraded" if health_percentage >= 50 else "unhealthy",
        timestamp=datetime.now(),
        version="2.0.0",
        uptime=uptime,
        components=components,
        metrics=metrics
    )

@app.post("/generate", response_model=DocumentResponseModel)
async def generate_document(
    request: DocumentRequestModel, 
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    engine: BULEngine = Depends(get_engine_dependency),
    agent_mgr: SMEAgentManager = Depends(get_agent_manager_dependency)
):
    """Generate a business document based on the request"""
    global request_count
    request_count += 1
    
    # Rate limiting
    rate_limiter = get_rate_limiter()
    client_id = getattr(current_user, 'user_id', 'unknown')
    
    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    try:
        # Convert API model to internal model
        business_area = None
        if request.business_area:
            try:
                business_area = BusinessArea(request.business_area.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid business area: {request.business_area}")
        
        document_type = None
        if request.document_type:
            try:
                document_type = DocumentType(request.document_type.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid document type: {request.document_type}")
        
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
        
        # Generate document
        response = await engine.generate_document(doc_request)
        
        # Add agent information to response
        agent_name = best_agent.name if best_agent else "Default Agent"
        
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
            processing_time=response.processing_time,
            confidence_score=response.confidence_score,
            created_at=response.created_at,
            agent_used=agent_name,
            format=request.format.value,
            style=request.style.value,
            metadata=response.metadata
        )
        
        # Log the generation
        background_tasks.add_task(log_document_generation, doc_request, response, agent_name)
        
        return api_response
        
    except Exception as e:
        logger.error(f"Error generating document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/batch", response_model=List[DocumentResponseModel])
async def generate_documents_batch(
    request: BatchDocumentRequestModel,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    engine: BULEngine = Depends(get_engine_dependency),
    agent_mgr: SMEAgentManager = Depends(get_agent_manager_dependency)
):
    """Generate multiple documents in batch"""
    global request_count
    request_count += len(request.requests)
    
    if not engine or not agent_mgr:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        responses = []
        
        if request.parallel:
            # Process in parallel
            tasks = []
            for doc_request in request.requests:
                task = generate_single_document(
                    doc_request, engine, agent_mgr
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
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
                    response = await generate_single_document(
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
        background_tasks.add_task(log_batch_generation, request, responses, current_user)
        
        return responses
        
    except Exception as e:
        logger.error(f"Error generating batch documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def generate_single_document(
    request: DocumentRequestModel,
    engine: BULEngine,
    agent_mgr: SMEAgentManager
) -> DocumentResponseModel:
    """Generate a single document"""
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
        metadata=response.metadata
    )

# System information endpoints
@app.get("/business-areas", response_model=List[str])
async def get_business_areas():
    """Get available business areas"""
    return [area.value for area in BusinessArea]

@app.get("/document-types", response_model=List[str])
async def get_document_types():
    """Get available document types"""
    return [doc_type.value for doc_type in DocumentType]

@app.get("/agents", response_model=List[Dict[str, Any]])
async def get_agents():
    """Get all available agents"""
    global agent_manager
    
    if not agent_manager:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        agents = await agent_manager.get_all_agents()
        return [
            {
                "id": agent.id,
                "name": agent.name,
                "agent_type": agent.agent_type.value,
                "experience_years": agent.experience_years,
                "success_rate": agent.success_rate,
                "total_documents_generated": agent.total_documents_generated,
                "average_rating": agent.average_rating,
                "is_active": agent.is_active,
                "created_at": agent.created_at.isoformat(),
                "last_used": agent.last_used.isoformat()
            }
            for agent in agents
        ]
    except Exception as e:
        logger.error(f"Error getting agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents/stats", response_model=AgentStatsModel)
async def get_agent_stats():
    """Get agent statistics"""
    global agent_manager
    
    if not agent_manager:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        stats = await agent_manager.get_agent_stats()
        return AgentStatsModel(**stats)
    except Exception as e:
        logger.error(f"Error getting agent stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time updates
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket, user_id)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get("type") == "ping":
                await manager.send_personal_message(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                }), websocket)
            elif message.get("type") == "subscribe":
                # Subscribe to specific updates
                await manager.send_personal_message(json.dumps({
                    "type": "subscribed",
                    "channel": message.get("channel"),
                    "timestamp": datetime.now().isoformat()
                }), websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)

# Background task functions
async def log_document_generation(request: DocumentRequest, response: DocumentResponse, agent_name: str):
    """Background task to log document generation"""
    logger.info(f"Document generated - ID: {response.id}, Agent: {agent_name}, Processing time: {response.processing_time:.2f}s")

async def log_batch_generation(request: BatchDocumentRequestModel, results: List[Any], user: dict):
    """Background task to log batch generation"""
    success_count = sum(1 for r in results if hasattr(r, 'id') or r.get("success", False))
    logger.info(f"Batch generation completed - {success_count}/{len(results)} successful")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found", "available_endpoints": [
            "/", "/health", "/generate", "/generate/batch", 
            "/business-areas", "/document-types", "/agents", 
            "/agents/stats", "/ws/{user_id}", "/docs"
        ]}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "timestamp": datetime.now().isoformat()}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
















