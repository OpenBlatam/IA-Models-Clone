"""
Unified API - Refactored Architecture

This module provides a unified API for the refactored HeyGen AI system
with consolidated endpoints and optimized performance.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import asyncio
import logging
import time
from datetime import datetime, timezone
import uuid

from ..core.unified_ai_system import RefactoredHeyGenAI, AILevel

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="HeyGen AI - Refactored Unified API",
    description="Unified API for the refactored HeyGen AI system with consolidated capabilities",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Global AI system instance
ai_system: Optional[RefactoredHeyGenAI] = None

# Pydantic models
class AIRequest(BaseModel):
    """AI request model."""
    request_type: str = Field(..., description="Type of AI request")
    data: Dict[str, Any] = Field(..., description="Request data")
    system_id: Optional[str] = Field(None, description="Specific system ID to use")

class AIResponse(BaseModel):
    """AI response model."""
    success: bool
    result: Dict[str, Any]
    system_used: str
    response_time: float
    capabilities_used: int
    timestamp: datetime

class SystemStatus(BaseModel):
    """System status model."""
    total_capabilities: int
    total_systems: int
    performance_metrics: Dict[str, Any]
    capability_levels: Dict[str, Dict[str, Any]]
    system_performance: Dict[str, Dict[str, Any]]

class CapabilityRequest(BaseModel):
    """Capability creation request model."""
    name: str
    level: str
    intelligence: float = 0.0
    mastery: float = 0.0
    execution: float = 0.0
    understanding: float = 0.0
    precision: float = 0.0
    wisdom: float = 0.0
    power: float = 0.0
    authority: float = 0.0

class SystemRequest(BaseModel):
    """System creation request model."""
    name: str
    capabilities: List[str]
    configuration: Optional[Dict[str, Any]] = None

# Dependency to get AI system
async def get_ai_system() -> RefactoredHeyGenAI:
    """Get the AI system instance."""
    global ai_system
    if ai_system is None:
        ai_system = RefactoredHeyGenAI()
        await ai_system.initialize_all_capabilities()
    return ai_system

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the AI system on startup."""
    global ai_system
    ai_system = RefactoredHeyGenAI()
    await ai_system.initialize_all_capabilities()
    logger.info("HeyGen AI system initialized successfully")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "2.0.0"
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with system information."""
    return {
        "message": "HeyGen AI - Refactored Unified API",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "status": "/status",
            "ai": "/ai/process",
            "capabilities": "/capabilities",
            "systems": "/systems"
        }
    }

# System status endpoint
@app.get("/status", response_model=SystemStatus)
async def get_system_status(ai: RefactoredHeyGenAI = Depends(get_ai_system)):
    """Get comprehensive system status."""
    try:
        status = ai.get_system_status()
        return SystemStatus(**status)
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# AI processing endpoint
@app.post("/ai/process", response_model=AIResponse)
async def process_ai_request(
    request: AIRequest,
    background_tasks: BackgroundTasks,
    ai: RefactoredHeyGenAI = Depends(get_ai_system)
):
    """Process an AI request."""
    try:
        start_time = time.time()
        
        result = await ai.process_request(
            request_type=request.request_type,
            data=request.data,
            system_id=request.system_id
        )
        
        response_time = time.time() - start_time
        
        # Log request in background
        background_tasks.add_task(
            log_request,
            request.request_type,
            response_time,
            result.get('success', False)
        )
        
        return AIResponse(
            success=result.get('success', False),
            result=result.get('result', {}),
            system_used=result.get('system_used', 'Unknown'),
            response_time=response_time,
            capabilities_used=result.get('capabilities_used', 0),
            timestamp=datetime.now(timezone.utc)
        )
        
    except Exception as e:
        logger.error(f"Error processing AI request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Capabilities endpoints
@app.get("/capabilities")
async def get_capabilities(ai: RefactoredHeyGenAI = Depends(get_ai_system)):
    """Get all AI capabilities."""
    try:
        status = ai.get_system_status()
        return {
            "capabilities": status.get('capability_levels', {}),
            "total": status.get('total_capabilities', 0)
        }
    except Exception as e:
        logger.error(f"Error getting capabilities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/capabilities")
async def create_capability(
    request: CapabilityRequest,
    ai: RefactoredHeyGenAI = Depends(get_ai_system)
):
    """Create a new AI capability."""
    try:
        # Validate AI level
        try:
            level = AILevel(request.level)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid AI level: {request.level}")
        
        capability = await ai.unified_engine.create_ai_capability(
            name=request.name,
            level=level,
            intelligence=request.intelligence,
            mastery=request.mastery,
            execution=request.execution,
            understanding=request.understanding,
            precision=request.precision,
            wisdom=request.wisdom,
            power=request.power,
            authority=request.authority
        )
        
        return {
            "success": True,
            "capability_id": capability.capability_id,
            "message": f"Capability '{request.name}' created successfully"
        }
        
    except Exception as e:
        logger.error(f"Error creating capability: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Systems endpoints
@app.get("/systems")
async def get_systems(ai: RefactoredHeyGenAI = Depends(get_ai_system)):
    """Get all AI systems."""
    try:
        status = ai.get_system_status()
        return {
            "systems": status.get('system_performance', {}),
            "total": status.get('total_systems', 0)
        }
    except Exception as e:
        logger.error(f"Error getting systems: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/systems")
async def create_system(
    request: SystemRequest,
    ai: RefactoredHeyGenAI = Depends(get_ai_system)
):
    """Create a new AI system."""
    try:
        system = await ai.unified_engine.create_ai_system(
            name=request.name,
            capabilities=request.capabilities,
            configuration=request.configuration
        )
        
        return {
            "success": True,
            "system_id": system.system_id,
            "message": f"System '{request.name}' created successfully"
        }
        
    except Exception as e:
        logger.error(f"Error creating system: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Specific AI capability endpoints
@app.post("/ai/intelligence")
async def process_intelligence(
    data: Dict[str, Any],
    ai: RefactoredHeyGenAI = Depends(get_ai_system)
):
    """Process intelligence request."""
    return await process_ai_request(
        AIRequest(request_type="intelligence", data=data),
        BackgroundTasks(),
        ai
    )

@app.post("/ai/mastery")
async def process_mastery(
    data: Dict[str, Any],
    ai: RefactoredHeyGenAI = Depends(get_ai_system)
):
    """Process mastery request."""
    return await process_ai_request(
        AIRequest(request_type="mastery", data=data),
        BackgroundTasks(),
        ai
    )

@app.post("/ai/execution")
async def process_execution(
    data: Dict[str, Any],
    ai: RefactoredHeyGenAI = Depends(get_ai_system)
):
    """Process execution request."""
    return await process_ai_request(
        AIRequest(request_type="execution", data=data),
        BackgroundTasks(),
        ai
    )

@app.post("/ai/understanding")
async def process_understanding(
    data: Dict[str, Any],
    ai: RefactoredHeyGenAI = Depends(get_ai_system)
):
    """Process understanding request."""
    return await process_ai_request(
        AIRequest(request_type="understanding", data=data),
        BackgroundTasks(),
        ai
    )

@app.post("/ai/precision")
async def process_precision(
    data: Dict[str, Any],
    ai: RefactoredHeyGenAI = Depends(get_ai_system)
):
    """Process precision request."""
    return await process_ai_request(
        AIRequest(request_type="precision", data=data),
        BackgroundTasks(),
        ai
    )

@app.post("/ai/wisdom")
async def process_wisdom(
    data: Dict[str, Any],
    ai: RefactoredHeyGenAI = Depends(get_ai_system)
):
    """Process wisdom request."""
    return await process_ai_request(
        AIRequest(request_type="wisdom", data=data),
        BackgroundTasks(),
        ai
    )

@app.post("/ai/power")
async def process_power(
    data: Dict[str, Any],
    ai: RefactoredHeyGenAI = Depends(get_ai_system)
):
    """Process power request."""
    return await process_ai_request(
        AIRequest(request_type="power", data=data),
        BackgroundTasks(),
        ai
    )

@app.post("/ai/authority")
async def process_authority(
    data: Dict[str, Any],
    ai: RefactoredHeyGenAI = Depends(get_ai_system)
):
    """Process authority request."""
    return await process_ai_request(
        AIRequest(request_type="authority", data=data),
        BackgroundTasks(),
        ai
    )

# Batch processing endpoint
@app.post("/ai/batch")
async def process_batch_requests(
    requests: List[AIRequest],
    ai: RefactoredHeyGenAI = Depends(get_ai_system)
):
    """Process multiple AI requests in batch."""
    try:
        results = []
        for request in requests:
            result = await ai.process_request(
                request_type=request.request_type,
                data=request.data,
                system_id=request.system_id
            )
            results.append(result)
        
        return {
            "success": True,
            "results": results,
            "total_processed": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error processing batch requests: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Performance metrics endpoint
@app.get("/metrics")
async def get_metrics(ai: RefactoredHeyGenAI = Depends(get_ai_system)):
    """Get performance metrics."""
    try:
        status = ai.get_system_status()
        return {
            "performance_metrics": status.get('performance_metrics', {}),
            "capability_metrics": {
                cap_id: {
                    "intelligence": cap_data.get('intelligence', 0),
                    "mastery": cap_data.get('mastery', 0),
                    "execution": cap_data.get('execution', 0),
                    "understanding": cap_data.get('understanding', 0),
                    "precision": cap_data.get('precision', 0),
                    "wisdom": cap_data.get('wisdom', 0),
                    "power": cap_data.get('power', 0),
                    "authority": cap_data.get('authority', 0)
                }
                for cap_id, cap_data in status.get('capability_levels', {}).items()
            }
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background task for logging
async def log_request(request_type: str, response_time: float, success: bool):
    """Log request in background."""
    logger.info(f"Request: {request_type}, Response Time: {response_time:.3f}s, Success: {success}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "detail": str(exc)}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "An unexpected error occurred"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
