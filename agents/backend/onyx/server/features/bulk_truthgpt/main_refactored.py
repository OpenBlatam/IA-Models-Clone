"""
Bulk TruthGPT Main Application - Refactored
==========================================

FastAPI application for continuous document generation using TruthGPT architecture.
Refactored for improved architecture, performance, and maintainability.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from contextlib import asynccontextmanager

# Core imports
from .core.base import BaseComponent, BaseService, BaseEngine, BaseManager, component_registry
from .core.truthgpt_engine import TruthGPTEngine
from .core.document_generator import DocumentGenerator
from .core.optimization_core import OptimizationCore
from .core.knowledge_base import KnowledgeBase
from .core.prompt_optimizer import PromptOptimizer
from .core.content_analyzer import ContentAnalyzer

# Service imports
from .services.queue_manager import QueueManager
from .services.monitor import SystemMonitor
from .services.notification_service import NotificationService
from .services.analytics_service import AnalyticsService

# Utility imports
from .utils.logging import setup_logger, LogContext
from .utils.exceptions import (
    BulkTruthGPTException, 
    fastapi_error_handler, 
    create_error_context
)
from .utils.metrics import metrics_collector, metrics_context
from .utils.template_engine import TemplateEngine
from .utils.format_converter import FormatConverter
from .utils.optimization_engine import OptimizationEngine
from .utils.learning_system import LearningSystem

# Configuration
from .config.settings import settings

# Models
from .models.schemas import (
    BulkGenerationRequest,
    BulkGenerationResponse,
    DocumentStatus,
    GenerationConfig,
    TruthGPTConfig
)

# Setup logging
logger = setup_logger(__name__)

# Global component instances
components = {}

async def initialize_components():
    """Initialize all system components."""
    try:
        logger.info("Initializing Bulk TruthGPT System components...")
        
        # Initialize core components
        components['truthgpt_engine'] = TruthGPTEngine()
        await components['truthgpt_engine'].initialize()
        component_registry.register(components['truthgpt_engine'])
        
        components['document_generator'] = DocumentGenerator(components['truthgpt_engine'])
        await components['document_generator'].initialize()
        component_registry.register(components['document_generator'])
        
        components['optimization_core'] = OptimizationCore()
        await components['optimization_core'].initialize()
        component_registry.register(components['optimization_core'])
        
        # Initialize advanced components
        components['knowledge_base'] = KnowledgeBase()
        await components['knowledge_base'].initialize()
        component_registry.register(components['knowledge_base'])
        
        components['prompt_optimizer'] = PromptOptimizer()
        await components['prompt_optimizer'].initialize()
        component_registry.register(components['prompt_optimizer'])
        
        components['content_analyzer'] = ContentAnalyzer()
        await components['content_analyzer'].initialize()
        component_registry.register(components['content_analyzer'])
        
        # Initialize services
        components['queue_manager'] = QueueManager()
        await components['queue_manager'].initialize()
        component_registry.register(components['queue_manager'])
        
        components['system_monitor'] = SystemMonitor()
        await components['system_monitor'].initialize()
        component_registry.register(components['system_monitor'])
        
        components['notification_service'] = NotificationService()
        await components['notification_service'].initialize()
        component_registry.register(components['notification_service'])
        
        components['analytics_service'] = AnalyticsService()
        await components['analytics_service'].initialize()
        component_registry.register(components['analytics_service'])
        
        # Initialize utilities
        components['template_engine'] = TemplateEngine()
        await components['template_engine'].initialize()
        component_registry.register(components['template_engine'])
        
        components['format_converter'] = FormatConverter()
        await components['format_converter'].initialize()
        component_registry.register(components['format_converter'])
        
        await metrics_collector.start()
        component_registry.register(metrics_collector)
        
        components['optimization_engine'] = OptimizationEngine()
        await components['optimization_engine'].initialize()
        component_registry.register(components['optimization_engine'])
        
        components['learning_system'] = LearningSystem()
        await components['learning_system'].initialize()
        component_registry.register(components['learning_system'])
        
        logger.info("All components initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {str(e)}")
        raise

async def cleanup_components():
    """Cleanup all system components."""
    try:
        logger.info("Cleaning up Bulk TruthGPT System components...")
        
        # Cleanup in reverse order
        for component_name in reversed(list(components.keys())):
            if component_name in components:
                try:
                    await components[component_name].cleanup()
                    logger.info(f"Cleaned up component: {component_name}")
                except Exception as e:
                    logger.error(f"Failed to cleanup component {component_name}: {str(e)}")
        
        # Cleanup metrics collector
        await metrics_collector.stop()
        
        # Cleanup component registry
        await component_registry.cleanup_all()
        
        logger.info("All components cleaned up successfully")
        
    except Exception as e:
        logger.error(f"Failed to cleanup components: {str(e)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global components
    
    logger.info("Starting Bulk TruthGPT System...")
    
    try:
        # Validate configuration
        if not settings.validate_config():
            raise Exception("Configuration validation failed")
        
        # Initialize components
        await initialize_components()
        
        logger.info("Bulk TruthGPT System started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start system: {str(e)}")
        raise
    finally:
        logger.info("Shutting down Bulk TruthGPT System...")
        
        # Cleanup components
        await cleanup_components()
        
        logger.info("Bulk TruthGPT System shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Bulk TruthGPT System",
    description="Continuous document generation system based on TruthGPT architecture",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add exception handlers
@app.exception_handler(BulkTruthGPTException)
async def bulk_truthgpt_exception_handler(request: Request, exc: BulkTruthGPTException):
    """Handle BulkTruthGPT exceptions."""
    return fastapi_error_handler.handle_bulk_truthgpt_exception(exc)

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    return fastapi_error_handler.handle_general_exception(exc)

# Middleware for request logging and metrics
@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """Middleware for request logging and metrics."""
    start_time = time.time()
    
    # Create request context
    request_id = str(uuid.uuid4())
    context = LogContext(
        component="api",
        operation="request",
        request_id=request_id
    )
    
    # Log request
    logger.info(f"Request started: {request.method} {request.url}")
    
    try:
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Record metrics
        metrics_collector.record_request(
            method=request.method,
            endpoint=str(request.url.path),
            status=response.status_code,
            duration=duration
        )
        
        # Log response
        logger.info(f"Request completed: {request.method} {request.url} - {response.status_code} ({duration:.3f}s)")
        
        return response
        
    except Exception as e:
        # Calculate duration
        duration = time.time() - start_time
        
        # Record error metrics
        metrics_collector.record_error(
            error_type=type(e).__name__,
            component="api"
        )
        
        # Log error
        logger.error(f"Request failed: {request.method} {request.url} - {str(e)} ({duration:.3f}s)")
        
        raise

# Health check endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check component health
        healthy_components = component_registry.get_healthy_components()
        total_components = len(component_registry.get_all())
        
        health_status = {
            "status": "healthy" if len(healthy_components) == total_components else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                name: component.get_info().__dict__ 
                for name, component in healthy_components.items()
            },
            "total_components": total_components,
            "healthy_components": len(healthy_components)
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.get("/health/database")
async def database_health():
    """Database health check."""
    try:
        # This would check database connectivity
        return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Database health check failed")

@app.get("/health/redis")
async def redis_health():
    """Redis health check."""
    try:
        # This would check Redis connectivity
        return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"Redis health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Redis health check failed")

@app.get("/health/model")
async def model_health():
    """Model health check."""
    try:
        # Check if TruthGPT model is loaded and ready
        if 'truthgpt_engine' in components:
            model_status = components['truthgpt_engine'].is_healthy()
            return {
                "status": "healthy" if model_status else "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "model_loaded": model_status
            }
        else:
            return {"status": "unhealthy", "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"Model health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Model health check failed")

# Metrics endpoints
@app.get("/metrics")
async def get_metrics():
    """Get Prometheus metrics."""
    try:
        from fastapi.responses import PlainTextResponse
        metrics_data = metrics_collector.get_prometheus_metrics()
        return PlainTextResponse(metrics_data, media_type="text/plain")
    except Exception as e:
        logger.error(f"Failed to get metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get metrics")

@app.get("/api/metrics/summary")
async def get_metrics_summary():
    """Get metrics summary."""
    try:
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics_collector.get_all_metrics(),
            "error_stats": metrics_collector.get_error_stats()
        }
    except Exception as e:
        logger.error(f"Failed to get metrics summary: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get metrics summary")

# API endpoints
@app.post("/api/generate/bulk", response_model=BulkGenerationResponse)
async def generate_bulk_documents(request: BulkGenerationRequest):
    """Generate multiple documents from a single query."""
    try:
        with metrics_context(metrics_collector, "bulk_generation", 
                           query=request.query, max_documents=request.max_documents):
            
            # Validate request
            if not request.query or not request.query.strip():
                raise HTTPException(status_code=400, detail="Query cannot be empty")
            
            if request.max_documents <= 0:
                raise HTTPException(status_code=400, detail="Max documents must be positive")
            
            if request.max_documents > settings.max_documents_per_task:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Max documents cannot exceed {settings.max_documents_per_task}"
                )
            
            # Start generation task
            if 'queue_manager' not in components:
                raise HTTPException(status_code=500, detail="Queue manager not available")
            
            task_id = await components['queue_manager'].start_bulk_generation(
                query=request.query,
                max_documents=request.max_documents,
                config=request.config
            )
            
            # Record metrics
            metrics_collector.record_generation(
                task_id=task_id,
                status="started",
                duration=0.0,
                quality=0.0
            )
            
            return BulkGenerationResponse(
                task_id=task_id,
                status="started",
                message="Bulk generation task started successfully"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start bulk generation: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start bulk generation")

@app.get("/api/tasks/{task_id}/status")
async def get_task_status(task_id: str):
    """Get task status."""
    try:
        if 'queue_manager' not in components:
            raise HTTPException(status_code=500, detail="Queue manager not available")
        
        status = await components['queue_manager'].get_task_status(task_id)
        if not status:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get task status")

@app.get("/api/tasks/{task_id}/results")
async def get_task_results(task_id: str):
    """Get task results."""
    try:
        if 'queue_manager' not in components:
            raise HTTPException(status_code=500, detail="Queue manager not available")
        
        results = await components['queue_manager'].get_task_results(task_id)
        if not results:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task results: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get task results")

@app.post("/api/analyze/quality")
async def analyze_content_quality(request: dict):
    """Analyze content quality."""
    try:
        if 'content_analyzer' not in components:
            raise HTTPException(status_code=500, detail="Content analyzer not available")
        
        content = request.get('content', '')
        if not content:
            raise HTTPException(status_code=400, detail="Content cannot be empty")
        
        analysis = await components['content_analyzer'].analyze_content(content)
        
        return {
            "quality_score": analysis.quality_score,
            "readability_score": analysis.readability_score,
            "coherence_score": analysis.coherence_score,
            "engagement_score": analysis.engagement_score,
            "metrics": analysis.metrics,
            "suggestions": analysis.suggestions
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to analyze content quality: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to analyze content quality")

@app.get("/api/system/status")
async def get_system_status():
    """Get system status."""
    try:
        if 'system_monitor' not in components:
            raise HTTPException(status_code=500, detail="System monitor not available")
        
        status = await components['system_monitor'].get_system_status()
        return status
        
    except Exception as e:
        logger.error(f"Failed to get system status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get system status")

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        log_level=settings.log_level.lower(),
        reload=settings.debug
    )











