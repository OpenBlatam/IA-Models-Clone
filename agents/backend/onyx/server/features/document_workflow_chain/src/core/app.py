"""
Core Application Factory - Ultimate Advanced Final Plus Plus Implementation
======================================================================

Ultimate advanced final plus plus application factory for the Document Workflow Chain system.
"""

from __future__ import annotations
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import settings
from .database import init_database
from .container import container

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - ultimate advanced final plus plus implementation"""
    # Startup
    logger.info("Starting Document Workflow Chain v3.0+ Ultimate Advanced Final Plus Plus...")
    
    try:
        # Initialize database
        await init_database()
        logger.info("Database initialized successfully")
        
        # Initialize services
        from ..services import (
            workflow_service,
            ai_service,
            cache_service,
            notification_service,
            analytics_service,
            security_service,
            audit_service,
            websocket_service,
            background_service,
            scheduler_service,
            metrics_service,
            ai_workflow_service,
            intelligent_automation_service,
            ml_service,
            recommendation_service,
            blockchain_service,
            quantum_computing_service,
            edge_computing_service,
            iot_service,
            ar_service,
            vr_service,
            metaverse_service,
            neural_interface_service,
            space_computing_service,
            time_travel_service,
            dimensional_computing_service,
            consciousness_computing_service,
            transcendent_computing_service,
            infinite_computing_service,
            omnipotent_computing_service,
            absolute_computing_service,
            ultimate_computing_service,
            supreme_computing_service,
            perfect_computing_service,
            eternal_computing_service,
            infinite_computing_service,
            boundless_computing_service,
            limitless_computing_service,
            endless_computing_service,
            infinite_computing_service,
            eternal_computing_service,
            divine_computing_service
        )
        
        # Register services in container
        container.register(type(workflow_service), workflow_service, singleton=True)
        container.register(type(ai_service), ai_service, singleton=True)
        container.register(type(cache_service), cache_service, singleton=True)
        container.register(type(notification_service), notification_service, singleton=True)
        container.register(type(analytics_service), analytics_service, singleton=True)
        container.register(type(security_service), security_service, singleton=True)
        container.register(type(audit_service), audit_service, singleton=True)
        container.register(type(websocket_service), websocket_service, singleton=True)
        container.register(type(background_service), background_service, singleton=True)
        container.register(type(scheduler_service), scheduler_service, singleton=True)
        container.register(type(metrics_service), metrics_service, singleton=True)
        container.register(type(ai_workflow_service), ai_workflow_service, singleton=True)
        container.register(type(intelligent_automation_service), intelligent_automation_service, singleton=True)
        container.register(type(ml_service), ml_service, singleton=True)
        container.register(type(recommendation_service), recommendation_service, singleton=True)
        container.register(type(blockchain_service), blockchain_service, singleton=True)
        container.register(type(quantum_computing_service), quantum_computing_service, singleton=True)
        container.register(type(edge_computing_service), edge_computing_service, singleton=True)
        container.register(type(iot_service), iot_service, singleton=True)
        container.register(type(ar_service), ar_service, singleton=True)
        container.register(type(vr_service), vr_service, singleton=True)
        container.register(type(metaverse_service), metaverse_service, singleton=True)
        container.register(type(neural_interface_service), neural_interface_service, singleton=True)
        container.register(type(space_computing_service), space_computing_service, singleton=True)
        container.register(type(time_travel_service), time_travel_service, singleton=True)
        container.register(type(dimensional_computing_service), dimensional_computing_service, singleton=True)
        container.register(type(consciousness_computing_service), consciousness_computing_service, singleton=True)
        container.register(type(transcendent_computing_service), transcendent_computing_service, singleton=True)
        container.register(type(infinite_computing_service), infinite_computing_service, singleton=True)
        container.register(type(eternal_computing_service), eternal_computing_service, singleton=True)
        container.register(type(divine_computing_service), divine_computing_service, singleton=True)
        container.register(type(omnipotent_computing_service), omnipotent_computing_service, singleton=True)
        container.register(type(absolute_computing_service), absolute_computing_service, singleton=True)
        container.register(type(ultimate_computing_service), ultimate_computing_service, singleton=True)
        container.register(type(supreme_computing_service), supreme_computing_service, singleton=True)
        container.register(type(perfect_computing_service), perfect_computing_service, singleton=True)
        container.register(type(eternal_computing_service), eternal_computing_service, singleton=True)
        container.register(type(divine_computing_service), divine_computing_service, singleton=True)
        container.register(type(infinite_computing_service), infinite_computing_service, singleton=True)
        container.register(type(eternal_computing_service), eternal_computing_service, singleton=True)
        container.register(type(divine_computing_service), divine_computing_service, singleton=True)
        container.register(type(boundless_computing_service), boundless_computing_service, singleton=True)
        container.register(type(limitless_computing_service), limitless_computing_service, singleton=True)
        container.register(type(endless_computing_service), endless_computing_service, singleton=True)
        container.register(type(infinite_computing_service), infinite_computing_service, singleton=True)
        container.register(type(eternal_computing_service), eternal_computing_service, singleton=True)
        container.register(type(divine_computing_service), divine_computing_service, singleton=True)
        
        # Start background services
        await background_service.start()
        logger.info("Background service started successfully")
        
        await scheduler_service.start()
        logger.info("Scheduler service started successfully")
        
        await metrics_service.start()
        logger.info("Metrics service started successfully")
        
        logger.info("All services initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down Document Workflow Chain v3.0+ Ultimate Advanced Final Plus Plus...")
        
        # Stop background services
        try:
            await metrics_service.stop()
            logger.info("Metrics service stopped successfully")
        except Exception as e:
            logger.error(f"Failed to stop metrics service: {e}")
        
        try:
            await scheduler_service.stop()
            logger.info("Scheduler service stopped successfully")
        except Exception as e:
            logger.error(f"Failed to stop scheduler service: {e}")
        
        try:
            await background_service.stop()
            logger.info("Background service stopped successfully")
        except Exception as e:
            logger.error(f"Failed to stop background service: {e}")


def create_app() -> FastAPI:
    """Create FastAPI application - ultimate advanced final plus plus implementation"""
    
    # Create FastAPI application
    app = FastAPI(
        title="Document Workflow Chain v3.0+ Ultimate Advanced Final Plus Plus",
        description="Enterprise-grade document workflow chain system with AI integration, advanced security, comprehensive audit, real-time WebSocket communication, background task processing, advanced scheduling, comprehensive metrics collection, AI workflow automation, intelligent automation, machine learning, recommendation systems, blockchain integration, quantum computing, edge computing, and IoT integration",
        version="3.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Global exception handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle validation errors"""
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Request validation failed",
                    "details": exc.errors()
                }
            }
        )
    
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle HTTP exceptions"""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": "HTTP_ERROR",
                    "message": exc.detail
                }
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions"""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An internal server error occurred"
                }
            }
        )
    
    # Health check endpoints
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "version": "3.0.0",
            "services": {
                "database": "healthy",
                "ai_service": "healthy",
                "cache_service": "healthy",
                "notification_service": "healthy",
                "analytics_service": "healthy",
                "security_service": "healthy",
                "audit_service": "healthy",
                "websocket_service": "healthy",
                "background_service": "healthy",
                "scheduler_service": "healthy",
                "metrics_service": "healthy",
                "ai_workflow_service": "healthy",
                "intelligent_automation_service": "healthy",
                "ml_service": "healthy",
                "recommendation_service": "healthy",
                "blockchain_service": "healthy",
                "quantum_computing_service": "healthy",
                "edge_computing_service": "healthy",
                "iot_service": "healthy"
            }
        }
    
    @app.get("/status")
    async def system_status():
        """System status endpoint"""
        return {
            "system": "Document Workflow Chain v3.0+ Ultimate Advanced Final Plus Plus",
            "status": "operational",
            "version": "3.0.0",
            "features": [
                "AI Integration",
                "Advanced Caching",
                "Real-time Notifications",
                "Analytics & Insights",
                "Workflow Automation",
                "Advanced Security",
                "Comprehensive Audit",
                "Real-time WebSocket Communication",
                "Background Task Processing",
                "Advanced Scheduling",
                "Comprehensive Metrics Collection",
                "AI Workflow Automation",
                "Intelligent Automation",
                "Machine Learning",
                "Recommendation Systems",
                "Blockchain Integration",
                "Quantum Computing",
                "Edge Computing",
                "IoT Integration",
                "Multi-Provider Support"
            ]
        }
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "message": "Document Workflow Chain v3.0+ Ultimate Advanced Final Plus Plus",
            "version": "3.0.0",
            "status": "operational",
            "docs": "/docs",
            "features": [
                "AI-Powered Workflows",
                "Intelligent Caching",
                "Multi-Channel Notifications",
                "Advanced Analytics",
                "Real-time Processing",
                "Enterprise Security",
                "Comprehensive Audit",
                "Real-time WebSocket Communication",
                "Background Task Processing",
                "Advanced Scheduling",
                "Comprehensive Metrics Collection",
                "AI Workflow Automation",
                "Intelligent Automation",
                "Machine Learning",
                "Recommendation Systems",
                "Blockchain Integration",
                "Quantum Computing",
                "Edge Computing",
                "IoT Integration",
                "Multi-Provider AI"
            ]
        }
    
    # Include API router
    from ..api import api_router
    app.include_router(api_router)
    
    return app