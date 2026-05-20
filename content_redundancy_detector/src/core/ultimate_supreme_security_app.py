"""
Ultimate Supreme Security App - Enterprise-level content security system
Following FastAPI best practices: functional programming, RORO pattern, async operations
"""

import logging
import sys
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .content_security_engine import initialize_content_security_engine
from ..api.security_routes import router as security_router

# Import all other existing routers
from ..api.analytics_routes import router as analytics_router
from ..api.websocket_routes import router as websocket_router
from ..api.ai_routes import router as ai_router
from ..api.optimization_routes import router as optimization_router
from ..api.workflow_routes import router as workflow_router
from ..api.intelligence_routes import router as intelligence_router
from ..api.ml_routes import router as ml_router

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Setup structured logging"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("ultimate_supreme_security_app.log", encoding="utf-8")
        ]
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    setup_logging()
    logger.info("Ultimate Supreme Security App starting up...")
    
    try:
        # Initialize security engine
        await initialize_content_security_engine()
        logger.info("Content Security Engine initialized successfully")
        
        logger.info("All systems initialized successfully")
        logger.info("Server ready to handle requests")
        
    except Exception as e:
        logger.error(f"Failed to initialize systems: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Ultimate Supreme Security App shutting down...")


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    app = FastAPI(
        title="Ultimate Supreme Security Content Redundancy Detector",
        description="Enterprise-level content security, threat detection, and redundancy analysis system with advanced AI, ML, and security capabilities",
        version="2.0.0",
        debug=False,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include all routers
    app.include_router(security_router)
    app.include_router(analytics_router)
    app.include_router(websocket_router)
    app.include_router(ai_router)
    app.include_router(optimization_router)
    app.include_router(workflow_router)
    app.include_router(intelligence_router)
    app.include_router(ml_router)
    
    return app


# Create app instance
app = create_app()


# Global exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions"""
    logger.warning(f"HTTP error: {exc.status_code} - {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time()
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions"""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if app.debug else "Internal server error",
            "timestamp": time.time()
        }
    )


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Ultimate Supreme Security Content Redundancy Detector",
        "version": "2.0.0",
        "timestamp": time.time(),
        "features": [
            "Content Security Engine",
            "Advanced Analytics",
            "Real-time Processing",
            "AI Content Analysis",
            "Content Optimization",
            "Workflow Automation",
            "Content Intelligence",
            "Machine Learning",
            "Threat Detection",
            "Encryption/Decryption",
            "Compliance Monitoring",
            "Security Auditing"
        ]
    }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "Ultimate Supreme Security Content Redundancy Detector",
        "version": "2.0.0",
        "description": "Enterprise-level content security, threat detection, and redundancy analysis system",
        "documentation": "/docs",
        "health_check": "/health",
        "features": {
            "security": {
                "threat_detection": "Detect SQL injection, XSS, path traversal, command injection, and malicious content",
                "encryption": "AES-256-GCM encryption with password protection",
                "compliance": "GDPR, HIPAA, PCI DSS compliance monitoring",
                "auditing": "Comprehensive security auditing and reporting"
            },
            "analytics": {
                "similarity_analysis": "TF-IDF, Jaccard, Cosine similarity analysis",
                "redundancy_detection": "Advanced redundancy detection with clustering",
                "content_metrics": "Readability, sentiment, quality metrics",
                "batch_processing": "Efficient batch processing with caching"
            },
            "ai": {
                "sentiment_analysis": "RoBERTa-based sentiment and emotion analysis",
                "topic_classification": "BART-based topic classification",
                "language_detection": "XLM-RoBERTa language detection",
                "entity_recognition": "BERT-based named entity recognition",
                "summarization": "BART-based automatic summarization"
            },
            "optimization": {
                "readability": "Flesch-Kincaid, Gunning Fog readability enhancement",
                "seo": "SEO optimization with keyword analysis",
                "engagement": "Engagement boosting with readability improvements",
                "grammar": "Grammar correction and style alignment"
            },
            "workflow": {
                "automation": "Workflow automation with step handlers",
                "dependencies": "Dependency management and error handling",
                "templates": "Pre-built workflow templates",
                "monitoring": "Real-time workflow monitoring"
            },
            "intelligence": {
                "trend_analysis": "Content trend analysis and insights",
                "strategy_planning": "Content strategy planning and recommendations",
                "audience_analysis": "Audience analysis and targeting",
                "competitive_analysis": "Competitive content analysis"
            },
            "ml": {
                "classification": "Content classification with multiple algorithms",
                "clustering": "Content clustering with K-means and DBSCAN",
                "topic_modeling": "LDA and BERTopic topic modeling",
                "neural_networks": "Deep learning models for content analysis"
            }
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Ultimate Supreme Security Content Redundancy Detector...")
    
    uvicorn.run(
        "ultimate_supreme_security_app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )


