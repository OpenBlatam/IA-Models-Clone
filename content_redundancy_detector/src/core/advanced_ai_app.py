"""
Advanced AI App - Next-generation AI-powered content analysis system
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

from .ai_predictive_engine import initialize_ai_predictive_engine
from .performance_optimizer import initialize_performance_optimizer
from .advanced_caching_engine import initialize_advanced_caching_engine
from .content_security_engine import initialize_content_security_engine

# Import all existing routers
from ..api.ai_predictive_routes import router as ai_predictive_router
from ..api.performance_routes import router as performance_router
from ..api.caching_routes import router as caching_router
from ..api.security_routes import router as security_router
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
            logging.FileHandler("advanced_ai_app.log", encoding="utf-8")
        ]
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    setup_logging()
    logger.info("Advanced AI App starting up...")
    
    try:
        # Initialize AI predictive engine
        await initialize_ai_predictive_engine()
        logger.info("AI Predictive Engine initialized successfully")
        
        # Initialize performance optimizer
        await initialize_performance_optimizer()
        logger.info("Performance Optimizer initialized successfully")
        
        # Initialize advanced caching engine
        await initialize_advanced_caching_engine()
        logger.info("Advanced Caching Engine initialized successfully")
        
        # Initialize content security engine
        await initialize_content_security_engine()
        logger.info("Content Security Engine initialized successfully")
        
        logger.info("All AI and optimization systems initialized successfully")
        logger.info("Server ready to handle requests with advanced AI capabilities")
        
    except Exception as e:
        logger.error(f"Failed to initialize AI systems: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Advanced AI App shutting down...")


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    app = FastAPI(
        title="Advanced AI Content Redundancy Detector",
        description="Next-generation AI-powered content security, threat detection, redundancy analysis, and predictive analytics system with advanced machine learning, performance optimization, and intelligent caching",
        version="4.0.0",
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
    app.include_router(ai_predictive_router)
    app.include_router(performance_router)
    app.include_router(caching_router)
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
        "service": "Advanced AI Content Redundancy Detector",
        "version": "4.0.0",
        "timestamp": time.time(),
        "ai_features": [
            "AI Predictive Analytics",
            "Advanced Machine Learning",
            "Performance Optimization",
            "Advanced Caching",
            "Content Security",
            "Real-time Analytics",
            "AI Content Analysis",
            "Content Optimization",
            "Workflow Automation",
            "Content Intelligence",
            "Machine Learning",
            "Threat Detection",
            "Encryption/Decryption",
            "Compliance Monitoring",
            "Security Auditing",
            "Time Series Forecasting",
            "Anomaly Detection",
            "Sentiment Analysis",
            "Topic Classification"
        ]
    }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "Advanced AI Content Redundancy Detector",
        "version": "4.0.0",
        "description": "Next-generation AI-powered content security, threat detection, redundancy analysis, and predictive analytics system with advanced machine learning and optimization",
        "documentation": "/docs",
        "health_check": "/health",
        "ai_features": {
            "predictive_analytics": {
                "content_classification": "AI-powered content classification with multiple algorithms",
                "sentiment_analysis": "Advanced sentiment analysis with RoBERTa and BERT",
                "topic_prediction": "Intelligent topic prediction and classification",
                "anomaly_detection": "Statistical and ML-based anomaly detection",
                "time_series_forecasting": "Prophet-based time series forecasting",
                "model_retraining": "Automatic model retraining and improvement"
            },
            "machine_learning": {
                "supervised_learning": "Classification and regression with multiple algorithms",
                "unsupervised_learning": "Clustering and dimensionality reduction",
                "deep_learning": "Neural networks and transformer models",
                "ensemble_methods": "Random forests and gradient boosting",
                "feature_engineering": "Automatic feature extraction and selection",
                "model_selection": "Automatic model selection and hyperparameter tuning"
            },
            "performance": {
                "monitoring": "Real-time performance monitoring with CPU, memory, disk, and network metrics",
                "optimization": "Automatic system optimization with intelligent recommendations",
                "profiling": "Memory profiling and performance analysis",
                "health_checks": "Comprehensive system health monitoring"
            },
            "caching": {
                "memory_cache": "High-performance in-memory cache with LRU/LFU eviction",
                "redis_cache": "Distributed Redis-based cache with persistence",
                "compression": "Data compression for optimal memory usage",
                "serialization": "Multiple serialization formats (JSON, Pickle, MessagePack)"
            },
            "security": {
                "threat_detection": "Advanced threat detection with multiple attack patterns",
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
            "ai_analysis": {
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
            "ml_engine": {
                "classification": "Content classification with multiple algorithms",
                "clustering": "Content clustering with K-means and DBSCAN",
                "topic_modeling": "LDA and BERTopic topic modeling",
                "neural_networks": "Deep learning models for content analysis"
            }
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Advanced AI Content Redundancy Detector...")
    
    uvicorn.run(
        "advanced_ai_app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )


