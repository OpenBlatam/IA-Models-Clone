"""
Enhanced AI Document Processor Application
Real, working FastAPI application with advanced features
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import time

from practical_ai_processor import initialize_practical_ai_processor
from enhanced_ai_processor import initialize_enhanced_ai_processor
from practical_routes import router as practical_router
from enhanced_routes import router as enhanced_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Enhanced AI Document Processor...")
    
    try:
        # Initialize practical AI processor
        await initialize_practical_ai_processor()
        logger.info("Practical AI processor initialized")
        
        # Initialize enhanced AI processor
        await initialize_enhanced_ai_processor()
        logger.info("Enhanced AI processor initialized")
        
        logger.info("Enhanced AI Document Processor started successfully")
    except Exception as e:
        logger.error(f"Failed to start Enhanced AI Document Processor: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Enhanced AI Document Processor...")

# Create FastAPI app
app = FastAPI(
    title="Enhanced AI Document Processor",
    description="A real, working AI document processing system with advanced features",
    version="2.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Include routers
app.include_router(practical_router)
app.include_router(enhanced_router)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Enhanced AI Document Processor",
        "version": "2.0.0",
        "status": "running",
        "description": "A real, working AI document processing system with advanced features",
        "features": [
            "Basic AI Processing",
            "Enhanced AI Analysis",
            "Complexity Analysis",
            "Readability Analysis",
            "Language Pattern Analysis",
            "Quality Metrics",
            "Advanced Keyword Analysis",
            "Similarity Analysis",
            "Topic Analysis",
            "Batch Processing",
            "Caching",
            "Performance Monitoring"
        ],
        "endpoints": {
            "practical": "/api/v1/practical",
            "enhanced": "/api/v1/enhanced",
            "docs": "/docs",
            "health": "/health"
        }
    }

@app.get("/health")
async def health():
    """Basic health check"""
    return {
        "status": "healthy",
        "service": "Enhanced AI Document Processor",
        "version": "2.0.0",
        "timestamp": time.time()
    }

@app.get("/status")
async def status():
    """Detailed status endpoint"""
    try:
        from enhanced_ai_processor import enhanced_ai_processor
        
        stats = enhanced_ai_processor.get_processing_stats()
        
        return {
            "status": "healthy",
            "version": "2.0.0",
            "uptime": time.time(),
            "processing_stats": stats["stats"],
            "models_loaded": stats["models_loaded"],
            "features_available": {
                "basic_ai": True,
                "enhanced_ai": True,
                "complexity_analysis": True,
                "readability_analysis": True,
                "language_patterns": True,
                "quality_metrics": True,
                "keyword_analysis": True,
                "similarity_analysis": stats["models_loaded"]["tfidf_vectorizer"],
                "topic_analysis": stats["models_loaded"]["tfidf_vectorizer"],
                "caching": stats["redis_available"] or stats["cache_size"] > 0,
                "batch_processing": True
            }
        }
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/metrics")
async def metrics():
    """Prometheus-style metrics endpoint"""
    try:
        from enhanced_ai_processor import enhanced_ai_processor
        
        stats = enhanced_ai_processor.get_processing_stats()
        
        metrics_data = {
            "ai_requests_total": stats["stats"]["total_requests"],
            "ai_cache_hits_total": stats["stats"]["cache_hits"],
            "ai_cache_misses_total": stats["stats"]["cache_misses"],
            "ai_errors_total": stats["stats"]["error_count"],
            "ai_processing_time_seconds": stats["stats"]["average_processing_time"],
            "ai_cache_size": stats["cache_size"],
            "ai_redis_available": 1 if stats["redis_available"] else 0,
            "ai_spacy_available": 1 if stats["models_loaded"]["spacy"] else 0,
            "ai_nltk_available": 1 if stats["models_loaded"]["nltk"] else 0,
            "ai_transformers_classifier_available": 1 if stats["models_loaded"]["transformers_classifier"] else 0,
            "ai_transformers_summarizer_available": 1 if stats["models_loaded"]["transformers_summarizer"] else 0,
            "ai_transformers_qa_available": 1 if stats["models_loaded"]["transformers_qa"] else 0,
            "ai_tfidf_vectorizer_available": 1 if stats["models_loaded"]["tfidf_vectorizer"] else 0
        }
        
        # Format as Prometheus metrics
        metrics_text = ""
        for key, value in metrics_data.items():
            metrics_text += f"# HELP {key} AI Document Processor Metrics\n"
            metrics_text += f"# TYPE {key} gauge\n"
            metrics_text += f"{key} {value}\n"
        
        return metrics_text
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return f"# Error getting metrics: {e}"

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "path": str(request.url),
            "method": request.method
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "enhanced_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )













