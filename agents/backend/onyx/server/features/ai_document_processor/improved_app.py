"""
Improved Real AI Document Processor Application
Enhanced FastAPI application with advanced features
"""

import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import time

from real_ai_processor import initialize_real_ai_processor
from advanced_ai_processor import initialize_advanced_ai_processor
from real_routes import router as real_router
from advanced_routes import router as advanced_router

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
    logger.info("Starting Improved Real AI Document Processor...")
    
    try:
        # Initialize basic AI processor
        await initialize_real_ai_processor()
        logger.info("Basic AI processor initialized")
        
        # Initialize advanced AI processor
        await initialize_advanced_ai_processor()
        logger.info("Advanced AI processor initialized")
        
        logger.info("Improved Real AI Document Processor started successfully")
    except Exception as e:
        logger.error(f"Failed to start Improved Real AI Document Processor: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Improved Real AI Document Processor...")

# Create FastAPI app
app = FastAPI(
    title="Improved Real AI Document Processor",
    description="An enhanced AI document processing system with advanced features",
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
app.include_router(real_router)
app.include_router(advanced_router)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Improved Real AI Document Processor",
        "version": "2.0.0",
        "status": "running",
        "features": [
            "Basic AI Processing",
            "Advanced AI Analysis",
            "Document Upload & Parsing",
            "Caching & Performance",
            "Batch Processing",
            "Real-time Monitoring"
        ],
        "endpoints": {
            "basic": "/api/v1/real-documents",
            "advanced": "/api/v1/advanced-documents",
            "docs": "/docs",
            "health": "/health"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Improved Real AI Document Processor",
        "version": "2.0.0",
        "timestamp": time.time()
    }

@app.get("/status")
async def status():
    """Detailed status endpoint"""
    try:
        from advanced_ai_processor import advanced_ai_processor
        from document_parser import document_parser
        
        stats = advanced_ai_processor.get_processing_stats()
        formats = document_parser.get_supported_formats()
        
        return {
            "status": "healthy",
            "version": "2.0.0",
            "uptime": time.time(),
            "processing_stats": stats["stats"],
            "supported_formats": formats,
            "features_available": {
                "basic_ai": True,
                "advanced_ai": True,
                "document_parsing": any(formats.values()),
                "caching": stats["redis_available"] or stats["cache_size"] > 0,
                "similarity_analysis": stats["sentence_transformer_available"],
                "topic_analysis": stats["vectorizer_available"]
            }
        }
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

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

@app.get("/metrics")
async def metrics():
    """Prometheus-style metrics endpoint"""
    try:
        from advanced_ai_processor import advanced_ai_processor
        
        stats = advanced_ai_processor.get_processing_stats()
        
        metrics_data = {
            "ai_requests_total": stats["stats"]["total_requests"],
            "ai_cache_hits_total": stats["stats"]["cache_hits"],
            "ai_cache_misses_total": stats["stats"]["cache_misses"],
            "ai_errors_total": stats["stats"]["error_count"],
            "ai_processing_time_seconds": stats["stats"]["average_processing_time"],
            "ai_cache_size": stats["cache_size"],
            "ai_redis_available": 1 if stats["redis_available"] else 0,
            "ai_sentence_transformer_available": 1 if stats["sentence_transformer_available"] else 0,
            "ai_vectorizer_available": 1 if stats["vectorizer_available"] else 0
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

if __name__ == "__main__":
    uvicorn.run(
        "improved_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )













