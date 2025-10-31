"""
Complete Real AI Document Processor Application
A comprehensive, real, working FastAPI application for document processing
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import time

from real_working_processor import initialize_real_working_processor
from advanced_real_processor import initialize_advanced_real_processor
from improved_real_routes import router as basic_router
from advanced_real_routes import router as advanced_router

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
    logger.info("Starting Complete Real AI Document Processor...")
    
    try:
        # Initialize basic AI processor
        await initialize_real_working_processor()
        logger.info("Basic AI processor initialized")
        
        # Initialize advanced AI processor
        await initialize_advanced_real_processor()
        logger.info("Advanced AI processor initialized")
        
        logger.info("Complete Real AI Document Processor started successfully")
    except Exception as e:
        logger.error(f"Failed to start Complete Real AI Document Processor: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Complete Real AI Document Processor...")

# Create FastAPI app
app = FastAPI(
    title="Complete Real AI Document Processor",
    description="A comprehensive, real, working AI document processing system",
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
app.include_router(basic_router)
app.include_router(advanced_router)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Complete Real AI Document Processor",
        "version": "2.0.0",
        "status": "running",
        "description": "A comprehensive, real, working AI document processing system",
        "features": [
            "Basic AI Processing",
            "Advanced AI Analysis",
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
            "basic": "/api/v1/real",
            "advanced": "/api/v1/advanced-real",
            "docs": "/docs",
            "health": "/health"
        }
    }

@app.get("/health")
async def health():
    """Basic health check"""
    return {
        "status": "healthy",
        "service": "Complete Real AI Document Processor",
        "version": "2.0.0",
        "timestamp": time.time()
    }

@app.get("/status")
async def status():
    """Detailed status endpoint"""
    try:
        from real_working_processor import real_working_processor
        from advanced_real_processor import advanced_real_processor
        
        basic_stats = real_working_processor.get_stats()
        advanced_stats = advanced_real_processor.get_stats()
        basic_capabilities = real_working_processor.get_capabilities()
        advanced_capabilities = advanced_real_processor.get_capabilities()
        
        return {
            "status": "healthy",
            "version": "2.0.0",
            "uptime": time.time(),
            "basic_processor": {
                "stats": basic_stats["stats"],
                "capabilities": basic_capabilities
            },
            "advanced_processor": {
                "stats": advanced_stats["stats"],
                "capabilities": advanced_capabilities
            },
            "features_available": {
                "basic_ai": True,
                "advanced_ai": True,
                "complexity_analysis": True,
                "readability_analysis": True,
                "language_patterns": True,
                "quality_metrics": True,
                "keyword_analysis": True,
                "similarity_analysis": advanced_capabilities["similarity_analysis"],
                "topic_analysis": advanced_capabilities["topic_analysis"],
                "caching": True,
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
        from real_working_processor import real_working_processor
        from advanced_real_processor import advanced_real_processor
        
        basic_stats = real_working_processor.get_stats()
        advanced_stats = advanced_real_processor.get_stats()
        basic_capabilities = real_working_processor.get_capabilities()
        advanced_capabilities = advanced_real_processor.get_capabilities()
        
        metrics_data = {
            "ai_requests_total": basic_stats["stats"]["total_requests"] + advanced_stats["stats"]["total_requests"],
            "ai_successful_requests_total": basic_stats["stats"]["successful_requests"] + advanced_stats["stats"]["successful_requests"],
            "ai_failed_requests_total": basic_stats["stats"]["failed_requests"] + advanced_stats["stats"]["failed_requests"],
            "ai_processing_time_seconds": (basic_stats["stats"]["average_processing_time"] + advanced_stats["stats"]["average_processing_time"]) / 2,
            "ai_uptime_seconds": basic_stats["uptime_seconds"],
            "ai_success_rate_percent": (basic_stats["success_rate"] + advanced_stats["success_rate"]) / 2,
            "ai_cache_hit_rate_percent": advanced_stats["cache_hit_rate"],
            "ai_spacy_available": 1 if basic_capabilities["nlp_analysis"] else 0,
            "ai_nltk_available": 1 if basic_capabilities["sentiment_analysis"] else 0,
            "ai_transformers_classifier_available": 1 if basic_capabilities["text_classification"] else 0,
            "ai_transformers_summarizer_available": 1 if basic_capabilities["text_summarization"] else 0,
            "ai_tfidf_vectorizer_available": 1 if advanced_capabilities["similarity_analysis"] else 0
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

@app.get("/comparison")
async def comparison():
    """Compare basic vs advanced processing"""
    try:
        from real_working_processor import real_working_processor
        from advanced_real_processor import advanced_real_processor
        
        basic_stats = real_working_processor.get_stats()
        advanced_stats = advanced_real_processor.get_stats()
        basic_capabilities = real_working_processor.get_capabilities()
        advanced_capabilities = advanced_real_processor.get_capabilities()
        
        return {
            "comparison": {
                "basic_processor": {
                    "capabilities": basic_capabilities,
                    "stats": basic_stats["stats"],
                    "success_rate": basic_stats["success_rate"]
                },
                "advanced_processor": {
                    "capabilities": advanced_capabilities,
                    "stats": advanced_stats["stats"],
                    "success_rate": advanced_stats["success_rate"],
                    "cache_hit_rate": advanced_stats["cache_hit_rate"]
                },
                "differences": {
                    "advanced_features": [
                        "complexity_analysis",
                        "readability_analysis",
                        "language_patterns",
                        "quality_metrics",
                        "similarity_analysis",
                        "topic_analysis",
                        "caching",
                        "batch_processing"
                    ],
                    "performance_improvements": {
                        "caching": "Advanced processor includes caching",
                        "batch_processing": "Advanced processor supports batch processing",
                        "more_analysis": "Advanced processor provides deeper analysis"
                    }
                }
            }
        }
    except Exception as e:
        logger.error(f"Error getting comparison: {e}")
        return {
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

if __name__ == "__main__":
    uvicorn.run(
        "complete_real_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )













