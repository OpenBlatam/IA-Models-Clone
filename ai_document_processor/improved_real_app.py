"""
Improved Real AI Document Processor Application
A real, working FastAPI application for document processing
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import time

from real_working_processor import initialize_real_working_processor
from improved_real_routes import router

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
        await initialize_real_working_processor()
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
    description="A real, working AI document processing system",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include router
app.include_router(router)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Improved Real AI Document Processor",
        "version": "1.0.0",
        "status": "running",
        "description": "A real, working AI document processing system",
        "features": [
            "Text Analysis",
            "Sentiment Analysis", 
            "Text Classification",
            "Text Summarization",
            "Keyword Extraction",
            "Language Detection",
            "Named Entity Recognition",
            "Part-of-Speech Tagging"
        ],
        "endpoints": {
            "analyze": "/api/v1/real/analyze-text",
            "sentiment": "/api/v1/real/analyze-sentiment",
            "classify": "/api/v1/real/classify-text",
            "summarize": "/api/v1/real/summarize-text",
            "keywords": "/api/v1/real/extract-keywords",
            "language": "/api/v1/real/detect-language",
            "health": "/api/v1/real/health",
            "capabilities": "/api/v1/real/capabilities",
            "stats": "/api/v1/real/stats",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health():
    """Basic health check"""
    return {
        "status": "healthy",
        "service": "Improved Real AI Document Processor",
        "version": "1.0.0",
        "timestamp": time.time()
    }

@app.get("/status")
async def status():
    """Detailed status endpoint"""
    try:
        from real_working_processor import real_working_processor
        
        stats = real_working_processor.get_stats()
        capabilities = real_working_processor.get_capabilities()
        
        return {
            "status": "healthy",
            "version": "1.0.0",
            "uptime": time.time(),
            "processing_stats": stats["stats"],
            "capabilities": capabilities,
            "features_available": {
                "basic_ai": True,
                "nlp_analysis": capabilities["nlp_analysis"],
                "sentiment_analysis": capabilities["sentiment_analysis"],
                "text_classification": capabilities["text_classification"],
                "text_summarization": capabilities["text_summarization"],
                "keyword_extraction": capabilities["keyword_extraction"],
                "language_detection": capabilities["language_detection"]
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
        
        stats = real_working_processor.get_stats()
        capabilities = real_working_processor.get_capabilities()
        
        metrics_data = {
            "ai_requests_total": stats["stats"]["total_requests"],
            "ai_successful_requests_total": stats["stats"]["successful_requests"],
            "ai_failed_requests_total": stats["stats"]["failed_requests"],
            "ai_processing_time_seconds": stats["stats"]["average_processing_time"],
            "ai_uptime_seconds": stats["uptime_seconds"],
            "ai_success_rate_percent": stats["success_rate"],
            "ai_spacy_available": 1 if capabilities["nlp_analysis"] else 0,
            "ai_nltk_available": 1 if capabilities["sentiment_analysis"] else 0,
            "ai_transformers_classifier_available": 1 if capabilities["text_classification"] else 0,
            "ai_transformers_summarizer_available": 1 if capabilities["text_summarization"] else 0
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
        "improved_real_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )













