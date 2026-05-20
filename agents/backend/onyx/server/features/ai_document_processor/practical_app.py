"""
Practical AI Document Processor Application
A real, working FastAPI application for document processing
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from practical_ai_processor import initialize_practical_ai_processor
from practical_routes import router

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
    logger.info("Starting Practical AI Document Processor...")
    try:
        await initialize_practical_ai_processor()
        logger.info("Practical AI Document Processor started successfully")
    except Exception as e:
        logger.error(f"Failed to start Practical AI Document Processor: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Practical AI Document Processor...")

# Create FastAPI app
app = FastAPI(
    title="Practical AI Document Processor",
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
        "message": "Practical AI Document Processor",
        "version": "1.0.0",
        "status": "running",
        "description": "A real, working AI document processing system",
        "endpoints": {
            "analyze": "/api/v1/practical/analyze-text",
            "sentiment": "/api/v1/practical/analyze-sentiment",
            "classify": "/api/v1/practical/classify-text",
            "summarize": "/api/v1/practical/summarize-text",
            "keywords": "/api/v1/practical/extract-keywords",
            "language": "/api/v1/practical/detect-language",
            "health": "/api/v1/practical/health",
            "capabilities": "/api/v1/practical/capabilities",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health():
    """Basic health check"""
    return {
        "status": "healthy",
        "service": "Practical AI Document Processor",
        "version": "1.0.0"
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
        "practical_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )













