"""
Real AI Document Processor Application
A functional FastAPI application for document processing
"""

import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from real_ai_processor import initialize_real_ai_processor
from real_routes import router

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
    logger.info("Starting Real AI Document Processor...")
    try:
        await initialize_real_ai_processor()
        logger.info("Real AI Document Processor started successfully")
    except Exception as e:
        logger.error(f"Failed to start Real AI Document Processor: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Real AI Document Processor...")

# Create FastAPI app
app = FastAPI(
    title="Real AI Document Processor",
    description="A practical AI document processing system using real technologies",
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

# Include routers
app.include_router(router)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Real AI Document Processor",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/real-documents/health"
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Real AI Document Processor",
        "version": "1.0.0"
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

if __name__ == "__main__":
    uvicorn.run(
        "real_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )













