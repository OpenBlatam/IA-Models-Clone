"""
Main FastAPI Application for AI Document Classifier
==================================================

This is the main entry point for the AI Document Classifier service.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import os
from contextlib import asynccontextmanager

from api_endpoints import router as classifier_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting AI Document Classifier service...")
    
    # Check for OpenAI API key
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        logger.info("✅ OpenAI API key found - AI classification enabled")
    else:
        logger.warning("⚠️  OpenAI API key not found - using pattern-based classification only")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down AI Document Classifier service...")

# Create FastAPI application
app = FastAPI(
    title="AI Document Classifier",
    description="""
    An AI-powered system that can identify document types from text queries 
    and export appropriate template designs.
    
    ## Features
    
    * **Document Classification**: Identify document types (novel, contract, design, etc.) from text descriptions
    * **Template Export**: Export document templates in multiple formats (JSON, YAML, Markdown)
    * **AI-Powered**: Uses OpenAI GPT models for enhanced classification accuracy
    * **Pattern Matching**: Fallback classification using keyword and pattern matching
    
    ## Supported Document Types
    
    * Novel (Fiction)
    * Contract (Legal)
    * Design (Technical/Architectural)
    * Business Plan
    * Academic Paper
    * Technical Manual
    * Marketing Material
    * User Manual
    * Report
    * Proposal
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(classifier_router)

# Include enhanced router
try:
    from enhanced_api import router as enhanced_router
    app.include_router(enhanced_router)
    logger.info("Enhanced API endpoints loaded")
except ImportError as e:
    logger.warning(f"Enhanced API not available: {e}")

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "AI Document Classifier",
        "version": "1.0.0",
        "description": "AI-powered document type classification and template export",
        "endpoints": {
            "classify": "/ai-document-classifier/classify",
            "templates": "/ai-document-classifier/templates/{document_type}",
            "export": "/ai-document-classifier/export-template",
            "health": "/ai-document-classifier/health"
        },
        "documentation": "/docs"
    }

@app.get("/health")
async def health_check():
    """Global health check endpoint"""
    return {
        "status": "healthy",
        "service": "AI Document Classifier",
        "version": "1.0.0"
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc) if os.getenv("DEBUG") == "true" else "An unexpected error occurred"
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )
