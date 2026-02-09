"""
BUL API - Production Main Application
====================================

Real-world BUL API implementation following FastAPI best practices:
- Functional programming approach
- RORO pattern implementation
- Early returns and guard clauses
- Async/await throughout
- Production-ready error handling
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from api.bul_api import (
    DocumentRequest,
    DocumentResponse,
    BatchDocumentRequest,
    create_response_context,
    process_document,
    process_batch_documents,
    handle_validation_error,
    handle_processing_error
)
from middleware.real_middleware import (
    create_real_middleware_stack,
    create_real_error_handlers
)
from utils.real_utils import (
    validate_required_fields,
    log_info,
    log_error
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan management"""
    # Startup
    log_info("BUL API starting up")
    yield
    # Shutdown
    log_info("BUL API shutting down")

# Create FastAPI application
app = FastAPI(
    title="BUL API",
    version="1.0.0",
    description="Business Universal Language API",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# Apply real-world middleware stack
app = create_real_middleware_stack(app)

# Add error handlers
create_real_error_handlers(app)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint with early returns"""
    return create_response_context({
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00Z"
    })

# Document generation endpoints
@app.post("/generate", response_model=dict)
async def generate_document(request: DocumentRequest):
    """Generate single document with early returns"""
    try:
        # Early validation
        validate_required_fields(request.dict(), ['query'])
        
        # Process document
        result = await process_document(request)
        
        # Log success
        log_info(f"Document generated: {result.id}")
        
        return create_response_context(result)
        
    except ValueError as e:
        raise handle_validation_error(e)
    except Exception as e:
        raise handle_processing_error(e)

@app.post("/generate/batch", response_model=dict)
async def generate_documents_batch(request: BatchDocumentRequest):
    """Generate multiple documents in batch with early returns"""
    try:
        # Early validation
        if not request.requests:
            raise ValueError("At least one request is required")
        
        # Process batch
        results = await process_batch_documents(request)
        
        # Log success
        log_info(f"Batch processed: {len(results)} documents")
        
        return create_response_context(results)
        
    except ValueError as e:
        raise handle_validation_error(e)
    except Exception as e:
        raise handle_processing_error(e)

# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Get application metrics"""
    return create_response_context({
        "timestamp": "2024-01-01T00:00:00Z",
        "status": "operational"
    })

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return create_response_context({
        "message": "BUL API is running",
        "version": "1.0.0",
        "timestamp": "2024-01-01T00:00:00Z"
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)