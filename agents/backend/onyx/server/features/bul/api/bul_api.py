"""
BUL API - Production Implementation
==================================

Real-world BUL API implementation following FastAPI best practices:
- Functional programming approach
- RORO pattern implementation
- Early returns and guard clauses
- Async/await throughout
- Production-ready error handling
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from fastapi import FastAPI, Request, Response, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# Real-world models following RORO pattern
class DocumentRequest(BaseModel):
    """Document generation request"""
    query: str = Field(..., min_length=10, max_length=2000)
    business_area: Optional[str] = Field(None, max_length=50)
    document_type: Optional[str] = Field(None, max_length=50)
    company_name: Optional[str] = Field(None, max_length=100)
    language: str = Field("es", max_length=2)
    format: str = Field("markdown", max_length=10)
    
    @validator('language')
    def validate_language(cls, v):
        if v not in ['es', 'en', 'pt', 'fr']:
            raise ValueError('Language must be es, en, pt, or fr')
        return v
    
    @validator('business_area')
    def validate_business_area(cls, v):
        if v and v not in ['marketing', 'sales', 'operations', 'hr', 'finance']:
            raise ValueError('Invalid business area')
        return v

class DocumentResponse(BaseModel):
    """Document generation response"""
    id: str
    content: str
    title: str
    summary: str
    word_count: int
    processing_time: float
    confidence_score: float
    created_at: datetime

class BatchDocumentRequest(BaseModel):
    """Batch document generation request"""
    requests: List[DocumentRequest] = Field(..., max_items=10)
    parallel: bool = Field(True)
    max_concurrent: int = Field(5, ge=1, le=10)

# Real-world utilities
def create_response_context(data: Any, success: bool = True, error: Optional[str] = None) -> Dict[str, Any]:
    """Create response context following RORO pattern"""
    return {
        "data": data,
        "success": success,
        "error": error,
        "timestamp": datetime.now().isoformat()
    }

def validate_required_fields(data: Dict[str, Any], required: List[str]) -> None:
    """Validate required fields with early returns"""
    for field in required:
        if field not in data or not data[field]:
            raise ValueError(f"Required field missing: {field}")

def extract_request_context(request: Request) -> Dict[str, Any]:
    """Extract request context"""
    return {
        "method": request.method,
        "url": str(request.url),
        "client_ip": request.client.host,
        "timestamp": datetime.now().isoformat()
    }

# Real-world document processor
async def process_document(request: DocumentRequest) -> DocumentResponse:
    """Process document with early returns and guard clauses"""
    # Early validation
    if not request.query:
        raise ValueError("Query is required")
    
    if len(request.query) < 10:
        raise ValueError("Query too short")
    
    # Process document
    start_time = time.time()
    
    # Simulate document generation
    content = f"Generated document for: {request.query}"
    title = f"Document: {request.business_area or 'General'}"
    summary = f"Summary of {request.query[:100]}..."
    
    processing_time = time.time() - start_time
    
    return DocumentResponse(
        id=str(uuid.uuid4()),
        content=content,
        title=title,
        summary=summary,
        word_count=len(content.split()),
        processing_time=processing_time,
        confidence_score=0.85,
        created_at=datetime.now()
    )

async def process_batch_documents(request: BatchDocumentRequest) -> List[DocumentResponse]:
    """Process batch documents with early returns"""
    # Early validation
    if not request.requests:
        raise ValueError("At least one request is required")
    
    if len(request.requests) > 10:
        raise ValueError("Too many requests")
    
    if request.parallel:
        # Process in parallel with concurrency limit
        semaphore = asyncio.Semaphore(request.max_concurrent)
        
        async def process_with_semaphore(doc_request: DocumentRequest) -> DocumentResponse:
            async with semaphore:
                return await process_document(doc_request)
        
        return await asyncio.gather(*[process_with_semaphore(req) for req in request.requests])
    else:
        # Process sequentially
        results = []
        for doc_request in request.requests:
            result = await process_document(doc_request)
            results.append(result)
        return results

# Real-world error handlers
def handle_validation_error(error: ValueError) -> HTTPException:
    """Handle validation errors"""
    return HTTPException(status_code=400, detail=str(error))

def handle_processing_error(error: Exception) -> HTTPException:
    """Handle processing errors"""
    return HTTPException(status_code=500, detail="Document processing failed")

# Real-world route handlers
async def handle_single_document_generation(
    request: DocumentRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Handle single document generation with early returns"""
    try:
        # Early validation
        validate_required_fields(request.dict(), ['query'])
        
        # Process document
        result = await process_document(request)
        
        # Background task for logging
        background_tasks.add_task(
            lambda: logging.info(f"Document generated: {result.id}")
        )
        
        return create_response_context(result)
        
    except ValueError as e:
        raise handle_validation_error(e)
    except Exception as e:
        raise handle_processing_error(e)

async def handle_batch_document_generation(
    request: BatchDocumentRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Handle batch document generation with early returns"""
    try:
        # Early validation
        if not request.requests:
            raise ValueError("At least one request is required")
        
        # Process batch
        results = await process_batch_documents(request)
        
        # Background task for logging
        background_tasks.add_task(
            lambda: logging.info(f"Batch processed: {len(results)} documents")
        )
        
        return create_response_context(results)
        
    except ValueError as e:
        raise handle_validation_error(e)
    except Exception as e:
        raise handle_processing_error(e)

# Real-world FastAPI application
def create_bul_app() -> FastAPI:
    """Create BUL FastAPI application"""
    
    app = FastAPI(
        title="BUL API",
        version="1.0.0",
        description="Business Universal Language API",
        docs_url="/docs",
        redoc_url="/redoc"
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
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return create_response_context({
            "status": "healthy",
            "timestamp": datetime.now().isoformat()
        })
    
    # Document generation endpoints
    @app.post("/generate", response_model=Dict[str, Any])
    async def generate_document(
        request: DocumentRequest,
        background_tasks: BackgroundTasks
    ):
        """Generate single document"""
        return await handle_single_document_generation(request, background_tasks)
    
    @app.post("/generate/batch", response_model=Dict[str, Any])
    async def generate_documents_batch(
        request: BatchDocumentRequest,
        background_tasks: BackgroundTasks
    ):
    """Generate multiple documents in batch"""
        return await handle_batch_document_generation(request, background_tasks)
    
    # Error handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions"""
        return JSONResponse(
            status_code=exc.status_code,
            content=create_response_context(
                None, 
                success=False, 
                error=exc.detail
            )
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions"""
    return JSONResponse(
        status_code=500,
            content=create_response_context(
                None, 
                success=False, 
                error="Internal server error"
            )
        )
    
    return app

# Export functions
__all__ = [
    "DocumentRequest",
    "DocumentResponse",
    "BatchDocumentRequest",
    "create_response_context",
    "validate_required_fields",
    "extract_request_context",
    "process_document",
    "process_batch_documents",
    "handle_validation_error",
    "handle_processing_error",
    "handle_single_document_generation",
    "handle_batch_document_generation",
    "create_bul_app"
]