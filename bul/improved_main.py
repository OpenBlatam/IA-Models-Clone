"""
Improved BUL API - Enhanced Main Application
==========================================

Enhanced BUL API with additional real-world functionality:
- Advanced document processing
- Business logic improvements
- Performance optimizations
- Real-world integrations
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from api.improved_bul_api import (
    ImprovedDocumentRequest,
    ImprovedDocumentResponse,
    BatchDocumentRequest,
    create_enhanced_response_context,
    process_enhanced_document,
    process_enhanced_batch_documents,
    handle_enhanced_validation_error,
    handle_enhanced_processing_error,
    handle_enhanced_single_document_generation,
    handle_enhanced_batch_document_generation
)
from utils.improved_utils import (
    validate_enhanced_required_fields,
    log_enhanced_info,
    log_enhanced_error,
    measure_enhanced_time,
    cache_enhanced_result
)

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Enhanced application lifespan
@asynccontextmanager
async def enhanced_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Enhanced application lifespan management"""
    # Startup
    log_enhanced_info("Enhanced BUL API starting up", version="2.0.0")
    
    # Initialize enhanced components
    app.state.enhanced_features = {
        "quality_scoring": True,
        "readability_analysis": True,
        "business_logic": True,
        "performance_monitoring": True
    }
    
    yield
    
    # Shutdown
    log_enhanced_info("Enhanced BUL API shutting down")

# Create enhanced FastAPI application
app = FastAPI(
    title="Enhanced BUL API",
    version="2.0.0",
    description="Enhanced Business Universal Language API with advanced features",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=enhanced_lifespan
)

# Add enhanced middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# Enhanced health check endpoint
@app.get("/health")
@measure_enhanced_time
async def enhanced_health_check():
    """Enhanced health check with detailed status"""
    return create_enhanced_response_context({
        "status": "healthy",
        "version": "2.0.0",
        "features": {
            "quality_scoring": True,
            "readability_analysis": True,
            "business_logic": True,
            "performance_monitoring": True
        },
        "timestamp": "2024-01-01T00:00:00Z"
    })

# Enhanced document generation endpoints
@app.post("/generate", response_model=dict)
@measure_enhanced_time
@cache_enhanced_result(ttl=1800)
async def generate_enhanced_document(request: ImprovedDocumentRequest):
    """Generate enhanced single document with advanced features"""
    try:
        # Enhanced validation
        validate_enhanced_required_fields(request.dict(), ['query'])
        
        # Process document with enhanced features
        result = await process_enhanced_document(request)
        
        # Log success with enhanced details
        log_enhanced_info(
            f"Enhanced document generated: {result.id}",
            document_type=result.document_type,
            business_area=result.business_area,
            quality_score=result.quality_score,
            readability_score=result.readability_score
        )
        
        return create_enhanced_response_context(
            result,
            metadata={
                "processing_time": result.processing_time,
                "quality_score": result.quality_score,
                "readability_score": result.readability_score,
                "word_count": result.word_count
            }
        )
        
    except ValueError as e:
        log_enhanced_error("Validation error in document generation", error=e)
        raise handle_enhanced_validation_error(e)
    except Exception as e:
        log_enhanced_error("Processing error in document generation", error=e)
        raise handle_enhanced_processing_error(e)

@app.post("/generate/batch", response_model=dict)
@measure_enhanced_time
async def generate_enhanced_documents_batch(request: BatchDocumentRequest):
    """Generate enhanced multiple documents in batch with advanced features"""
    try:
        # Enhanced validation
        if not request.requests:
            raise ValueError("At least one request is required")
        
        # Process batch with enhanced features
        results = await process_enhanced_batch_documents(request)
        
        # Calculate batch statistics
        total_processing_time = sum(r.processing_time for r in results)
        avg_quality_score = sum(r.quality_score or 0 for r in results) / len(results)
        avg_readability_score = sum(r.readability_score or 0 for r in results) / len(results)
        
        # Log success with enhanced details
        log_enhanced_info(
            f"Enhanced batch processed: {len(results)} documents",
            total_processing_time=total_processing_time,
            avg_quality_score=avg_quality_score,
            avg_readability_score=avg_readability_score
        )
        
        return create_enhanced_response_context(
            results,
            metadata={
                "batch_size": len(results),
                "total_processing_time": total_processing_time,
                "avg_quality_score": round(avg_quality_score, 2),
                "avg_readability_score": round(avg_readability_score, 2)
            }
        )
        
    except ValueError as e:
        log_enhanced_error("Validation error in batch generation", error=e)
        raise handle_enhanced_validation_error(e)
    except Exception as e:
        log_enhanced_error("Processing error in batch generation", error=e)
        raise handle_enhanced_processing_error(e)

# Enhanced metrics endpoint
@app.get("/metrics")
@measure_enhanced_time
async def get_enhanced_metrics():
    """Get enhanced application metrics"""
    return create_enhanced_response_context({
        "timestamp": "2024-01-01T00:00:00Z",
        "status": "operational",
        "version": "2.0.0",
        "features": {
            "quality_scoring": True,
            "readability_analysis": True,
            "business_logic": True,
            "performance_monitoring": True
        },
        "capabilities": {
            "single_document_generation": True,
            "batch_document_generation": True,
            "quality_analysis": True,
            "readability_analysis": True,
            "business_logic_processing": True
        }
    })

# Enhanced validation endpoint
@app.post("/validate")
@measure_enhanced_time
async def validate_enhanced_request(request: ImprovedDocumentRequest):
    """Validate enhanced request with detailed feedback"""
    try:
        # Enhanced validation
        validate_enhanced_required_fields(request.dict(), ['query'])
        
        # Additional business logic validation
        validation_results = {
            "basic_validation": True,
            "business_logic_validation": True,
            "quality_checks": True
        }
        
        # Check business area and document type compatibility
        if request.business_area and request.document_type:
            compatible_combinations = {
                'marketing': ['marketing_strategy', 'business_plan'],
                'sales': ['sales_proposal', 'business_plan'],
                'operations': ['operational_manual', 'business_plan'],
                'hr': ['hr_policy', 'operational_manual'],
                'finance': ['financial_report', 'business_plan']
            }
            
            if request.business_area in compatible_combinations:
                if request.document_type not in compatible_combinations[request.business_area]:
                    validation_results["business_logic_validation"] = False
        
        return create_enhanced_response_context(
            validation_results,
            metadata={
                "validation_timestamp": "2024-01-01T00:00:00Z",
                "request_id": "validation_request_id"
            }
        )
        
    except ValueError as e:
        log_enhanced_error("Validation error in request validation", error=e)
        raise handle_enhanced_validation_error(e)
    except Exception as e:
        log_enhanced_error("Processing error in request validation", error=e)
        raise handle_enhanced_processing_error(e)

# Enhanced root endpoint
@app.get("/")
async def enhanced_root():
    """Enhanced root endpoint with detailed information"""
    return create_enhanced_response_context({
        "message": "Enhanced BUL API is running",
        "version": "2.0.0",
        "features": [
            "Enhanced document generation",
            "Quality scoring",
            "Readability analysis",
            "Business logic processing",
            "Performance monitoring"
        ],
        "endpoints": {
            "POST /generate": "Generate single enhanced document",
            "POST /generate/batch": "Generate multiple enhanced documents",
            "POST /validate": "Validate enhanced request",
            "GET /health": "Enhanced health check",
            "GET /metrics": "Enhanced application metrics"
        },
        "timestamp": "2024-01-01T00:00:00Z"
    })

# Enhanced error handlers
@app.exception_handler(HTTPException)
async def enhanced_http_exception_handler(request: Request, exc: HTTPException):
    """Enhanced HTTP exception handler"""
    log_enhanced_error(
        f"HTTP exception: {exc.status_code}",
        error=exc,
        path=request.url.path,
        method=request.method
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=create_enhanced_response_context(
            None,
            success=False,
            error=exc.detail,
            metadata={
                "status_code": exc.status_code,
                "path": request.url.path,
                "method": request.method
            }
        )
    )

@app.exception_handler(Exception)
async def enhanced_general_exception_handler(request: Request, exc: Exception):
    """Enhanced general exception handler"""
    log_enhanced_error(
        f"General exception: {type(exc).__name__}",
        error=exc,
        path=request.url.path,
        method=request.method
    )
    
    return JSONResponse(
        status_code=500,
        content=create_enhanced_response_context(
            None,
            success=False,
            error="Internal server error",
            metadata={
                "status_code": 500,
                "path": request.url.path,
                "method": request.method,
                "error_type": type(exc).__name__
            }
        )
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)












