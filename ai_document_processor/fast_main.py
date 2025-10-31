"""
Fast Main API - Ultra High Performance Version
=============================================

Optimized FastAPI application with maximum speed and efficiency.
Implements async processing, caching, and performance monitoring.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import logging
import os
import tempfile
import asyncio
import time
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
from pathlib import Path
import json

# Import optimized services
from services.fast_document_processor import get_fast_processor, close_fast_processor
from services.enhanced_cache_service import get_cache_service, close_cache_service
from services.performance_monitor import get_performance_monitor, close_performance_monitor
from models.document_models import DocumentProcessingRequest, DocumentProcessingResponse
from config import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global settings
settings = Settings()

# Global services
fast_processor = None
cache_service = None
performance_monitor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global fast_processor, cache_service, performance_monitor
    
    # Startup
    logger.info("Starting Fast AI Document Processor...")
    
    try:
        # Initialize services
        fast_processor = await get_fast_processor()
        cache_service = await get_cache_service()
        performance_monitor = await get_performance_monitor()
        
        logger.info("All services initialized successfully")
        
        yield
        
    finally:
        # Shutdown
        logger.info("Shutting down Fast AI Document Processor...")
        
        if fast_processor:
            await close_fast_processor()
        if cache_service:
            await close_cache_service()
        if performance_monitor:
            await close_performance_monitor()
        
        logger.info("Shutdown complete")

# Create FastAPI app with optimizations
app = FastAPI(
    title="Fast AI Document Processor",
    description="Ultra high-performance AI document processing API",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware for performance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Mount static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint with system info"""
    return {
        "message": "Fast AI Document Processor API",
        "version": "2.0.0",
        "status": "running",
        "features": [
            "Ultra-fast document processing",
            "Parallel processing",
            "Intelligent caching",
            "Performance monitoring",
            "Streaming support"
        ]
    }

@app.get("/health", response_class=JSONResponse)
async def health_check():
    """Comprehensive health check"""
    try:
        if not performance_monitor:
            raise HTTPException(status_code=503, detail="Performance monitor not available")
        
        health_status = performance_monitor.get_health_status()
        
        return {
            "status": health_status['overall_status'],
            "timestamp": time.time(),
            "checks": health_status['checks'],
            "system_metrics": health_status['system_metrics'],
            "recommendations": performance_monitor.get_performance_recommendations()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@app.get("/metrics", response_class=JSONResponse)
async def get_metrics():
    """Get performance metrics"""
    try:
        if not performance_monitor or not fast_processor:
            raise HTTPException(status_code=503, detail="Services not available")
        
        # Get processor stats
        processor_stats = fast_processor.get_performance_stats()
        
        # Get performance metrics
        metrics_summary = await performance_monitor.get_metrics_summary()
        operation_stats = performance_monitor.get_operation_stats()
        
        # Get cache stats
        cache_stats = cache_service.get_stats() if cache_service else {}
        
        return {
            "processor_stats": processor_stats,
            "performance_metrics": metrics_summary,
            "operation_stats": operation_stats,
            "cache_stats": cache_stats,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")

@app.post("/process", response_class=JSONResponse)
async def process_document_fast(
    file: UploadFile = File(...),
    processing_options: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Process document with maximum speed"""
    start_time = time.time()
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Check file size
        file_size = 0
        content = await file.read()
        file_size = len(content)
        
        if file_size > settings.max_file_size:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Maximum size: {settings.max_file_size} bytes"
            )
        
        # Check file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in settings.allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type not supported. Allowed: {settings.allowed_extensions}"
            )
        
        # Parse processing options
        options = {}
        if processing_options:
            try:
                options = json.loads(processing_options)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid processing options JSON")
        
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Process document with fast processor
            if not fast_processor:
                raise HTTPException(status_code=503, detail="Fast processor not available")
            
            result = await fast_processor.process_document_fast(
                temp_file_path, 
                file.filename, 
                options
            )
            
            # Add processing metadata
            result.metadata = result.metadata or {}
            result.metadata.update({
                'api_version': '2.0.0',
                'processing_time_ms': (time.time() - start_time) * 1000,
                'file_size_bytes': file_size,
                'file_type': file_ext
            })
            
            # Record metrics
            if performance_monitor:
                await performance_monitor.record_metric(
                    "api.document_processing_time_ms",
                    (time.time() - start_time) * 1000,
                    tags={'file_type': file_ext, 'file_size': str(file_size)},
                    unit="ms"
                )
            
            return result.dict()
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {temp_file_path}: {e}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/process/batch", response_class=JSONResponse)
async def process_documents_batch_fast(
    files: List[UploadFile] = File(...),
    processing_options: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Process multiple documents in parallel"""
    start_time = time.time()
    
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        if len(files) > 10:  # Limit batch size
            raise HTTPException(status_code=400, detail="Maximum 10 files per batch")
        
        # Parse processing options
        options = {}
        if processing_options:
            try:
                options = json.loads(processing_options)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid processing options JSON")
        
        # Save files temporarily
        temp_files = []
        try:
            for file in files:
                if not file.filename:
                    continue
                
                # Check file size
                content = await file.read()
                if len(content) > settings.max_file_size:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File {file.filename} too large"
                    )
                
                # Check file extension
                file_ext = Path(file.filename).suffix.lower()
                if file_ext not in settings.allowed_extensions:
                    raise HTTPException(
                        status_code=400,
                        detail=f"File type {file_ext} not supported for {file.filename}"
                    )
                
                # Save temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                    temp_file.write(content)
                    temp_files.append((temp_file.name, file.filename))
            
            # Process documents in parallel
            if not fast_processor:
                raise HTTPException(status_code=503, detail="Fast processor not available")
            
            file_paths = [temp_file[0] for temp_file in temp_files]
            results = await fast_processor.process_batch_fast(file_paths, options)
            
            # Add metadata to results
            for i, result in enumerate(results):
                if i < len(temp_files):
                    result.metadata = result.metadata or {}
                    result.metadata.update({
                        'api_version': '2.0.0',
                        'batch_processing': True,
                        'batch_size': len(files),
                        'file_index': i
                    })
            
            # Record metrics
            if performance_monitor:
                await performance_monitor.record_metric(
                    "api.batch_processing_time_ms",
                    (time.time() - start_time) * 1000,
                    tags={'batch_size': str(len(files))},
                    unit="ms"
                )
            
            return {
                "success": True,
                "results": [result.dict() for result in results],
                "batch_metadata": {
                    "total_files": len(files),
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "api_version": "2.0.0"
                }
            }
            
        finally:
            # Clean up temporary files
            for temp_file_path, _ in temp_files:
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {temp_file_path}: {e}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

@app.get("/cache/stats", response_class=JSONResponse)
async def get_cache_stats():
    """Get cache statistics"""
    try:
        if not cache_service:
            raise HTTPException(status_code=503, detail="Cache service not available")
        
        return cache_service.get_stats()
    except Exception as e:
        logger.error(f"Cache stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cache stats retrieval failed: {str(e)}")

@app.post("/cache/clear", response_class=JSONResponse)
async def clear_cache():
    """Clear all cache entries"""
    try:
        if not cache_service:
            raise HTTPException(status_code=503, detail="Cache service not available")
        
        await cache_service.clear()
        return {"success": True, "message": "Cache cleared successfully"}
    except Exception as e:
        logger.error(f"Cache clear failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {str(e)}")

@app.get("/performance/recommendations", response_class=JSONResponse)
async def get_performance_recommendations():
    """Get performance optimization recommendations"""
    try:
        if not performance_monitor:
            raise HTTPException(status_code=503, detail="Performance monitor not available")
        
        recommendations = performance_monitor.get_performance_recommendations()
        return {
            "recommendations": recommendations,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Performance recommendations retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Performance recommendations retrieval failed: {str(e)}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": time.time()
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    # Run with optimized settings
    uvicorn.run(
        "fast_main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1,  # Single worker for async processing
        loop="asyncio",
        access_log=True,
        log_level="info"
    )

















