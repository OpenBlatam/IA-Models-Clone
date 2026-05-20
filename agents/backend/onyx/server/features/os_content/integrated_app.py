from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
from contextlib import asynccontextmanager
import uuid
from optimized_video_pipeline import OptimizedVideoPipeline, VideoConfig, AudioConfig
from optimized_nlp_service import OptimizedNLPService, ProcessingConfig
from optimized_cache_manager import OptimizedCacheManager, CacheConfig, cached
from optimized_async_processor import OptimizedAsyncProcessor, ProcessorConfig, TaskPriority, TaskType, TaskStatus
from optimized_performance_monitor import OptimizedPerformanceMonitor, PerformanceConfig, AlertLevel
from refactored_architecture import (
        import psutil
from typing import Any, List, Dict, Optional
"""
Integrated OS Content Application
Combines all optimized components with refactored architecture
"""


# Import optimized components

# Import refactored architecture
    RefactoredOSContentApplication,
    VideoProcessingUseCase,
    NLPProcessingUseCase,
    CacheManagementUseCase,
    PerformanceMonitoringUseCase,
    ProcessingRequest,
    ProcessingResult,
    Priority,
    ProcessingMode
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global application instance
app_instance: Optional[RefactoredOSContentApplication] = None

# Pydantic models for API
class VideoRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for video generation")
    duration: int = Field(10, ge=1, le=60, description="Video duration in seconds")
    resolution: str = Field("1920x1080", description="Video resolution")
    fps: int = Field(30, ge=15, le=60, description="Frames per second")
    quality: str = Field("high", description="Video quality (low, medium, high)")
    priority: str = Field("normal", description="Processing priority (low, normal, high, critical)")

class VideoResponse(BaseModel):
    request_id: str
    status: str
    message: str
    estimated_time: Optional[int] = None
    processing_time: float = 0.0

class NLPRequest(BaseModel):
    text: str = Field(..., description="Text to analyze")
    analysis_type: str = Field("full", description="Type of analysis (sentiment, entities, keywords, full)")
    priority: str = Field("normal", description="Processing priority")

class NLPResponse(BaseModel):
    request_id: str
    success: bool
    data: Dict[str, Any]
    processing_time: float
    metadata: Dict[str, Any] = {}

class CacheRequest(BaseModel):
    key: str
    value: Any
    ttl: Optional[int] = 3600

class CacheResponse(BaseModel):
    request_id: str
    success: bool
    message: str
    processing_time: float

class PerformanceStats(BaseModel):
    system_metrics: Dict[str, Any]
    application_metrics: Dict[str, Any]
    cache_stats: Dict[str, Any]
    processor_stats: Dict[str, Any]
    video_stats: Dict[str, Any]
    nlp_stats: Dict[str, Any]

class BatchRequest(BaseModel):
    requests: List[Dict[str, Any]]
    batch_type: str = Field(..., description="Type of batch processing (video, nlp, cache)")

class BatchResponse(BaseModel):
    batch_id: str
    total_requests: int
    completed_requests: int
    failed_requests: int
    results: List[Dict[str, Any]]
    processing_time: float

# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    
    """lifespan function."""
# Startup
    await startup_event()
    yield
    # Shutdown
    await shutdown_event()

async def startup_event():
    """Initialize the integrated application"""
    global app_instance
    
    try:
        logger.info("Initializing integrated OS Content application...")
        
        # Create application instance
        app_instance = RefactoredOSContentApplication()
        
        # Initialize application
        await app_instance.initialize()
        
        logger.info("Integrated application initialized successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

async def shutdown_event():
    """Cleanup the integrated application"""
    global app_instance
    
    try:
        if app_instance:
            logger.info("Shutting down integrated application...")
            await app_instance.shutdown()
            logger.info("Application shut down successfully")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# Create FastAPI app
app = FastAPI(
    title="Integrated OS Content API",
    description="High-performance content generation and analysis API with clean architecture",
    version="3.0.0",
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

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Dependency injection
def get_app() -> RefactoredOSContentApplication:
    if not app_instance:
        raise HTTPException(status_code=503, detail="Application not available")
    return app_instance

def get_video_use_case() -> VideoProcessingUseCase:
    app = get_app()
    return app.get_video_use_case()

def get_nlp_use_case() -> NLPProcessingUseCase:
    app = get_app()
    return app.get_nlp_use_case()

def get_cache_use_case() -> CacheManagementUseCase:
    app = get_app()
    return app.get_cache_use_case()

def get_performance_use_case() -> PerformanceMonitoringUseCase:
    app = get_app()
    return app.get_performance_use_case()

# Middleware for request timing and monitoring
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    
    """add_process_time_header function."""
start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Request-ID"] = str(uuid.uuid4())
    return response

# Health check endpoint
@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    if not app_instance:
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": "Application not initialized"
        }
    
    try:
        # Check all components
        video_use_case = app_instance.get_video_use_case()
        nlp_use_case = app_instance.get_nlp_use_case()
        cache_use_case = app_instance.get_cache_use_case()
        perf_use_case = app_instance.get_performance_use_case()
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "components": {
                "video_processing": video_use_case is not None,
                "nlp_processing": nlp_use_case is not None,
                "cache_management": cache_use_case is not None,
                "performance_monitoring": perf_use_case is not None
            },
            "version": "3.0.0"
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e)
        }

# Video processing endpoints
@app.post("/video/generate", response_model=VideoResponse)
async def generate_video(
    request: VideoRequest,
    background_tasks: BackgroundTasks,
    video_use_case: VideoProcessingUseCase = Depends(get_video_use_case)
):
    """Generate video from text prompt using clean architecture"""
    start_time = time.time()
    
    try:
        # Convert priority string to enum
        priority_map = {
            "low": Priority.LOW,
            "normal": Priority.NORMAL,
            "high": Priority.HIGH,
            "critical": Priority.CRITICAL
        }
        priority = priority_map.get(request.priority, Priority.NORMAL)
        
        # Create processing request
        processing_request = ProcessingRequest(
            data={
                "prompt": request.prompt,
                "duration": request.duration,
                "resolution": request.resolution,
                "fps": request.fps,
                "quality": request.quality
            },
            priority=priority,
            mode=ProcessingMode.ASYNC
        )
        
        # Process video generation
        result = await video_use_case.generate_video(
            request.prompt,
            request.duration,
            resolution=request.resolution,
            fps=request.fps,
            quality=request.quality
        )
        
        processing_time = time.time() - start_time
        
        return VideoResponse(
            request_id=result.request_id,
            status="processing" if result.success else "failed",
            message=result.error if not result.success else "Video generation started",
            estimated_time=request.duration * 2 if result.success else None,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error generating video: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/video/status/{request_id}")
async def get_video_status(
    request_id: str,
    video_use_case: VideoProcessingUseCase = Depends(get_video_use_case)
):
    """Get video generation status"""
    start_time = time.time()
    
    try:
        result = await video_use_case.get_video_status(request_id)
        
        processing_time = time.time() - start_time
        
        return {
            "request_id": request_id,
            "success": result.success,
            "status": "completed" if result.success and result.data else "processing",
            "data": result.data,
            "error": result.error,
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"Error getting video status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/video/cancel/{request_id}")
async def cancel_video(
    request_id: str,
    video_use_case: VideoProcessingUseCase = Depends(get_video_use_case)
):
    """Cancel video generation"""
    start_time = time.time()
    
    try:
        result = await video_use_case.cancel_video(request_id)
        
        processing_time = time.time() - start_time
        
        return {
            "request_id": request_id,
            "success": result.success,
            "message": "Video cancelled successfully" if result.success else "Failed to cancel video",
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"Error cancelling video: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# NLP processing endpoints
@app.post("/nlp/analyze", response_model=NLPResponse)
async def analyze_text(
    request: NLPRequest,
    nlp_use_case: NLPProcessingUseCase = Depends(get_nlp_use_case)
):
    """Analyze text using NLP service with clean architecture"""
    start_time = time.time()
    
    try:
        result = await nlp_use_case.analyze_text(request.text, request.analysis_type)
        
        processing_time = time.time() - start_time
        
        return NLPResponse(
            request_id=result.request_id,
            success=result.success,
            data=result.data if result.success else {},
            processing_time=processing_time,
            metadata=result.metadata
        )
        
    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/nlp/batch-analyze")
async def batch_analyze_texts(
    texts: List[str],
    nlp_use_case: NLPProcessingUseCase = Depends(get_nlp_use_case)
):
    """Analyze multiple texts in batch"""
    start_time = time.time()
    
    try:
        result = await nlp_use_case.batch_analyze(texts)
        
        processing_time = time.time() - start_time
        
        return {
            "batch_id": result.request_id,
            "success": result.success,
            "total_texts": len(texts),
            "results": result.data if result.success else [],
            "processing_time": processing_time,
            "error": result.error
        }
        
    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/nlp/qa")
async def question_answering(
    question: str,
    context: str,
    nlp_use_case: NLPProcessingUseCase = Depends(get_nlp_use_case)
):
    """Answer questions using context"""
    start_time = time.time()
    
    try:
        result = await nlp_use_case.answer_question(question, context)
        
        processing_time = time.time() - start_time
        
        return {
            "request_id": result.request_id,
            "success": result.success,
            "answer": result.data if result.success else None,
            "processing_time": processing_time,
            "error": result.error
        }
        
    except Exception as e:
        logger.error(f"Error in question answering: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Cache management endpoints
@app.post("/cache/set", response_model=CacheResponse)
async def set_cache(
    request: CacheRequest,
    cache_use_case: CacheManagementUseCase = Depends(get_cache_use_case)
):
    """Set value in cache"""
    start_time = time.time()
    
    try:
        result = await cache_use_case.set(request.key, request.value, request.ttl)
        
        processing_time = time.time() - start_time
        
        return CacheResponse(
            request_id=result.request_id,
            success=result.success,
            message="Value set successfully" if result.success else "Failed to set value",
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error setting cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cache/get/{key}")
async def get_cache(
    key: str,
    cache_use_case: CacheManagementUseCase = Depends(get_cache_use_case)
):
    """Get value from cache"""
    start_time = time.time()
    
    try:
        result = await cache_use_case.get(key)
        
        processing_time = time.time() - start_time
        
        if not result.success:
            raise HTTPException(status_code=404, detail="Key not found")
        
        return {
            "key": key,
            "value": result.data,
            "processing_time": processing_time
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/cache/delete/{key}")
async def delete_cache(
    key: str,
    cache_use_case: CacheManagementUseCase = Depends(get_cache_use_case)
):
    """Delete value from cache"""
    start_time = time.time()
    
    try:
        result = await cache_use_case.delete(key)
        
        processing_time = time.time() - start_time
        
        return CacheResponse(
            request_id=result.request_id,
            success=result.success,
            message="Value deleted successfully" if result.success else "Failed to delete value",
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error deleting cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cache/stats")
async def get_cache_stats(
    cache_use_case: CacheManagementUseCase = Depends(get_cache_use_case)
):
    """Get cache statistics"""
    try:
        # This would need to be implemented in the cache use case
        # For now, return a placeholder
        return {
            "cache_stats": "Not implemented in use case yet",
            "message": "Use direct cache service for stats"
        }
        
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Performance monitoring endpoints
@app.get("/performance/stats", response_model=PerformanceStats)
async def get_performance_stats(
    perf_use_case: PerformanceMonitoringUseCase = Depends(get_performance_use_case)
):
    """Get comprehensive performance statistics"""
    try:
        # Get metrics for all system components
        metrics_result = await perf_use_case.get_metrics([
            "system.cpu.usage",
            "system.memory.usage",
            "system.disk.usage",
            "application.process.cpu",
            "application.process.memory.percent"
        ])
        
        # Get alerts
        alerts_result = await perf_use_case.get_alerts()
        
        return PerformanceStats(
            system_metrics=metrics_result.data.get('current', {}) if metrics_result.success else {},
            application_metrics=metrics_result.data.get('historical', {}) if metrics_result.success else {},
            cache_stats={"status": "Not available through use case"},
            processor_stats={"status": "Not available through use case"},
            video_stats={"status": "Not available through use case"},
            nlp_stats={"status": "Not available through use case"}
        )
        
    except Exception as e:
        logger.error(f"Error getting performance stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/performance/alerts")
async def get_alerts(
    perf_use_case: PerformanceMonitoringUseCase = Depends(get_performance_use_case),
    level: Optional[str] = None
):
    """Get performance alerts"""
    try:
        result = await perf_use_case.get_alerts(level)
        
        return {
            "alerts": result.data if result.success else [],
            "total_alerts": len(result.data) if result.success else 0
        }
        
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/performance/report")
async def generate_performance_report(
    perf_use_case: PerformanceMonitoringUseCase = Depends(get_performance_use_case)
):
    """Generate performance report"""
    try:
        result = await perf_use_case.generate_report()
        
        return {
            "report": result.data if result.success else "Failed to generate report",
            "success": result.success
        }
        
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Batch processing endpoints
@app.post("/batch/process", response_model=BatchResponse)
async def process_batch(
    request: BatchRequest,
    background_tasks: BackgroundTasks
):
    """Process multiple requests in batch"""
    start_time = time.time()
    
    try:
        batch_id = str(uuid.uuid4())
        total_requests = len(request.requests)
        completed_requests = 0
        failed_requests = 0
        results = []
        
        # Process based on batch type
        if request.batch_type == "nlp":
            nlp_use_case = get_nlp_use_case()
            texts = [req.get("text", "") for req in request.requests]
            batch_result = await nlp_use_case.batch_analyze(texts)
            
            if batch_result.success:
                completed_requests = len(batch_result.data)
                results = batch_result.data
            else:
                failed_requests = total_requests
                
        elif request.batch_type == "video":
            video_use_case = get_video_use_case()
            video_results = []
            
            for req in request.requests:
                try:
                    result = await video_use_case.generate_video(
                        req.get("prompt", ""),
                        req.get("duration", 10)
                    )
                    if result.success:
                        completed_requests += 1
                    else:
                        failed_requests += 1
                    video_results.append(result)
                except Exception:
                    failed_requests += 1
                    video_results.append({"error": "Failed to process"})
            
            results = video_results
            
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported batch type: {request.batch_type}")
        
        processing_time = time.time() - start_time
        
        return BatchResponse(
            batch_id=batch_id,
            total_requests=total_requests,
            completed_requests=completed_requests,
            failed_requests=failed_requests,
            results=results,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Utility endpoints
@app.get("/metrics/{metric_name}")
async def get_metric_data(
    metric_name: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    perf_use_case: PerformanceMonitoringUseCase = Depends(get_performance_use_case)
):
    """Get specific metric data"""
    try:
        result = await perf_use_case.get_metrics([metric_name])
        
        if not result.success:
            raise HTTPException(status_code=404, detail="Metric not found")
        
        return {
            "metric_name": metric_name,
            "data": result.data.get('historical', {}).get(metric_name, [])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting metric data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system/info")
async def get_system_info():
    """Get system information"""
    try:
        
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "disk_total": psutil.disk_usage('/').total,
            "disk_free": psutil.disk_usage('/').free,
            "python_version": "3.8+",
            "platform": "Linux/Windows/macOS"
        }
        
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "integrated_app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info"
    ) 