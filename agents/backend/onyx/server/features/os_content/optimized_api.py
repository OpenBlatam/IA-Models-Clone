from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

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

from optimized_video_pipeline import OptimizedVideoPipeline, VideoConfig, AudioConfig
from optimized_nlp_service import OptimizedNLPService, ProcessingConfig
from optimized_cache_manager import OptimizedCacheManager, CacheConfig, cached
from optimized_async_processor import OptimizedAsyncProcessor, ProcessorConfig
from optimized_performance_monitor import OptimizedPerformanceMonitor, PerformanceConfig, AlertLevel
from typing import Any, List, Dict, Optional
# Import optimized components

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
video_pipeline: Optional[OptimizedVideoPipeline] = None
nlp_service: Optional[OptimizedNLPService] = None
cache_manager: Optional[OptimizedCacheManager] = None
async_processor: Optional[OptimizedAsyncProcessor] = None
performance_monitor: Optional[OptimizedPerformanceMonitor] = None

# Pydantic models
class VideoRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for video generation")
    duration: int = Field(10, ge=1, le=60, description="Video duration in seconds")
    resolution: str = Field("1920x1080", description="Video resolution")
    fps: int = Field(30, ge=15, le=60, description="Frames per second")
    quality: str = Field("high", description="Video quality (low, medium, high)")

class VideoResponse(BaseModel):
    task_id: str
    status: str
    message: str
    estimated_time: Optional[int] = None

class NLPRequest(BaseModel):
    text: str = Field(..., description="Text to analyze")
    analysis_type: str = Field("full", description="Type of analysis (sentiment, entities, keywords, full)")

class NLPResponse(BaseModel):
    sentiment: Dict[str, Any]
    entities: List[Dict[str, Any]]
    keywords: List[str]
    summary: str
    language: str
    confidence: float

class CacheRequest(BaseModel):
    key: str
    value: Any
    ttl: Optional[int] = 3600

class CacheResponse(BaseModel):
    success: bool
    message: str

class PerformanceStats(BaseModel):
    system_metrics: Dict[str, Any]
    application_metrics: Dict[str, Any]
    cache_stats: Dict[str, Any]
    processor_stats: Dict[str, Any]

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
    """Initialize all optimized components"""
    global video_pipeline, nlp_service, cache_manager, async_processor, performance_monitor
    
    try:
        logger.info("Initializing optimized components...")
        
        # Initialize performance monitor first
        perf_config = PerformanceConfig(
            collection_interval=1.0,
            enable_prometheus=True,
            enable_alerting=True,
            alert_thresholds={
                'system.cpu.usage': {'warning': 80.0, 'error': 90.0},
                'system.memory.usage': {'warning': 85.0, 'error': 95.0}
            }
        )
        performance_monitor = OptimizedPerformanceMonitor(perf_config)
        await performance_monitor.start()
        
        # Initialize cache manager
        cache_config = CacheConfig(
            max_memory_size=100 * 1024 * 1024,  # 100MB
            compression_level=3,
            enable_stats=True
        )
        cache_manager = OptimizedCacheManager(config=cache_config)
        
        # Initialize async processor
        proc_config = ProcessorConfig(
            max_workers=4,
            enable_priority_queue=True,
            enable_auto_scaling=True
        )
        async_processor = OptimizedAsyncProcessor(proc_config)
        await async_processor.start()
        
        # Initialize video pipeline
        video_pipeline = OptimizedVideoPipeline(
            device="cuda" if torch.cuda.is_available() else "cpu",
            processing_mode="gpu" if torch.cuda.is_available() else "cpu"
        )
        
        # Initialize NLP service
        nlp_config = ProcessingConfig(
            max_length=512,
            batch_size=8,
            use_gpu=True,
            cache_embeddings=True
        )
        nlp_service = OptimizedNLPService(config=nlp_config)
        
        logger.info("All optimized components initialized successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

async def shutdown_event():
    """Cleanup all optimized components"""
    global video_pipeline, nlp_service, cache_manager, async_processor, performance_monitor
    
    try:
        logger.info("Shutting down optimized components...")
        
        if async_processor:
            await async_processor.stop()
        
        if performance_monitor:
            await performance_monitor.stop()
        
        if cache_manager:
            await cache_manager.close()
        
        if video_pipeline:
            await video_pipeline.close()
        
        if nlp_service:
            await nlp_service.close()
        
        logger.info("All components shut down successfully")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# Create FastAPI app
app = FastAPI(
    title="Optimized OS Content API",
    description="High-performance content generation and analysis API",
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

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Dependency injection
async def get_cache_manager() -> OptimizedCacheManager:
    if not cache_manager:
        raise HTTPException(status_code=503, detail="Cache manager not available")
    return cache_manager

async def get_video_pipeline() -> OptimizedVideoPipeline:
    if not video_pipeline:
        raise HTTPException(status_code=503, detail="Video pipeline not available")
    return video_pipeline

async def get_nlp_service() -> OptimizedNLPService:
    if not nlp_service:
        raise HTTPException(status_code=503, detail="NLP service not available")
    return nlp_service

async def get_async_processor() -> OptimizedAsyncProcessor:
    if not async_processor:
        raise HTTPException(status_code=503, detail="Async processor not available")
    return async_processor

async def get_performance_monitor() -> OptimizedPerformanceMonitor:
    if not performance_monitor:
        raise HTTPException(status_code=503, detail="Performance monitor not available")
    return performance_monitor

# Middleware for request timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    
    """add_process_time_header function."""
start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "components": {
            "video_pipeline": video_pipeline is not None,
            "nlp_service": nlp_service is not None,
            "cache_manager": cache_manager is not None,
            "async_processor": async_processor is not None,
            "performance_monitor": performance_monitor is not None
        }
    }

# Video generation endpoints
@app.post("/video/generate", response_model=VideoResponse)
async def generate_video(
    request: VideoRequest,
    background_tasks: BackgroundTasks,
    video_pipe: OptimizedVideoPipeline = Depends(get_video_pipeline),
    cache: OptimizedCacheManager = Depends(get_cache_manager)
):
    """Generate video from text prompt"""
    try:
        # Check cache first
        cache_key = f"video:{hash(request.prompt + str(request.duration))}"
        cached_result = await cache.get(cache_key)
        if cached_result:
            return VideoResponse(
                task_id=cached_result['task_id'],
                status="completed",
                message="Video retrieved from cache",
                estimated_time=0
            )
        
        # Create video configuration
        width, height = map(int, request.resolution.split('x'))
        video_config = VideoConfig(
            fps=request.fps,
            resolution=(width, height),
            bitrate="5000k" if request.quality == "high" else "2000k"
        )
        
        # Submit task to async processor
        task_id = await async_processor.submit_task(
            video_pipe.create_video,
            request.prompt,
            request.duration,
            f"output_{task_id}.mp4",
            video_config,
            priority=TaskPriority.HIGH,
            task_type=TaskType.CPU_INTENSIVE
        )
        
        # Store in cache
        await cache.set(cache_key, {
            'task_id': task_id,
            'status': 'processing'
        }, ttl=3600)
        
        return VideoResponse(
            task_id=task_id,
            status="processing",
            message="Video generation started",
            estimated_time=request.duration * 2  # Rough estimate
        )
        
    except Exception as e:
        logger.error(f"Error generating video: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/video/status/{task_id}")
async def get_video_status(
    task_id: str,
    cache: OptimizedCacheManager = Depends(get_cache_manager)
):
    """Get video generation status"""
    try:
        # Check cache first
        cached_status = await cache.get(f"video_status:{task_id}")
        if cached_status:
            return cached_status
        
        # Get status from async processor
        status = await async_processor.get_task_status(task_id)
        
        if status == TaskStatus.COMPLETED:
            result = await async_processor.get_task_result(task_id)
            response = {
                "task_id": task_id,
                "status": "completed",
                "result": result,
                "progress": 100
            }
        else:
            response = {
                "task_id": task_id,
                "status": status.value,
                "progress": 0
            }
        
        # Cache the status
        await cache.set(f"video_status:{task_id}", response, ttl=300)
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting video status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# NLP analysis endpoints
@app.post("/nlp/analyze", response_model=NLPResponse)
@cached(ttl=1800, key_prefix="nlp")
async def analyze_text(
    request: NLPRequest,
    nlp: OptimizedNLPService = Depends(get_nlp_service)
):
    """Analyze text using NLP service"""
    try:
        result = await nlp.analyze_text(request.text)
        
        # Filter based on analysis type
        if request.analysis_type == "sentiment":
            return NLPResponse(
                sentiment=result.sentiment,
                entities=[],
                keywords=[],
                summary="",
                language=result.language,
                confidence=result.confidence
            )
        elif request.analysis_type == "entities":
            return NLPResponse(
                sentiment={},
                entities=result.entities,
                keywords=[],
                summary="",
                language=result.language,
                confidence=result.confidence
            )
        elif request.analysis_type == "keywords":
            return NLPResponse(
                sentiment={},
                entities=[],
                keywords=result.keywords,
                summary="",
                language=result.language,
                confidence=result.confidence
            )
        else:
            return NLPResponse(
                sentiment=result.sentiment,
                entities=result.entities,
                keywords=result.keywords,
                summary=result.summary,
                language=result.language,
                confidence=result.confidence
            )
        
    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/nlp/batch-analyze")
async def batch_analyze_texts(
    texts: List[str],
    nlp: OptimizedNLPService = Depends(get_nlp_service)
):
    """Analyze multiple texts in batch"""
    try:
        results = await nlp.batch_analyze(texts)
        return {"results": results}
        
    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/nlp/qa")
async def question_answering(
    question: str,
    context: str,
    nlp: OptimizedNLPService = Depends(get_nlp_service)
):
    """Answer questions using context"""
    try:
        result = await nlp.answer_question(question, context)
        return result
        
    except Exception as e:
        logger.error(f"Error in question answering: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Cache management endpoints
@app.post("/cache/set", response_model=CacheResponse)
async def set_cache(
    request: CacheRequest,
    cache: OptimizedCacheManager = Depends(get_cache_manager)
):
    """Set value in cache"""
    try:
        success = await cache.set(request.key, request.value, request.ttl)
        return CacheResponse(
            success=success,
            message="Value set successfully" if success else "Failed to set value"
        )
        
    except Exception as e:
        logger.error(f"Error setting cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cache/get/{key}")
async def get_cache(
    key: str,
    cache: OptimizedCacheManager = Depends(get_cache_manager)
):
    """Get value from cache"""
    try:
        value = await cache.get(key)
        if value is None:
            raise HTTPException(status_code=404, detail="Key not found")
        return {"key": key, "value": value}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/cache/delete/{key}")
async def delete_cache(
    key: str,
    cache: OptimizedCacheManager = Depends(get_cache_manager)
):
    """Delete value from cache"""
    try:
        success = await cache.delete(key)
        return CacheResponse(
            success=success,
            message="Value deleted successfully" if success else "Failed to delete value"
        )
        
    except Exception as e:
        logger.error(f"Error deleting cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cache/stats")
async def get_cache_stats(
    cache: OptimizedCacheManager = Depends(get_cache_manager)
):
    """Get cache statistics"""
    try:
        stats = cache.get_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Performance monitoring endpoints
@app.get("/performance/stats", response_model=PerformanceStats)
async def get_performance_stats(
    monitor: OptimizedPerformanceMonitor = Depends(get_performance_monitor),
    cache: OptimizedCacheManager = Depends(get_cache_manager),
    processor: OptimizedAsyncProcessor = Depends(get_async_processor)
):
    """Get comprehensive performance statistics"""
    try:
        # Get system metrics
        system_metrics = {
            'cpu_usage': monitor.get_metric_statistics('system.cpu.usage'),
            'memory_usage': monitor.get_metric_statistics('system.memory.usage'),
            'disk_usage': monitor.get_metric_statistics('system.disk.usage')
        }
        
        # Get application metrics
        app_metrics = {
            'process_cpu': monitor.get_metric_statistics('application.process.cpu'),
            'process_memory': monitor.get_metric_statistics('application.process.memory.percent')
        }
        
        # Get cache stats
        cache_stats = cache.get_stats()
        
        # Get processor stats
        processor_stats = processor.get_stats()
        
        return PerformanceStats(
            system_metrics=system_metrics,
            application_metrics=app_metrics,
            cache_stats=cache_stats,
            processor_stats=processor_stats
        )
        
    except Exception as e:
        logger.error(f"Error getting performance stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/performance/alerts")
async def get_alerts(
    monitor: OptimizedPerformanceMonitor = Depends(get_performance_monitor),
    level: Optional[str] = None,
    acknowledged: Optional[bool] = None
):
    """Get performance alerts"""
    try:
        alert_level = AlertLevel(level) if level else None
        alerts = monitor.get_alerts(level=alert_level, acknowledged=acknowledged)
        
        return {
            "alerts": [
                {
                    "id": alert.id,
                    "level": alert.level.value,
                    "message": alert.message,
                    "metric_name": alert.metric_name,
                    "threshold": alert.threshold,
                    "current_value": alert.current_value,
                    "timestamp": alert.timestamp,
                    "acknowledged": alert.acknowledged
                }
                for alert in alerts
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/performance/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    monitor: OptimizedPerformanceMonitor = Depends(get_performance_monitor)
):
    """Acknowledge an alert"""
    try:
        success = monitor.acknowledge_alert(alert_id)
        return {"success": success, "message": "Alert acknowledged" if success else "Alert not found"}
        
    except Exception as e:
        logger.error(f"Error acknowledging alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/performance/prometheus")
async def get_prometheus_metrics(
    monitor: OptimizedPerformanceMonitor = Depends(get_performance_monitor)
):
    """Get Prometheus metrics"""
    try:
        metrics = monitor.get_prometheus_metrics()
        return StreamingResponse(
            iter([metrics]),
            media_type="text/plain"
        )
        
    except Exception as e:
        logger.error(f"Error getting Prometheus metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/performance/report")
async def generate_performance_report(
    monitor: OptimizedPerformanceMonitor = Depends(get_performance_monitor)
):
    """Generate performance report"""
    try:
        report = monitor.generate_report()
        return {"report": report}
        
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Utility endpoints
@app.get("/metrics/{metric_name}")
async def get_metric_data(
    metric_name: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    monitor: OptimizedPerformanceMonitor = Depends(get_performance_monitor)
):
    """Get specific metric data"""
    try:
        data = monitor.get_metric(metric_name, start_time, end_time)
        return {"metric_name": metric_name, "data": data}
        
    except Exception as e:
        logger.error(f"Error getting metric data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tasks")
async def get_all_tasks(
    status: Optional[str] = None,
    processor: OptimizedAsyncProcessor = Depends(get_async_processor)
):
    """Get all tasks"""
    try:
        task_status = TaskStatus(status) if status else None
        tasks = await processor.get_all_tasks(status=task_status)
        
        return {
            "tasks": [
                {
                    "id": task.id,
                    "status": task.status.value,
                    "priority": task.priority.value,
                    "created_at": task.created_at,
                    "started_at": task.started_at,
                    "completed_at": task.completed_at,
                    "progress": task.progress
                }
                for task in tasks
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
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
        "optimized_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info"
    ) 