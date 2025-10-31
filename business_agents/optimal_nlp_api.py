"""
Optimal NLP API
===============

API REST óptima para el sistema NLP con máximo rendimiento,
optimizaciones avanzadas y capacidades de producción.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
import asyncio
import logging

from .optimal_nlp_system import optimal_nlp_system, OptimalConfig, OptimizationLevel, ProcessingMode
from .nlp_cache import nlp_cache
from .nlp_metrics import nlp_monitoring
from .nlp_trends import nlp_trend_analyzer
from .exceptions import NLPProcessingError, ModelLoadError

logger = logging.getLogger(__name__)

# Create optimal router
router = APIRouter(prefix="/optimal-nlp", tags=["Optimal NLP"])

# Optimal request/response models
class OptimalAnalysisRequest(BaseModel):
    """Request model for optimal text analysis."""
    text: str = Field(..., description="Text to analyze", min_length=1, max_length=100000)
    language: Optional[str] = Field(default="en", description="Language code")
    tasks: Optional[List[str]] = Field(
        default=["sentiment", "entities", "keywords", "readability"],
        description="Analysis tasks to perform"
    )
    use_cache: bool = Field(default=True, description="Use intelligent caching")
    quality_check: bool = Field(default=True, description="Perform quality assessment")
    parallel_processing: bool = Field(default=True, description="Use parallel processing")
    optimization_level: Optional[str] = Field(
        default="maximum",
        description="Optimization level: minimal, balanced, maximum, ultra"
    )

class OptimalAnalysisResponse(BaseModel):
    """Response model for optimal text analysis."""
    text: str
    language: str
    analysis: Dict[str, Any]
    processing_time: float
    cache_hit: bool
    quality_score: Optional[float] = None
    quality_assessment: Optional[Dict[str, Any]] = None
    optimization_level: str
    parallel_processing: bool
    timestamp: datetime

class OptimalBatchRequest(BaseModel):
    """Request model for optimal batch analysis."""
    texts: List[str] = Field(..., description="Texts to analyze", min_items=1, max_items=1000)
    language: Optional[str] = Field(default="en", description="Language code")
    tasks: Optional[List[str]] = Field(
        default=["sentiment", "entities", "keywords"],
        description="Analysis tasks to perform"
    )
    use_cache: bool = Field(default=True, description="Use intelligent caching")
    parallel_processing: bool = Field(default=True, description="Use parallel processing")
    batch_size: Optional[int] = Field(default=64, description="Batch size for processing", ge=1, le=256)

class OptimalBatchResponse(BaseModel):
    """Response model for optimal batch analysis."""
    results: List[OptimalAnalysisResponse]
    total_processing_time: float
    average_processing_time: float
    success_count: int
    error_count: int
    cache_hit_rate: float
    average_quality_score: float
    throughput_per_second: float
    parallel_processing: bool
    batch_size: int
    timestamp: datetime

class SystemOptimizationRequest(BaseModel):
    """Request model for system optimization."""
    optimization_level: str = Field(..., description="Optimization level")
    processing_mode: str = Field(..., description="Processing mode")
    max_workers: Optional[int] = Field(default=None, description="Maximum workers")
    batch_size: Optional[int] = Field(default=None, description="Batch size")
    memory_limit_gb: Optional[float] = Field(default=None, description="Memory limit in GB")
    cache_size_mb: Optional[int] = Field(default=None, description="Cache size in MB")

class SystemOptimizationResponse(BaseModel):
    """Response model for system optimization."""
    optimization_level: str
    processing_mode: str
    max_workers: int
    batch_size: int
    memory_limit_gb: float
    cache_size_mb: int
    gpu_available: bool
    gpu_device: str
    optimization_applied: bool
    timestamp: datetime

class PerformanceMetricsResponse(BaseModel):
    """Response model for performance metrics."""
    system_status: Dict[str, Any]
    performance_stats: Dict[str, Any]
    quality_stats: Dict[str, Any]
    memory_status: Dict[str, Any]
    cache_status: Dict[str, Any]
    optimization_level: str
    processing_mode: str
    timestamp: datetime

class QualityAssessmentRequest(BaseModel):
    """Request model for quality assessment."""
    text: str = Field(..., description="Text to assess", min_length=1, max_length=10000)
    language: Optional[str] = Field(default="en", description="Language code")
    detailed_assessment: bool = Field(default=True, description="Include detailed assessment")

class QualityAssessmentResponse(BaseModel):
    """Response model for quality assessment."""
    text: str
    overall_quality: float
    sentiment_quality: float
    entity_quality: float
    keyword_quality: float
    readability_quality: float
    completeness: float
    recommendations: List[str]
    processing_time: float
    timestamp: datetime

# Dependency to ensure optimal NLP system is initialized
async def get_optimal_nlp_system():
    """Get initialized optimal NLP system."""
    if not optimal_nlp_system.is_initialized:
        await optimal_nlp_system.initialize()
    return optimal_nlp_system

# Optimal API Endpoints

@router.get("/health", response_model=Dict[str, Any])
async def get_optimal_health():
    """Get comprehensive optimal system health status."""
    try:
        status = await optimal_nlp_system.get_optimal_status()
        return status
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {e}")

@router.post("/analyze", response_model=OptimalAnalysisResponse)
async def analyze_text_optimal(request: OptimalAnalysisRequest):
    """Perform optimal text analysis with maximum performance."""
    try:
        # Apply optimization level if specified
        if request.optimization_level:
            try:
                opt_level = OptimizationLevel(request.optimization_level)
                optimal_nlp_system.config.optimization_level = opt_level
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid optimization level: {request.optimization_level}"
                )
        
        result = await optimal_nlp_system.analyze_text_optimal(
            text=request.text,
            language=request.language or "en",
            tasks=request.tasks,
            use_cache=request.use_cache,
            quality_check=request.quality_check,
            parallel_processing=request.parallel_processing
        )
        
        return OptimalAnalysisResponse(
            text=result['text'] if 'text' in result else request.text,
            language=request.language or "en",
            analysis=result,
            processing_time=result.get('processing_time', 0),
            cache_hit=result.get('cache_hit', False),
            quality_score=result.get('quality_score'),
            quality_assessment=result.get('quality_assessment'),
            optimization_level=result.get('optimization_level', 'maximum'),
            parallel_processing=request.parallel_processing,
            timestamp=datetime.fromisoformat(result.get('timestamp', datetime.now().isoformat()))
        )
        
    except NLPProcessingError as e:
        logger.error(f"NLP processing error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Optimal analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Optimal analysis failed: {e}")

@router.post("/batch", response_model=OptimalBatchResponse)
async def batch_analyze_optimal(request: OptimalBatchRequest):
    """Perform optimal batch analysis with maximum performance."""
    try:
        start_time = datetime.now()
        
        # Apply batch size if specified
        if request.batch_size:
            optimal_nlp_system.config.batch_size = request.batch_size
        
        results = await optimal_nlp_system.batch_analyze_optimal(
            texts=request.texts,
            language=request.language or "en",
            tasks=request.tasks,
            use_cache=request.use_cache,
            parallel_processing=request.parallel_processing
        )
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate statistics
        success_count = len([r for r in results if 'error' not in r])
        error_count = len(results) - success_count
        cache_hits = len([r for r in results if r.get('cache_hit', False)])
        cache_hit_rate = cache_hits / len(results) if results else 0
        
        quality_scores = [r.get('quality_score', 0) for r in results if 'quality_score' in r]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        throughput = len(results) / total_time if total_time > 0 else 0
        
        # Convert to response format
        response_results = []
        for result in results:
            if 'error' in result:
                response_results.append(OptimalAnalysisResponse(
                    text=result.get('text', ''),
                    language=request.language or "en",
                    analysis={'error': result['error']},
                    processing_time=0,
                    cache_hit=False,
                    optimization_level='maximum',
                    parallel_processing=request.parallel_processing,
                    timestamp=datetime.now()
                ))
            else:
                response_results.append(OptimalAnalysisResponse(
                    text=result.get('text', ''),
                    language=request.language or "en",
                    analysis=result,
                    processing_time=result.get('processing_time', 0),
                    cache_hit=result.get('cache_hit', False),
                    quality_score=result.get('quality_score'),
                    quality_assessment=result.get('quality_assessment'),
                    optimization_level=result.get('optimization_level', 'maximum'),
                    parallel_processing=request.parallel_processing,
                    timestamp=datetime.fromisoformat(result.get('timestamp', datetime.now().isoformat()))
                ))
        
        return OptimalBatchResponse(
            results=response_results,
            total_processing_time=total_time,
            average_processing_time=total_time / len(results) if results else 0,
            success_count=success_count,
            error_count=error_count,
            cache_hit_rate=cache_hit_rate,
            average_quality_score=avg_quality,
            throughput_per_second=throughput,
            parallel_processing=request.parallel_processing,
            batch_size=request.batch_size or 64,
            timestamp=start_time
        )
        
    except Exception as e:
        logger.error(f"Optimal batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Optimal batch analysis failed: {e}")

@router.post("/optimize", response_model=SystemOptimizationResponse)
async def optimize_system(request: SystemOptimizationRequest):
    """Optimize system configuration for maximum performance."""
    try:
        # Apply optimization level
        try:
            opt_level = OptimizationLevel(request.optimization_level)
            optimal_nlp_system.config.optimization_level = opt_level
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid optimization level: {request.optimization_level}"
            )
        
        # Apply processing mode
        try:
            proc_mode = ProcessingMode(request.processing_mode)
            optimal_nlp_system.config.processing_mode = proc_mode
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid processing mode: {request.processing_mode}"
            )
        
        # Apply other optimizations
        if request.max_workers:
            optimal_nlp_system.config.max_workers = request.max_workers
        
        if request.batch_size:
            optimal_nlp_system.config.batch_size = request.batch_size
        
        if request.memory_limit_gb:
            optimal_nlp_system.config.memory_limit_gb = request.memory_limit_gb
        
        if request.cache_size_mb:
            optimal_nlp_system.config.cache_size_mb = request.cache_size_mb
        
        return SystemOptimizationResponse(
            optimization_level=optimal_nlp_system.config.optimization_level.value,
            processing_mode=optimal_nlp_system.config.processing_mode.value,
            max_workers=optimal_nlp_system.config.max_workers,
            batch_size=optimal_nlp_system.config.batch_size,
            memory_limit_gb=optimal_nlp_system.config.memory_limit_gb,
            cache_size_mb=optimal_nlp_system.config.cache_size_mb,
            gpu_available=optimal_nlp_system.gpu_available,
            gpu_device=optimal_nlp_system.gpu_device,
            optimization_applied=True,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"System optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"System optimization failed: {e}")

@router.get("/metrics", response_model=PerformanceMetricsResponse)
async def get_optimal_metrics():
    """Get comprehensive optimal system metrics."""
    try:
        # Get system status
        system_status = await optimal_nlp_system.get_optimal_status()
        
        # Get performance statistics
        performance_stats = system_status.get('performance', {})
        
        # Get quality statistics
        quality_stats = system_status.get('quality', {})
        
        # Get memory status
        memory_status = system_status.get('memory', {})
        
        # Get cache status
        cache_status = system_status.get('cache', {})
        
        return PerformanceMetricsResponse(
            system_status=system_status.get('system', {}),
            performance_stats=performance_stats,
            quality_stats=quality_stats,
            memory_status=memory_status,
            cache_status=cache_status,
            optimization_level=optimal_nlp_system.config.optimization_level.value,
            processing_mode=optimal_nlp_system.config.processing_mode.value,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Failed to get optimal metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get optimal metrics: {e}")

@router.post("/quality", response_model=QualityAssessmentResponse)
async def assess_quality_optimal(request: QualityAssessmentRequest):
    """Assess text quality with optimal performance."""
    try:
        start_time = datetime.now()
        
        # Perform analysis
        result = await optimal_nlp_system.analyze_text_optimal(
            text=request.text,
            language=request.language or "en",
            tasks=["sentiment", "entities", "keywords", "readability"],
            use_cache=True,
            quality_check=True,
            parallel_processing=True
        )
        
        # Extract quality assessment
        quality_assessment = result.get('quality_assessment', {})
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return QualityAssessmentResponse(
            text=request.text,
            overall_quality=quality_assessment.get('overall_quality', 0.0),
            sentiment_quality=quality_assessment.get('sentiment_quality', 0.0),
            entity_quality=quality_assessment.get('entity_quality', 0.0),
            keyword_quality=quality_assessment.get('keyword_quality', 0.0),
            readability_quality=quality_assessment.get('readability_quality', 0.0),
            completeness=quality_assessment.get('completeness', 0.0),
            recommendations=quality_assessment.get('recommendations', []),
            processing_time=processing_time,
            timestamp=start_time
        )
        
    except Exception as e:
        logger.error(f"Quality assessment failed: {e}")
        raise HTTPException(status_code=500, detail=f"Quality assessment failed: {e}")

@router.get("/performance", response_model=Dict[str, Any])
async def get_performance_benchmark():
    """Get performance benchmark results."""
    try:
        # Get current performance metrics
        status = await optimal_nlp_system.get_optimal_status()
        
        # Calculate performance indicators
        performance = status.get('performance', {})
        
        benchmark_results = {
            'throughput': {
                'requests_per_second': performance.get('requests_processed', 0) / max(performance.get('average_processing_time', 1), 1),
                'average_processing_time': performance.get('average_processing_time', 0),
                'success_rate': performance.get('success_rate', 0)
            },
            'cache_performance': {
                'hit_rate': performance.get('cache_hit_rate', 0),
                'cache_hits': performance.get('cache_hits', 0),
                'cache_misses': performance.get('cache_misses', 0)
            },
            'quality_metrics': status.get('quality', {}),
            'memory_usage': status.get('memory', {}),
            'optimization_level': optimal_nlp_system.config.optimization_level.value,
            'processing_mode': optimal_nlp_system.config.processing_mode.value,
            'gpu_available': optimal_nlp_system.gpu_available,
            'timestamp': datetime.now().isoformat()
        }
        
        return benchmark_results
        
    except Exception as e:
        logger.error(f"Performance benchmark failed: {e}")
        raise HTTPException(status_code=500, detail=f"Performance benchmark failed: {e}")

@router.post("/stress-test")
async def run_stress_test(
    concurrent_requests: int = Query(10, description="Number of concurrent requests", ge=1, le=100),
    text_length: int = Query(100, description="Text length for testing", ge=10, le=1000)
):
    """Run stress test with optimal performance."""
    try:
        # Generate test text
        test_text = "This is a stress test for the optimal NLP system. " * (text_length // 50)
        
        start_time = datetime.now()
        
        # Create concurrent tasks
        tasks = []
        for i in range(concurrent_requests):
            task = optimal_nlp_system.analyze_text_optimal(
                text=f"{test_text} Request #{i+1}",
                use_cache=True,
                quality_check=True,
                parallel_processing=True
            )
            tasks.append(task)
        
        # Execute stress test
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Analyze results
        success_count = len([r for r in results if not isinstance(r, Exception)])
        error_count = len(results) - success_count
        throughput = concurrent_requests / total_time if total_time > 0 else 0
        
        stress_test_results = {
            'concurrent_requests': concurrent_requests,
            'text_length': text_length,
            'total_time': total_time,
            'throughput_per_second': throughput,
            'success_count': success_count,
            'error_count': error_count,
            'success_rate': (success_count / concurrent_requests) * 100,
            'average_time_per_request': total_time / concurrent_requests,
            'optimization_level': optimal_nlp_system.config.optimization_level.value,
            'processing_mode': optimal_nlp_system.config.processing_mode.value,
            'timestamp': start_time.isoformat()
        }
        
        return stress_test_results
        
    except Exception as e:
        logger.error(f"Stress test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Stress test failed: {e}")

@router.get("/capabilities", response_model=Dict[str, Any])
async def get_optimal_capabilities():
    """Get optimal system capabilities and features."""
    return {
        "optimization_levels": [
            "minimal", "balanced", "maximum", "ultra"
        ],
        "processing_modes": [
            "cpu_only", "gpu_accelerated", "hybrid", "distributed"
        ],
        "analysis_tasks": [
            "sentiment", "entities", "keywords", "readability", "topics", "embeddings"
        ],
        "optimizations": [
            "intelligent_caching",
            "parallel_processing",
            "gpu_acceleration",
            "memory_optimization",
            "quality_assessment",
            "performance_monitoring"
        ],
        "performance_features": [
            "batch_processing",
            "concurrent_processing",
            "model_caching",
            "memory_monitoring",
            "quality_tracking",
            "stress_testing"
        ],
        "supported_languages": [
            "en", "es", "fr", "de", "it", "pt", "zh", "ja", "ko", "ru", "ar"
        ],
        "max_text_length": 100000,
        "max_batch_size": 1000,
        "max_concurrent_requests": 100,
        "cache_enabled": True,
        "gpu_acceleration": optimal_nlp_system.gpu_available,
        "parallel_processing": True,
        "quality_assessment": True,
        "performance_monitoring": True
    }

# Utility endpoints
@router.get("/supported-languages", response_model=List[str])
async def get_supported_languages():
    """Get list of supported languages."""
    return ["en", "es", "fr", "de", "it", "pt", "zh", "ja", "ko", "ru", "ar"]

@router.get("/optimization-levels", response_model=List[str])
async def get_optimization_levels():
    """Get available optimization levels."""
    return ["minimal", "balanced", "maximum", "ultra"]

@router.get("/processing-modes", response_model=List[str])
async def get_processing_modes():
    """Get available processing modes."""
    return ["cpu_only", "gpu_accelerated", "hybrid", "distributed"]

@router.get("/analysis-tasks", response_model=List[str])
async def get_analysis_tasks():
    """Get available analysis tasks."""
    return ["sentiment", "entities", "keywords", "readability", "topics", "embeddings"]












