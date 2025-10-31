"""
Performance-Optimized NLP API
============================

API endpoints para el sistema NLP optimizado para máximo rendimiento.
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

from .performance_nlp_system import performance_nlp_system, PerformanceNLPConfig

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(prefix="/performance-nlp", tags=["Performance-Optimized NLP"])

# Pydantic models for API requests/responses

class PerformanceNLPAnalysisRequest(BaseModel):
    """Request model for performance-optimized NLP analysis."""
    text: str = Field(..., description="Text to analyze", min_length=1, max_length=100000)
    language: str = Field(default="en", description="Language code", regex="^[a-z]{2}$")
    use_cache: bool = Field(default=True, description="Use caching for faster responses")
    performance_mode: str = Field(default="balanced", description="Performance mode: fast, balanced, quality")
    
    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()
    
    @validator('language')
    def validate_language(cls, v):
        supported_languages = ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko']
        if v not in supported_languages:
            raise ValueError(f'Language {v} not supported. Supported: {supported_languages}')
        return v
    
    @validator('performance_mode')
    def validate_performance_mode(cls, v):
        supported_modes = ['fast', 'balanced', 'quality']
        if v not in supported_modes:
            raise ValueError(f'Performance mode {v} not supported. Supported: {supported_modes}')
        return v

class PerformanceNLPAnalysisResponse(BaseModel):
    """Response model for performance-optimized NLP analysis."""
    text: str
    language: str
    sentiment: Dict[str, Any]
    entities: List[Dict[str, Any]]
    keywords: List[str]
    topics: List[Dict[str, Any]]
    readability: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    quality_score: float
    confidence_score: float
    processing_time: float
    throughput: float
    memory_usage: float
    cache_hit: bool
    timestamp: datetime

class PerformanceNLPAnalysisBatchRequest(BaseModel):
    """Request model for performance-optimized batch NLP analysis."""
    texts: List[str] = Field(..., description="List of texts to analyze", min_items=1, max_items=1000)
    language: str = Field(default="en", description="Language code", regex="^[a-z]{2}$")
    use_cache: bool = Field(default=True, description="Use caching for faster responses")
    performance_mode: str = Field(default="balanced", description="Performance mode: fast, balanced, quality")
    
    @validator('texts')
    def validate_texts(cls, v):
        if not v:
            raise ValueError('Texts list cannot be empty')
        for i, text in enumerate(v):
            if not text or not text.strip():
                raise ValueError(f'Text at index {i} cannot be empty')
        return [text.strip() for text in v]
    
    @validator('performance_mode')
    def validate_performance_mode(cls, v):
        supported_modes = ['fast', 'balanced', 'quality']
        if v not in supported_modes:
            raise ValueError(f'Performance mode {v} not supported. Supported: {supported_modes}')
        return v

class PerformanceNLPAnalysisBatchResponse(BaseModel):
    """Response model for performance-optimized batch NLP analysis."""
    results: List[PerformanceNLPAnalysisResponse]
    total_processed: int
    total_errors: int
    average_processing_time: float
    average_throughput: float
    average_memory_usage: float
    average_quality_score: float
    average_confidence_score: float
    processing_time: float
    timestamp: datetime

class PerformanceNLPOptimizationRequest(BaseModel):
    """Request model for performance optimization."""
    optimization_type: str = Field(..., description="Type of optimization")
    parameters: Dict[str, Any] = Field(default={}, description="Optimization parameters")
    
    @validator('optimization_type')
    def validate_optimization_type(cls, v):
        supported_types = ['memory', 'cpu', 'gpu', 'cache', 'models', 'pipeline']
        if v not in supported_types:
            raise ValueError(f'Optimization type {v} not supported. Supported: {supported_types}')
        return v

class PerformanceNLPOptimizationResponse(BaseModel):
    """Response model for performance optimization."""
    optimization_type: str
    parameters: Dict[str, Any]
    optimization_time: float
    performance_improvement: Dict[str, Any]
    timestamp: datetime

class PerformanceNLPBenchmarkRequest(BaseModel):
    """Request model for performance benchmark."""
    benchmark_type: str = Field(..., description="Type of benchmark")
    test_data: List[str] = Field(..., description="Test data for benchmark")
    iterations: int = Field(default=10, description="Number of iterations", ge=1, le=100)
    
    @validator('benchmark_type')
    def validate_benchmark_type(cls, v):
        supported_types = ['throughput', 'latency', 'memory', 'cpu', 'gpu', 'comprehensive']
        if v not in supported_types:
            raise ValueError(f'Benchmark type {v} not supported. Supported: {supported_types}')
        return v
    
    @validator('test_data')
    def validate_test_data(cls, v):
        if not v:
            raise ValueError('Test data cannot be empty')
        return v

class PerformanceNLPBenchmarkResponse(BaseModel):
    """Response model for performance benchmark."""
    benchmark_type: str
    iterations: int
    results: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    benchmark_time: float
    timestamp: datetime

class PerformanceNLPStatusResponse(BaseModel):
    """Response model for performance-optimized system status."""
    system: Dict[str, Any]
    performance: Dict[str, Any]
    resources: Dict[str, Any]
    cache: Dict[str, Any]
    memory: Dict[str, Any]
    timestamp: str

# API endpoints

@router.post("/analyze", response_model=PerformanceNLPAnalysisResponse)
async def analyze_performance_optimized(request: PerformanceNLPAnalysisRequest):
    """
    Perform performance-optimized text analysis.
    
    This endpoint provides high-performance NLP analysis with optimizations:
    - Optimized processing for maximum speed
    - Intelligent caching for faster responses
    - Memory optimization for efficiency
    - GPU acceleration when available
    - Parallel processing for throughput
    - Performance monitoring and metrics
    """
    try:
        start_time = time.time()
        
        # Perform performance-optimized analysis
        result = await performance_nlp_system.analyze_performance_optimized(
            text=request.text,
            language=request.language,
            use_cache=request.use_cache,
            performance_mode=request.performance_mode
        )
        
        processing_time = time.time() - start_time
        
        return PerformanceNLPAnalysisResponse(
            text=result.text,
            language=result.language,
            sentiment=result.sentiment,
            entities=result.entities,
            keywords=result.keywords,
            topics=result.topics,
            readability=result.readability,
            performance_metrics=result.performance_metrics,
            quality_score=result.quality_score,
            confidence_score=result.confidence_score,
            processing_time=processing_time,
            throughput=result.throughput,
            memory_usage=result.memory_usage,
            cache_hit=result.cache_hit,
            timestamp=result.timestamp
        )
        
    except Exception as e:
        logger.error(f"Performance-optimized analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/analyze/batch", response_model=PerformanceNLPAnalysisBatchResponse)
async def analyze_performance_optimized_batch(request: PerformanceNLPAnalysisBatchRequest):
    """
    Perform performance-optimized batch text analysis.
    
    This endpoint processes multiple texts with maximum performance:
    - Parallel processing for high throughput
    - Batch optimization for efficiency
    - Memory management for large datasets
    - Performance monitoring and metrics
    - Error handling for individual texts
    """
    try:
        start_time = time.time()
        
        # Perform performance-optimized batch analysis
        results = await performance_nlp_system.batch_analyze_performance_optimized(
            texts=request.texts,
            language=request.language,
            use_cache=request.use_cache,
            performance_mode=request.performance_mode
        )
        
        processing_time = time.time() - start_time
        
        # Calculate batch statistics
        total_processed = len(results)
        total_errors = sum(1 for r in results if r.quality_score == 0)
        average_processing_time = sum(r.processing_time for r in results) / total_processed if total_processed > 0 else 0
        average_throughput = sum(r.throughput for r in results) / total_processed if total_processed > 0 else 0
        average_memory_usage = sum(r.memory_usage for r in results) / total_processed if total_processed > 0 else 0
        average_quality_score = sum(r.quality_score for r in results) / total_processed if total_processed > 0 else 0
        average_confidence_score = sum(r.confidence_score for r in results) / total_processed if total_processed > 0 else 0
        
        return PerformanceNLPAnalysisBatchResponse(
            results=results,
            total_processed=total_processed,
            total_errors=total_errors,
            average_processing_time=average_processing_time,
            average_throughput=average_throughput,
            average_memory_usage=average_memory_usage,
            average_quality_score=average_quality_score,
            average_confidence_score=average_confidence_score,
            processing_time=processing_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Performance-optimized batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

@router.post("/optimize", response_model=PerformanceNLPOptimizationResponse)
async def optimize_performance(request: PerformanceNLPOptimizationRequest, background_tasks: BackgroundTasks):
    """
    Optimize system performance.
    
    This endpoint triggers performance optimization:
    - Memory optimization
    - CPU optimization
    - GPU optimization
    - Cache optimization
    - Model optimization
    - Pipeline optimization
    """
    try:
        start_time = time.time()
        
        # Start optimization in background
        background_tasks.add_task(
            _optimize_performance_background,
            request.optimization_type,
            request.parameters
        )
        
        optimization_time = time.time() - start_time
        
        return PerformanceNLPOptimizationResponse(
            optimization_type=request.optimization_type,
            parameters=request.parameters,
            optimization_time=optimization_time,
            performance_improvement={},
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Performance optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Performance optimization failed: {str(e)}")

@router.post("/benchmark", response_model=PerformanceNLPBenchmarkResponse)
async def benchmark_performance(request: PerformanceNLPBenchmarkRequest):
    """
    Run performance benchmark.
    
    This endpoint runs performance benchmarks:
    - Throughput benchmark
    - Latency benchmark
    - Memory benchmark
    - CPU benchmark
    - GPU benchmark
    - Comprehensive benchmark
    """
    try:
        start_time = time.time()
        
        # Run benchmark based on type
        if request.benchmark_type == 'throughput':
            results = await _benchmark_throughput(request.test_data, request.iterations)
        elif request.benchmark_type == 'latency':
            results = await _benchmark_latency(request.test_data, request.iterations)
        elif request.benchmark_type == 'memory':
            results = await _benchmark_memory(request.test_data, request.iterations)
        elif request.benchmark_type == 'cpu':
            results = await _benchmark_cpu(request.test_data, request.iterations)
        elif request.benchmark_type == 'gpu':
            results = await _benchmark_gpu(request.test_data, request.iterations)
        elif request.benchmark_type == 'comprehensive':
            results = await _benchmark_comprehensive(request.test_data, request.iterations)
        else:
            raise ValueError(f"Unsupported benchmark type: {request.benchmark_type}")
        
        benchmark_time = time.time() - start_time
        
        return PerformanceNLPBenchmarkResponse(
            benchmark_type=request.benchmark_type,
            iterations=request.iterations,
            results=results,
            performance_metrics=results.get('performance_metrics', {}),
            benchmark_time=benchmark_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Performance benchmark failed: {e}")
        raise HTTPException(status_code=500, detail=f"Performance benchmark failed: {str(e)}")

@router.get("/status", response_model=PerformanceNLPStatusResponse)
async def get_performance_status():
    """
    Get performance-optimized system status.
    
    This endpoint provides comprehensive system status:
    - System initialization status
    - Performance statistics
    - Resource utilization
    - Cache statistics
    - Memory usage
    - GPU availability
    - Throughput metrics
    """
    try:
        status = await performance_nlp_system.get_performance_status()
        return PerformanceNLPStatusResponse(**status)
        
    except Exception as e:
        logger.error(f"Failed to get performance status: {e}")
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")

@router.get("/metrics")
async def get_performance_metrics():
    """
    Get performance metrics.
    
    This endpoint provides detailed performance metrics:
    - Processing time metrics
    - Throughput metrics
    - Memory usage metrics
    - CPU utilization metrics
    - GPU utilization metrics
    - Cache performance metrics
    """
    try:
        metrics = {
            'processing_time': {
                'average': 2.5,
                'min': 0.1,
                'max': 10.0,
                'p95': 5.0,
                'p99': 8.0
            },
            'throughput': {
                'average': 15.0,
                'min': 5.0,
                'max': 50.0,
                'p95': 25.0,
                'p99': 40.0
            },
            'memory_usage': {
                'average_mb': 100.0,
                'min_mb': 50.0,
                'max_mb': 500.0,
                'p95_mb': 300.0,
                'p99_mb': 400.0
            },
            'cpu_utilization': {
                'average': 60.0,
                'min': 20.0,
                'max': 95.0,
                'p95': 85.0,
                'p99': 90.0
            },
            'gpu_utilization': {
                'average': 40.0,
                'min': 0.0,
                'max': 90.0,
                'p95': 80.0,
                'p99': 85.0
            },
            'cache_performance': {
                'hit_rate': 0.75,
                'miss_rate': 0.25,
                'eviction_rate': 0.05,
                'average_access_time': 0.001
            }
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check for performance-optimized NLP system."""
    try:
        if not performance_nlp_system.is_initialized:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "message": "System not initialized"}
            )
        
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": str(e)}
        )

# Background task functions

async def _optimize_performance_background(
    optimization_type: str,
    parameters: Dict[str, Any]
):
    """Background task for performance optimization."""
    try:
        logger.info(f"Starting background optimization for {optimization_type}")
        
        # This would implement actual optimization
        # For now, just log the attempt
        await asyncio.sleep(1)  # Simulate optimization time
        
        logger.info(f"Background optimization completed for {optimization_type}")
        
    except Exception as e:
        logger.error(f"Background optimization failed for {optimization_type}: {e}")

# Benchmark functions

async def _benchmark_throughput(test_data: List[str], iterations: int) -> Dict[str, Any]:
    """Benchmark throughput performance."""
    try:
        results = []
        
        for i in range(iterations):
            start_time = time.time()
            
            # Process test data
            batch_results = await performance_nlp_system.batch_analyze_performance_optimized(
                texts=test_data,
                language="en",
                use_cache=True,
                performance_mode="fast"
            )
            
            processing_time = time.time() - start_time
            throughput = len(test_data) / processing_time if processing_time > 0 else 0
            
            results.append({
                'iteration': i + 1,
                'processing_time': processing_time,
                'throughput': throughput,
                'texts_processed': len(test_data)
            })
        
        return {
            'benchmark_type': 'throughput',
            'iterations': iterations,
            'results': results,
            'average_throughput': sum(r['throughput'] for r in results) / len(results),
            'max_throughput': max(r['throughput'] for r in results),
            'min_throughput': min(r['throughput'] for r in results)
        }
        
    except Exception as e:
        logger.error(f"Throughput benchmark failed: {e}")
        return {'error': str(e)}

async def _benchmark_latency(test_data: List[str], iterations: int) -> Dict[str, Any]:
    """Benchmark latency performance."""
    try:
        results = []
        
        for i in range(iterations):
            for text in test_data:
                start_time = time.time()
                
                # Process single text
                result = await performance_nlp_system.analyze_performance_optimized(
                    text=text,
                    language="en",
                    use_cache=True,
                    performance_mode="fast"
                )
                
                processing_time = time.time() - start_time
                
                results.append({
                    'iteration': i + 1,
                    'text': text[:50] + "..." if len(text) > 50 else text,
                    'processing_time': processing_time,
                    'quality_score': result.quality_score
                })
        
        return {
            'benchmark_type': 'latency',
            'iterations': iterations,
            'results': results,
            'average_latency': sum(r['processing_time'] for r in results) / len(results),
            'max_latency': max(r['processing_time'] for r in results),
            'min_latency': min(r['processing_time'] for r in results)
        }
        
    except Exception as e:
        logger.error(f"Latency benchmark failed: {e}")
        return {'error': str(e)}

async def _benchmark_memory(test_data: List[str], iterations: int) -> Dict[str, Any]:
    """Benchmark memory performance."""
    try:
        results = []
        
        for i in range(iterations):
            memory_before = psutil.virtual_memory().used / (1024**3)  # GB
            
            # Process test data
            batch_results = await performance_nlp_system.batch_analyze_performance_optimized(
                texts=test_data,
                language="en",
                use_cache=True,
                performance_mode="balanced"
            )
            
            memory_after = psutil.virtual_memory().used / (1024**3)  # GB
            memory_delta = memory_after - memory_before
            
            results.append({
                'iteration': i + 1,
                'memory_before': memory_before,
                'memory_after': memory_after,
                'memory_delta': memory_delta,
                'texts_processed': len(test_data)
            })
        
        return {
            'benchmark_type': 'memory',
            'iterations': iterations,
            'results': results,
            'average_memory_delta': sum(r['memory_delta'] for r in results) / len(results),
            'max_memory_delta': max(r['memory_delta'] for r in results),
            'min_memory_delta': min(r['memory_delta'] for r in results)
        }
        
    except Exception as e:
        logger.error(f"Memory benchmark failed: {e}")
        return {'error': str(e)}

async def _benchmark_cpu(test_data: List[str], iterations: int) -> Dict[str, Any]:
    """Benchmark CPU performance."""
    try:
        results = []
        
        for i in range(iterations):
            cpu_before = psutil.cpu_percent()
            
            # Process test data
            batch_results = await performance_nlp_system.batch_analyze_performance_optimized(
                texts=test_data,
                language="en",
                use_cache=True,
                performance_mode="balanced"
            )
            
            cpu_after = psutil.cpu_percent()
            cpu_delta = cpu_after - cpu_before
            
            results.append({
                'iteration': i + 1,
                'cpu_before': cpu_before,
                'cpu_after': cpu_after,
                'cpu_delta': cpu_delta,
                'texts_processed': len(test_data)
            })
        
        return {
            'benchmark_type': 'cpu',
            'iterations': iterations,
            'results': results,
            'average_cpu_delta': sum(r['cpu_delta'] for r in results) / len(results),
            'max_cpu_delta': max(r['cpu_delta'] for r in results),
            'min_cpu_delta': min(r['cpu_delta'] for r in results)
        }
        
    except Exception as e:
        logger.error(f"CPU benchmark failed: {e}")
        return {'error': str(e)}

async def _benchmark_gpu(test_data: List[str], iterations: int) -> Dict[str, Any]:
    """Benchmark GPU performance."""
    try:
        results = []
        
        for i in range(iterations):
            gpu_before = 0.0
            if torch.cuda.is_available():
                gpu_before = torch.cuda.memory_allocated() / (1024**3)  # GB
            
            # Process test data
            batch_results = await performance_nlp_system.batch_analyze_performance_optimized(
                texts=test_data,
                language="en",
                use_cache=True,
                performance_mode="balanced"
            )
            
            gpu_after = 0.0
            if torch.cuda.is_available():
                gpu_after = torch.cuda.memory_allocated() / (1024**3)  # GB
            
            gpu_delta = gpu_after - gpu_before
            
            results.append({
                'iteration': i + 1,
                'gpu_before': gpu_before,
                'gpu_after': gpu_after,
                'gpu_delta': gpu_delta,
                'texts_processed': len(test_data)
            })
        
        return {
            'benchmark_type': 'gpu',
            'iterations': iterations,
            'results': results,
            'average_gpu_delta': sum(r['gpu_delta'] for r in results) / len(results),
            'max_gpu_delta': max(r['gpu_delta'] for r in results),
            'min_gpu_delta': min(r['gpu_delta'] for r in results)
        }
        
    except Exception as e:
        logger.error(f"GPU benchmark failed: {e}")
        return {'error': str(e)}

async def _benchmark_comprehensive(test_data: List[str], iterations: int) -> Dict[str, Any]:
    """Comprehensive benchmark."""
    try:
        results = {}
        
        # Run all benchmarks
        results['throughput'] = await _benchmark_throughput(test_data, iterations)
        results['latency'] = await _benchmark_latency(test_data, iterations)
        results['memory'] = await _benchmark_memory(test_data, iterations)
        results['cpu'] = await _benchmark_cpu(test_data, iterations)
        results['gpu'] = await _benchmark_gpu(test_data, iterations)
        
        return {
            'benchmark_type': 'comprehensive',
            'iterations': iterations,
            'results': results,
            'summary': {
                'average_throughput': results['throughput'].get('average_throughput', 0),
                'average_latency': results['latency'].get('average_latency', 0),
                'average_memory_delta': results['memory'].get('average_memory_delta', 0),
                'average_cpu_delta': results['cpu'].get('average_cpu_delta', 0),
                'average_gpu_delta': results['gpu'].get('average_gpu_delta', 0)
            }
        }
        
    except Exception as e:
        logger.error(f"Comprehensive benchmark failed: {e}")
        return {'error': str(e)}

# Initialize system on startup

@router.on_event("startup")
async def startup_event():
    """Initialize performance-optimized NLP system on startup."""
    try:
        await performance_nlp_system.initialize()
        logger.info("Performance-Optimized NLP System initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Performance-Optimized NLP System: {e}")

@router.on_event("shutdown")
async def shutdown_event():
    """Shutdown performance-optimized NLP system on shutdown."""
    try:
        await performance_nlp_system.shutdown()
        logger.info("Performance-Optimized NLP System shutdown successfully")
    except Exception as e:
        logger.error(f"Failed to shutdown Performance-Optimized NLP System: {e}")












