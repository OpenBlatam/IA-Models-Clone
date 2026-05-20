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
import json
import hashlib
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import gzip
import pickle
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, validator
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
import aiohttp
import aiofiles
    from prometheus_client import Counter, Histogram, Gauge, Summary
from ..core.document_intelligence_engine import DocumentIntelligenceEngine, ProcessingConfig
from ..core.citation_manager import CitationManager, CitationConfig
from ..core.document_pipeline import DocumentPipeline, PipelineConfig
from ..optimization.ultra_optimization_system import UltraOptimizationSystem, OptimizationConfig
from ..nlp import NLPEngine
from ..ml_integration import MLModelManager
    import uvicorn
from typing import Any, List, Dict, Optional
"""
Ultra Optimized API
==================

API ultra-optimizada con todas las mejoras de rendimiento:
- Caché inteligente multi-nivel
- Procesamiento paralelo y asíncrono
- Rate limiting adaptativo
- Compresión y serialización optimizada
- Monitoreo de rendimiento en tiempo real
- Auto-scaling y load balancing
"""



# Prometheus metrics
try:
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Core imports

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Pydantic models
class UltraAPIRequest(BaseModel):
    """Request model for ultra-optimized API"""
    operation: str
    data: Any
    use_cache: bool = True
    use_gpu: bool = True
    batch_size: Optional[int] = None
    priority: str = "normal"  # low, normal, high, critical
    timeout: Optional[int] = None

class UltraAPIResponse(BaseModel):
    """Response model for ultra-optimized API"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    message: Optional[str] = None
    processing_time: float
    cache_hit: bool = False
    optimization_level: str = "standard"
    timestamp: datetime = Field(default_factory=datetime.now)

class RateLimitConfig(BaseModel):
    """Rate limiting configuration"""
    requests_per_minute: int = 100
    burst_size: int = 20
    window_size: int = 60  # seconds

class CacheConfig(BaseModel):
    """Cache configuration"""
    enable_l1_cache: bool = True
    enable_l2_cache: bool = True
    enable_l3_cache: bool = False
    ttl: int = 3600
    max_size: int = 10000
    compression_level: int = 6

class UltraOptimizedAPI:
    """
    Ultra Optimized API
    
    Características:
    - Caché inteligente multi-nivel
    - Procesamiento paralelo y asíncrono
    - Rate limiting adaptativo
    - Compresión y serialización optimizada
    - Monitoreo de rendimiento en tiempo real
    - Auto-scaling y load balancing
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        db_session: AsyncSession = None
    ):
        
    """__init__ function."""
self.redis_url = redis_url
        self.db_session = db_session
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="Ultra Optimized API",
            description="API ultra-optimizada con todas las mejoras de rendimiento",
            version="3.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Initialize components
        self.optimization_system = None
        self.document_pipeline = None
        self.citation_manager = None
        self.nlp_engine = None
        self.ml_manager = None
        
        # Rate limiting
        self.rate_limit_config = RateLimitConfig()
        self.request_counts = {}
        self.rate_limit_lock = asyncio.Lock()
        
        # Cache configuration
        self.cache_config = CacheConfig()
        
        # Performance metrics
        self.metrics = {
            'requests_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'processing_time_avg': 0.0,
            'error_rate': 0.0,
            'throughput': 0.0
        }
        
        # Setup middleware and routes
        self._setup_middleware()
        self._setup_routes()
        
        logger.info("Ultra Optimized API initialized")
    
    async def startup(self) -> Any:
        """Initialize all components"""
        try:
            # Initialize optimization system
            optimization_config = OptimizationConfig(
                enable_gpu_optimization=True,
                enable_memory_optimization=True,
                enable_multi_level_cache=True,
                enable_parallel_processing=True,
                enable_performance_monitoring=True,
                enable_auto_tuning=True,
                max_workers=16,
                batch_size=64,
                cache_ttl=self.cache_config.ttl,
                cache_max_size=self.cache_config.max_size
            )
            
            self.optimization_system = UltraOptimizationSystem(
                config=optimization_config,
                redis_url=self.redis_url,
                db_session=self.db_session
            )
            await self.optimization_system.startup()
            
            # Initialize document pipeline
            pipeline_config = PipelineConfig(
                enable_document_intelligence=True,
                enable_citation_management=True,
                enable_nlp_analysis=True,
                enable_ml_integration=True,
                enable_performance_optimization=True,
                enable_ocr=True,
                enable_sentiment_analysis=True,
                enable_keyword_extraction=True,
                enable_topic_modeling=True,
                enable_entity_recognition=True,
                enable_summarization=True,
                enable_citation_generation=True,
                enable_insight_generation=True,
                batch_size=32,
                max_workers=8,
                output_format="json",
                include_metadata=True,
                include_metrics=True,
                include_insights=True
            )
            
            self.document_pipeline = DocumentPipeline(
                config=pipeline_config,
                redis_url=self.redis_url,
                db_session=self.db_session
            )
            await self.document_pipeline.startup()
            
            # Initialize citation manager
            citation_config = CitationConfig(
                enable_auto_detection=True,
                enable_validation=True,
                enable_formatting=True,
                enable_database_lookup=True,
                enable_doi_resolution=True,
                enable_arxiv_lookup=True,
                enable_google_scholar=True,
                enable_crossref=True,
                max_citations_per_doc=100,
                confidence_threshold=0.7,
                cache_ttl=86400,
                request_timeout=30
            )
            
            self.citation_manager = CitationManager(
                config=citation_config,
                redis_url=self.redis_url,
                db_session=self.db_session
            )
            await self.citation_manager.startup()
            
            # Initialize other components
            self.nlp_engine = NLPEngine()
            self.ml_manager = MLModelManager()
            await self.ml_manager.initialize()
            
            logger.info("Ultra Optimized API started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Ultra Optimized API: {e}")
            raise
    
    async def shutdown(self) -> Any:
        """Cleanup and shutdown"""
        try:
            if self.optimization_system:
                await self.optimization_system.shutdown()
            
            if self.document_pipeline:
                await self.document_pipeline.shutdown()
            
            if self.citation_manager:
                await self.citation_manager.shutdown()
            
            if self.ml_manager:
                await self.ml_manager.shutdown()
            
            logger.info("Ultra Optimized API shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def _setup_middleware(self) -> Any:
        """Setup FastAPI middleware"""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Trusted host middleware
        self.app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"]
        )
        
        # GZip compression middleware
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Custom middleware for rate limiting and metrics
        @self.app.middleware("http")
        async def ultra_middleware(request: Request, call_next):
            
    """ultra_middleware function."""
start_time = time.time()
            
            # Rate limiting
            client_ip = request.client.host
            if not await self._check_rate_limit(client_ip):
                return JSONResponse(
                    status_code=429,
                    content={"error": "Rate limit exceeded", "retry_after": 60}
                )
            
            # Process request
            response = await call_next(request)
            
            # Add performance headers
            processing_time = time.time() - start_time
            response.headers["X-Processing-Time"] = str(processing_time)
            response.headers["X-Cache-Hit"] = str(request.state.cache_hit if hasattr(request.state, 'cache_hit') else False)
            
            # Update metrics
            self._update_metrics(processing_time, response.status_code < 400)
            
            return response
    
    def _setup_routes(self) -> Any:
        """Setup API routes"""
        
        # Health check
        @self.app.get("/health", response_model=UltraAPIResponse)
        async def health_check():
            """Health check endpoint"""
            start_time = time.time()
            
            try:
                health_status = await self._get_health_status()
                
                return UltraAPIResponse(
                    success=True,
                    data=health_status,
                    message="API is healthy",
                    processing_time=time.time() - start_time,
                    optimization_level="ultra"
                )
                
            except Exception as e:
                return UltraAPIResponse(
                    success=False,
                    error=str(e),
                    message="Health check failed",
                    processing_time=time.time() - start_time
                )
        
        # Ultra optimized document processing
        @self.app.post("/ultra-process-document", response_model=UltraAPIResponse)
        async def ultra_process_document(
            request: UltraAPIRequest,
            background_tasks: BackgroundTasks,
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ):
            """Ultra-optimized document processing"""
            start_time = time.time()
            
            try:
                # Validate authentication
                await self._validate_auth(credentials)
                
                # Check cache first
                cache_key = self._generate_cache_key("ultra_process_document", request.dict())
                cached_result = await self._get_cached_result(cache_key)
                
                if cached_result and request.use_cache:
                    return UltraAPIResponse(
                        success=True,
                        data=cached_result,
                        message="Document processed (cached)",
                        processing_time=time.time() - start_time,
                        cache_hit=True,
                        optimization_level="ultra"
                    )
                
                # Process document with ultra optimization
                result = await self._ultra_process_document(request)
                
                # Cache result
                if request.use_cache:
                    await self._cache_result(cache_key, result)
                
                return UltraAPIResponse(
                    success=True,
                    data=result,
                    message="Document processed successfully",
                    processing_time=time.time() - start_time,
                    cache_hit=False,
                    optimization_level="ultra"
                )
                
            except Exception as e:
                return UltraAPIResponse(
                    success=False,
                    error=str(e),
                    message="Document processing failed",
                    processing_time=time.time() - start_time
                )
        
        # Ultra optimized batch processing
        @self.app.post("/ultra-process-batch", response_model=UltraAPIResponse)
        async def ultra_process_batch(
            request: UltraAPIRequest,
            background_tasks: BackgroundTasks,
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ):
            """Ultra-optimized batch processing"""
            start_time = time.time()
            
            try:
                # Validate authentication
                await self._validate_auth(credentials)
                
                # Check cache first
                cache_key = self._generate_cache_key("ultra_process_batch", request.dict())
                cached_result = await self._get_cached_result(cache_key)
                
                if cached_result and request.use_cache:
                    return UltraAPIResponse(
                        success=True,
                        data=cached_result,
                        message="Batch processed (cached)",
                        processing_time=time.time() - start_time,
                        cache_hit=True,
                        optimization_level="ultra"
                    )
                
                # Process batch with ultra optimization
                result = await self._ultra_process_batch(request)
                
                # Cache result
                if request.use_cache:
                    await self._cache_result(cache_key, result)
                
                return UltraAPIResponse(
                    success=True,
                    data=result,
                    message="Batch processed successfully",
                    processing_time=time.time() - start_time,
                    cache_hit=False,
                    optimization_level="ultra"
                )
                
            except Exception as e:
                return UltraAPIResponse(
                    success=False,
                    error=str(e),
                    message="Batch processing failed",
                    processing_time=time.time() - start_time
                )
        
        # Ultra optimized citation extraction
        @self.app.post("/ultra-extract-citations", response_model=UltraAPIResponse)
        async def ultra_extract_citations(
            request: UltraAPIRequest,
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ):
            """Ultra-optimized citation extraction"""
            start_time = time.time()
            
            try:
                # Validate authentication
                await self._validate_auth(credentials)
                
                # Check cache first
                cache_key = self._generate_cache_key("ultra_extract_citations", request.dict())
                cached_result = await self._get_cached_result(cache_key)
                
                if cached_result and request.use_cache:
                    return UltraAPIResponse(
                        success=True,
                        data=cached_result,
                        message="Citations extracted (cached)",
                        processing_time=time.time() - start_time,
                        cache_hit=True,
                        optimization_level="ultra"
                    )
                
                # Extract citations with ultra optimization
                result = await self._ultra_extract_citations(request)
                
                # Cache result
                if request.use_cache:
                    await self._cache_result(cache_key, result)
                
                return UltraAPIResponse(
                    success=True,
                    data=result,
                    message="Citations extracted successfully",
                    processing_time=time.time() - start_time,
                    cache_hit=False,
                    optimization_level="ultra"
                )
                
            except Exception as e:
                return UltraAPIResponse(
                    success=False,
                    error=str(e),
                    message="Citation extraction failed",
                    processing_time=time.time() - start_time
                )
        
        # Ultra optimized NLP analysis
        @self.app.post("/ultra-analyze-text", response_model=UltraAPIResponse)
        async def ultra_analyze_text(
            request: UltraAPIRequest,
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ):
            """Ultra-optimized NLP analysis"""
            start_time = time.time()
            
            try:
                # Validate authentication
                await self._validate_auth(credentials)
                
                # Check cache first
                cache_key = self._generate_cache_key("ultra_analyze_text", request.dict())
                cached_result = await self._get_cached_result(cache_key)
                
                if cached_result and request.use_cache:
                    return UltraAPIResponse(
                        success=True,
                        data=cached_result,
                        message="Text analyzed (cached)",
                        processing_time=time.time() - start_time,
                        cache_hit=True,
                        optimization_level="ultra"
                    )
                
                # Analyze text with ultra optimization
                result = await self._ultra_analyze_text(request)
                
                # Cache result
                if request.use_cache:
                    await self._cache_result(cache_key, result)
                
                return UltraAPIResponse(
                    success=True,
                    data=result,
                    message="Text analyzed successfully",
                    processing_time=time.time() - start_time,
                    cache_hit=False,
                    optimization_level="ultra"
                )
                
            except Exception as e:
                return UltraAPIResponse(
                    success=False,
                    error=str(e),
                    message="Text analysis failed",
                    processing_time=time.time() - start_time
                )
        
        # Streaming ultra processing
        @self.app.post("/ultra-stream-process")
        async def ultra_stream_process(
            request: UltraAPIRequest,
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ):
            """Streaming ultra-optimized processing"""
            try:
                # Validate authentication
                await self._validate_auth(credentials)
                
                async def generate_stream():
                    """Generate streaming response"""
                    try:
                        # Start processing
                        yield f"data: {json.dumps({'status': 'started', 'message': 'Ultra processing started...'})}\n\n"
                        
                        # Process with progress updates
                        result = await self._ultra_stream_process(request)
                        
                        # Send final result
                        yield f"data: {json.dumps({'status': 'completed', 'data': result})}\n\n"
                        
                    except Exception as e:
                        yield f"data: {json.dumps({'status': 'error', 'error': str(e)})}\n\n"
                
                return StreamingResponse(
                    generate_stream(),
                    media_type="text/plain",
                    headers={"Cache-Control": "no-cache"}
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # Performance metrics
        @self.app.get("/ultra-metrics", response_model=UltraAPIResponse)
        async def get_ultra_metrics(
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ):
            """Get ultra performance metrics"""
            start_time = time.time()
            
            try:
                # Validate authentication
                await self._validate_auth(credentials)
                
                # Collect all metrics
                metrics = {
                    'api_metrics': self.metrics,
                    'optimization_metrics': await self.optimization_system.get_metrics(),
                    'pipeline_metrics': await self.document_pipeline.get_metrics(),
                    'citation_metrics': await self.citation_manager.get_metrics(),
                    'performance_history': await self.optimization_system.get_performance_history()
                }
                
                return UltraAPIResponse(
                    success=True,
                    data=metrics,
                    message="Ultra metrics retrieved successfully",
                    processing_time=time.time() - start_time,
                    optimization_level="ultra"
                )
                
            except Exception as e:
                return UltraAPIResponse(
                    success=False,
                    error=str(e),
                    message="Failed to retrieve ultra metrics",
                    processing_time=time.time() - start_time
                )
        
        # Cache management
        @self.app.post("/ultra-clear-cache", response_model=UltraAPIResponse)
        async def ultra_clear_cache(
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ):
            """Clear all ultra caches"""
            start_time = time.time()
            
            try:
                # Validate authentication
                await self._validate_auth(credentials)
                
                # Clear all caches
                await self.optimization_system.clear_cache()
                await self.document_pipeline.clear_cache()
                await self.citation_manager.clear_cache()
                
                return UltraAPIResponse(
                    success=True,
                    message="All ultra caches cleared successfully",
                    processing_time=time.time() - start_time,
                    optimization_level="ultra"
                )
                
            except Exception as e:
                return UltraAPIResponse(
                    success=False,
                    error=str(e),
                    message="Failed to clear ultra caches",
                    processing_time=time.time() - start_time
                )
    
    async def _ultra_process_document(self, request: UltraAPIRequest) -> Dict[str, Any]:
        """Ultra-optimized document processing"""
        try:
            # Use optimization system for processing
            result = await self.optimization_system.optimize_operation(
                self.document_pipeline.process_document,
                request.data.get('document_path'),
                use_cache=request.use_cache,
                use_gpu=request.use_gpu,
                batch_size=request.batch_size
            )
            
            return result.dict() if hasattr(result, 'dict') else result
            
        except Exception as e:
            logger.error(f"Error in ultra document processing: {e}")
            raise
    
    async def _ultra_process_batch(self, request: UltraAPIRequest) -> List[Dict[str, Any]]:
        """Ultra-optimized batch processing"""
        try:
            # Use optimization system for batch processing
            results = await self.optimization_system.optimize_batch_operation(
                self.document_pipeline.process_documents_batch,
                request.data.get('document_paths', []),
                batch_size=request.batch_size,
                use_cache=request.use_cache,
                use_gpu=request.use_gpu
            )
            
            return [result.dict() if hasattr(result, 'dict') else result for result in results]
            
        except Exception as e:
            logger.error(f"Error in ultra batch processing: {e}")
            raise
    
    async def _ultra_extract_citations(self, request: UltraAPIRequest) -> Dict[str, Any]:
        """Ultra-optimized citation extraction"""
        try:
            # Use optimization system for citation extraction
            text = request.data.get('text', '')
            
            citations = await self.optimization_system.optimize_operation(
                self.citation_manager.extract_citations,
                text,
                use_cache=request.use_cache,
                use_gpu=False  # Citations don't need GPU
            )
            
            # Validate citations
            validated_citations = await self.optimization_system.optimize_operation(
                self.citation_manager.validate_citations,
                citations,
                use_cache=request.use_cache,
                use_gpu=False
            )
            
            # Format citations
            format_name = request.data.get('format_name', 'APA')
            formatted_citations = await self.optimization_system.optimize_operation(
                self.citation_manager.format_citations,
                validated_citations,
                format_name,
                use_cache=request.use_cache,
                use_gpu=False
            )
            
            return {
                'citations': [c.dict() for c in validated_citations],
                'formatted_citations': formatted_citations,
                'citation_count': len(validated_citations)
            }
            
        except Exception as e:
            logger.error(f"Error in ultra citation extraction: {e}")
            raise
    
    async def _ultra_analyze_text(self, request: UltraAPIRequest) -> Dict[str, Any]:
        """Ultra-optimized NLP analysis"""
        try:
            text = request.data.get('text', '')
            
            # Parallel analysis tasks
            analysis_tasks = []
            
            if request.data.get('enable_sentiment', True):
                analysis_tasks.append(
                    self.optimization_system.optimize_operation(
                        self.nlp_engine.analyze_sentiment,
                        text,
                        use_cache=request.use_cache,
                        use_gpu=request.use_gpu
                    )
                )
            
            if request.data.get('enable_keywords', True):
                analysis_tasks.append(
                    self.optimization_system.optimize_operation(
                        self.nlp_engine.extract_keywords,
                        text,
                        use_cache=request.use_cache,
                        use_gpu=request.use_gpu
                    )
                )
            
            if request.data.get('enable_topics', True):
                analysis_tasks.append(
                    self.optimization_system.optimize_operation(
                        self.nlp_engine.model_topics,
                        text,
                        use_cache=request.use_cache,
                        use_gpu=request.use_gpu
                    )
                )
            
            if request.data.get('enable_entities', True):
                analysis_tasks.append(
                    self.optimization_system.optimize_operation(
                        self.nlp_engine.recognize_entities,
                        text,
                        use_cache=request.use_cache,
                        use_gpu=request.use_gpu
                    )
                )
            
            if request.data.get('enable_summary', True):
                analysis_tasks.append(
                    self.optimization_system.optimize_operation(
                        self.nlp_engine.summarize_text,
                        text,
                        use_cache=request.use_cache,
                        use_gpu=request.use_gpu
                    )
                )
            
            # Execute all analysis tasks
            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            analysis_result = {
                'sentiment': results[0] if len(results) > 0 and not isinstance(results[0], Exception) else None,
                'keywords': results[1] if len(results) > 1 and not isinstance(results[1], Exception) else None,
                'topics': results[2] if len(results) > 2 and not isinstance(results[2], Exception) else None,
                'entities': results[3] if len(results) > 3 and not isinstance(results[3], Exception) else None,
                'summary': results[4] if len(results) > 4 and not isinstance(results[4], Exception) else None
            }
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in ultra text analysis: {e}")
            raise
    
    async def _ultra_stream_process(self, request: UltraAPIRequest) -> Dict[str, Any]:
        """Ultra-optimized streaming processing"""
        # This would be implemented with progress callbacks
        return await self._ultra_process_document(request)
    
    async def _check_rate_limit(self, client_ip: str) -> bool:
        """Check rate limiting"""
        try:
            async with self.rate_limit_lock:
                current_time = time.time()
                window_start = current_time - self.rate_limit_config.window_size
                
                # Clean old entries
                if client_ip in self.request_counts:
                    self.request_counts[client_ip] = [
                        timestamp for timestamp in self.request_counts[client_ip]
                        if timestamp > window_start
                    ]
                else:
                    self.request_counts[client_ip] = []
                
                # Check if limit exceeded
                if len(self.request_counts[client_ip]) >= self.rate_limit_config.requests_per_minute:
                    return False
                
                # Add current request
                self.request_counts[client_ip].append(current_time)
                return True
                
        except Exception as e:
            logger.error(f"Error in rate limiting: {e}")
            return True  # Allow request if rate limiting fails
    
    async def _validate_auth(self, credentials: HTTPAuthorizationCredentials):
        """Validate authentication"""
        # Implement your authentication logic here
        if not credentials or not credentials.credentials:
            raise HTTPException(status_code=401, detail="Invalid authentication")
    
    def _generate_cache_key(self, operation: str, data: Dict[str, Any]) -> str:
        """Generate cache key"""
        try:
            content = f"{operation}_{json.dumps(data, sort_keys=True)}"
            return hashlib.md5(content.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Error generating cache key: {e}")
            return str(hash((operation, data)))
    
    async def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached result"""
        try:
            # Use optimization system's cache
            return await self.optimization_system._get_cached_result(cache_key)
        except Exception as e:
            logger.error(f"Error getting cached result: {e}")
            return None
    
    async def _cache_result(self, cache_key: str, result: Any):
        """Cache result"""
        try:
            # Use optimization system's cache
            await self.optimization_system._cache_result(cache_key, result)
        except Exception as e:
            logger.error(f"Error caching result: {e}")
    
    async def _get_health_status(self) -> Dict[str, Any]:
        """Get health status of all components"""
        health_status = {
            'api': 'healthy',
            'components': {}
        }
        
        # Check optimization system health
        if self.optimization_system:
            optimization_health = await self.optimization_system.health_check()
            health_status['components']['optimization_system'] = optimization_health
        
        # Check pipeline health
        if self.document_pipeline:
            pipeline_health = await self.document_pipeline.health_check()
            health_status['components']['document_pipeline'] = pipeline_health
        
        # Check citation manager health
        if self.citation_manager:
            try:
                citation_metrics = await self.citation_manager.get_metrics()
                health_status['components']['citation_manager'] = {
                    'status': 'healthy',
                    'metrics': citation_metrics
                }
            except Exception as e:
                health_status['components']['citation_manager'] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
        
        # Check other components
        health_status['components']['nlp_engine'] = {
            'status': 'healthy' if self.nlp_engine else 'unavailable'
        }
        
        health_status['components']['ml_manager'] = {
            'status': 'healthy' if self.ml_manager else 'unavailable'
        }
        
        return health_status
    
    def _update_metrics(self, processing_time: float, success: bool):
        """Update API metrics"""
        try:
            # Update metrics
            self.metrics['requests_processed'] += 1
            self.metrics['processing_time_avg'] = (
                (self.metrics['processing_time_avg'] * (self.metrics['requests_processed'] - 1) + 
                 processing_time) / self.metrics['requests_processed']
            )
            self.metrics['throughput'] += 1
            
            if not success:
                self.metrics['error_rate'] = (
                    (self.metrics['error_rate'] * (self.metrics['requests_processed'] - 1) + 1) /
                    self.metrics['requests_processed']
                )
                
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")


# Create FastAPI app instance
app = FastAPI(
    title="Ultra Optimized API",
    description="API ultra-optimizada con todas las mejoras de rendimiento",
    version="3.0.0"
)

# Global API instance
api_instance = None

@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    global api_instance
    api_instance = UltraOptimizedAPI()
    await api_instance.startup()
    
    # Mount the API routes
    app.mount("/api", api_instance.app)

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    global api_instance
    if api_instance:
        await api_instance.shutdown()

# Example usage
if __name__ == "__main__":
    
    uvicorn.run(
        "ultra_optimized_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 