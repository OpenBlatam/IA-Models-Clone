"""
Blaze AI REST API Module v7.3.0

RESTful API interface for accessing all Blaze AI system capabilities
including NLP, vision, reasoning, ML, and optimization features.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable
import time
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import httpx

from .base import BaseModule, ModuleConfig, ModuleStatus
from ..modules import (
    ModuleRegistry, AIIntelligenceModule, MLModule, DataAnalysisModule,
    CacheModule, MonitoringModule, OptimizationModule, StorageModule,
    ExecutionModule, EnginesModule
)

logger = logging.getLogger(__name__)

# ============================================================================
# ENUMS AND CONFIGURATION
# ============================================================================

class APIVersion(Enum):
    """API versions."""
    V1 = "v1"
    V2 = "v2"
    BETA = "beta"

class EndpointType(Enum):
    """API endpoint types."""
    NLP = "nlp"
    VISION = "vision"
    REASONING = "reasoning"
    MULTIMODAL = "multimodal"
    ML = "ml"
    DATA_ANALYSIS = "data_analysis"
    CACHE = "cache"
    MONITORING = "monitoring"
    OPTIMIZATION = "optimization"
    STORAGE = "storage"
    EXECUTION = "execution"
    ENGINES = "engines"
    SYSTEM = "system"

class AuthenticationMethod(Enum):
    """Authentication methods."""
    NONE = "none"
    API_KEY = "api_key"
    JWT = "jwt"
    OAUTH2 = "oauth2"

@dataclass
class APIRESTConfig(ModuleConfig):
    """Configuration for REST API module."""
    host: str = "0.0.0.0"
    port: int = 8000
    api_version: APIVersion = APIVersion.V1
    enable_cors: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    authentication_method: AuthenticationMethod = AuthenticationMethod.API_KEY
    api_keys: List[str] = field(default_factory=list)
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    enable_documentation: bool = True
    enable_metrics: bool = True
    max_request_size: int = 100 * 1024 * 1024  # 100MB
    timeout_seconds: int = 30

@dataclass
class APIMetrics:
    """Metrics for REST API module."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    active_connections: int = 0
    requests_per_endpoint: Dict[str, int] = field(default_factory=dict)
    error_counts: Dict[str, int] = field(default_factory=dict)

@dataclass
class APIEndpoint:
    """API endpoint definition."""
    path: str
    method: str
    endpoint_type: EndpointType
    description: str
    tags: List[str] = field(default_factory=list)
    rate_limit: Optional[int] = None
    authentication_required: bool = True

# ============================================================================
# PYDANTIC MODELS FOR API
# ============================================================================

class NLPRequest(BaseModel):
    """NLP processing request."""
    text: str = Field(..., description="Text to process")
    task: str = Field(..., description="NLP task type (sentiment, classification, summarization, translation)")
    language: Optional[str] = Field("en", description="Source language")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional options")

class VisionRequest(BaseModel):
    """Computer vision processing request."""
    image_data: str = Field(..., description="Base64 encoded image data")
    task: str = Field(..., description="Vision task type (detection, classification, segmentation)")
    format: str = Field("jpeg", description="Image format")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional options")

class ReasoningRequest(BaseModel):
    """Automated reasoning request."""
    query: str = Field(..., description="Reasoning query")
    reasoning_type: str = Field("logical", description="Reasoning type (logical, symbolic, fuzzy, quantum)")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")

class MultimodalRequest(BaseModel):
    """Multimodal processing request."""
    text: str = Field(..., description="Text description")
    image_data: str = Field(..., description="Base64 encoded image data")
    task: str = Field(..., description="Analysis task")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional options")

class MLTrainingRequest(BaseModel):
    """Machine learning training request."""
    model_type: str = Field(..., description="Model type (transformer, cnn, rnn)")
    training_data: str = Field(..., description="Path to training data")
    optimization_strategy: str = Field("standard", description="Optimization strategy")
    hyperparameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Model hyperparameters")

class DataAnalysisRequest(BaseModel):
    """Data analysis request."""
    data_source: str = Field(..., description="Data source identifier")
    analysis_type: str = Field(..., description="Analysis type (descriptive, exploratory, clustering)")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Analysis options")

class CacheRequest(BaseModel):
    """Cache operation request."""
    key: str = Field(..., description="Cache key")
    value: Optional[Any] = Field(None, description="Value to store")
    ttl: Optional[int] = Field(None, description="Time to live in seconds")
    tags: Optional[List[str]] = Field(default_factory=list, description="Cache tags")

class OptimizationRequest(BaseModel):
    """Optimization request."""
    name: str = Field(..., description="Optimization task name")
    objective_function: str = Field(..., description="Objective function definition")
    constraints: Optional[List[str]] = Field(default_factory=list, description="Constraint functions")
    bounds: Optional[Dict[str, List[float]]] = Field(default_factory=dict, description="Parameter bounds")
    algorithm: str = Field("genetic", description="Optimization algorithm")

class APIResponse(BaseModel):
    """Standard API response."""
    success: bool = Field(..., description="Request success status")
    data: Optional[Any] = Field(None, description="Response data")
    message: str = Field("", description="Response message")
    timestamp: float = Field(default_factory=time.time, description="Response timestamp")
    request_id: Optional[str] = Field(None, description="Unique request identifier")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")

class ErrorResponse(BaseModel):
    """Error response."""
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    timestamp: float = Field(default_factory=time.time, description="Error timestamp")

# ============================================================================
# AUTHENTICATION AND SECURITY
# ============================================================================

class APIAuthenticator:
    """API authentication handler."""
    
    def __init__(self, config: APIRESTConfig):
        self.config = config
        self.security = HTTPBearer(auto_error=False)
        self.valid_api_keys = set(config.api_keys)
    
    async def authenticate(self, credentials: Optional[HTTPAuthorizationCredentials] = None) -> bool:
        """Authenticate API request."""
        if self.config.authentication_method == AuthenticationMethod.NONE:
            return True
        
        if self.config.authentication_method == AuthenticationMethod.API_KEY:
            if not credentials:
                return False
            return credentials.credentials in self.valid_api_keys
        
        # Add other authentication methods as needed
        return False
    
    def get_auth_dependency(self):
        """Get FastAPI dependency for authentication."""
        async def auth_dependency(credentials: HTTPAuthorizationCredentials = Depends(self.security)):
            if not await self.authenticate(credentials):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials"
                )
            return credentials
        return auth_dependency

# ============================================================================
# RATE LIMITING
# ============================================================================

class RateLimiter:
    """API rate limiting handler."""
    
    def __init__(self, config: APIRESTConfig):
        self.config = config
        self.request_counts: Dict[str, List[float]] = {}
    
    async def check_rate_limit(self, client_id: str) -> bool:
        """Check if request is within rate limit."""
        if not self.config.rate_limit_enabled:
            return True
        
        current_time = time.time()
        window_start = current_time - self.config.rate_limit_window
        
        # Clean old requests
        if client_id in self.request_counts:
            self.request_counts[client_id] = [
                req_time for req_time in self.request_counts[client_id]
                if req_time > window_start
            ]
        else:
            self.request_counts[client_id] = []
        
        # Check limit
        if len(self.request_counts[client_id]) >= self.config.rate_limit_requests:
            return False
        
        # Add current request
        self.request_counts[client_id].append(current_time)
        return True

# ============================================================================
# MAIN REST API MODULE
# ============================================================================

class APIRESTModule(BaseModule):
    """REST API module providing HTTP access to Blaze AI capabilities."""
    
    def __init__(self, config: APIRESTConfig):
        super().__init__(config)
        self.config = config
        self.metrics = APIMetrics()
        
        # FastAPI application
        self.app = FastAPI(
            title="Blaze AI REST API",
            description="Advanced AI capabilities through RESTful interface",
            version="7.3.0",
            docs_url="/docs" if config.enable_documentation else None,
            redoc_url="/redoc" if config.enable_documentation else None
        )
        
        # Authentication and rate limiting
        self.authenticator = APIAuthenticator(config)
        self.rate_limiter = RateLimiter(config)
        
        # Module references
        self.registry: Optional[ModuleRegistry] = None
        self.ai_intelligence: Optional[AIIntelligenceModule] = None
        self.ml_module: Optional[MLModule] = None
        self.data_analysis: Optional[DataAnalysisModule] = None
        self.cache: Optional[CacheModule] = None
        self.monitoring: Optional[MonitoringModule] = None
        self.optimization: Optional[OptimizationModule] = None
        self.storage: Optional[StorageModule] = None
        self.execution: Optional[ExecutionModule] = None
        self.engines: Optional[EnginesModule] = None
        
        # Server
        self.server: Optional[uvicorn.Server] = None
        self._setup_middleware()
        self._setup_routes()
    
    def _setup_middleware(self):
        """Setup FastAPI middleware."""
        # CORS middleware
        if self.config.enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=self.config.cors_origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"]
            )
        
        # Request size middleware
        @self.app.middleware("http")
        async def request_size_middleware(request, call_next):
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > self.config.max_request_size:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="Request too large"
                )
            response = await call_next(request)
            return response
        
        # Metrics middleware
        @self.app.middleware("http")
        async def metrics_middleware(request, call_next):
            start_time = time.time()
            self.metrics.total_requests += 1
            self.metrics.active_connections += 1
            
            # Track endpoint usage
            endpoint = f"{request.method} {request.url.path}"
            self.metrics.requests_per_endpoint[endpoint] = self.metrics.requests_per_endpoint.get(endpoint, 0) + 1
            
            try:
                response = await call_next(request)
                self.metrics.successful_requests += 1
                return response
            except Exception as e:
                self.metrics.failed_requests += 1
                error_type = type(e).__name__
                self.metrics.error_counts[error_type] = self.metrics.error_counts.get(error_type, 0) + 1
                raise
            finally:
                processing_time = time.time() - start_time
                self.metrics.active_connections -= 1
                self._update_average_response_time(processing_time)
    
    def _setup_routes(self):
        """Setup API routes."""
        # Health check
        @self.app.get("/health", tags=["System"])
        async def health_check():
            """System health check."""
            return {"status": "healthy", "timestamp": time.time()}
        
        # API info
        @self.app.get("/", tags=["System"])
        async def api_info():
            """API information."""
            return {
                "name": "Blaze AI REST API",
                "version": "7.3.0",
                "description": "Advanced AI capabilities through RESTful interface",
                "endpoints": self._get_endpoints_info()
            }
        
        # NLP endpoints
        @self.app.post("/api/v1/nlp/process", response_model=APIResponse, tags=["NLP"])
        async def process_nlp(
            request: NLPRequest,
            background_tasks: BackgroundTasks,
            auth: HTTPAuthorizationCredentials = Depends(self.authenticator.get_auth_dependency())
        ):
            """Process NLP task."""
            return await self._handle_nlp_request(request)
        
        @self.app.post("/api/v1/nlp/sentiment", response_model=APIResponse, tags=["NLP"])
        async def analyze_sentiment(
            request: NLPRequest,
            auth: HTTPAuthorizationCredentials = Depends(self.authenticator.get_auth_dependency())
        ):
            """Analyze text sentiment."""
            request.task = "sentiment"
            return await self._handle_nlp_request(request)
        
        # Vision endpoints
        @self.app.post("/api/v1/vision/process", response_model=APIResponse, tags=["Vision"])
        async def process_vision(
            request: VisionRequest,
            auth: HTTPAuthorizationCredentials = Depends(self.authenticator.get_auth_dependency())
        ):
            """Process computer vision task."""
            return await self._handle_vision_request(request)
        
        @self.app.post("/api/v1/vision/detect", response_model=APIResponse, tags=["Vision"])
        async def detect_objects(
            request: VisionRequest,
            auth: HTTPAuthorizationCredentials = Depends(self.authenticator.get_auth_dependency())
        ):
            """Detect objects in image."""
            request.task = "object_detection"
            return await self._handle_vision_request(request)
        
        # Reasoning endpoints
        @self.app.post("/api/v1/reasoning/process", response_model=APIResponse, tags=["Reasoning"])
        async def process_reasoning(
            request: ReasoningRequest,
            auth: HTTPAuthorizationCredentials = Depends(self.authenticator.get_auth_dependency())
        ):
            """Process reasoning task."""
            return await self._handle_reasoning_request(request)
        
        # Multimodal endpoints
        @self.app.post("/api/v1/multimodal/process", response_model=APIResponse, tags=["Multimodal"])
        async def process_multimodal(
            request: MultimodalRequest,
            auth: HTTPAuthorizationCredentials = Depends(self.authenticator.get_auth_dependency())
        ):
            """Process multimodal task."""
            return await self._handle_multimodal_request(request)
        
        # ML endpoints
        @self.app.post("/api/v1/ml/train", response_model=APIResponse, tags=["Machine Learning"])
        async def train_model(
            request: MLTrainingRequest,
            auth: HTTPAuthorizationCredentials = Depends(self.authenticator.get_auth_dependency())
        ):
            """Train machine learning model."""
            return await self._handle_ml_request(request)
        
        # Data analysis endpoints
        @self.app.post("/api/v1/data/analyze", response_model=APIResponse, tags=["Data Analysis"])
        async def analyze_data(
            request: DataAnalysisRequest,
            auth: HTTPAuthorizationCredentials = Depends(self.authenticator.get_auth_dependency())
        ):
            """Analyze data."""
            return await self._handle_data_analysis_request(request)
        
        # Cache endpoints
        @self.app.get("/api/v1/cache/{key}", response_model=APIResponse, tags=["Cache"])
        async def get_cache(
            key: str,
            auth: HTTPAuthorizationCredentials = Depends(self.authenticator.get_auth_dependency())
        ):
            """Get value from cache."""
            return await self._handle_cache_get(key)
        
        @self.app.post("/api/v1/cache", response_model=APIResponse, tags=["Cache"])
        async def set_cache(
            request: CacheRequest,
            auth: HTTPAuthorizationCredentials = Depends(self.authenticator.get_auth_dependency())
        ):
            """Set value in cache."""
            return await self._handle_cache_set(request)
        
        # Optimization endpoints
        @self.app.post("/api/v1/optimization/submit", response_model=APIResponse, tags=["Optimization"])
        async def submit_optimization(
            request: OptimizationRequest,
            auth: HTTPAuthorizationCredentials = Depends(self.authenticator.get_auth_dependency())
        ):
            """Submit optimization task."""
            return await self._handle_optimization_request(request)
        
        # System endpoints
        @self.app.get("/api/v1/system/status", response_model=APIResponse, tags=["System"])
        async def system_status(
            auth: HTTPAuthorizationCredentials = Depends(self.authenticator.get_auth_dependency())
        ):
            """Get system status."""
            return await self._handle_system_status()
        
        @self.app.get("/api/v1/system/metrics", response_model=APIResponse, tags=["System"])
        async def system_metrics(
            auth: HTTPAuthorizationCredentials = Depends(self.authenticator.get_auth_dependency())
        ):
            """Get system metrics."""
            return await self._handle_system_metrics()
    
    async def initialize(self) -> bool:
        """Initialize the REST API module."""
        try:
            await super().initialize()
            
            # Start FastAPI server
            config = uvicorn.Config(
                app=self.app,
                host=self.config.host,
                port=self.config.port,
                log_level="info"
            )
            self.server = uvicorn.Server(config)
            
            # Start server in background
            asyncio.create_task(self._start_server())
            
            self.status = ModuleStatus.ACTIVE
            logger.info(f"REST API module initialized on {self.config.host}:{self.config.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize REST API module: {e}")
            self.status = ModuleStatus.ERROR
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the REST API module."""
        try:
            if self.server:
                self.server.should_exit = True
                await asyncio.sleep(1)  # Give server time to shutdown
            
            await super().shutdown()
            logger.info("REST API module shutdown successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during REST API module shutdown: {e}")
            return False
    
    async def _start_server(self):
        """Start the FastAPI server."""
        try:
            await self.server.serve()
        except Exception as e:
            logger.error(f"Server error: {e}")
            self.status = ModuleStatus.ERROR
    
    def _get_endpoints_info(self) -> List[Dict[str, Any]]:
        """Get information about available endpoints."""
        endpoints = []
        for route in self.app.routes:
            if hasattr(route, 'methods') and hasattr(route, 'path'):
                endpoints.append({
                    "path": route.path,
                    "methods": list(route.methods),
                    "tags": getattr(route, 'tags', [])
                })
        return endpoints
    
    async def _handle_nlp_request(self, request: NLPRequest) -> APIResponse:
        """Handle NLP processing request."""
        if not self.ai_intelligence:
            raise HTTPException(status_code=503, detail="AI Intelligence module not available")
        
        start_time = time.time()
        try:
            result = await self.ai_intelligence.process_nlp_task(
                request.text,
                request.task
            )
            processing_time = time.time() - start_time
            
            return APIResponse(
                success=result.get("success", False),
                data=result,
                message="NLP processing completed",
                processing_time=processing_time
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"NLP processing failed: {str(e)}")
    
    async def _handle_vision_request(self, request: VisionRequest) -> APIResponse:
        """Handle computer vision request."""
        if not self.ai_intelligence:
            raise HTTPException(status_code=503, detail="AI Intelligence module not available")
        
        start_time = time.time()
        try:
            # Decode base64 image data
            import base64
            image_data = base64.b64decode(request.image_data)
            
            result = await self.ai_intelligence.process_vision_task(
                image_data,
                request.task
            )
            processing_time = time.time() - start_time
            
            return APIResponse(
                success=result.get("success", False),
                data=result,
                message="Vision processing completed",
                processing_time=processing_time
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Vision processing failed: {str(e)}")
    
    async def _handle_reasoning_request(self, request: ReasoningRequest) -> APIResponse:
        """Handle reasoning request."""
        if not self.ai_intelligence:
            raise HTTPException(status_code=503, detail="AI Intelligence module not available")
        
        start_time = time.time()
        try:
            from ..modules.ai_intelligence import ReasoningType
            reasoning_type = ReasoningType(request.reasoning_type)
            
            result = await self.ai_intelligence.process_reasoning_task(
                request.query,
                reasoning_type
            )
            processing_time = time.time() - start_time
            
            return APIResponse(
                success=result.get("success", False),
                data=result,
                message="Reasoning completed",
                processing_time=processing_time
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Reasoning failed: {str(e)}")
    
    async def _handle_multimodal_request(self, request: MultimodalRequest) -> APIResponse:
        """Handle multimodal request."""
        if not self.ai_intelligence:
            raise HTTPException(status_code=503, detail="AI Intelligence module not available")
        
        start_time = time.time()
        try:
            # Decode base64 image data
            import base64
            image_data = base64.b64decode(request.image_data)
            
            result = await self.ai_intelligence.process_multimodal_task(
                request.text,
                image_data,
                request.task
            )
            processing_time = time.time() - start_time
            
            return APIResponse(
                success=result.get("success", False),
                data=result,
                message="Multimodal processing completed",
                processing_time=processing_time
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Multimodal processing failed: {str(e)}")
    
    async def _handle_ml_request(self, request: MLTrainingRequest) -> APIResponse:
        """Handle ML training request."""
        if not self.ml_module:
            raise HTTPException(status_code=503, detail="ML module not available")
        
        start_time = time.time()
        try:
            # This would need to be implemented based on actual ML module interface
            result = {"status": "training_started", "task_id": "ml_001"}
            processing_time = time.time() - start_time
            
            return APIResponse(
                success=True,
                data=result,
                message="ML training started",
                processing_time=processing_time
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"ML training failed: {str(e)}")
    
    async def _handle_data_analysis_request(self, request: DataAnalysisRequest) -> APIResponse:
        """Handle data analysis request."""
        if not self.data_analysis:
            raise HTTPException(status_code=503, detail="Data Analysis module not available")
        
        start_time = time.time()
        try:
            # This would need to be implemented based on actual Data Analysis module interface
            result = {"status": "analysis_started", "job_id": "da_001"}
            processing_time = time.time() - start_time
            
            return APIResponse(
                success=True,
                data=result,
                message="Data analysis started",
                processing_time=processing_time
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Data analysis failed: {str(e)}")
    
    async def _handle_cache_get(self, key: str) -> APIResponse:
        """Handle cache get request."""
        if not self.cache:
            raise HTTPException(status_code=503, detail="Cache module not available")
        
        try:
            value = await self.cache.get(key)
            return APIResponse(
                success=True,
                data={"key": key, "value": value},
                message="Cache value retrieved"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Cache get failed: {str(e)}")
    
    async def _handle_cache_set(self, request: CacheRequest) -> APIResponse:
        """Handle cache set request."""
        if not self.cache:
            raise HTTPException(status_code=503, detail="Cache module not available")
        
        try:
            await self.cache.set(request.key, request.value, ttl=request.ttl, tags=request.tags)
            return APIResponse(
                success=True,
                data={"key": request.key, "status": "stored"},
                message="Cache value stored"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Cache set failed: {str(e)}")
    
    async def _handle_optimization_request(self, request: OptimizationRequest) -> APIResponse:
        """Handle optimization request."""
        if not self.optimization:
            raise HTTPException(status_code=503, detail="Optimization module not available")
        
        try:
            # This would need to be implemented based on actual Optimization module interface
            result = {"status": "optimization_started", "task_id": "opt_001"}
            return APIResponse(
                success=True,
                data=result,
                message="Optimization task submitted"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")
    
    async def _handle_system_status(self) -> APIResponse:
        """Handle system status request."""
        try:
            status_info = {
                "api_status": self.status.value,
                "modules_available": {
                    "ai_intelligence": self.ai_intelligence is not None,
                    "ml": self.ml_module is not None,
                    "data_analysis": self.data_analysis is not None,
                    "cache": self.cache is not None,
                    "monitoring": self.monitoring is not None,
                    "optimization": self.optimization is not None,
                    "storage": self.storage is not None,
                    "execution": self.execution is not None,
                    "engines": self.engines is not None
                }
            }
            
            return APIResponse(
                success=True,
                data=status_info,
                message="System status retrieved"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")
    
    async def _handle_system_metrics(self) -> APIResponse:
        """Handle system metrics request."""
        try:
            metrics_data = {
                "api_metrics": self.metrics,
                "module_metrics": {}
            }
            
            # Collect metrics from available modules
            if self.ai_intelligence:
                metrics_data["module_metrics"]["ai_intelligence"] = await self.ai_intelligence.get_metrics()
            if self.monitoring:
                metrics_data["module_metrics"]["monitoring"] = self.monitoring.get_metric_summary()
            
            return APIResponse(
                success=True,
                data=metrics_data,
                message="System metrics retrieved"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")
    
    def _update_average_response_time(self, response_time: float):
        """Update average response time metric."""
        total_time = self.metrics.average_response_time * (self.metrics.total_requests - 1)
        self.metrics.average_response_time = (total_time + response_time) / self.metrics.total_requests
    
    async def get_metrics(self) -> APIMetrics:
        """Get module metrics."""
        return self.metrics
    
    async def health_check(self) -> Dict[str, Any]:
        """Check module health."""
        health = await super().health_check()
        health["server_running"] = self.server is not None and not getattr(self.server, 'should_exit', False)
        health["active_connections"] = self.metrics.active_connections
        health["total_requests"] = self.metrics.total_requests
        return health

# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_api_rest_module(config: Optional[APIRESTConfig] = None) -> APIRESTModule:
    """Create REST API module."""
    if config is None:
        config = APIRESTConfig()
    return APIRESTModule(config)

def create_api_rest_module_with_defaults(**kwargs) -> APIRESTModule:
    """Create REST API module with default configuration."""
    config = APIRESTConfig(**kwargs)
    return APIRESTModule(config)

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "APIVersion",
    "EndpointType",
    "AuthenticationMethod",
    
    # Configuration and Data Classes
    "APIRESTConfig",
    "APIMetrics",
    "APIEndpoint",
    
    # Pydantic Models
    "NLPRequest",
    "VisionRequest",
    "ReasoningRequest",
    "MultimodalRequest",
    "MLTrainingRequest",
    "DataAnalysisRequest",
    "CacheRequest",
    "OptimizationRequest",
    "APIResponse",
    "ErrorResponse",
    
    # Main Module
    "APIRESTModule",
    
    # Factory Functions
    "create_api_rest_module",
    "create_api_rest_module_with_defaults"
]
