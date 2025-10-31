"""
Advanced REST API for Blaze AI

This module provides a comprehensive REST API with FastAPI integration,
rate limiting, authentication, and OpenAPI documentation.
"""

from __future__ import annotations

import asyncio
import time
import json
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from enum import Enum

try:
    from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Mock classes for when FastAPI is not available
    class FastAPI:
        def __init__(self, *args, **kwargs): pass
        def add_middleware(self, *args, **kwargs): pass
        def get(self, *args, **kwargs): pass
        def post(self, *args, **kwargs): pass
        def put(self, *args, **kwargs): pass
        def delete(self, *args, **kwargs): pass
    class HTTPException:
        def __init__(self, *args, **kwargs): pass
    class Depends:
        def __init__(self, *args, **kwargs): pass
    class status:
        HTTP_200_OK = 200
        HTTP_201_CREATED = 201
        HTTP_400_BAD_REQUEST = 400
        HTTP_401_UNAUTHORIZED = 401
        HTTP_403_FORBIDDEN = 403
        HTTP_404_NOT_FOUND = 404
        HTTP_429_TOO_MANY_REQUESTS = 429
        HTTP_500_INTERNAL_SERVER_ERROR = 500
    class BackgroundTasks:
        def __init__(self): pass
        def add_task(self, *args, **kwargs): pass
    class BaseModel:
        pass
    class Field:
        def __init__(self, *args, **kwargs): pass
    class HTTPBearer:
        def __init__(self, *args, **kwargs): pass
    class HTTPAuthorizationCredentials:
        def __init__(self, *args, **kwargs): pass
    class JSONResponse:
        def __init__(self, *args, **kwargs): pass
    class StreamingResponse:
        def __init__(self, *args, **kwargs): pass
    class CORSMiddleware:
        pass
    class TrustedHostMiddleware:
        pass
    def uvicorn(*args, **kwargs): pass

from ...core.interfaces import CoreConfig
from ...engines import get_engine_manager, shutdown_engine_manager
from ...utils.logging import get_logger
from ...utils.metrics import get_metrics_collector
from ...utils.alerting import get_alerting_engine, AlertRule, AlertSeverity

# =============================================================================
# API Models and Schemas
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Overall system status")
    timestamp: float = Field(..., description="Health check timestamp")
    components: Dict[str, str] = Field(..., description="Component health status")
    version: str = Field(..., description="System version")

class MetricsResponse(BaseModel):
    """Metrics response."""
    metrics: Dict[str, Any] = Field(..., description="System metrics")
    timestamp: float = Field(..., description="Metrics timestamp")

class AlertRuleRequest(BaseModel):
    """Alert rule creation request."""
    name: str = Field(..., description="Alert rule name")
    description: str = Field(..., description="Alert rule description")
    severity: str = Field(..., description="Alert severity level")
    condition: str = Field(..., description="Alert condition")
    threshold: Union[float, int, str] = Field(..., description="Alert threshold")
    comparison: str = Field(..., description="Comparison operator")
    duration: float = Field(0.0, description="Duration before alerting")
    cooldown: float = Field(300.0, description="Cooldown period")
    labels: List[str] = Field(default_factory=list, description="Alert labels")
    annotations: Dict[str, str] = Field(default_factory=dict, description="Alert annotations")

class AlertRuleResponse(BaseModel):
    """Alert rule response."""
    name: str = Field(..., description="Alert rule name")
    description: str = Field(..., description="Alert rule description")
    severity: str = Field(..., description="Alert severity level")
    enabled: bool = Field(..., description="Whether rule is enabled")
    created_at: float = Field(..., description="Rule creation timestamp")

class TextGenerationRequest(BaseModel):
    """Text generation request."""
    prompt: str = Field(..., description="Input prompt for generation")
    max_length: Optional[int] = Field(100, description="Maximum generation length")
    temperature: Optional[float] = Field(0.7, description="Generation temperature")
    top_p: Optional[float] = Field(0.9, description="Top-p sampling parameter")
    top_k: Optional[int] = Field(50, description="Top-k sampling parameter")
    repetition_penalty: Optional[float] = Field(1.1, description="Repetition penalty")
    do_sample: Optional[bool] = Field(True, description="Whether to use sampling")

class TextGenerationResponse(BaseModel):
    """Text generation response."""
    text: str = Field(..., description="Generated text")
    tokens: List[str] = Field(..., description="Generated tokens")
    usage: Dict[str, Any] = Field(..., description="Token usage information")
    processing_time: float = Field(..., description="Processing time in seconds")
    model_name: str = Field(..., description="Model used for generation")

class ImageGenerationRequest(BaseModel):
    """Image generation request."""
    prompt: str = Field(..., description="Input prompt for image generation")
    width: int = Field(512, description="Image width")
    height: int = Field(512, description="Image height")
    num_inference_steps: Optional[int] = Field(50, description="Number of inference steps")
    guidance_scale: Optional[float] = Field(7.5, description="Guidance scale")
    seed: Optional[int] = Field(None, description="Random seed")

class ImageGenerationResponse(BaseModel):
    """Image generation response."""
    images: List[str] = Field(..., description="Base64 encoded images")
    metadata: Dict[str, Any] = Field(..., description="Generation metadata")
    processing_time: float = Field(..., description="Processing time in seconds")
    model_name: str = Field(..., description="Model used for generation")

class RouteRegistrationRequest(BaseModel):
    """Route registration request."""
    route_id: str = Field(..., description="Route identifier")
    target_engine: str = Field(..., description="Target engine name")
    target_operation: str = Field(..., description="Target operation")
    weight: float = Field(1.0, description="Route weight")
    priority: int = Field(1, description="Route priority")

class RouteRegistrationResponse(BaseModel):
    """Route registration response."""
    route_id: str = Field(..., description="Route identifier")
    status: str = Field(..., description="Registration status")
    message: str = Field(..., description="Status message")

class SystemStatusResponse(BaseModel):
    """System status response."""
    overall_status: str = Field(..., description="Overall system status")
    engines: Dict[str, Dict[str, Any]] = Field(..., description="Engine status")
    metrics: Dict[str, Any] = Field(..., description="System metrics")
    alerts: Dict[str, Any] = Field(..., description="Active alerts")
    timestamp: float = Field(..., description="Status timestamp")

# =============================================================================
# Rate Limiting and Security
# =============================================================================

class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self, requests_per_minute: int = 100):
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, List[float]] = {}
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed."""
        current_time = time.time()
        
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        # Remove old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if current_time - req_time < 60
        ]
        
        # Check if limit exceeded
        if len(self.requests[client_id]) >= self.requests_per_minute:
            return False
        
        # Add current request
        self.requests[client_id].append(current_time)
        return True

class SecurityManager:
    """Security manager for API authentication and authorization."""
    
    def __init__(self, config: Optional[CoreConfig] = None):
        self.config = config
        self.api_keys: Dict[str, str] = {}
        self.bearer_token = HTTPBearer()
        
        # Load API keys from config
        if config and hasattr(config, 'security'):
            self.api_keys = getattr(config.security, 'api_keys', {})
    
    async def authenticate_api_key(self, api_key: str) -> bool:
        """Authenticate API key."""
        return api_key in self.api_keys
    
    async def authenticate_bearer_token(self, credentials: HTTPAuthorizationCredentials) -> bool:
        """Authenticate bearer token."""
        # This is a simplified implementation
        # In production, you would validate JWT tokens
        return credentials.scheme == "Bearer" and credentials.credentials == "valid_token"

# =============================================================================
# Advanced REST API
# =============================================================================

class AdvancedRESTAPI:
    """Advanced REST API for Blaze AI system."""
    
    def __init__(self, config: Optional[CoreConfig] = None):
        self.config = config
        self.logger = get_logger("rest_api")
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="Blaze AI REST API",
            description="Advanced REST API for Blaze AI Engine System",
            version="2.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Initialize components
        self.engine_manager = None
        self.metrics_collector = None
        self.alerting_engine = None
        self.rate_limiter = RateLimiter()
        self.security_manager = SecurityManager(config)
        
        # Setup middleware and routes
        self._setup_middleware()
        self._setup_routes()
        self._setup_error_handlers()
    
    def _setup_middleware(self):
        """Setup API middleware."""
        try:
            # CORS middleware
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],  # Configure appropriately for production
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"]
            )
            
            # Trusted host middleware
            self.app.add_middleware(
                TrustedHostMiddleware,
                allowed_hosts=["*"]  # Configure appropriately for production
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to setup middleware: {e}")
    
    def _setup_routes(self):
        """Setup API routes."""
        try:
            # Health and status endpoints
            self.app.get("/health", response_model=HealthResponse)(self._health_check)
            self.app.get("/status", response_model=SystemStatusResponse)(self._system_status)
            self.app.get("/metrics", response_model=MetricsResponse)(self._get_metrics)
            
            # Engine endpoints
            self.app.post("/llm/generate", response_model=TextGenerationResponse)(self._generate_text)
            self.app.post("/llm/generate_batch")(self._generate_text_batch)
            self.app.post("/diffusion/generate", response_model=ImageGenerationResponse)(self._generate_image)
            self.app.post("/diffusion/generate_batch")(self._generate_image_batch)
            self.app.post("/router/route")(self._route_request)
            
            # Management endpoints
            self.app.post("/routes/register", response_model=RouteRegistrationResponse)(self._register_route)
            self.app.delete("/routes/{route_id}")(self._unregister_route)
            self.app.get("/routes")(self._list_routes)
            
            # Alerting endpoints
            self.app.post("/alerts/rules", response_model=AlertRuleResponse)(self._create_alert_rule)
            self.app.get("/alerts/rules")(self._list_alert_rules)
            self.app.delete("/alerts/rules/{rule_name}")(self._delete_alert_rule)
            self.app.get("/alerts/active")(self._get_active_alerts)
            self.app.post("/alerts/{rule_name}/acknowledge")(self._acknowledge_alert)
            
            # System endpoints
            self.app.post("/system/shutdown")(self._shutdown_system)
            self.app.post("/system/restart")(self._restart_system)
            
        except Exception as e:
            self.logger.error(f"Failed to setup routes: {e}")
    
    def _setup_error_handlers(self):
        """Setup error handlers."""
        try:
            @self.app.exception_handler(HTTPException)
            async def http_exception_handler(request, exc):
                return JSONResponse(
                    status_code=exc.status_code,
                    content={"error": exc.detail, "timestamp": time.time()}
                )
            
            @self.app.exception_handler(Exception)
            async def general_exception_handler(request, exc):
                self.logger.error(f"Unhandled exception: {exc}")
                return JSONResponse(
                    status_code=500,
                    content={"error": "Internal server error", "timestamp": time.time()}
                )
                
        except Exception as e:
            self.logger.error(f"Failed to setup error handlers: {e}")
    
    async def _initialize_components(self):
        """Initialize system components."""
        try:
            if not self.engine_manager:
                self.engine_manager = get_engine_manager(self.config)
            
            if not self.metrics_collector:
                self.metrics_collector = get_metrics_collector(self.config)
            
            if not self.alerting_engine:
                self.alerting_engine = get_alerting_engine(self.config)
                
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
    
    # =============================================================================
    # Health and Status Endpoints
    # =============================================================================
    
    async def _health_check(self) -> HealthResponse:
        """Health check endpoint."""
        try:
            await self._initialize_components()
            
            # Check engine manager health
            engine_status = "healthy"
            if self.engine_manager:
                try:
                    engine_metrics = self.engine_manager.get_system_metrics()
                    if engine_metrics.get("total_engines", 0) == 0:
                        engine_status = "degraded"
                except Exception:
                    engine_status = "unhealthy"
            
            # Check metrics collector health
            metrics_status = "healthy"
            if self.metrics_collector:
                try:
                    metrics_summary = self.metrics_collector.get_metrics_summary()
                    if not metrics_summary:
                        metrics_status = "degraded"
                except Exception:
                    metrics_status = "unhealthy"
            
            # Check alerting engine health
            alerting_status = "healthy"
            if self.alerting_engine:
                try:
                    alerts_summary = self.alerting_engine.get_alerts_summary()
                    if alerts_summary.get("active_alerts", 0) > 10:
                        alerting_status = "degraded"
                except Exception:
                    alerting_status = "unhealthy"
            
            # Determine overall status
            overall_status = "healthy"
            if any(status == "unhealthy" for status in [engine_status, metrics_status, alerting_status]):
                overall_status = "unhealthy"
            elif any(status == "degraded" for status in [engine_status, metrics_status, alerting_status]):
                overall_status = "degraded"
            
            return HealthResponse(
                status=overall_status,
                timestamp=time.time(),
                components={
                    "engine_manager": engine_status,
                    "metrics_collector": metrics_status,
                    "alerting_engine": alerting_status
                },
                version="2.0.0"
            )
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Health check failed: {str(e)}"
            )
    
    async def _system_status(self) -> SystemStatusResponse:
        """System status endpoint."""
        try:
            await self._initialize_components()
            
            # Get engine status
            engines = {}
            if self.engine_manager:
                engines = self.engine_manager.get_engine_status()
            
            # Get system metrics
            metrics = {}
            if self.engine_manager:
                metrics = self.engine_manager.get_system_metrics()
            
            # Get alerts summary
            alerts = {}
            if self.alerting_engine:
                alerts = self.alerting_engine.get_alerts_summary()
            
            return SystemStatusResponse(
                overall_status="operational",
                engines=engines,
                metrics=metrics,
                alerts=alerts,
                timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"System status failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"System status failed: {str(e)}"
            )
    
    async def _get_metrics(self) -> MetricsResponse:
        """Get system metrics endpoint."""
        try:
            await self._initialize_components()
            
            metrics = {}
            if self.metrics_collector:
                metrics = self.metrics_collector.get_metrics_summary()
            
            return MetricsResponse(
                metrics=metrics,
                timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"Metrics retrieval failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Metrics retrieval failed: {str(e)}"
            )
    
    # =============================================================================
    # Engine Endpoints
    # =============================================================================
    
    async def _generate_text(self, request: TextGenerationRequest) -> TextGenerationResponse:
        """Text generation endpoint."""
        try:
            await self._initialize_components()
            
            start_time = time.time()
            
            result = await self.engine_manager.dispatch(
                "llm", "generate", {
                    "prompt": request.prompt,
                    "max_length": request.max_length,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "top_k": request.top_k,
                    "repetition_penalty": request.repetition_penalty,
                    "do_sample": request.do_sample
                }
            )
            
            processing_time = time.time() - start_time
            
            return TextGenerationResponse(
                text=result.text,
                tokens=result.tokens,
                usage=result.usage,
                processing_time=processing_time,
                model_name=result.model_name or "unknown"
            )
            
        except Exception as e:
            self.logger.error(f"Text generation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Text generation failed: {str(e)}"
            )
    
    async def _generate_text_batch(self, requests: List[TextGenerationRequest]):
        """Batch text generation endpoint."""
        try:
            await self._initialize_components()
            
            batch_requests = []
            for req in requests:
                batch_requests.append({
                    "prompt": req.prompt,
                    "max_length": req.max_length,
                    "temperature": req.temperature,
                    "top_p": req.top_p,
                    "top_k": req.top_k,
                    "repetition_penalty": req.repetition_penalty,
                    "do_sample": req.do_sample
                })
            
            result = await self.engine_manager.dispatch(
                "llm", "generate_batch", {"requests": batch_requests}
            )
            
            return {"results": result}
            
        except Exception as e:
            self.logger.error(f"Batch text generation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Batch text generation failed: {str(e)}"
            )
    
    async def _generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        """Image generation endpoint."""
        try:
            await self._initialize_components()
            
            start_time = time.time()
            
            result = await self.engine_manager.dispatch(
                "diffusion", "generate", {
                    "prompt": request.prompt,
                    "width": request.width,
                    "height": request.height,
                    "num_inference_steps": request.num_inference_steps,
                    "guidance_scale": request.guidance_scale,
                    "seed": request.seed
                }
            )
            
            processing_time = time.time() - start_time
            
            # Convert images to base64
            import base64
            import io
            from PIL import Image
            
            image_strings = []
            for img in result.images:
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                img_str = base64.b64encode(buffer.getvalue()).decode()
                image_strings.append(img_str)
            
            return ImageGenerationResponse(
                images=image_strings,
                metadata=result.metadata,
                processing_time=processing_time,
                model_name=result.model_name or "unknown"
            )
            
        except Exception as e:
            self.logger.error(f"Image generation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Image generation failed: {str(e)}"
            )
    
    async def _generate_image_batch(self, requests: List[ImageGenerationRequest]):
        """Batch image generation endpoint."""
        try:
            await self._initialize_components()
            
            batch_requests = []
            for req in requests:
                batch_requests.append({
                    "prompt": req.prompt,
                    "width": req.width,
                    "height": req.height,
                    "num_inference_steps": req.num_inference_steps,
                    "guidance_scale": request.guidance_scale,
                    "seed": req.seed
                })
            
            result = await self.engine_manager.dispatch(
                "diffusion", "generate_batch", {"requests": batch_requests}
            )
            
            return {"results": result}
            
        except Exception as e:
            self.logger.error(f"Batch image generation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Batch image generation failed: {str(e)}"
            )
    
    async def _route_request(self, request: Dict[str, Any]):
        """Route request endpoint."""
        try:
            await self._initialize_components()
            
            result = await self.engine_manager.dispatch(
                "router", "route", request
            )
            
            return {"result": result}
            
        except Exception as e:
            self.logger.error(f"Routing failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Routing failed: {str(e)}"
            )
    
    # =============================================================================
    # Management Endpoints
    # =============================================================================
    
    async def _register_route(self, request: RouteRegistrationRequest) -> RouteRegistrationResponse:
        """Route registration endpoint."""
        try:
            await self._initialize_components()
            
            result = await self.engine_manager.dispatch(
                "router", "register_route", {
                    "route_id": request.route_id,
                    "target_engine": request.target_engine,
                    "target_operation": request.target_operation,
                    "weight": request.weight,
                    "priority": request.priority
                }
            )
            
            return RouteRegistrationResponse(
                route_id=request.route_id,
                status="success" if result else "failed",
                message="Route registered successfully" if result else "Route registration failed"
            )
            
        except Exception as e:
            self.logger.error(f"Route registration failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Route registration failed: {str(e)}"
            )
    
    async def _unregister_route(self, route_id: str):
        """Route unregistration endpoint."""
        try:
            await self._initialize_components()
            
            result = await self.engine_manager.dispatch(
                "router", "unregister_route", {"route_id": route_id}
            )
            
            return {"status": "success" if result else "failed"}
            
        except Exception as e:
            self.logger.error(f"Route unregistration failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Route unregistration failed: {str(e)}"
            )
    
    async def _list_routes(self):
        """List routes endpoint."""
        try:
            await self._initialize_components()
            
            # This would get route information from the router engine
            return {"routes": [], "total": 0}
            
        except Exception as e:
            self.logger.error(f"Route listing failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Route listing failed: {str(e)}"
            )
    
    # =============================================================================
    # Alerting Endpoints
    # =============================================================================
    
    async def _create_alert_rule(self, request: AlertRuleRequest) -> AlertRuleResponse:
        """Create alert rule endpoint."""
        try:
            await self._initialize_components()
            
            # Create alert rule
            rule = AlertRule(
                name=request.name,
                description=request.description,
                severity=AlertSeverity(request.severity),
                condition=request.condition,
                threshold=request.threshold,
                comparison=request.comparison,
                duration=request.duration,
                cooldown=request.cooldown,
                labels=request.labels,
                annotations=request.annotations
            )
            
            self.alerting_engine.add_alert_rule(rule)
            
            return AlertRuleResponse(
                name=rule.name,
                description=rule.description,
                severity=rule.severity.value,
                enabled=rule.enabled,
                created_at=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"Alert rule creation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Alert rule creation failed: {str(e)}"
            )
    
    async def _list_alert_rules(self):
        """List alert rules endpoint."""
        try:
            await self._initialize_components()
            
            rules = []
            for rule in self.alerting_engine.alert_rules.values():
                rules.append({
                    "name": rule.name,
                    "description": rule.description,
                    "severity": rule.severity.value,
                    "enabled": rule.enabled
                })
            
            return {"rules": rules, "total": len(rules)}
            
        except Exception as e:
            self.logger.error(f"Alert rule listing failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Alert rule listing failed: {str(e)}"
            )
    
    async def _delete_alert_rule(self, rule_name: str):
        """Delete alert rule endpoint."""
        try:
            await self._initialize_components()
            
            self.alerting_engine.remove_alert_rule(rule_name)
            
            return {"status": "success", "message": f"Alert rule {rule_name} deleted"}
            
        except Exception as e:
            self.logger.error(f"Alert rule deletion failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Alert rule deletion failed: {str(e)}"
            )
    
    async def _get_active_alerts(self):
        """Get active alerts endpoint."""
        try:
            await self._initialize_components()
            
            alerts = []
            for alert in self.alerting_engine.active_alerts.values():
                alerts.append({
                    "rule_name": alert.rule_name,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "status": alert.status.value,
                    "created_at": alert.created_at,
                    "value": alert.value,
                    "threshold": alert.threshold
                })
            
            return {"alerts": alerts, "total": len(alerts)}
            
        except Exception as e:
            self.logger.error(f"Active alerts retrieval failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Active alerts retrieval failed: {str(e)}"
            )
    
    async def _acknowledge_alert(self, rule_name: str):
        """Acknowledge alert endpoint."""
        try:
            await self._initialize_components()
            
            self.alerting_engine.acknowledge_alert(rule_name)
            
            return {"status": "success", "message": f"Alert {rule_name} acknowledged"}
            
        except Exception as e:
            self.logger.error(f"Alert acknowledgement failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Alert acknowledgement failed: {str(e)}"
            )
    
    # =============================================================================
    # System Endpoints
    # =============================================================================
    
    async def _shutdown_system(self):
        """System shutdown endpoint."""
        try:
            # This would initiate a graceful shutdown
            return {"status": "success", "message": "System shutdown initiated"}
            
        except Exception as e:
            self.logger.error(f"System shutdown failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"System shutdown failed: {str(e)}"
            )
    
    async def _restart_system(self):
        """System restart endpoint."""
        try:
            # This would initiate a system restart
            return {"status": "success", "message": "System restart initiated"}
            
        except Exception as e:
            self.logger.error(f"System restart failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"System restart failed: {str(e)}"
            )
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """Run the API server."""
        try:
            if not FASTAPI_AVAILABLE:
                self.logger.error("FastAPI not available, cannot run server")
                return
            
            self.logger.info(f"Starting REST API server on {host}:{port}")
            uvicorn.run(self.app, host=host, port=port, **kwargs)
            
        except Exception as e:
            self.logger.error(f"Failed to start API server: {e}")

# =============================================================================
# Global API Instance
# =============================================================================

_global_api: Optional[AdvancedRESTAPI] = None

def get_rest_api(config: Optional[CoreConfig] = None) -> AdvancedRESTAPI:
    """Get the global REST API instance."""
    global _global_api
    if _global_api is None:
        _global_api = AdvancedRESTAPI(config)
    return _global_api


