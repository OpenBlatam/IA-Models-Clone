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
import traceback
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request, Response, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse
import uvicorn
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from .config import config, Environment
from .monitoring import ProductionMonitoring, monitoring_context
from ..core.math_service import create_math_service, MathOperation, OperationType, CalculationMethod
from ..workflow.workflow_engine import MathWorkflowEngine
from ..analytics.analytics_engine import MathAnalyticsEngine
from ..optimization.optimization_engine import MathOptimizationEngine
from ..platform.unified_platform import UnifiedMathPlatform, PlatformConfig
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Production FastAPI Application
Production-ready FastAPI application with comprehensive features.
"""





# Configure logging
logger = logging.getLogger(__name__)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting API metrics."""
    
    def __init__(self, app, monitoring: ProductionMonitoring):
        
    """__init__ function."""
super().__init__(app)
        self.monitoring = monitoring
    
    async def dispatch(self, request: StarletteRequest, call_next) -> StarletteResponse:
        start_time = time.time()
        
        # Increment active requests
        self.monitoring.app_metrics.api_active_requests.inc()
        
        try:
            response = await call_next(request)
            
            # Record metrics
            duration = time.time() - start_time
            self.monitoring.app_metrics.api_requests.labels(
                method=request.method,
                endpoint=request.url.path,
                status_code=response.status_code
            ).inc()
            self.monitoring.app_metrics.api_duration.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(duration)
            
            return response
            
        except Exception as e:
            # Record error metrics
            duration = time.time() - start_time
            self.monitoring.app_metrics.api_requests.labels(
                method=request.method,
                endpoint=request.url.path,
                status_code=500
            ).inc()
            self.monitoring.app_metrics.api_duration.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(duration)
            
            raise
        finally:
            # Decrement active requests
            self.monitoring.app_metrics.api_active_requests.dec()


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for request validation and rate limiting."""
    
    def __init__(self, app, config) -> Any:
        super().__init__(app)
        self.config = config
        self.request_counts = {}
        self.last_cleanup = time.time()
    
    async def dispatch(self, request: StarletteRequest, call_next) -> StarletteResponse:
        # Rate limiting
        client_ip = request.client.host
        current_time = time.time()
        
        # Cleanup old entries
        if current_time - self.last_cleanup > 60:
            self._cleanup_old_entries(current_time)
            self.last_cleanup = current_time
        
        # Check rate limit
        if not self._check_rate_limit(client_ip, current_time):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"error": "Rate limit exceeded"}
            )
        
        # Request size validation
        content_length = request.headers.get("content-length")
        if content_length:
            size = int(content_length)
            if size > self.config.security.max_request_size:
                return JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content={"error": "Request too large"}
                )
        
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response
    
    def _check_rate_limit(self, client_ip: str, current_time: float) -> bool:
        """Check if client has exceeded rate limit."""
        window_start = current_time - self.config.security.rate_limit_window
        
        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = []
        
        # Remove old requests
        self.request_counts[client_ip] = [
            req_time for req_time in self.request_counts[client_ip]
            if req_time > window_start
        ]
        
        # Check limit
        if len(self.request_counts[client_ip]) >= self.config.security.rate_limit_requests:
            return False
        
        # Add current request
        self.request_counts[client_ip].append(current_time)
        return True
    
    def _cleanup_old_entries(self, current_time: float):
        """Clean up old rate limiting entries."""
        window_start = current_time - self.config.security.rate_limit_window
        for client_ip in list(self.request_counts.keys()):
            self.request_counts[client_ip] = [
                req_time for req_time in self.request_counts[client_ip]
                if req_time > window_start
            ]
            if not self.request_counts[client_ip]:
                del self.request_counts[client_ip]


# Global variables for app lifecycle
monitoring: Optional[ProductionMonitoring] = None
math_platform: Optional[UnifiedMathPlatform] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global monitoring, math_platform
    
    # Startup
    logger.info("Starting Math Platform API...")
    
    # Initialize monitoring
    monitoring = ProductionMonitoring(config)
    
    # Initialize math platform
    platform_config = PlatformConfig(
        max_workers=config.performance.max_workers,
        cache_size=config.performance.cache_size,
        analytics_enabled=True,
        optimization_enabled=True,
        workflow_enabled=True
    )
    
    math_platform = UnifiedMathPlatform(platform_config)
    await math_platform.initialize()
    
    logger.info("Math Platform API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Math Platform API...")
    
    if math_platform:
        await math_platform.shutdown()
    
    if monitoring:
        monitoring.shutdown()
    
    logger.info("Math Platform API shutdown complete")


def create_production_app() -> FastAPI:
    """Create production FastAPI application."""
    
    # Create FastAPI app
    app = FastAPI(
        title=config.api.title,
        version=config.api.version,
        description=config.api.description,
        docs_url=config.api.docs_url if config.settings.debug else None,
        redoc_url=config.api.redoc_url if config.settings.debug else None,
        openapi_url=config.api.openapi_url if config.settings.debug else None,
        lifespan=lifespan
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.security.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"] if config.settings.debug else ["localhost", "127.0.0.1"]
    )
    
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Add custom middleware
    @app.middleware("http")
    async def add_monitoring_middleware(request: Request, call_next):
        
    """add_monitoring_middleware function."""
if monitoring:
            return await MetricsMiddleware(app, monitoring).dispatch(request, call_next)
        return await call_next(request)
    
    @app.middleware("http")
    async def add_security_middleware(request: Request, call_next):
        
    """add_security_middleware function."""
return await SecurityMiddleware(app, config).dispatch(request, call_next)
    
    # Exception handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        
    """validation_exception_handler function."""
logger.warning(f"Validation error: {exc.errors()}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"error": "Validation error", "details": exc.errors()}
        )
    
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        
    """http_exception_handler function."""
logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail}
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        
    """general_exception_handler function."""
logger.error(f"Unhandled exception: {exc}", exc_info=True)
        
        if monitoring:
            monitoring.performance_monitor.record_error("unhandled_exception")
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "Internal server error"}
        )
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        if not monitoring:
            raise HTTPException(status_code=503, detail="Monitoring not available")
        
        health_data = monitoring.get_monitoring_data()
        
        # Determine overall status
        overall_status = health_data["health"]["overall_status"]
        if overall_status == "critical":
            status_code = 503
        elif overall_status == "unhealthy":
            status_code = 503
        elif overall_status == "degraded":
            status_code = 200
        else:
            status_code = 200
        
        return JSONResponse(
            status_code=status_code,
            content=health_data
        )
    
    # Metrics endpoint
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        if not config.monitoring.prometheus_enabled:
            raise HTTPException(status_code=404, detail="Metrics not enabled")
        
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )
    
    # Math operation endpoints
    @app.post("/math/add")
    async def add_numbers(request: Dict[str, Any]):
        """Add multiple numbers."""
        try:
            numbers = request.get("numbers", [])
            method = request.get("method", "basic")
            
            if not numbers or len(numbers) < 2:
                raise HTTPException(status_code=400, detail="At least 2 numbers required")
            
            result = await math_platform.execute_operation("add", numbers, method=method)
            
            if monitoring:
                monitoring.app_metrics.record_operation("add", method, result.execution_time, result.success)
            
            return {
                "result": result.value,
                "execution_time": result.execution_time,
                "method": method
            }
            
        except Exception as e:
            logger.error(f"Error in add operation: {e}")
            if monitoring:
                monitoring.performance_monitor.record_error("add_operation_error")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/math/multiply")
    async def multiply_numbers(request: Dict[str, Any]):
        """Multiply multiple numbers."""
        try:
            numbers = request.get("numbers", [])
            method = request.get("method", "basic")
            
            if not numbers or len(numbers) < 2:
                raise HTTPException(status_code=400, detail="At least 2 numbers required")
            
            result = await math_platform.execute_operation("multiply", numbers, method=method)
            
            if monitoring:
                monitoring.app_metrics.record_operation("multiply", method, result.execution_time, result.success)
            
            return {
                "result": result.value,
                "execution_time": result.execution_time,
                "method": method
            }
            
        except Exception as e:
            logger.error(f"Error in multiply operation: {e}")
            if monitoring:
                monitoring.performance_monitor.record_error("multiply_operation_error")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/math/divide")
    async def divide_numbers(request: Dict[str, Any]):
        """Divide two numbers."""
        try:
            a = request.get("a")
            b = request.get("b")
            method = request.get("method", "basic")
            
            if a is None or b is None:
                raise HTTPException(status_code=400, detail="Both 'a' and 'b' parameters required")
            
            if b == 0:
                raise HTTPException(status_code=400, detail="Division by zero")
            
            result = await math_platform.execute_operation("divide", [a, b], method=method)
            
            if monitoring:
                monitoring.app_metrics.record_operation("divide", method, result.execution_time, result.success)
            
            return {
                "result": result.value,
                "execution_time": result.execution_time,
                "method": method
            }
            
        except Exception as e:
            logger.error(f"Error in divide operation: {e}")
            if monitoring:
                monitoring.performance_monitor.record_error("divide_operation_error")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Batch operations endpoint
    @app.post("/math/batch")
    async def batch_operations(request: Dict[str, Any]):
        """Execute multiple operations in batch."""
        try:
            operations = request.get("operations", [])
            
            if not operations:
                raise HTTPException(status_code=400, detail="No operations provided")
            
            if len(operations) > 100:
                raise HTTPException(status_code=400, detail="Maximum 100 operations per batch")
            
            results = []
            for op in operations:
                op_type = op.get("type")
                operands = op.get("operands", [])
                method = op.get("method", "basic")
                
                if not op_type or not operands:
                    results.append({"error": "Invalid operation"})
                    continue
                
                try:
                    result = await math_platform.execute_operation(op_type, operands, method=method)
                    results.append({
                        "result": result.value,
                        "execution_time": result.execution_time,
                        "method": method
                    })
                except Exception as e:
                    results.append({"error": str(e)})
            
            return {"results": results}
            
        except Exception as e:
            logger.error(f"Error in batch operations: {e}")
            if monitoring:
                monitoring.performance_monitor.record_error("batch_operation_error")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Workflow endpoint
    @app.post("/workflow/execute")
    async def execute_workflow(request: Dict[str, Any]):
        """Execute a workflow."""
        try:
            workflow_name = request.get("name", "Default Workflow")
            steps = request.get("steps", [])
            
            if not steps:
                raise HTTPException(status_code=400, detail="No workflow steps provided")
            
            result = await math_platform.execute_workflow(workflow_name, steps)
            
            if monitoring:
                monitoring.app_metrics.record_workflow(workflow_name, result.execution_time, result.success)
            
            return {
                "workflow_name": workflow_name,
                "result": result.output,
                "execution_time": result.execution_time,
                "steps_executed": len(result.step_results)
            }
            
        except Exception as e:
            logger.error(f"Error in workflow execution: {e}")
            if monitoring:
                monitoring.performance_monitor.record_error("workflow_execution_error")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Analytics endpoint
    @app.get("/analytics/dashboard")
    async def get_analytics_dashboard():
        """Get analytics dashboard data."""
        try:
            if not math_platform:
                raise HTTPException(status_code=503, detail="Math platform not available")
            
            analytics_data = await math_platform.get_analytics_dashboard()
            
            return analytics_data
            
        except Exception as e:
            logger.error(f"Error getting analytics dashboard: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Platform status endpoint
    @app.get("/platform/status")
    async def get_platform_status():
        """Get platform status."""
        try:
            if not math_platform:
                raise HTTPException(status_code=503, detail="Math platform not available")
            
            status_data = math_platform.get_platform_status()
            
            return status_data
            
        except Exception as e:
            logger.error(f"Error getting platform status: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


def run_production_server():
    """Run the production server."""
    # Validate production readiness
    issues = config.validate_production_ready()
    if issues:
        logger.error("Production configuration issues:")
        for issue in issues:
            logger.error(f"  - {issue}")
        raise ValueError("Configuration not production-ready")
    
    # Create app
    app = create_production_app()
    
    # Run server
    uvicorn.run(
        app,
        host=config.api.host,
        port=config.api.port,
        workers=config.api.workers,
        log_level=config.api.log_level,
        access_log=config.api.access_log
    )


match __name__:
    case "__main__":
    run_production_server() 