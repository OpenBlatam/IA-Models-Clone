"""
Enhanced main application for Blaze AI with advanced security, monitoring, and error handling.

This module demonstrates the improved Blaze AI module with enterprise-grade features
including comprehensive security, performance monitoring, and robust error handling.
"""

import asyncio
import uvicorn
from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import yaml
from pathlib import Path
import time
import logging

from . import create_modular_ai, get_logger
from .api.router import router as blaze_ai_router
from .core.interfaces import CoreConfig
from .utils.error_handling import (
    BlazeAIError, ServiceUnavailableError, RateLimitExceededError,
    ValidationError, get_error_monitor, CircuitBreaker, CircuitBreakerConfig
)
from .utils.rate_limiting import (
    RateLimitManager, RateLimitConfig, RateLimitAlgorithm, ThrottleAction
)
from .utils.performance_monitoring import (
    PerformanceMonitor, MonitoringConfig, ProfilingLevel, set_performance_monitor
)
from .middleware.security import (
    SecurityMiddleware, SecurityConfig, ThreatDetectionConfig
)

# =============================================================================
# Configuration
# =============================================================================

def load_config(config_path: str = "config.yaml") -> CoreConfig:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        return CoreConfig(**config_data)
    except FileNotFoundError:
        logger = get_logger("main")
        logger.warning(f"Config file {config_path} not found, using defaults")
        return CoreConfig()
    except Exception as e:
        logger = get_logger("main")
        logger.error(f"Error loading config: {e}")
        return CoreConfig()

def create_security_config() -> SecurityConfig:
    """Create security configuration."""
    return SecurityConfig(
        enable_authentication=True,
        enable_authorization=True,
        enable_input_validation=True,
        enable_threat_detection=True,
        enable_rate_limiting=True,
        enable_audit_logging=True,
        jwt_secret_key="your-super-secret-key-change-in-production",
        jwt_algorithm="HS256",
        jwt_expiration=3600,
        api_key_header="X-API-Key",
        enable_cors=True,
        allowed_origins=["*"],
        max_request_size=10 * 1024 * 1024,  # 10MB
        enable_encryption=False
    )

def create_threat_detection_config() -> ThreatDetectionConfig:
    """Create threat detection configuration."""
    return ThreatDetectionConfig(
        enable_sql_injection_detection=True,
        enable_xss_detection=True,
        enable_path_traversal_detection=True,
        enable_command_injection_detection=True,
        enable_rate_limit_bypass_detection=True,
        suspicious_patterns=[
            r"eval\s*\(",
            r"exec\s*\(",
            r"system\s*\(",
            r"subprocess\s*\(",
        ],
        max_failed_attempts=5,
        lockout_duration=300,
        enable_ip_blacklisting=True,
        enable_behavioral_analysis=True
    )

def create_rate_limit_config() -> RateLimitConfig:
    """Create rate limiting configuration."""
    return RateLimitConfig(
        algorithm=RateLimitAlgorithm.ADAPTIVE,
        requests_per_minute=100,
        requests_per_hour=1000,
        requests_per_day=10000,
        burst_limit=50,
        window_size=60,
        action=ThrottleAction.REJECT,
        enable_user_limits=True,
        enable_ip_limits=True,
        enable_global_limits=True,
        enable_distributed=False
    )

def create_monitoring_config() -> MonitoringConfig:
    """Create performance monitoring configuration."""
    return MonitoringConfig(
        enable_monitoring=True,
        enable_profiling=True,
        enable_memory_tracking=True,
        enable_system_metrics=True,
        enable_custom_metrics=True,
        metrics_interval=1.0,
        profiling_level=ProfilingLevel.DETAILED,
        max_metrics_history=1000,
        enable_alerting=True,
        alert_thresholds={
            "system.cpu.percent": 80.0,
            "system.memory.percent": 85.0,
            "system.disk.percent": 90.0
        }
    )

# =============================================================================
# Enhanced FastAPI Application
# =============================================================================

def create_enhanced_app(config: CoreConfig) -> FastAPI:
    """Create and configure enhanced FastAPI application."""
    app = FastAPI(
        title="Blaze AI Enhanced API",
        description="Advanced AI Module with Enterprise-Grade Security and Monitoring",
        version="2.1.0",
        docs_url="/docs" if config.api.enable_docs else None,
        redoc_url="/redoc" if config.api.enable_docs else None
    )
    
    # Add security middleware
    security_config = create_security_config()
    security_middleware = SecurityMiddleware(security_config)
    app.add_middleware(security_middleware)
    
    # Add trusted host middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # Configure appropriately for production
    )
    
    # Add CORS middleware
    if security_config.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=security_config.allowed_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Include Blaze AI router
    app.include_router(blaze_ai_router, prefix=config.api.api_prefix)
    
    # Add custom exception handlers
    app.add_exception_handler(BlazeAIError, handle_blaze_ai_error)
    app.add_exception_handler(ServiceUnavailableError, handle_service_unavailable)
    app.add_exception_handler(RateLimitExceededError, handle_rate_limit_exceeded)
    app.add_exception_handler(ValidationError, handle_validation_error)
    
    # Add startup and shutdown events
    @app.on_event("startup")
    async def startup_event():
        logger = get_logger("main")
        logger.info("Starting Enhanced Blaze AI application...")
        
        # Initialize performance monitoring
        monitoring_config = create_monitoring_config()
        performance_monitor = PerformanceMonitor(monitoring_config)
        set_performance_monitor(performance_monitor)
        performance_monitor.start_monitoring()
        
        # Initialize rate limiting
        rate_limit_config = create_rate_limit_config()
        rate_limit_manager = RateLimitManager(rate_limit_config)
        app.state.rate_limit_manager = rate_limit_manager
        
        # Initialize circuit breakers
        circuit_breakers = {
            "llm_engine": CircuitBreaker("llm_engine", CircuitBreakerConfig()),
            "diffusion_engine": CircuitBreaker("diffusion_engine", CircuitBreakerConfig()),
            "seo_service": CircuitBreaker("seo_service", CircuitBreakerConfig())
        }
        app.state.circuit_breakers = circuit_breakers
        
        # Initialize the AI module
        try:
            ai = create_modular_ai(config=config)
            app.state.ai = ai
            logger.info("Enhanced Blaze AI module initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced Blaze AI module: {e}")
            raise
    
    @app.on_event("shutdown")
    async def shutdown_event():
        logger = get_logger("main")
        logger.info("Shutting down Enhanced Blaze AI application...")
        
        # Stop performance monitoring
        performance_monitor = get_performance_monitor()
        if performance_monitor:
            performance_monitor.stop_monitoring()
        
        # Shutdown the AI module
        if hasattr(app.state, 'ai'):
            try:
                await app.state.ai.shutdown()
                logger.info("Enhanced Blaze AI module shutdown successfully")
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")
    
    return app

# =============================================================================
# Exception Handlers
# =============================================================================

async def handle_blaze_ai_error(request: Request, exc: BlazeAIError):
    """Handle Blaze AI specific errors."""
    error_monitor = get_error_monitor()
    if error_monitor:
        error_monitor.record_error(exc, {
            "request_path": request.url.path,
            "request_method": request.method,
            "client_ip": request.client.host if request.client else "unknown"
        })
    
    return JSONResponse(
        status_code=400,
        content={
            "error": "Blaze AI Error",
            "message": str(exc),
            "severity": exc.severity.value,
            "timestamp": exc.timestamp
        }
    )

async def handle_service_unavailable(request: Request, exc: ServiceUnavailableError):
    """Handle service unavailable errors."""
    error_monitor = get_error_monitor()
    if error_monitor:
        error_monitor.record_error(exc, {
            "request_path": request.url.path,
            "request_method": request.method,
            "client_ip": request.client.host if request.client else "unknown"
        })
    
    return JSONResponse(
        status_code=503,
        content={
            "error": "Service Unavailable",
            "message": str(exc),
            "severity": exc.severity.value,
            "timestamp": exc.timestamp
        }
    )

async def handle_rate_limit_exceeded(request: Request, exc: RateLimitExceededError):
    """Handle rate limit exceeded errors."""
    error_monitor = get_error_monitor()
    if error_monitor:
        error_monitor.record_error(exc, {
            "request_path": request.url.path,
            "request_method": request.method,
            "client_ip": request.client.host if request.client else "unknown"
        })
    
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate Limit Exceeded",
            "message": str(exc),
            "severity": exc.severity.value,
            "timestamp": exc.timestamp,
            "retry_after": 60
        }
    )

async def handle_validation_error(request: Request, exc: ValidationError):
    """Handle validation errors."""
    error_monitor = get_error_monitor()
    if error_monitor:
        error_monitor.record_error(exc, {
            "request_path": request.url.path,
            "request_method": request.method,
            "client_ip": request.client.host if request.client else "unknown"
        })
    
    return JSONResponse(
        status_code=400,
        content={
            "error": "Validation Error",
            "message": str(exc),
            "severity": exc.severity.value,
            "timestamp": exc.timestamp
        }
    )

# =============================================================================
# Enhanced Health Check Endpoints
# =============================================================================

def add_health_endpoints(app: FastAPI):
    """Add enhanced health check endpoints."""
    
    @app.get("/health")
    async def health_check():
        """Basic health check."""
        return {"status": "healthy", "timestamp": time.time()}
    
    @app.get("/health/detailed")
    async def detailed_health_check():
        """Detailed health check with all systems."""
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "systems": {}
        }
        
        # Check AI module
        if hasattr(app.state, 'ai'):
            try:
                ai_status = await app.state.ai.get_health_status()
                health_status["systems"]["ai_module"] = ai_status
            except Exception as e:
                health_status["systems"]["ai_module"] = {"status": "error", "error": str(e)}
                health_status["status"] = "degraded"
        
        # Check performance monitoring
        performance_monitor = get_performance_monitor()
        if performance_monitor:
            try:
                metrics_summary = performance_monitor.get_metrics_summary()
                health_status["systems"]["performance_monitoring"] = {
                    "status": "healthy",
                    "metrics_count": metrics_summary.get("metrics_count", 0)
                }
            except Exception as e:
                health_status["systems"]["performance_monitoring"] = {"status": "error", "error": str(e)}
                health_status["status"] = "degraded"
        
        # Check circuit breakers
        if hasattr(app.state, 'circuit_breakers'):
            circuit_breaker_status = {}
            for name, cb in app.state.circuit_breakers.items():
                circuit_breaker_status[name] = cb.get_status()
            health_status["systems"]["circuit_breakers"] = circuit_breaker_status
        
        # Check rate limiting
        if hasattr(app.state, 'rate_limit_manager'):
            try:
                rate_limit_status = app.state.rate_limit_manager.get_limiter_status()
                health_status["systems"]["rate_limiting"] = {
                    "status": "healthy",
                    "limiters": rate_limit_status
                }
            except Exception as e:
                health_status["systems"]["rate_limiting"] = {"status": "error", "error": str(e)}
                health_status["status"] = "degraded"
        
        return health_status
    
    @app.get("/metrics")
    async def get_metrics():
        """Get performance metrics."""
        performance_monitor = get_performance_monitor()
        if not performance_monitor:
            raise HTTPException(status_code=503, detail="Performance monitoring not available")
        
        return performance_monitor.get_metrics_summary()
    
    @app.get("/metrics/prometheus")
    async def get_prometheus_metrics():
        """Get metrics in Prometheus format."""
        performance_monitor = get_performance_monitor()
        if not performance_monitor:
            raise HTTPException(status_code=503, detail="Performance monitoring not available")
        
        prometheus_metrics = performance_monitor.export_metrics("prometheus")
        return Response(content=prometheus_metrics, media_type="text/plain")
    
    @app.get("/security/status")
    async def get_security_status():
        """Get security status and threat summary."""
        # This would typically get from the security middleware
        return {
            "status": "secure",
            "timestamp": time.time(),
            "threats_blocked": 0,
            "ips_blocked": 0,
            "recent_incidents": []
        }
    
    @app.get("/errors/summary")
    async def get_error_summary():
        """Get error summary from error monitor."""
        error_monitor = get_error_monitor()
        if not error_monitor:
            return {"error": "Error monitoring not available"}
        
        return error_monitor.get_error_summary()

# =============================================================================
# Main Application Entry Point
# =============================================================================

async def main_enhanced():
    """Enhanced main application entry point."""
    # Load configuration
    config = load_config()
    
    # Setup logging
    from .utils.logging import setup_logging
    setup_logging(
        level=config.log_level.value,
        log_file=config.log_file
    )
    
    logger = get_logger("main")
    logger.info("Initializing Enhanced Blaze AI application...")
    
    # Create enhanced FastAPI app
    app = create_enhanced_app(config)
    
    # Add health endpoints
    add_health_endpoints(app)
    
    # Start server
    logger.info(f"Starting enhanced server on {config.api.host}:{config.api.port}")
    config_dict = uvicorn.Config(
        app=app,
        host=config.api.host,
        port=config.api.port,
        workers=config.api.workers,
        log_level=config.log_level.value.lower()
    )
    
    server = uvicorn.Server(config_dict)
    await server.serve()

# =============================================================================
# Development Server
# =============================================================================

def run_development_server():
    """Run development server with hot reload."""
    import uvicorn
    
    uvicorn.run(
        "main_enhanced:create_enhanced_app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["."],
        log_level="info"
    )

if __name__ == "__main__":
    # For development, use the development server
    if "--dev" in sys.argv:
        run_development_server()
    else:
        # For production, use the async main
        asyncio.run(main_enhanced())
