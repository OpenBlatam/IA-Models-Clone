"""
Advanced Monitoring and Observability Module
Features: OpenTelemetry tracing, Prometheus metrics, structured logging, health checks
"""

import asyncio
import time
import logging
import json
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import traceback
from datetime import datetime, timezone

# OpenTelemetry imports
try:
    from opentelemetry import trace, metrics
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.redis import RedisInstrumentor
    from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.trace import Status, StatusCode
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

# Prometheus imports
try:
    from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Structured logging
import structlog
from structlog.stdlib import LoggerFactory

logger = logging.getLogger(__name__)

class LogLevel(Enum):
    """Log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class MetricType(Enum):
    """Metric types"""
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    INFO = "info"

@dataclass
class TracingConfig:
    """Tracing configuration"""
    enabled: bool = True
    service_name: str = "fastapi-service"
    service_version: str = "1.0.0"
    jaeger_endpoint: Optional[str] = None
    otlp_endpoint: Optional[str] = None
    sampling_rate: float = 1.0
    max_attributes: int = 32
    max_events: int = 128
    max_links: int = 128

@dataclass
class MetricsConfig:
    """Metrics configuration"""
    enabled: bool = True
    prometheus_port: int = 8000
    prometheus_path: str = "/metrics"
    custom_metrics: List[str] = field(default_factory=list)
    enable_default_metrics: bool = True

@dataclass
class LoggingConfig:
    """Logging configuration"""
    enabled: bool = True
    level: LogLevel = LogLevel.INFO
    format: str = "json"
    include_trace_id: bool = True
    include_span_id: bool = True
    log_file: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5

@dataclass
class HealthCheckConfig:
    """Health check configuration"""
    enabled: bool = True
    endpoint: str = "/health"
    readiness_endpoint: str = "/health/ready"
    liveness_endpoint: str = "/health/live"
    check_interval: float = 30.0
    timeout: float = 5.0
    dependencies: List[str] = field(default_factory=list)

class CustomMetrics:
    """
    Custom Prometheus metrics for FastAPI applications
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        
        # HTTP metrics
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.http_request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        self.http_request_size = Histogram(
            'http_request_size_bytes',
            'HTTP request size in bytes',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        self.http_response_size = Histogram(
            'http_response_size_bytes',
            'HTTP response size in bytes',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # Application metrics
        self.active_connections = Gauge(
            'active_connections',
            'Number of active connections',
            registry=self.registry
        )
        
        self.database_connections = Gauge(
            'database_connections_active',
            'Number of active database connections',
            registry=self.registry
        )
        
        self.cache_hits = Counter(
            'cache_hits_total',
            'Total cache hits',
            ['cache_type'],
            registry=self.registry
        )
        
        self.cache_misses = Counter(
            'cache_misses_total',
            'Total cache misses',
            ['cache_type'],
            registry=self.registry
        )
        
        # Business metrics
        self.users_registered = Counter(
            'users_registered_total',
            'Total users registered',
            registry=self.registry
        )
        
        self.videos_processed = Counter(
            'videos_processed_total',
            'Total videos processed',
            ['status'],
            registry=self.registry
        )
        
        # System metrics
        self.memory_usage = Gauge(
            'memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        
        self.cpu_usage = Gauge(
            'cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        # Service info
        self.service_info = Info(
            'service_info',
            'Service information',
            registry=self.registry
        )

class StructuredLogger:
    """
    Structured logger with OpenTelemetry integration
    """
    
    def __init__(self, config: LoggingConfig):
        self.config = config
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> structlog.BoundLogger:
        """Setup structured logger"""
        
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                self._add_trace_context,
                structlog.processors.JSONRenderer() if self.config.format == "json" 
                else structlog.dev.ConsoleRenderer()
            ],
            context_class=dict,
            logger_factory=LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        return structlog.get_logger()
    
    def _add_trace_context(self, logger, method_name, event_dict):
        """Add OpenTelemetry trace context to logs"""
        if OPENTELEMETRY_AVAILABLE:
            span = trace.get_current_span()
            if span and span.is_recording():
                span_context = span.get_span_context()
                if self.config.include_trace_id:
                    event_dict["trace_id"] = format(span_context.trace_id, '032x')
                if self.config.include_span_id:
                    event_dict["span_id"] = format(span_context.span_id, '016x')
        return event_dict
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.logger.critical(message, **kwargs)

class HealthChecker:
    """
    Health check system for microservices
    """
    
    def __init__(self, config: HealthCheckConfig):
        self.config = config
        self.checks: Dict[str, Callable] = {}
        self.last_check_time: Dict[str, float] = {}
        self.check_results: Dict[str, Dict[str, Any]] = {}
    
    def add_check(self, name: str, check_func: Callable):
        """Add a health check"""
        self.checks[name] = check_func
    
    async def run_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": {}
        }
        
        all_healthy = True
        
        for name, check_func in self.checks.items():
            try:
                start_time = time.time()
                
                if asyncio.iscoroutinefunction(check_func):
                    check_result = await asyncio.wait_for(
                        check_func(), 
                        timeout=self.config.timeout
                    )
                else:
                    check_result = check_func()
                
                duration = time.time() - start_time
                
                if isinstance(check_result, dict):
                    check_status = check_result.get("status", "healthy")
                    check_message = check_result.get("message", "OK")
                else:
                    check_status = "healthy" if check_result else "unhealthy"
                    check_message = "OK" if check_result else "Check failed"
                
                results["checks"][name] = {
                    "status": check_status,
                    "message": check_message,
                    "duration": duration
                }
                
                if check_status != "healthy":
                    all_healthy = False
                
                self.last_check_time[name] = time.time()
                self.check_results[name] = results["checks"][name]
                
            except asyncio.TimeoutError:
                results["checks"][name] = {
                    "status": "unhealthy",
                    "message": "Check timeout",
                    "duration": self.config.timeout
                }
                all_healthy = False
                
            except Exception as e:
                results["checks"][name] = {
                    "status": "unhealthy",
                    "message": f"Check error: {str(e)}",
                    "duration": 0
                }
                all_healthy = False
        
        results["status"] = "healthy" if all_healthy else "unhealthy"
        return results
    
    async def get_readiness(self) -> Dict[str, Any]:
        """Get readiness status"""
        return await self.run_checks()
    
    async def get_liveness(self) -> Dict[str, Any]:
        """Get liveness status"""
        return {
            "status": "alive",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

class ObservabilityManager:
    """
    Central observability manager for FastAPI applications
    """
    
    def __init__(
        self,
        tracing_config: Optional[TracingConfig] = None,
        metrics_config: Optional[MetricsConfig] = None,
        logging_config: Optional[LoggingConfig] = None,
        health_config: Optional[HealthCheckConfig] = None
    ):
        self.tracing_config = tracing_config or TracingConfig()
        self.metrics_config = metrics_config or MetricsConfig()
        self.logging_config = logging_config or LoggingConfig()
        self.health_config = health_config or HealthCheckConfig()
        
        self.tracer = None
        self.meter = None
        self.custom_metrics = None
        self.structured_logger = None
        self.health_checker = None
        
        self._initialized = False
    
    async def initialize(self):
        """Initialize observability components"""
        if self._initialized:
            return
        
        try:
            # Initialize tracing
            if self.tracing_config.enabled and OPENTELEMETRY_AVAILABLE:
                await self._setup_tracing()
            
            # Initialize metrics
            if self.metrics_config.enabled and PROMETHEUS_AVAILABLE:
                await self._setup_metrics()
            
            # Initialize logging
            if self.logging_config.enabled:
                self._setup_logging()
            
            # Initialize health checks
            if self.health_config.enabled:
                self._setup_health_checks()
            
            self._initialized = True
            logger.info("Observability manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize observability manager: {e}")
            raise
    
    async def _setup_tracing(self):
        """Setup OpenTelemetry tracing"""
        try:
            # Create resource
            resource = Resource.create({
                "service.name": self.tracing_config.service_name,
                "service.version": self.tracing_config.service_version,
            })
            
            # Create tracer provider
            tracer_provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(tracer_provider)
            
            # Add span processors
            if self.tracing_config.jaeger_endpoint:
                jaeger_exporter = JaegerExporter(
                    agent_host_name=self.tracing_config.jaeger_endpoint.split(":")[0],
                    agent_port=int(self.tracing_config.jaeger_endpoint.split(":")[1])
                )
                span_processor = BatchSpanProcessor(jaeger_exporter)
                tracer_provider.add_span_processor(span_processor)
            
            if self.tracing_config.otlp_endpoint:
                otlp_exporter = OTLPSpanExporter(endpoint=self.tracing_config.otlp_endpoint)
                span_processor = BatchSpanProcessor(otlp_exporter)
                tracer_provider.add_span_processor(span_processor)
            
            # Get tracer
            self.tracer = trace.get_tracer(self.tracing_config.service_name)
            
            logger.info("Tracing setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup tracing: {e}")
    
    async def _setup_metrics(self):
        """Setup Prometheus metrics"""
        try:
            # Create custom metrics
            self.custom_metrics = CustomMetrics()
            
            # Create meter provider
            metric_reader = PrometheusMetricReader()
            meter_provider = MeterProvider(metric_readers=[metric_reader])
            metrics.set_meter_provider(meter_provider)
            
            # Get meter
            self.meter = metrics.get_meter(self.tracing_config.service_name)
            
            logger.info("Metrics setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup metrics: {e}")
    
    def _setup_logging(self):
        """Setup structured logging"""
        try:
            self.structured_logger = StructuredLogger(self.logging_config)
            logger.info("Structured logging setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup logging: {e}")
    
    def _setup_health_checks(self):
        """Setup health checks"""
        try:
            self.health_checker = HealthChecker(self.health_config)
            
            # Add default health checks
            self.health_checker.add_check("application", self._check_application)
            self.health_checker.add_check("memory", self._check_memory)
            self.health_checker.add_check("dependencies", self._check_dependencies)
            
            logger.info("Health checks setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup health checks: {e}")
    
    def _check_application(self) -> Dict[str, Any]:
        """Check application health"""
        return {
            "status": "healthy",
            "message": "Application is running",
            "uptime": time.time() - getattr(self, '_start_time', time.time())
        }
    
    def _check_memory(self) -> Dict[str, Any]:
        """Check memory usage"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {
                "status": "healthy" if memory.percent < 90 else "unhealthy",
                "message": f"Memory usage: {memory.percent}%",
                "usage_percent": memory.percent,
                "available_mb": memory.available // (1024 * 1024)
            }
        except ImportError:
            return {
                "status": "unknown",
                "message": "psutil not available"
            }
    
    async def _check_dependencies(self) -> Dict[str, Any]:
        """Check external dependencies"""
        # This is a placeholder - implement your dependency checks
        return {
            "status": "healthy",
            "message": "All dependencies are available"
        }
    
    def instrument_fastapi(self, app):
        """Instrument FastAPI application"""
        if OPENTELEMETRY_AVAILABLE and self.tracing_config.enabled:
            FastAPIInstrumentor.instrument_app(app)
            RequestsInstrumentor().instrument()
            RedisInstrumentor().instrument()
            AsyncioInstrumentor().instrument()
    
    def get_metrics(self) -> str:
        """Get Prometheus metrics"""
        if self.custom_metrics:
            return generate_latest(self.custom_metrics.registry)
        return ""
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status"""
        if self.health_checker:
            return await self.health_checker.run_checks()
        return {"status": "unknown"}
    
    async def get_readiness(self) -> Dict[str, Any]:
        """Get readiness status"""
        if self.health_checker:
            return await self.health_checker.get_readiness()
        return {"status": "unknown"}
    
    async def get_liveness(self) -> Dict[str, Any]:
        """Get liveness status"""
        if self.health_checker:
            return await self.health_checker.get_liveness()
        return {"status": "unknown"}

# Decorators for observability
def trace_function(operation_name: str = None):
    """Decorator to trace function execution"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            if OPENTELEMETRY_AVAILABLE:
                tracer = trace.get_tracer(__name__)
                with tracer.start_as_current_span(operation_name or func.__name__) as span:
                    try:
                        result = await func(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.set_attribute("error", True)
                        span.set_attribute("error.message", str(e))
                        raise
            else:
                return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            if OPENTELEMETRY_AVAILABLE:
                tracer = trace.get_tracer(__name__)
                with tracer.start_as_current_span(operation_name or func.__name__) as span:
                    try:
                        result = func(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.set_attribute("error", True)
                        span.set_attribute("error.message", str(e))
                        raise
            else:
                return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def measure_duration(metric_name: str):
    """Decorator to measure function duration"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                if PROMETHEUS_AVAILABLE:
                    # Record duration metric
                    pass
        
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                if PROMETHEUS_AVAILABLE:
                    # Record duration metric
                    pass
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# Global observability manager instance
observability_manager = ObservabilityManager()






























