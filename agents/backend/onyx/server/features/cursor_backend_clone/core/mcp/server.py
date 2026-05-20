"""
MCP Server - Servidor Model Context Protocol para Cursor
========================================================

Servidor MCP que expone el agente como recurso para Cursor IDE.
Permite que Cursor IDE se conecte y envíe comandos al agente.
"""

import asyncio
import logging
import time
import hashlib
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
import uvicorn

# Importar módulos refactorizados
from .mcp_models import CommandRequest, BatchCommandRequest, LoginRequest, JSONRPCRequest
from .mcp_utils import validate_jsonrpc, get_client_id, get_json_dumps, get_json_loads
from .mcp_auth import authenticate_request, check_permission, require_auth, require_admin
from .mcp_rate_limiter import RateLimiter
from .circuit_breaker import CircuitBreaker, CircuitState

# Funciones JSON
_json_dumps = get_json_dumps()
_json_loads = get_json_loads()

logger = logging.getLogger(__name__)

try:
    from .mcp_metrics import MCPServerMetrics
except ImportError:
    MCPServerMetrics = None

try:
    from .mcp_config import MCPServerConfig
except ImportError:
    MCPServerConfig = None

try:
    from .auth import AuthManager
except ImportError:
    AuthManager = None

try:
    from .cache import Cache
except ImportError:
    Cache = None

try:
    from .mcp_middleware import RequestIDMiddleware, CompressionMiddleware, SecurityHeadersMiddleware
except ImportError:
    RequestIDMiddleware = None
    CompressionMiddleware = None
    SecurityHeadersMiddleware = None

try:
    from .mcp_events import EventBus, EventType, Event
except ImportError:
    EventBus = None
    EventType = None
    Event = None

try:
    from .mcp_errors import MCPException, MCPErrorCode, create_mcp_exception
except ImportError:
    MCPException = None
    MCPErrorCode = None
    create_mcp_exception = None

try:
    from .command_validator import CommandValidator
except ImportError:
    CommandValidator = None

try:
    from .mcp_connection_pool import HTTPConnectionPool
except ImportError:
    HTTPConnectionPool = None

try:
    from .mcp_request_queue import RequestQueue, RequestPriority
except ImportError:
    RequestQueue = None
    RequestPriority = None

try:
    from .mcp_token_bucket import TokenBucketRateLimiter
except ImportError:
    TokenBucketRateLimiter = None

try:
    from .mcp_request_deduplication import RequestDeduplicator
except ImportError:
    RequestDeduplicator = None

try:
    from .mcp_prometheus import PrometheusExporter
except ImportError:
    PrometheusExporter = None

try:
    from .mcp_adaptive_rate_limiter import AdaptiveRateLimiter, AdaptiveLimiterConfig
except ImportError:
    AdaptiveRateLimiter = None
    AdaptiveLimiterConfig = None


class MCPServer:
    """Servidor MCP para exponer el agente a Cursor IDE"""
    
    def __init__(
        self,
        agent,
        host: str = "localhost",
        port: int = 8025,
        enable_cors: bool = True,
        config: Optional[Any] = None
    ):
        self.agent = agent
        self.config = config or (MCPServerConfig() if MCPServerConfig else None)
        
        if self.config:
            self.host = self.config.host
            self.port = self.config.port
            enable_cors = self.config.enable_cors
        else:
            self.host = host
            self.port = port
        
        self.app = FastAPI(
            title="Cursor Agent MCP Server",
            description="Model Context Protocol server for Cursor Agent 24/7",
            version="1.0.0",
            docs_url="/mcp/v1/docs",
            redoc_url="/mcp/v1/redoc",
            openapi_url="/mcp/v1/openapi.json"
        )
        self._websocket_connections: Dict[str, WebSocket] = {}
        
        max_requests = self.config.rate_limit_max_requests if self.config else 100
        window_seconds = self.config.rate_limit_window_seconds if self.config else 60
        self._rate_limiter = RateLimiter(max_requests=max_requests, window_seconds=window_seconds)
        
        if self.config and self.config.enable_circuit_breaker:
            failure_threshold = self.config.circuit_breaker_failure_threshold
            recovery_timeout = self.config.circuit_breaker_recovery_timeout
            self._circuit_breaker = CircuitBreaker(
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout
            )
        else:
            self._circuit_breaker = None
        
        self._metrics = MCPServerMetrics() if (MCPServerMetrics and (not self.config or self.config.enable_metrics)) else None
        
        self._auth_manager = AuthManager() if (AuthManager and self.config and self.config.enable_auth) else None
        
        if self.config and self.config.enable_cache and Cache:
            cache_max_size = self.config.cache_max_size
            cache_ttl = self.config.cache_default_ttl
            self._cache = Cache(max_size=cache_max_size, default_ttl=cache_ttl)
        else:
            self._cache = None
        
        self._webhook_urls = self.config.webhook_urls if (self.config and self.config.enable_webhooks) else []
        
        self._event_bus = EventBus() if EventBus else None
        self._user_rate_limiters: Dict[str, RateLimiter] = {}
        
        max_cmd_length = self.config.max_command_length if self.config else 10000
        self._command_validator = CommandValidator(max_length=max_cmd_length) if CommandValidator else None
        
        use_token_bucket = getattr(self.config, 'use_token_bucket_rate_limiting', False) if self.config else False
        if use_token_bucket and TokenBucketRateLimiter:
            capacity = self.config.rate_limit_max_requests if self.config else 100
            refill_rate = capacity / (self.config.rate_limit_window_seconds if self.config else 60)
            self._token_bucket_limiter = TokenBucketRateLimiter(
                capacity=float(capacity),
                refill_rate=refill_rate
            )
        else:
            self._token_bucket_limiter = None
        
        enable_request_queue = getattr(self.config, 'enable_request_queue', False) if self.config else False
        if enable_request_queue and RequestQueue:
            max_queue_size = getattr(self.config, 'request_queue_max_size', 1000) if self.config else 1000
            max_workers = getattr(self.config, 'request_queue_max_workers', 10) if self.config else 10
            self._request_queue = RequestQueue(max_size=max_queue_size, max_workers=max_workers)
        else:
            self._request_queue = None
        
        if HTTPConnectionPool and self.config and self.config.enable_webhooks:
            self._connection_pool = HTTPConnectionPool(
                max_connections=getattr(self.config, 'connection_pool_max_connections', 10) if self.config else 10,
                timeout=5.0
            )
        else:
            self._connection_pool = None
        
        enable_deduplication = getattr(self.config, 'enable_request_deduplication', False) if self.config else False
        if enable_deduplication and RequestDeduplicator:
            window_seconds = getattr(self.config, 'deduplication_window_seconds', 60.0) if self.config else 60.0
            max_cache_size = getattr(self.config, 'deduplication_max_cache_size', 10000) if self.config else 10000
            self._deduplicator = RequestDeduplicator(
                window_seconds=window_seconds,
                max_cache_size=max_cache_size
            )
        else:
            self._deduplicator = None
        
        max_websocket_connections = getattr(self.config, 'max_websocket_connections', None) if self.config else None
        self._max_websocket_connections = max_websocket_connections
        
        enable_adaptive_rate_limiting = getattr(self.config, 'enable_adaptive_rate_limiting', False) if self.config else False
        if enable_adaptive_rate_limiting and AdaptiveRateLimiter:
            self._adaptive_rate_limiter = AdaptiveRateLimiter()
        else:
            self._adaptive_rate_limiter = None
        
        if PrometheusExporter and self._metrics:
            self._prometheus_exporter = PrometheusExporter(self._metrics)
        else:
            self._prometheus_exporter = None
        
        self._shutdown_event = asyncio.Event()
        
        self._setup_middleware(enable_cors)
        self._setup_exception_handlers()
        self._setup_routes()
        self._start_time = datetime.now()
        
    def _setup_middleware(self, enable_cors: bool) -> None:
        """Configurar middleware"""
        if RequestIDMiddleware:
            self.app.add_middleware(RequestIDMiddleware)
        
        if SecurityHeadersMiddleware:
            self.app.add_middleware(SecurityHeadersMiddleware)
        
        if CompressionMiddleware:
            self.app.add_middleware(CompressionMiddleware, min_size=1024)
        
        if enable_cors:
            origins = self.config.allowed_origins if self.config else ["*"]
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
    
    async def _authenticate_request(self, request: Request) -> Optional[str]:
        """Autenticar request si auth está habilitado"""
        return await authenticate_request(request, self._auth_manager)
    
    async def _check_permission(self, username: Optional[str], permission: str) -> bool:
        """Verificar permisos"""
        return await check_permission(username, permission, self._auth_manager)
    
    async def _send_webhook(self, event: str, data: Dict[str, Any]) -> None:
        """Enviar webhook si está habilitado"""
        if not self._webhook_urls:
            return
        
        payload = {
            "event": event,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        if self._connection_pool:
            for url in self._webhook_urls:
                try:
                    await self._connection_pool.post(url, json=payload)
                except Exception as e:
                    logger.debug(f"Failed to send webhook to {url}: {e}")
        else:
            try:
                import httpx
                async with httpx.AsyncClient(timeout=5.0) as client:
                    for url in self._webhook_urls:
                        try:
                            await client.post(url, json=payload)
                        except Exception as e:
                            logger.debug(f"Failed to send webhook to {url}: {e}")
            except ImportError:
                logger.warning("httpx not available for webhooks")
            except Exception as e:
                logger.debug(f"Error sending webhook: {e}")
    
    def _validate_jsonrpc(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validar formato JSON-RPC"""
        return validate_jsonrpc(message)
    
    def _get_client_id(self, request: Request) -> str:
        """Obtener ID del cliente desde request"""
        return get_client_id(request)
    
    def _get_user_rate_limiter(self, username: Optional[str]) -> Optional[RateLimiter]:
        """Obtener rate limiter para usuario específico"""
        if not username or not self.config:
            return None
        
        if username not in self._user_rate_limiters:
            max_requests = getattr(self.config, 'user_rate_limit_max_requests', 50)
            window_seconds = getattr(self.config, 'user_rate_limit_window_seconds', 60)
            self._user_rate_limiters[username] = RateLimiter(
                max_requests=max_requests,
                window_seconds=window_seconds
            )
        
        return self._user_rate_limiters[username]
    
    async def _publish_event(self, event_type: Any, data: Dict[str, Any], source: Optional[str] = None):
        """Publicar evento si el event bus está disponible"""
        if self._event_bus and Event and EventType:
            event = Event(
                event_type=event_type,
                data=data,
                source=source or "mcp_server"
            )
            await self._event_bus.publish(event)
    
    def _setup_exception_handlers(self) -> None:
        """Configurar handlers de excepciones globales"""
        if MCPException:
            @self.app.exception_handler(MCPException)
            async def mcp_exception_handler(request: Request, exc: MCPException):
                return JSONResponse(
                    status_code=exc.status_code,
                    content=exc.to_dict()
                )
        
        @self.app.exception_handler(ValidationError)
        async def validation_exception_handler(request: Request, exc: ValidationError):
            if self._metrics:
                self._metrics.record_error("validation_error")
            return JSONResponse(
                status_code=422,
                content={
                    "error": {
                        "code": "VALIDATION_ERROR",
                        "message": "Validation error",
                        "details": exc.errors()
                    }
                }
            )
        
        @self.app.exception_handler(HTTPException)
        async def http_exception_handler(request: Request, exc: HTTPException):
            if self._metrics:
                self._metrics.record_error(f"http_{exc.status_code}")
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": {
                        "code": f"HTTP_{exc.status_code}",
                        "message": exc.detail
                    }
                }
            )
    
    def _setup_routes(self) -> None:
        """Configurar rutas del servidor MCP"""
        
        @self.app.get(
            "/mcp/v1/health",
            tags=["Health"],
            summary="Health Check",
            description="Verificar el estado de salud del servidor MCP y sus componentes"
        )
        async def health_check():
            """Health check del servidor MCP"""
            try:
                agent_status = "unknown"
                if hasattr(self.agent, 'status'):
                    agent_status = self.agent.status.value
                
                uptime = (datetime.now() - self._start_time).total_seconds()
                
                health_data = {
                    "status": "healthy",
                    "version": "1.0.0",
                    "agent_status": agent_status,
                    "uptime_seconds": int(uptime),
                    "websocket_connections": len(self._websocket_connections),
                    "timestamp": datetime.now().isoformat()
                }
                
                if self._metrics:
                    stats = self._metrics.get_stats()
                    health_data["metrics"] = {
                        "total_requests": stats.get("total_requests", 0),
                        "total_errors": stats.get("total_errors", 0),
                        "average_response_time_ms": stats.get("average_response_time_ms", 0)
                    }
                
                if self._circuit_breaker:
                    health_data["circuit_breaker"] = {
                        "state": self._circuit_breaker.state.value,
                        "failure_count": self._circuit_breaker.failure_count
                    }
                
                if self._cache:
                    cache_stats = await self._cache.get_stats()
                    health_data["cache"] = {
                        "size": cache_stats.get("size", 0),
                        "max_size": cache_stats.get("max_size", 0),
                        "usage_percent": cache_stats.get("usage_percent", 0),
                        "hit_rate": cache_stats.get("hit_rate", 0)
                    }
                
                if self._event_bus:
                    health_data["events"] = {
                        "recent_events": len(await self._event_bus.get_recent_events(limit=10))
                    }
                
                if self._request_queue:
                    queue_stats = self._request_queue.get_stats()
                    health_data["request_queue"] = queue_stats
                
                if self._connection_pool:
                    health_data["connection_pool"] = {
                        "available": True,
                        "closed": self._connection_pool._closed if hasattr(self._connection_pool, '_closed') else False
                    }
                
                if self._token_bucket_limiter:
                    bucket_stats = await self._token_bucket_limiter.get_stats("default")
                    health_data["rate_limiter"] = {
                        "type": "token_bucket",
                        "stats": bucket_stats
                    }
                else:
                    health_data["rate_limiter"] = {
                        "type": "sliding_window"
                    }
                
                if self._deduplicator:
                    dedup_stats = self._deduplicator.get_stats()
                    health_data["deduplication"] = dedup_stats
                
                if self._max_websocket_connections:
                    health_data["websocket_limits"] = {
                        "max_connections": self._max_websocket_connections,
                        "current_connections": len(self._websocket_connections),
                        "available": self._max_websocket_connections - len(self._websocket_connections)
                    }
                
                if self._adaptive_rate_limiter:
                    adaptive_stats = self._adaptive_rate_limiter.get_stats()
                    health_data["adaptive_rate_limiter"] = adaptive_stats
                
                if self._prometheus_exporter:
                    health_data["prometheus_export"] = {
                        "available": True,
                        "endpoint": "/mcp/v1/metrics/prometheus"
                    }
                
                return health_data
            except Exception as e:
                logger.error(f"Error in health check: {e}")
                return JSONResponse(
                    status_code=500,
                    content={"status": "unhealthy", "error": str(e)}
                )
        
        @self.app.get(
            "/mcp/v1/resources",
            tags=["Resources"],
            summary="List Resources",
            description="Listar todos los recursos disponibles en el servidor MCP"
        )
        async def list_resources():
            """Listar recursos disponibles"""
            try:
                return {
                    "resources": [
                        {
                            "id": "cursor-agent",
                            "type": "agent",
                            "name": "Cursor Agent 24/7",
                            "description": "Agente persistente que ejecuta comandos",
                            "capabilities": ["execute_command", "get_status", "list_tasks"],
                            "version": "1.0.0"
                        }
                    ]
                }
            except Exception as e:
                logger.error(f"Error listing resources: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post(
            "/mcp/v1/command",
            tags=["Commands"],
            summary="Execute Command",
            description="Ejecutar un comando a través del protocolo MCP"
        )
        async def execute_command(request: Request, command_request: CommandRequest):
            """Ejecutar comando a través de MCP"""
            start_time = time.time()
            client_id = self._get_client_id(request)
            endpoint = "/mcp/v1/command"
            
            username = await self._authenticate_request(request)
            if self._auth_manager and not username:
                await self._publish_event(
                    EventType.AUTH_FAILED if EventType else None,
                    {"client_id": client_id, "reason": "no_auth"},
                    "execute_command"
                )
                if create_mcp_exception:
                    raise create_mcp_exception(
                        MCPErrorCode.AUTHENTICATION_REQUIRED,
                        "Authentication required",
                        401
                    )
                raise HTTPException(status_code=401, detail="Authentication required")
            
            if username and not await self._check_permission(username, "execute_command"):
                await self._publish_event(
                    EventType.AUTH_FAILED if EventType else None,
                    {"username": username, "reason": "insufficient_permissions"},
                    "execute_command"
                )
                if create_mcp_exception:
                    raise create_mcp_exception(
                        MCPErrorCode.INSUFFICIENT_PERMISSIONS,
                        "Insufficient permissions",
                        403
                    )
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            
            if username:
                user_limiter = self._get_user_rate_limiter(username)
                if user_limiter and not user_limiter.is_allowed(username):
                    await self._publish_event(
                        EventType.RATE_LIMIT_EXCEEDED if EventType else None,
                        {"username": username, "type": "user"},
                        "execute_command"
                    )
                    if self._metrics:
                        self._metrics.record_error("user_rate_limit_exceeded")
                    raise HTTPException(
                        status_code=429,
                        detail="User rate limit exceeded. Please try again later."
                    )
            
            if self._adaptive_rate_limiter:
                if not self._adaptive_rate_limiter.is_allowed(client_id):
                    await self._publish_event(
                        EventType.RATE_LIMIT_EXCEEDED if EventType else None,
                        {"client_id": client_id, "type": "adaptive"},
                        "execute_command"
                    )
                    if self._metrics:
                        self._metrics.record_error("adaptive_rate_limit_exceeded")
                    current_limit = self._adaptive_rate_limiter.get_current_limit()
                    window_seconds = self.config.rate_limit_window_seconds if self.config else 60
                    if create_mcp_exception:
                        raise create_mcp_exception(
                            MCPErrorCode.RATE_LIMIT_EXCEEDED,
                            f"Rate limit exceeded (current limit: {current_limit} requests/{window_seconds}s).",
                            429
                        )
                    raise HTTPException(
                        status_code=429,
                        detail=f"Rate limit exceeded (current limit: {current_limit} requests/{window_seconds}s)."
                    )
            elif self._token_bucket_limiter:
                is_allowed = await self._token_bucket_limiter.is_allowed(client_id)
                if not is_allowed:
                    retry_after = await self._token_bucket_limiter.get_retry_after(client_id)
                    await self._publish_event(
                        EventType.RATE_LIMIT_EXCEEDED if EventType else None,
                        {"client_id": client_id, "type": "ip", "retry_after": retry_after},
                        "execute_command"
                    )
                    if self._metrics:
                        self._metrics.record_error("rate_limit_exceeded")
                    if create_mcp_exception:
                        raise create_mcp_exception(
                            MCPErrorCode.RATE_LIMIT_EXCEEDED,
                            f"Rate limit exceeded. Please try again in {retry_after:.1f} seconds.",
                            429
                        )
                    raise HTTPException(
                        status_code=429,
                        detail=f"Rate limit exceeded. Please try again in {retry_after:.1f} seconds.",
                        headers={"Retry-After": str(int(retry_after) + 1)}
                    )
            elif not self._rate_limiter.is_allowed(client_id):
                await self._publish_event(
                    EventType.RATE_LIMIT_EXCEEDED if EventType else None,
                    {"client_id": client_id, "type": "ip"},
                    "execute_command"
                )
                if self._metrics:
                    self._metrics.record_error("rate_limit_exceeded")
                if create_mcp_exception:
                    raise create_mcp_exception(
                        MCPErrorCode.RATE_LIMIT_EXCEEDED,
                        "Rate limit exceeded. Please try again later.",
                        429
                    )
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Please try again later."
                )
            
            try:
                command = command_request.command
                metadata = command_request.metadata or {}
                
                if self._deduplicator:
                    body_key = _json_dumps({"command": command, "metadata": metadata})
                    is_duplicate, age = self._deduplicator.is_duplicate(
                        method="POST",
                        path=endpoint,
                        body=body_key,
                        headers={k: v for k, v in request.headers.items() if k.lower() not in ['x-request-id', 'x-api-key', 'authorization', 'user-agent', 'date', 'content-length']}
                    )
                    if is_duplicate:
                        logger.debug(f"Duplicate request detected (age: {age:.2f}s)")
                        return JSONResponse(
                            status_code=409,
                            content={
                                "error": {
                                    "code": "DUPLICATE_REQUEST",
                                    "message": f"Duplicate request detected (age: {age:.2f}s)",
                                    "retry_after": max(0, self._deduplicator.window_seconds - age)
                                }
                            },
                            headers={"Retry-After": str(int(self._deduplicator.window_seconds - age) + 1)}
                        )
                
                if self._command_validator:
                    validation_result = self._command_validator.validate(command)
                    if not validation_result.is_valid:
                        if create_mcp_exception:
                            if "empty" in str(validation_result.errors[0]).lower():
                                raise create_mcp_exception(
                                    MCPErrorCode.COMMAND_EMPTY,
                                    validation_result.errors[0],
                                    400
                                )
                            elif "length" in str(validation_result.errors[0]).lower():
                                raise create_mcp_exception(
                                    MCPErrorCode.COMMAND_TOO_LONG,
                                    validation_result.errors[0],
                                    400
                                )
                            else:
                                raise create_mcp_exception(
                                    MCPErrorCode.INVALID_COMMAND,
                                    "; ".join(validation_result.errors),
                                    400
                                )
                        else:
                            raise HTTPException(status_code=400, detail="; ".join(validation_result.errors))
                    
                    if validation_result.warnings:
                        for warning in validation_result.warnings:
                            logger.warning(f"Command validation warning: {warning}")
                    
                    command = validation_result.sanitized_command or command
                else:
                    if not command or not command.strip():
                        if create_mcp_exception:
                            raise create_mcp_exception(
                                MCPErrorCode.COMMAND_EMPTY,
                                "Command cannot be empty",
                                400
                            )
                        else:
                            raise HTTPException(status_code=400, detail="Command cannot be empty")
                    
                    max_length = self.config.max_command_length if self.config else 10000
                    if len(command) > max_length:
                        if create_mcp_exception:
                            raise create_mcp_exception(
                                MCPErrorCode.COMMAND_TOO_LONG,
                                f"Command too long (max {max_length} chars)",
                                400
                            )
                        else:
                            raise HTTPException(status_code=400, detail=f"Command too long (max {max_length} chars)")
                
                cache_key = f"command:{hashlib.sha256(command.encode()).hexdigest()}"
                cached_result = None
                if self._cache:
                    cached_result = await self._cache.get(cache_key)
                
                if cached_result:
                    response_time = time.time() - start_time
                    if self._metrics:
                        self._metrics.record_request(client_id, response_time, success=True, endpoint=endpoint)
                    
                    await self._send_webhook("command_cached", {
                        "command": command[:50],
                        "client_id": client_id
                    })
                    
                    return {
                        "success": True,
                        "task_id": cached_result.get("task_id"),
                        "message": "Command result from cache",
                        "cached": True,
                        "timestamp": datetime.now().isoformat(),
                        "response_time_ms": round(response_time * 1000, 2)
                    }
                
                if self._circuit_breaker and self._circuit_breaker.is_open():
                    if create_mcp_exception:
                        raise create_mcp_exception(
                            MCPErrorCode.CIRCUIT_BREAKER_OPEN,
                            "Service temporarily unavailable due to high error rate",
                            503
                        )
                    raise HTTPException(status_code=503, detail="Service temporarily unavailable")
                
                try:
                    task_id = await self.agent.add_task(command)
                    if self._circuit_breaker:
                        await self._circuit_breaker.record_success()
                    
                    if self._cache:
                        await self._cache.set(cache_key, {"task_id": task_id}, ttl=300)
                    
                    await self._publish_event(
                        EventType.COMMAND_EXECUTED if EventType else None,
                        {
                            "task_id": task_id,
                            "command": command[:50],
                            "client_id": client_id,
                            "username": username
                        },
                        "execute_command"
                    )
                except Exception as e:
                    if self._circuit_breaker:
                        await self._circuit_breaker.record_failure()
                        if self._circuit_breaker.is_open():
                            await self._publish_event(
                                EventType.CIRCUIT_BREAKER_OPENED if EventType else None,
                                {"failure_count": self._circuit_breaker.failure_count},
                                "execute_command"
                            )
                    
                    await self._publish_event(
                        EventType.COMMAND_FAILED if EventType else None,
                        {
                            "command": command[:50],
                            "error": str(e),
                            "client_id": client_id,
                            "username": username
                        },
                        "execute_command"
                    )
                    raise
                
                response_time = time.time() - start_time
                
                if self._metrics:
                    self._metrics.record_request(client_id, response_time, success=True, endpoint=endpoint)
                
                if self._adaptive_rate_limiter:
                    self._adaptive_rate_limiter.record_response_time(response_time)
                    self._adaptive_rate_limiter.record_error(False)
                
                await self._send_webhook("command_executed", {
                    "task_id": task_id,
                    "command": command[:50],
                    "client_id": client_id,
                    "username": username
                })
                
                logger.info(f"✅ Command queued via MCP: {task_id[:8]}... (client: {client_id})")
                
                return {
                    "success": True,
                    "task_id": task_id,
                    "message": "Command queued for execution",
                    "timestamp": datetime.now().isoformat(),
                    "response_time_ms": round(response_time * 1000, 2)
                }
            except HTTPException as e:
                response_time = time.time() - start_time
                if self._metrics:
                    self._metrics.record_request(client_id, response_time, success=False, endpoint=endpoint)
                    if e.status_code == 503:
                        self._metrics.record_error("circuit_breaker_open")
                    else:
                        self._metrics.record_error("http_exception")
                
                if self._adaptive_rate_limiter:
                    self._adaptive_rate_limiter.record_response_time(response_time)
                    self._adaptive_rate_limiter.record_error(True)
                raise
            except ValueError as e:
                logger.warning(f"Validation error in command: {e}")
                response_time = time.time() - start_time
                if self._metrics:
                    self._metrics.record_request(client_id, response_time, success=False, endpoint=endpoint)
                    self._metrics.record_error("validation_error")
                
                if self._adaptive_rate_limiter:
                    self._adaptive_rate_limiter.record_response_time(response_time)
                    self._adaptive_rate_limiter.record_error(True)
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"Error executing command via MCP: {e}", exc_info=True)
                response_time = time.time() - start_time
                if self._metrics:
                    self._metrics.record_request(client_id, response_time, success=False, endpoint=endpoint)
                    self._metrics.record_error("internal_error")
                
                if self._adaptive_rate_limiter:
                    self._adaptive_rate_limiter.record_response_time(response_time)
                    self._adaptive_rate_limiter.record_error(True)
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        
        @self.app.get(
            "/mcp/v1/status",
            tags=["Status"],
            summary="Get Agent Status",
            description="Obtener el estado actual del agente"
        )
        async def get_agent_status(request: Request):
            """Obtener estado del agente"""
            username = await self._authenticate_request(request)
            if self._auth_manager and not username:
                raise HTTPException(status_code=401, detail="Authentication required")
            
            if username and not await self._check_permission(username, "read_status"):
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            
            try:
                status = await self.agent.get_status()
                return status
            except Exception as e:
                logger.error(f"Error getting agent status: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get(
            "/mcp/v1/metrics",
            tags=["Metrics"],
            summary="Get Metrics",
            description="Obtener métricas detalladas del servidor MCP (requiere permisos de admin)"
        )
        async def get_metrics(request: Request):
            """Obtener métricas del servidor MCP"""
            username = await self._authenticate_request(request)
            if self._auth_manager and not username:
                raise HTTPException(status_code=401, detail="Authentication required")
            
            if username and not await self._check_permission(username, "read_metrics"):
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            
            try:
                if self._metrics:
                    self._metrics.set_websocket_connections(len(self._websocket_connections))
                    stats = self._metrics.get_stats()
                    
                    if self._adaptive_rate_limiter:
                        stats["adaptive_rate_limiter"] = self._adaptive_rate_limiter.get_stats()
                    
                    return stats
                return {"message": "Metrics not available"}
            except Exception as e:
                logger.error(f"Error getting metrics: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get(
            "/mcp/v1/metrics/prometheus",
            tags=["Metrics"],
            summary="Prometheus Metrics",
            description="Exportar métricas en formato Prometheus"
        )
        async def get_prometheus_metrics():
            """Exportar métricas en formato Prometheus"""
            try:
                if self._prometheus_exporter:
                    from fastapi.responses import Response
                    prometheus_output = self._prometheus_exporter.export()
                    return Response(
                        content=prometheus_output,
                        media_type="text/plain; version=0.0.4"
                    )
                return Response(
                    content="# No metrics available\n",
                    media_type="text/plain; version=0.0.4"
                )
            except Exception as e:
                logger.error(f"Error exporting Prometheus metrics: {e}", exc_info=True)
                return Response(
                    content=f"# Error: {str(e)}\n",
                    media_type="text/plain; version=0.0.4",
                    status_code=500
                )
        
        @self.app.get(
            "/mcp/v1/config",
            tags=["Config"],
            summary="Get Configuration",
            description="Obtener la configuración actual del servidor MCP (requiere permisos de admin)"
        )
        async def get_config(request: Request):
            """Obtener configuración del servidor MCP"""
            username = await self._authenticate_request(request)
            if self._auth_manager and not username:
                raise HTTPException(status_code=401, detail="Authentication required")
            
            if username:
                from .auth import Role
                user = self._auth_manager.users.get(username)
                if not user or user.role != Role.ADMIN:
                    raise HTTPException(status_code=403, detail="Admin access required")
            
            try:
                if self.config:
                    config_dict = self.config.to_dict()
                    config_dict.pop("webhook_urls", None)
                    return config_dict
                return {"message": "Config not available"}
            except Exception as e:
                logger.error(f"Error getting config: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post(
            "/mcp/v1/auth/login",
            tags=["Authentication"],
            summary="Login",
            description="Autenticar y obtener API key para usar en requests posteriores"
        )
        async def login(login_request: LoginRequest):
            """Autenticar y obtener API key"""
            if not self._auth_manager:
                raise HTTPException(status_code=403, detail="Authentication not enabled")
            
            try:
                session_id = self._auth_manager.authenticate(login_request.username, login_request.password)
                if not session_id:
                    raise HTTPException(status_code=401, detail="Invalid credentials")
                
                user = self._auth_manager.users.get(login_request.username)
                return {
                    "success": True,
                    "session_id": session_id,
                    "api_key": user.api_key,
                    "role": user.role.value,
                    "expires_at": (datetime.now() + timedelta(hours=24)).isoformat()
                }
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error in login: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post(
            "/mcp/v1/commands/batch",
            tags=["Commands"],
            summary="Execute Batch Commands",
            description="Ejecutar múltiples comandos en una sola request (puede ejecutarse en paralelo o secuencialmente)"
        )
        async def execute_batch_commands(request: Request, batch_request: BatchCommandRequest):
            """Ejecutar múltiples comandos en lote"""
            if not (self.config and self.config.enable_batch_operations):
                raise HTTPException(status_code=403, detail="Batch operations not enabled")
            
            start_time = time.time()
            client_id = self._get_client_id(request)
            
            username = await self._authenticate_request(request)
            if self._auth_manager and not username:
                raise HTTPException(status_code=401, detail="Authentication required")
            
            if username and not await self._check_permission(username, "execute_command"):
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            
            try:
                commands = batch_request.commands
                parallel = batch_request.parallel
                
                if parallel:
                    task_ids = await asyncio.gather(*[self.agent.add_task(cmd) for cmd in commands])
                else:
                    task_ids = []
                    for cmd in commands:
                        task_id = await self.agent.add_task(cmd)
                        task_ids.append(task_id)
                
                response_time = time.time() - start_time
                
                if self._metrics:
                    self._metrics.record_request(client_id, response_time, success=True, endpoint="/mcp/v1/commands/batch")
                
                await self._send_webhook("batch_commands_executed", {
                    "count": len(commands),
                    "task_ids": task_ids,
                    "client_id": client_id,
                    "username": username
                })
                
                return {
                    "success": True,
                    "task_ids": task_ids,
                    "count": len(commands),
                    "parallel": parallel,
                    "timestamp": datetime.now().isoformat(),
                    "response_time_ms": round(response_time * 1000, 2)
                }
            except Exception as e:
                logger.error(f"Error executing batch commands: {e}", exc_info=True)
                if self._metrics:
                    self._metrics.record_request(client_id, time.time() - start_time, success=False, endpoint="/mcp/v1/commands/batch")
                    self._metrics.record_error("batch_error")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get(
            "/mcp/v1/tasks/{task_id}/stream",
            tags=["Tasks"],
            summary="Stream Task Result",
            description="Streaming de resultados de tarea usando Server-Sent Events (SSE)"
        )
        async def stream_task_result(task_id: str):
            """Streaming de resultados de tarea"""
            async def generate():
                try:
                    max_wait = 300
                    start_time = time.time()
                    
                    while time.time() - start_time < max_wait:
                        tasks = await self.agent.get_tasks(limit=1000)
                        task = next((t for t in tasks if t.get("id") == task_id), None)
                        
                        if task:
                            status = task.get("status")
                            
                            if status == "completed":
                                result = task.get("result")
                                if self._cache:
                                    await self._cache.set(f"task_result:{task_id}", {"result": result}, ttl=3600)
                                yield f"data: {_json_dumps({'status': 'completed', 'result': result})}\n\n"
                                break
                            elif status == "failed":
                                yield f"data: {_json_dumps({'status': 'failed', 'error': task.get('error')})}\n\n"
                                break
                            else:
                                yield f"data: {_json_dumps({'status': status, 'task_id': task_id})}\n\n"
                        
                        await asyncio.sleep(1)
                    
                    yield f"data: {_json_dumps({'status': 'timeout', 'message': 'Task result not available'})}\n\n"
                except Exception as e:
                    yield f"data: {_json_dumps({'status': 'error', 'error': str(e)})}\n\n"
            
            return StreamingResponse(generate(), media_type="text/event-stream")
        
        @self.app.get(
            "/mcp/v1/tasks/{task_id}/result",
            tags=["Tasks"],
            summary="Get Task Result",
            description="Obtener resultado de una tarea específica (con soporte de caché)"
        )
        async def get_task_result(task_id: str):
            """Obtener resultado de tarea (con caché)"""
            try:
                if self._cache:
                    cached = await self._cache.get(f"task_result:{task_id}")
                    if cached:
                        return {"cached": True, "result": cached.get("result")}
                
                tasks = await self.agent.get_tasks(limit=1000)
                task = next((t for t in tasks if t.get("id") == task_id), None)
                
                if not task:
                    raise HTTPException(status_code=404, detail="Task not found")
                
                return {
                    "task_id": task_id,
                    "status": task.get("status"),
                    "result": task.get("result"),
                    "error": task.get("error"),
                    "timestamp": task.get("timestamp")
                }
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error getting task result: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get(
            "/mcp/v1/events",
            tags=["Events"],
            summary="Get Recent Events",
            description="Obtener eventos recientes del sistema (requiere permisos de admin)"
        )
        async def get_events(request: Request, event_type: Optional[str] = None, limit: int = 100):
            """Obtener eventos recientes"""
            username = await self._authenticate_request(request)
            if self._auth_manager and not username:
                raise HTTPException(status_code=401, detail="Authentication required")
            
            if username:
                from .auth import Role
                user = self._auth_manager.users.get(username)
                if not user or user.role != Role.ADMIN:
                    raise HTTPException(status_code=403, detail="Admin access required")
            
            try:
                if not self._event_bus:
                    return {"message": "Event bus not available", "events": []}
                
                event_type_enum = None
                if event_type and EventType:
                    try:
                        event_type_enum = EventType(event_type)
                    except ValueError:
                        raise HTTPException(status_code=400, detail=f"Invalid event type: {event_type}")
                
                events = await self._event_bus.get_recent_events(
                    event_type=event_type_enum,
                    limit=min(limit, 1000)
                )
                
                return {
                    "count": len(events),
                    "events": [
                        {
                            "event_type": e.event_type.value if hasattr(e.event_type, 'value') else str(e.event_type),
                            "data": e.data,
                            "timestamp": e.timestamp.isoformat(),
                            "source": e.source
                        }
                        for e in events
                    ]
                }
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error getting events: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete(
            "/mcp/v1/cache",
            tags=["Cache"],
            summary="Clear Cache",
            description="Limpiar todo el caché del servidor (requiere permisos de admin)"
        )
        async def clear_cache(request: Request):
            """Limpiar caché"""
            username = await self._authenticate_request(request)
            if self._auth_manager and not username:
                raise HTTPException(status_code=401, detail="Authentication required")
            
            if username:
                from .auth import Role
                user = self._auth_manager.users.get(username)
                if not user or user.role != Role.ADMIN:
                    raise HTTPException(status_code=403, detail="Admin access required")
            
            try:
                if not self._cache:
                    return {"message": "Cache not available", "cleared": False}
                
                await self._cache.clear()
                return {
                    "success": True,
                    "message": "Cache cleared successfully",
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Error clearing cache: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get(
            "/mcp/v1/cache/stats",
            tags=["Cache"],
            summary="Get Cache Statistics",
            description="Obtener estadísticas del caché (requiere permisos de admin)"
        )
        async def get_cache_stats(request: Request):
            """Obtener estadísticas del caché"""
            username = await self._authenticate_request(request)
            if self._auth_manager and not username:
                raise HTTPException(status_code=401, detail="Authentication required")
            
            if username:
                from .auth import Role
                user = self._auth_manager.users.get(username)
                if not user or user.role != Role.ADMIN:
                    raise HTTPException(status_code=403, detail="Admin access required")
            
            try:
                if not self._cache:
                    return {"message": "Cache not available"}
                
                stats = await self._cache.get_stats()
                return stats
            except Exception as e:
                logger.error(f"Error getting cache stats: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.websocket("/mcp/v1/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint para comunicación en tiempo real"""
            if self._max_websocket_connections and len(self._websocket_connections) >= self._max_websocket_connections:
                await websocket.close(code=1008, reason="Maximum connections reached")
                logger.warning(f"WebSocket connection rejected: max connections ({self._max_websocket_connections}) reached")
                return
            
            connection_id = f"ws_{datetime.now().timestamp()}_{len(self._websocket_connections)}"
            await websocket.accept()
            self._websocket_connections[connection_id] = websocket
            if self._metrics:
                self._metrics.set_websocket_connections(len(self._websocket_connections))
            
            await self._publish_event(
                EventType.WEBSOCKET_CONNECTED if EventType else None,
                {"connection_id": connection_id},
                "websocket"
            )
            
            logger.info(f"📡 New MCP WebSocket connection: {connection_id} (total: {len(self._websocket_connections)})")
            
            heartbeat_task = None
            if hasattr(self.config, 'websocket_heartbeat_interval') and self.config.websocket_heartbeat_interval:
                heartbeat_task = asyncio.create_task(self._websocket_heartbeat(websocket, connection_id))
            
            try:
                while True:
                    try:
                        data = await asyncio.wait_for(websocket.receive_text(), timeout=300.0)
                    except asyncio.TimeoutError:
                        await websocket.send_json({
                            "jsonrpc": "2.0",
                            "error": {
                                "code": -32000,
                                "message": "Connection timeout"
                            }
                        })
                        break
                    
                    try:
                        message = _json_loads(data)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid JSON in WebSocket message: {e}")
                        await websocket.send_json({
                            "jsonrpc": "2.0",
                            "error": {
                                "code": -32700,
                                "message": "Parse error"
                            }
                        })
                        continue
                    
                    validated = self._validate_jsonrpc(message)
                    if not validated:
                        await websocket.send_json({
                            "jsonrpc": "2.0",
                            "id": message.get("id"),
                            "error": {
                                "code": -32600,
                                "message": "Invalid Request"
                            }
                        })
                        continue
                    
                    method = validated.get("method")
                    message_id = validated.get("id")
                    
                    if method == "mcp/command":
                        try:
                            command = validated.get("params", {}).get("command", "")
                            if not command:
                                await websocket.send_json({
                                    "jsonrpc": "2.0",
                                    "id": message_id,
                                    "error": {
                                        "code": -32602,
                                        "message": "Invalid params: command is required"
                                    }
                                })
                                continue
                            
                            task_id = await self.agent.add_task(command)
                            await websocket.send_json({
                                "jsonrpc": "2.0",
                                "id": message_id,
                                "result": {
                                    "task_id": task_id,
                                    "status": "queued",
                                    "timestamp": datetime.now().isoformat()
                                }
                            })
                        except Exception as e:
                            logger.error(f"Error executing command via WebSocket: {e}")
                            await websocket.send_json({
                                "jsonrpc": "2.0",
                                "id": message_id,
                                "error": {
                                    "code": -32000,
                                    "message": f"Execution error: {str(e)}"
                                }
                            })
                    
                    elif method == "mcp/ping":
                        await websocket.send_json({
                            "jsonrpc": "2.0",
                            "id": message_id,
                            "result": {
                                "pong": True,
                                "timestamp": datetime.now().isoformat()
                            }
                        })
                    
                    else:
                        await websocket.send_json({
                            "jsonrpc": "2.0",
                            "id": message_id,
                            "error": {
                                "code": -32601,
                                "message": f"Method not found: {method}"
                            }
                        })
                    
            except WebSocketDisconnect:
                logger.info(f"📡 MCP WebSocket disconnected: {connection_id}")
            except Exception as e:
                logger.error(f"Error in MCP WebSocket {connection_id}: {e}", exc_info=True)
            finally:
                if heartbeat_task:
                    heartbeat_task.cancel()
                
                if connection_id in self._websocket_connections:
                    del self._websocket_connections[connection_id]
                    if self._metrics:
                        self._metrics.set_websocket_connections(len(self._websocket_connections))
                    
                    await self._publish_event(
                        EventType.WEBSOCKET_DISCONNECTED if EventType else None,
                        {"connection_id": connection_id},
                        "websocket"
                    )
                    
                    logger.info(f"📡 WebSocket connection removed: {connection_id} (total: {len(self._websocket_connections)})")
    
    async def _websocket_heartbeat(self, websocket: WebSocket, connection_id: str):
        """Enviar heartbeat periódico a WebSocket"""
        try:
            interval = getattr(self.config, 'websocket_heartbeat_interval', 30) if self.config else 30
            while True:
                await asyncio.sleep(interval)
                try:
                    await websocket.send_json({
                        "jsonrpc": "2.0",
                        "method": "ping",
                        "params": {"timestamp": datetime.now().isoformat()}
                    })
                except Exception:
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"Heartbeat error for {connection_id}: {e}")
    
    async def broadcast_message(self, message: Dict[str, Any]) -> int:
        """Enviar mensaje a todas las conexiones WebSocket"""
        disconnected = []
        sent_count = 0
        
        for connection_id, ws in list(self._websocket_connections.items()):
            try:
                await ws.send_json(message)
                sent_count += 1
            except Exception as e:
                logger.debug(f"Error sending to WebSocket {connection_id}: {e}")
                disconnected.append(connection_id)
        
        for connection_id in disconnected:
            if connection_id in self._websocket_connections:
                del self._websocket_connections[connection_id]
        
        return sent_count
    
    async def run(self) -> None:
        """Ejecutar servidor MCP"""
        try:
            if self._request_queue:
                await self._request_queue.start()
            
            if self._adaptive_rate_limiter:
                await self._adaptive_rate_limiter.start()
            
            config = uvicorn.Config(
                self.app,
                host=self.host,
                port=self.port,
                log_level="info",
                access_log=True
            )
            server = uvicorn.Server(config)
            logger.info(f"🚀 Starting MCP server on http://{self.host}:{self.port}")
            
            try:
                await server.serve()
            finally:
                await self.shutdown()
        except Exception as e:
            logger.error(f"❌ Failed to start MCP server: {e}", exc_info=True)
            await self.shutdown()
            raise
    
    async def shutdown(self) -> None:
        """Cerrar el servidor de forma ordenada"""
        logger.info("🛑 Shutting down MCP server...")
        self._shutdown_event.set()
        
        if self._adaptive_rate_limiter:
            await self._adaptive_rate_limiter.stop()
        
        if self._request_queue:
            await self._request_queue.stop()
        
        if self._connection_pool:
            await self._connection_pool.close()
        
        for connection_id, ws in list(self._websocket_connections.items()):
            try:
                await ws.close()
            except Exception:
                pass
        
        self._websocket_connections.clear()
        logger.info("✅ MCP server shutdown complete")
