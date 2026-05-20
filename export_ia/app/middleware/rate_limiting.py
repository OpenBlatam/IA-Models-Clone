"""
Rate Limiting Middleware - Middleware de limitación de tasa avanzado
"""

import time
import asyncio
import logging
from typing import Dict, Any, Optional, Tuple
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from collections import defaultdict, deque
from datetime import datetime, timedelta
import ipaddress
import hashlib

logger = logging.getLogger(__name__)


class RateLimitConfig:
    """Configuración de límite de tasa."""
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        requests_per_day: int = 10000,
        burst_limit: int = 10,
        window_size: int = 60
    ):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.requests_per_day = requests_per_day
        self.burst_limit = burst_limit
        self.window_size = window_size


class RateLimitTracker:
    """Rastreador de límites de tasa."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.requests: Dict[str, deque] = defaultdict(lambda: deque())
        self.blocked_ips: Dict[str, datetime] = {}
        self.suspicious_ips: Dict[str, List[datetime]] = defaultdict(list)
        self._lock = asyncio.Lock()
    
    async def is_allowed(self, identifier: str, endpoint: Optional[str] = None) -> Tuple[bool, Dict[str, Any]]:
        """Verificar si la request está permitida."""
        async with self._lock:
            now = datetime.now()
            
            # Verificar si la IP está bloqueada
            if identifier in self.blocked_ips:
                block_until = self.blocked_ips[identifier]
                if now < block_until:
                    return False, {
                        "reason": "ip_blocked",
                        "blocked_until": block_until.isoformat(),
                        "retry_after": int((block_until - now).total_seconds())
                    }
                else:
                    # Desbloquear IP
                    del self.blocked_ips[identifier]
            
            # Limpiar requests antiguas
            cutoff_time = now - timedelta(seconds=self.config.window_size)
            while self.requests[identifier] and self.requests[identifier][0] < cutoff_time:
                self.requests[identifier].popleft()
            
            # Verificar límites
            current_requests = len(self.requests[identifier])
            
            # Límite por minuto
            if current_requests >= self.config.requests_per_minute:
                # Marcar como sospechoso
                self.suspicious_ips[identifier].append(now)
                
                # Bloquear si hay muchos intentos sospechosos
                recent_suspicious = [
                    t for t in self.suspicious_ips[identifier]
                    if now - t < timedelta(minutes=10)
                ]
                
                if len(recent_suspicious) >= 5:
                    # Bloquear por 1 hora
                    self.blocked_ips[identifier] = now + timedelta(hours=1)
                    logger.warning(f"IP {identifier} bloqueada por actividad sospechosa")
                
                return False, {
                    "reason": "rate_limit_exceeded",
                    "limit": self.config.requests_per_minute,
                    "current": current_requests,
                    "retry_after": 60,
                    "window": "minute"
                }
            
            # Límite por hora
            hour_cutoff = now - timedelta(hours=1)
            hour_requests = [
                req_time for req_time in self.requests[identifier]
                if req_time > hour_cutoff
            ]
            
            if len(hour_requests) >= self.config.requests_per_hour:
                return False, {
                    "reason": "rate_limit_exceeded",
                    "limit": self.config.requests_per_hour,
                    "current": len(hour_requests),
                    "retry_after": 3600,
                    "window": "hour"
                }
            
            # Límite por día
            day_cutoff = now - timedelta(days=1)
            day_requests = [
                req_time for req_time in self.requests[identifier]
                if req_time > day_cutoff
            ]
            
            if len(day_requests) >= self.config.requests_per_day:
                return False, {
                    "reason": "rate_limit_exceeded",
                    "limit": self.config.requests_per_day,
                    "current": len(day_requests),
                    "retry_after": 86400,
                    "window": "day"
                }
            
            # Verificar límite de ráfaga
            recent_requests = [
                req_time for req_time in self.requests[identifier]
                if now - req_time < timedelta(seconds=10)
            ]
            
            if len(recent_requests) >= self.config.burst_limit:
                return False, {
                    "reason": "burst_limit_exceeded",
                    "limit": self.config.burst_limit,
                    "current": len(recent_requests),
                    "retry_after": 10,
                    "window": "burst"
                }
            
            # Registrar request
            self.requests[identifier].append(now)
            
            return True, {
                "remaining": self.config.requests_per_minute - current_requests - 1,
                "reset_time": (now + timedelta(seconds=self.config.window_size)).isoformat(),
                "limit": self.config.requests_per_minute
            }
    
    async def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de límites de tasa."""
        async with self._lock:
            now = datetime.now()
            
            # Limpiar datos antiguos
            cutoff_time = now - timedelta(hours=24)
            for identifier in list(self.suspicious_ips.keys()):
                self.suspicious_ips[identifier] = [
                    t for t in self.suspicious_ips[identifier]
                    if t > cutoff_time
                ]
                if not self.suspicious_ips[identifier]:
                    del self.suspicious_ips[identifier]
            
            # Limpiar IPs bloqueadas expiradas
            for identifier in list(self.blocked_ips.keys()):
                if now >= self.blocked_ips[identifier]:
                    del self.blocked_ips[identifier]
            
            return {
                "active_identifiers": len(self.requests),
                "blocked_ips": len(self.blocked_ips),
                "suspicious_ips": len(self.suspicious_ips),
                "total_requests": sum(len(requests) for requests in self.requests.values()),
                "config": {
                    "requests_per_minute": self.config.requests_per_minute,
                    "requests_per_hour": self.config.requests_per_hour,
                    "requests_per_day": self.config.requests_per_day,
                    "burst_limit": self.config.burst_limit,
                    "window_size": self.config.window_size
                },
                "blocked_ips_list": list(self.blocked_ips.keys()),
                "suspicious_ips_list": list(self.suspicious_ips.keys())
            }
    
    async def unblock_ip(self, identifier: str) -> bool:
        """Desbloquear una IP."""
        async with self._lock:
            if identifier in self.blocked_ips:
                del self.blocked_ips[identifier]
                logger.info(f"IP {identifier} desbloqueada manualmente")
                return True
            return False
    
    async def block_ip(self, identifier: str, duration_hours: int = 1) -> bool:
        """Bloquear una IP manualmente."""
        async with self._lock:
            self.blocked_ips[identifier] = datetime.now() + timedelta(hours=duration_hours)
            logger.warning(f"IP {identifier} bloqueada manualmente por {duration_hours} horas")
            return True


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """
    Middleware de limitación de tasa avanzado.
    """
    
    def __init__(
        self,
        app: ASGIApp,
        default_config: Optional[RateLimitConfig] = None,
        endpoint_configs: Optional[Dict[str, RateLimitConfig]] = None,
        identifier_func: Optional[callable] = None
    ):
        super().__init__(app)
        self.default_config = default_config or RateLimitConfig()
        self.endpoint_configs = endpoint_configs or {}
        self.identifier_func = identifier_func or self._default_identifier
        self.trackers: Dict[str, RateLimitTracker] = {}
        self._cleanup_task = None
        self._start_cleanup_task()
    
    def _default_identifier(self, request: Request) -> str:
        """Función por defecto para identificar requests."""
        # Usar IP del cliente
        client_ip = request.client.host if request.client else "unknown"
        
        # Agregar información adicional si está disponible
        user_agent = request.headers.get("user-agent", "")
        if user_agent:
            # Crear hash del user agent para evitar identificadores muy largos
            user_agent_hash = hashlib.md5(user_agent.encode()).hexdigest()[:8]
            return f"{client_ip}:{user_agent_hash}"
        
        return client_ip
    
    def _start_cleanup_task(self):
        """Iniciar tarea de limpieza."""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(300)  # Cada 5 minutos
                    await self._cleanup_old_data()
                except Exception as e:
                    logger.error(f"Error en limpieza de rate limiting: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    async def _cleanup_old_data(self):
        """Limpiar datos antiguos."""
        try:
            for tracker in self.trackers.values():
                # La limpieza se hace automáticamente en get_stats
                await tracker.get_stats()
        except Exception as e:
            logger.error(f"Error en limpieza de datos: {e}")
    
    def _get_config_for_endpoint(self, endpoint: str) -> RateLimitConfig:
        """Obtener configuración para un endpoint específico."""
        return self.endpoint_configs.get(endpoint, self.default_config)
    
    def _get_tracker_for_config(self, config: RateLimitConfig) -> RateLimitTracker:
        """Obtener tracker para una configuración específica."""
        config_key = f"{config.requests_per_minute}_{config.requests_per_hour}_{config.requests_per_day}"
        
        if config_key not in self.trackers:
            self.trackers[config_key] = RateLimitTracker(config)
        
        return self.trackers[config_key]
    
    async def dispatch(self, request: Request, call_next):
        """Procesar request con limitación de tasa."""
        try:
            # Obtener identificador
            identifier = self.identifier_func(request)
            
            # Obtener endpoint
            endpoint = f"{request.method} {request.url.path}"
            
            # Obtener configuración
            config = self._get_config_for_endpoint(endpoint)
            tracker = self._get_tracker_for_config(config)
            
            # Verificar límite de tasa
            allowed, info = await tracker.is_allowed(identifier, endpoint)
            
            if not allowed:
                # Agregar headers informativos
                response = JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded",
                        "message": f"Too many requests. {info.get('reason', 'Unknown reason')}",
                        "retry_after": info.get("retry_after", 60),
                        "limit": info.get("limit"),
                        "current": info.get("current"),
                        "window": info.get("window"),
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
                # Agregar headers estándar
                response.headers["Retry-After"] = str(info.get("retry_after", 60))
                response.headers["X-RateLimit-Limit"] = str(info.get("limit", 0))
                response.headers["X-RateLimit-Remaining"] = str(info.get("remaining", 0))
                response.headers["X-RateLimit-Reset"] = info.get("reset_time", "")
                
                return response
            
            # Procesar request
            response = await call_next(request)
            
            # Agregar headers informativos
            response.headers["X-RateLimit-Limit"] = str(info.get("limit", 0))
            response.headers["X-RateLimit-Remaining"] = str(info.get("remaining", 0))
            response.headers["X-RateLimit-Reset"] = info.get("reset_time", "")
            
            return response
            
        except Exception as e:
            logger.error(f"Error en rate limiting middleware: {e}")
            # En caso de error, permitir la request
            return await call_next(request)
    
    async def get_rate_limit_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de límites de tasa."""
        try:
            all_stats = {}
            for config_key, tracker in self.trackers.items():
                all_stats[config_key] = await tracker.get_stats()
            
            return {
                "trackers": all_stats,
                "total_trackers": len(self.trackers),
                "default_config": {
                    "requests_per_minute": self.default_config.requests_per_minute,
                    "requests_per_hour": self.default_config.requests_per_hour,
                    "requests_per_day": self.default_config.requests_per_day,
                    "burst_limit": self.default_config.burst_limit,
                    "window_size": self.default_config.window_size
                },
                "endpoint_configs": {
                    endpoint: {
                        "requests_per_minute": config.requests_per_minute,
                        "requests_per_hour": config.requests_per_hour,
                        "requests_per_day": config.requests_per_day,
                        "burst_limit": config.burst_limit,
                        "window_size": config.window_size
                    }
                    for endpoint, config in self.endpoint_configs.items()
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error al obtener estadísticas de rate limiting: {e}")
            return {"error": str(e)}
    
    async def unblock_identifier(self, identifier: str) -> bool:
        """Desbloquear un identificador."""
        try:
            unblocked = False
            for tracker in self.trackers.values():
                if await tracker.unblock_ip(identifier):
                    unblocked = True
            
            return unblocked
            
        except Exception as e:
            logger.error(f"Error al desbloquear identificador: {e}")
            return False
    
    async def block_identifier(self, identifier: str, duration_hours: int = 1) -> bool:
        """Bloquear un identificador."""
        try:
            blocked = False
            for tracker in self.trackers.values():
                if await tracker.block_ip(identifier, duration_hours):
                    blocked = True
            
            return blocked
            
        except Exception as e:
            logger.error(f"Error al bloquear identificador: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """Verificar salud del middleware de rate limiting."""
        try:
            return {
                "status": "healthy",
                "total_trackers": len(self.trackers),
                "default_config": {
                    "requests_per_minute": self.default_config.requests_per_minute,
                    "requests_per_hour": self.default_config.requests_per_hour,
                    "requests_per_day": self.default_config.requests_per_day,
                    "burst_limit": self.default_config.burst_limit,
                    "window_size": self.default_config.window_size
                },
                "endpoint_configs_count": len(self.endpoint_configs),
                "cleanup_task_running": self._cleanup_task is not None and not self._cleanup_task.done(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en health check de rate limiting: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }




