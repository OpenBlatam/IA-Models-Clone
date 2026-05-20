"""
MCP Rate Limiter - Rate limiting para recursos MCP
===================================================

Sistema de rate limiting en memoria para controlar el número de solicitudes
por período de tiempo. Para producción, considerar usar Redis o similar.
"""

import time
import logging
from typing import Dict, Optional, List
from collections import defaultdict
from datetime import datetime, timedelta

from .exceptions import MCPRateLimitError

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Rate limiter simple en memoria.
    
    Controla el número de solicitudes permitidas por minuto y/o hora
    para diferentes claves (ej: user_id, resource_id, etc.).
    
    Nota: Para producción, usar Redis o similar para rate limiting distribuido.
    """
    
    def __init__(self) -> None:
        """
        Inicializar rate limiter.
        
        Crea estructuras de datos para rastrear solicitudes y límites.
        """
        self._requests: Dict[str, List[float]] = defaultdict(list)
        self._limits: Dict[str, Dict[str, int]] = {}
        logger.debug("RateLimiter initialized")
    
    def set_limit(
        self,
        key: str,
        requests_per_minute: int,
        requests_per_hour: Optional[int] = None,
    ) -> None:
        """
        Establece límite para una clave.
        
        Args:
            key: Clave única (ej: "user_id:resource_id" o "user_id").
            requests_per_minute: Requests permitidos por minuto.
            requests_per_hour: Requests permitidos por hora (opcional).
        
        Raises:
            ValueError: Si los parámetros son inválidos.
        """
        if not key or not key.strip():
            raise ValueError("key cannot be empty")
        
        if requests_per_minute < 1:
            raise ValueError("requests_per_minute must be at least 1")
        
        if requests_per_hour is not None and requests_per_hour < 1:
            raise ValueError("requests_per_hour must be at least 1")
        
        if requests_per_hour is not None and requests_per_hour < requests_per_minute:
            raise ValueError("requests_per_hour must be >= requests_per_minute")
        
        self._limits[key.strip()] = {
            "per_minute": requests_per_minute,
            "per_hour": requests_per_hour,
        }
        logger.debug(
            f"Rate limit set for key '{key}': "
            f"{requests_per_minute}/min, {requests_per_hour or 'unlimited'}/hour"
        )
    
    def check_limit(self, key: str) -> bool:
        """
        Verifica si se puede hacer un request.
        
        Args:
            key: Clave única para verificar límite.
        
        Returns:
            True si está dentro del límite.
        
        Raises:
            MCPRateLimitError: Si se excedió el límite.
            ValueError: Si key está vacío.
        """
        if not key or not key.strip():
            raise ValueError("key cannot be empty")
        
        key_clean = key.strip()
        now = time.time()
        limit_config = self._limits.get(key_clean)
        
        if not limit_config:
            # Sin límite configurado, permitir
            return True
        
        # Limpiar requests antiguos (mantener últimos 60 minutos)
        self._requests[key_clean] = [
            req_time for req_time in self._requests[key_clean]
            if now - req_time < 3600
        ]
        
        # Verificar límite por minuto
        minute_ago = now - 60
        recent_requests = [
            req_time for req_time in self._requests[key_clean]
            if req_time > minute_ago
        ]
        
        per_minute_limit = limit_config["per_minute"]
        if len(recent_requests) >= per_minute_limit:
            raise MCPRateLimitError(
                limit=per_minute_limit,
                window="per minute",
                details={
                    "key": key_clean,
                    "current": len(recent_requests),
                    "limit": per_minute_limit,
                }
            )
        
        # Verificar límite por hora
        per_hour_limit = limit_config.get("per_hour")
        if per_hour_limit:
            hour_ago = now - 3600
            hour_requests = [
                req_time for req_time in self._requests[key_clean]
                if req_time > hour_ago
            ]
            
            if len(hour_requests) >= per_hour_limit:
                raise MCPRateLimitError(
                    limit=per_hour_limit,
                    window="per hour",
                    details={
                        "key": key_clean,
                        "current": len(hour_requests),
                        "limit": per_hour_limit,
                    }
                )
        
        # Registrar request
        self._requests[key_clean].append(now)
        logger.debug(f"Request allowed for key '{key_clean}'")
        
        return True
    
    def reset(self, key: Optional[str] = None) -> int:
        """
        Resetea contadores de solicitudes.
        
        Args:
            key: Clave específica o None para resetear todas.
        
        Returns:
            Número de claves reseteadas.
        
        Raises:
            ValueError: Si key está vacío cuando se proporciona.
        """
        if key is not None:
            if not key.strip():
                raise ValueError("key cannot be empty when provided")
            
            key_clean = key.strip()
            if key_clean in self._requests:
                del self._requests[key_clean]
                logger.debug(f"Reset rate limit counters for key '{key_clean}'")
                return 1
            return 0
        else:
            count = len(self._requests)
            self._requests.clear()
            logger.info(f"Reset all rate limit counters ({count} keys)")
            return count
    
    def get_stats(self, key: str) -> Dict[str, int]:
        """
        Obtiene estadísticas de rate limiting para una clave.
        
        Args:
            key: Clave única.
        
        Returns:
            Diccionario con estadísticas:
            - total: Total de requests registrados
            - last_minute: Requests en el último minuto
            - last_hour: Requests en la última hora
            - limit_per_minute: Límite por minuto configurado
            - limit_per_hour: Límite por hora configurado
        
        Raises:
            ValueError: Si key está vacío.
        """
        if not key or not key.strip():
            raise ValueError("key cannot be empty")
        
        key_clean = key.strip()
        now = time.time()
        minute_ago = now - 60
        hour_ago = now - 3600
        
        requests = self._requests.get(key_clean, [])
        limit_config = self._limits.get(key_clean, {})
        
        stats = {
            "total": len(requests),
            "last_minute": len([r for r in requests if r > minute_ago]),
            "last_hour": len([r for r in requests if r > hour_ago]),
            "limit_per_minute": limit_config.get("per_minute", 0),
            "limit_per_hour": limit_config.get("per_hour", 0),
        }
        
        return stats
    
    def get_all_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Obtiene estadísticas de rate limiting para todas las claves.
        
        Returns:
            Diccionario con estadísticas por clave.
        """
        return {
            key: self.get_stats(key)
            for key in set(list(self._requests.keys()) + list(self._limits.keys()))
        }
