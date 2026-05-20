"""
Rate Limiter - Limitador de Tasa
=================================

Implementa rate limiting para proteger la API.
"""

import logging
import time
from typing import Dict, Any
from collections import defaultdict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RateLimiter:
    """Limitador de tasa de requests"""

    def __init__(self):
        """Inicializa el rate limiter"""
        self.requests: Dict[str, list] = defaultdict(list)
        self.limits = {
            "default": {"requests": 100, "window": 3600},  # 100 requests por hora
            "generate": {"requests": 10, "window": 3600},  # 10 generaciones por hora
            "search": {"requests": 50, "window": 3600},  # 50 búsquedas por hora
        }

    def is_allowed(
        self,
        client_id: str,
        endpoint: str = "default",
    ) -> tuple[bool, Dict[str, Any]]:
        """
        Verifica si un request está permitido.

        Args:
            client_id: ID del cliente (IP, API key, etc.)
            endpoint: Endpoint específico

        Returns:
            Tupla (allowed, info)
        """
        limit_config = self.limits.get(endpoint, self.limits["default"])
        max_requests = limit_config["requests"]
        window_seconds = limit_config["window"]

        key = f"{client_id}:{endpoint}"
        now = time.time()

        # Limpiar requests antiguos
        self.requests[key] = [
            req_time for req_time in self.requests[key]
            if now - req_time < window_seconds
        ]

        # Verificar límite
        if len(self.requests[key]) >= max_requests:
            reset_time = min(self.requests[key]) + window_seconds
            return False, {
                "allowed": False,
                "limit": max_requests,
                "remaining": 0,
                "reset_at": datetime.fromtimestamp(reset_time).isoformat(),
            }

        # Agregar request
        self.requests[key].append(now)

        remaining = max_requests - len(self.requests[key])
        return True, {
            "allowed": True,
            "limit": max_requests,
            "remaining": remaining,
            "reset_at": datetime.fromtimestamp(now + window_seconds).isoformat(),
        }

    def get_rate_limit_info(
        self,
        client_id: str,
        endpoint: str = "default",
    ) -> Dict[str, Any]:
        """Obtiene información de rate limit"""
        limit_config = self.limits.get(endpoint, self.limits["default"])
        key = f"{client_id}:{endpoint}"
        now = time.time()

        # Limpiar requests antiguos
        self.requests[key] = [
            req_time for req_time in self.requests[key]
            if now - req_time < limit_config["window"]
        ]

        return {
            "limit": limit_config["requests"],
            "remaining": limit_config["requests"] - len(self.requests[key]),
            "used": len(self.requests[key]),
            "window_seconds": limit_config["window"],
        }


