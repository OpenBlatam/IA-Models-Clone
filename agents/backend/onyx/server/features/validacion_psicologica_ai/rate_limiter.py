"""
Rate Limiting Avanzado
======================
Sistema de rate limiting con múltiples estrategias
"""

from typing import Dict, Any, Optional
from uuid import UUID
from datetime import datetime, timedelta
import structlog
from collections import defaultdict
import time

logger = structlog.get_logger()


class RateLimitStrategy:
    """Estrategia de rate limiting"""
    
    def __init__(
        self,
        max_requests: int,
        window_seconds: int,
        burst_allowance: int = 0
    ):
        """
        Inicializar estrategia
        
        Args:
            max_requests: Máximo de requests
            window_seconds: Ventana de tiempo en segundos
            burst_allowance: Permiso de burst adicional
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.burst_allowance = burst_allowance
        self._requests: Dict[str, List[float]] = defaultdict(list)
    
    def is_allowed(
        self,
        identifier: str,
        current_time: Optional[float] = None
    ) -> tuple[bool, Dict[str, Any]]:
        """
        Verificar si request está permitido
        
        Args:
            identifier: Identificador (user_id, IP, etc.)
            current_time: Tiempo actual (opcional)
            
        Returns:
            (allowed, info)
        """
        if current_time is None:
            current_time = time.time()
        
        # Limpiar requests antiguos
        cutoff = current_time - self.window_seconds
        self._requests[identifier] = [
            t for t in self._requests[identifier]
            if t > cutoff
        ]
        
        # Verificar límite
        request_count = len(self._requests[identifier])
        max_allowed = self.max_requests + self.burst_allowance
        
        if request_count >= max_allowed:
            # Calcular tiempo hasta reset
            oldest_request = min(self._requests[identifier]) if self._requests[identifier] else current_time
            reset_time = oldest_request + self.window_seconds
            
            return False, {
                "allowed": False,
                "limit": max_allowed,
                "remaining": 0,
                "reset_at": datetime.fromtimestamp(reset_time).isoformat(),
                "retry_after": max(0, int(reset_time - current_time))
            }
        
        # Registrar request
        self._requests[identifier].append(current_time)
        
        return True, {
            "allowed": True,
            "limit": max_allowed,
            "remaining": max_allowed - request_count - 1,
            "reset_at": datetime.fromtimestamp(
                current_time + self.window_seconds
            ).isoformat()
        }


class AdvancedRateLimiter:
    """Rate limiter avanzado"""
    
    def __init__(self):
        """Inicializar rate limiter"""
        self._strategies: Dict[str, RateLimitStrategy] = {}
        self._default_strategy = RateLimitStrategy(
            max_requests=100,
            window_seconds=60
        )
        logger.info("AdvancedRateLimiter initialized")
    
    def register_strategy(
        self,
        name: str,
        max_requests: int,
        window_seconds: int,
        burst_allowance: int = 0
    ) -> None:
        """
        Registrar estrategia de rate limiting
        
        Args:
            name: Nombre de la estrategia
            max_requests: Máximo de requests
            window_seconds: Ventana de tiempo
            burst_allowance: Permiso de burst
        """
        self._strategies[name] = RateLimitStrategy(
            max_requests=max_requests,
            window_seconds=window_seconds,
            burst_allowance=burst_allowance
        )
        logger.info("Rate limit strategy registered", name=name)
    
    def check_rate_limit(
        self,
        identifier: str,
        strategy_name: Optional[str] = None
    ) -> tuple[bool, Dict[str, Any]]:
        """
        Verificar rate limit
        
        Args:
            identifier: Identificador
            strategy_name: Nombre de estrategia (opcional)
            
        Returns:
            (allowed, info)
        """
        strategy = (
            self._strategies.get(strategy_name)
            if strategy_name
            else self._default_strategy
        )
        
        if not strategy:
            strategy = self._default_strategy
        
        return strategy.is_allowed(identifier)
    
    def get_rate_limit_info(
        self,
        identifier: str,
        strategy_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Obtener información de rate limit
        
        Args:
            identifier: Identificador
            strategy_name: Nombre de estrategia
            
        Returns:
            Información de rate limit
        """
        allowed, info = self.check_rate_limit(identifier, strategy_name)
        return info


# Instancia global del rate limiter
rate_limiter = AdvancedRateLimiter()

# Registrar estrategias por defecto
rate_limiter.register_strategy("api", max_requests=100, window_seconds=60)
rate_limiter.register_strategy("validation", max_requests=10, window_seconds=300)
rate_limiter.register_strategy("export", max_requests=20, window_seconds=60)




