"""
Advanced Rate Limiting para LLMs.

Rate limiting sofisticado con ventanas deslizantes, múltiples estrategias,
y soporte para diferentes tipos de límites (por usuario, por modelo, global).
"""

import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum

from config.logging_config import get_logger

logger = get_logger(__name__)


class RateLimitStrategy(str, Enum):
    """Estrategias de rate limiting."""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimitConfig:
    """Configuración de rate limit."""
    limit: int  # Número de requests permitidos
    window_seconds: int  # Ventana de tiempo en segundos
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    burst_size: Optional[int] = None  # Para token bucket
    refill_rate: Optional[float] = None  # Para token bucket/leaky bucket
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "limit": self.limit,
            "window_seconds": self.window_seconds,
            "strategy": self.strategy.value,
            "burst_size": self.burst_size,
            "refill_rate": self.refill_rate
        }


@dataclass
class RateLimitInfo:
    """Información sobre rate limit."""
    allowed: bool
    remaining: int
    reset_at: float
    limit: int
    retry_after: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "allowed": self.allowed,
            "remaining": self.remaining,
            "reset_at": self.reset_at,
            "limit": self.limit,
            "retry_after": self.retry_after
        }


class AdvancedRateLimiter:
    """
    Rate limiter avanzado con múltiples estrategias.
    
    Características:
    - Múltiples estrategias (fixed window, sliding window, token bucket, leaky bucket)
    - Rate limiting por clave (usuario, modelo, IP, etc.)
    - Ventanas deslizantes precisas
    - Burst handling
    - Estadísticas detalladas
    """
    
    def __init__(self):
        """Inicializar rate limiter."""
        # Configuraciones por clave
        self.configs: Dict[str, RateLimitConfig] = {}
        
        # Estados por estrategia
        self._fixed_windows: Dict[str, Tuple[int, float]] = {}  # (count, window_start)
        self._sliding_windows: Dict[str, deque] = {}  # Timestamps de requests
        self._token_buckets: Dict[str, Tuple[float, float]] = {}  # (tokens, last_refill)
        self._leaky_buckets: Dict[str, Tuple[float, float]] = {}  # (level, last_leak)
        
        # Estadísticas
        self.stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total_requests": 0,
            "allowed_requests": 0,
            "blocked_requests": 0,
            "last_reset": datetime.now().isoformat()
        })
    
    def configure(
        self,
        key: str,
        limit: int,
        window_seconds: int,
        strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW,
        burst_size: Optional[int] = None,
        refill_rate: Optional[float] = None
    ) -> None:
        """
        Configurar rate limit para una clave.
        
        Args:
            key: Clave única (usuario, modelo, IP, etc.)
            limit: Número máximo de requests
            window_seconds: Ventana de tiempo en segundos
            strategy: Estrategia a usar
            burst_size: Tamaño de burst para token bucket
            refill_rate: Tasa de relleno para token bucket/leaky bucket
        """
        config = RateLimitConfig(
            limit=limit,
            window_seconds=window_seconds,
            strategy=strategy,
            burst_size=burst_size or limit,
            refill_rate=refill_rate or (limit / window_seconds)
        )
        
        self.configs[key] = config
        
        # Inicializar estado según estrategia
        if strategy == RateLimitStrategy.SLIDING_WINDOW:
            self._sliding_windows[key] = deque()
        elif strategy == RateLimitStrategy.TOKEN_BUCKET:
            self._token_buckets[key] = (config.burst_size, time.time())
        elif strategy == RateLimitStrategy.LEAKY_BUCKET:
            self._leaky_buckets[key] = (0.0, time.time())
        
        logger.info(f"Rate limit configurado para {key}: {limit}/{window_seconds}s ({strategy.value})")
    
    def is_allowed(
        self,
        key: str,
        tokens: int = 1
    ) -> RateLimitInfo:
        """
        Verificar si una request está permitida.
        
        Args:
            key: Clave única
            tokens: Número de tokens/requests (útil para weighted limits)
            
        Returns:
            Información sobre el rate limit
        """
        if key not in self.configs:
            # Sin límite configurado, permitir
            return RateLimitInfo(
                allowed=True,
                remaining=999999,
                reset_at=time.time() + 3600,
                limit=999999
            )
        
        config = self.configs[key]
        now = time.time()
        
        # Actualizar estadísticas
        self.stats[key]["total_requests"] += 1
        
        # Verificar según estrategia
        allowed = False
        remaining = 0
        reset_at = now + config.window_seconds
        
        if config.strategy == RateLimitStrategy.FIXED_WINDOW:
            allowed, remaining, reset_at = self._check_fixed_window(key, config, now, tokens)
        
        elif config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            allowed, remaining, reset_at = self._check_sliding_window(key, config, now, tokens)
        
        elif config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            allowed, remaining, reset_at = self._check_token_bucket(key, config, now, tokens)
        
        elif config.strategy == RateLimitStrategy.LEAKY_BUCKET:
            allowed, remaining, reset_at = self._check_leaky_bucket(key, config, now, tokens)
        
        # Actualizar estadísticas
        if allowed:
            self.stats[key]["allowed_requests"] += 1
        else:
            self.stats[key]["blocked_requests"] += 1
        
        retry_after = None if allowed else (reset_at - now)
        
        return RateLimitInfo(
            allowed=allowed,
            remaining=remaining,
            reset_at=reset_at,
            limit=config.limit,
            retry_after=retry_after
        )
    
    def _check_fixed_window(
        self,
        key: str,
        config: RateLimitConfig,
        now: float,
        tokens: int
    ) -> Tuple[bool, int, float]:
        """Verificar rate limit con ventana fija."""
        if key not in self._fixed_windows:
            self._fixed_windows[key] = (0, now)
        
        count, window_start = self._fixed_windows[key]
        
        # Nueva ventana?
        if now - window_start >= config.window_seconds:
            count = 0
            window_start = now
        
        # Verificar límite
        if count + tokens <= config.limit:
            count += tokens
            self._fixed_windows[key] = (count, window_start)
            remaining = config.limit - count
            reset_at = window_start + config.window_seconds
            return True, remaining, reset_at
        else:
            remaining = max(0, config.limit - count)
            reset_at = window_start + config.window_seconds
            return False, remaining, reset_at
    
    def _check_sliding_window(
        self,
        key: str,
        config: RateLimitConfig,
        now: float,
        tokens: int
    ) -> Tuple[bool, int, float]:
        """Verificar rate limit con ventana deslizante."""
        if key not in self._sliding_windows:
            self._sliding_windows[key] = deque()
        
        window = self._sliding_windows[key]
        window_start = now - config.window_seconds
        
        # Limpiar timestamps fuera de la ventana
        while window and window[0] < window_start:
            window.popleft()
        
        # Verificar límite
        if len(window) + tokens <= config.limit:
            # Agregar tokens
            for _ in range(tokens):
                window.append(now)
            
            remaining = config.limit - len(window)
            reset_at = window[0] + config.window_seconds if window else now + config.window_seconds
            return True, remaining, reset_at
        else:
            remaining = max(0, config.limit - len(window))
            reset_at = window[0] + config.window_seconds if window else now + config.window_seconds
            return False, remaining, reset_at
    
    def _check_token_bucket(
        self,
        key: str,
        config: RateLimitConfig,
        now: float,
        tokens: int
    ) -> Tuple[bool, int, float]:
        """Verificar rate limit con token bucket."""
        if key not in self._token_buckets:
            self._token_buckets[key] = (config.burst_size, now)
        
        current_tokens, last_refill = self._token_buckets[key]
        
        # Rellenar tokens
        elapsed = now - last_refill
        tokens_to_add = elapsed * config.refill_rate
        current_tokens = min(config.burst_size, current_tokens + tokens_to_add)
        
        # Verificar si hay suficientes tokens
        if current_tokens >= tokens:
            current_tokens -= tokens
            self._token_buckets[key] = (current_tokens, now)
            remaining = int(current_tokens)
            reset_at = now + ((tokens / config.refill_rate) if config.refill_rate > 0 else config.window_seconds)
            return True, remaining, reset_at
        else:
            remaining = int(current_tokens)
            reset_at = now + ((tokens / config.refill_rate) if config.refill_rate > 0 else config.window_seconds)
            return False, remaining, reset_at
    
    def _check_leaky_bucket(
        self,
        key: str,
        config: RateLimitConfig,
        now: float,
        tokens: int
    ) -> Tuple[bool, int, float]:
        """Verificar rate limit con leaky bucket."""
        if key not in self._leaky_buckets:
            self._leaky_buckets[key] = (0.0, now)
        
        level, last_leak = self._leaky_buckets[key]
        
        # Fuga de tokens
        elapsed = now - last_leak
        leaked = elapsed * config.refill_rate
        level = max(0.0, level - leaked)
        
        # Verificar capacidad
        if level + tokens <= config.limit:
            level += tokens
            self._leaky_buckets[key] = (level, now)
            remaining = int(config.limit - level)
            reset_at = now + ((tokens / config.refill_rate) if config.refill_rate > 0 else config.window_seconds)
            return True, remaining, reset_at
        else:
            remaining = int(max(0, config.limit - level))
            reset_at = now + ((tokens / config.refill_rate) if config.refill_rate > 0 else config.window_seconds)
            return False, remaining, reset_at
    
    def reset(self, key: str) -> bool:
        """
        Resetear rate limit para una clave.
        
        Args:
            key: Clave a resetear
            
        Returns:
            True si se reseteó correctamente
        """
        if key in self._fixed_windows:
            del self._fixed_windows[key]
        if key in self._sliding_windows:
            self._sliding_windows[key].clear()
        if key in self._token_buckets:
            config = self.configs.get(key)
            if config:
                self._token_buckets[key] = (config.burst_size, time.time())
        if key in self._leaky_buckets:
            self._leaky_buckets[key] = (0.0, time.time())
        
        if key in self.stats:
            self.stats[key]["last_reset"] = datetime.now().isoformat()
        
        logger.info(f"Rate limit reseteado para {key}")
        return True
    
    def get_stats(self, key: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtener estadísticas de rate limiting.
        
        Args:
            key: Clave específica (opcional, todas si no se proporciona)
            
        Returns:
            Estadísticas
        """
        if key:
            return self.stats.get(key, {})
        return dict(self.stats)
    
    def get_config(self, key: str) -> Optional[RateLimitConfig]:
        """Obtener configuración de rate limit."""
        return self.configs.get(key)


def get_advanced_rate_limiter() -> AdvancedRateLimiter:
    """Factory function para obtener instancia singleton del rate limiter."""
    if not hasattr(get_advanced_rate_limiter, "_instance"):
        get_advanced_rate_limiter._instance = AdvancedRateLimiter()
    return get_advanced_rate_limiter._instance



