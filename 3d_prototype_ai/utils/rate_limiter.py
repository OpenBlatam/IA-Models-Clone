"""
Rate Limiter - Sistema de rate limiting avanzado
=================================================
"""

import logging
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum

logger = logging.getLogger(__name__)


class RateLimitStrategy(str, Enum):
    """Estrategias de rate limiting"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


class RateLimiter:
    """Sistema de rate limiting avanzado"""
    
    def __init__(self, strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW):
        self.strategy = strategy
        self.limits: Dict[str, Dict[str, int]] = {}
        self.windows: Dict[str, deque] = defaultdict(deque)
        self.tokens: Dict[str, Dict[str, float]] = {}
        self.blocked: Dict[str, datetime] = {}
    
    def set_limit(self, key: str, requests: int, window_seconds: int = 60):
        """Configura un límite de rate"""
        self.limits[key] = {
            "requests": requests,
            "window_seconds": window_seconds
        }
    
    def is_allowed(self, identifier: str, limit_key: str = "default") -> Tuple[bool, Optional[Dict[str, int]]]:
        """
        Verifica si una solicitud está permitida
        
        Returns:
            (allowed, info) donde info contiene remaining, reset_time, etc.
        """
        # Verificar si está bloqueado
        if identifier in self.blocked:
            block_until = self.blocked[identifier]
            if datetime.now() < block_until:
                remaining_seconds = int((block_until - datetime.now()).total_seconds())
                return False, {
                    "blocked": True,
                    "retry_after": remaining_seconds
                }
            else:
                del self.blocked[identifier]
        
        limit_config = self.limits.get(limit_key, {"requests": 100, "window_seconds": 60})
        requests_limit = limit_config["requests"]
        window_seconds = limit_config["window_seconds"]
        
        key = f"{identifier}:{limit_key}"
        
        if self.strategy == RateLimitStrategy.FIXED_WINDOW:
            return self._fixed_window_check(key, requests_limit, window_seconds)
        elif self.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return self._sliding_window_check(key, requests_limit, window_seconds)
        elif self.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return self._token_bucket_check(key, requests_limit, window_seconds)
        else:
            return self._leaky_bucket_check(key, requests_limit, window_seconds)
    
    def _fixed_window_check(self, key: str, limit: int, window: int) -> Tuple[bool, Dict[str, int]]:
        """Fixed window rate limiting"""
        now = datetime.now()
        window_start = now.replace(second=0, microsecond=0)
        
        if key not in self.windows:
            self.windows[key] = deque([(window_start, 1)])
            return True, {"remaining": limit - 1, "reset_in": window}
        
        window_data = self.windows[key]
        current_window_count = sum(1 for ts, _ in window_data if ts >= window_start)
        
        if current_window_count >= limit:
            reset_time = window_start + timedelta(seconds=window)
            reset_in = int((reset_time - now).total_seconds())
            return False, {"remaining": 0, "reset_in": reset_in}
        
        window_data.append((now, 1))
        # Limpiar ventanas antiguas
        cutoff = window_start - timedelta(seconds=window)
        while window_data and window_data[0][0] < cutoff:
            window_data.popleft()
        
        return True, {"remaining": limit - current_window_count - 1, "reset_in": window}
    
    def _sliding_window_check(self, key: str, limit: int, window: int) -> Tuple[bool, Dict[str, int]]:
        """Sliding window rate limiting"""
        now = datetime.now()
        cutoff = now - timedelta(seconds=window)
        
        if key not in self.windows:
            self.windows[key] = deque([now])
            return True, {"remaining": limit - 1, "reset_in": window}
        
        window_data = self.windows[key]
        
        # Limpiar timestamps fuera de la ventana
        while window_data and window_data[0] < cutoff:
            window_data.popleft()
        
        if len(window_data) >= limit:
            oldest = window_data[0]
            reset_in = int((oldest + timedelta(seconds=window) - now).total_seconds())
            return False, {"remaining": 0, "reset_in": max(0, reset_in)}
        
        window_data.append(now)
        return True, {"remaining": limit - len(window_data), "reset_in": window}
    
    def _token_bucket_check(self, key: str, limit: int, window: int) -> Tuple[bool, Dict[str, int]]:
        """Token bucket rate limiting"""
        now = datetime.now()
        
        if key not in self.tokens:
            self.tokens[key] = {
                "tokens": float(limit),
                "last_refill": now,
                "capacity": limit,
                "refill_rate": limit / window  # tokens per second
            }
        
        bucket = self.tokens[key]
        time_passed = (now - bucket["last_refill"]).total_seconds()
        
        # Refill tokens
        new_tokens = time_passed * bucket["refill_rate"]
        bucket["tokens"] = min(bucket["capacity"], bucket["tokens"] + new_tokens)
        bucket["last_refill"] = now
        
        if bucket["tokens"] >= 1.0:
            bucket["tokens"] -= 1.0
            return True, {
                "remaining": int(bucket["tokens"]),
                "reset_in": int((1.0 / bucket["refill_rate"]) if bucket["refill_rate"] > 0 else window)
            }
        
        reset_in = int((1.0 - bucket["tokens"]) / bucket["refill_rate"]) if bucket["refill_rate"] > 0 else window
        return False, {"remaining": 0, "reset_in": reset_in}
    
    def _leaky_bucket_check(self, key: str, limit: int, window: int) -> Tuple[bool, Dict[str, int]]:
        """Leaky bucket rate limiting"""
        # Similar a token bucket pero con diferentes semánticas
        return self._token_bucket_check(key, limit, window)
    
    def block_identifier(self, identifier: str, duration_seconds: int = 3600):
        """Bloquea un identificador temporalmente"""
        self.blocked[identifier] = datetime.now() + timedelta(seconds=duration_seconds)
        logger.warning(f"Identifier {identifier} bloqueado por {duration_seconds} segundos")
    
    def get_stats(self, identifier: Optional[str] = None) -> Dict[str, Any]:
        """Obtiene estadísticas de rate limiting"""
        if identifier:
            key_prefix = f"{identifier}:"
            relevant_windows = {k: v for k, v in self.windows.items() if k.startswith(key_prefix)}
            relevant_tokens = {k: v for k, v in self.tokens.items() if k.startswith(key_prefix)}
            
            return {
                "identifier": identifier,
                "active_windows": len(relevant_windows),
                "active_buckets": len(relevant_tokens),
                "blocked": identifier in self.blocked
            }
        
        return {
            "total_windows": len(self.windows),
            "total_buckets": len(self.tokens),
            "total_blocked": len(self.blocked),
            "limits_configured": len(self.limits)
        }




