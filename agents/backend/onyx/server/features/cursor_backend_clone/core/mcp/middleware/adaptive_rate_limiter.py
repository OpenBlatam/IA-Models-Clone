"""
MCP Adaptive Rate Limiter - Rate Limiting Adaptativo
=====================================================

Rate limiting que se adapta automáticamente según la carga del sistema.
"""

import time
import asyncio
import logging
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveLimiterConfig:
    """Configuración del rate limiter adaptativo"""
    base_max_requests: int = 100
    base_window_seconds: int = 60
    min_max_requests: int = 10
    max_max_requests: int = 1000
    adaptation_factor: float = 0.1
    check_interval: float = 30.0
    load_threshold_high: float = 0.8
    load_threshold_low: float = 0.3


class AdaptiveRateLimiter:
    """Rate limiter que se adapta según la carga del sistema"""
    
    def __init__(self, config: Optional[AdaptiveLimiterConfig] = None):
        self.config = config or AdaptiveLimiterConfig()
        self.max_requests = self.config.base_max_requests
        self.window_seconds = self.config.base_window_seconds
        self._requests: Dict[str, deque] = {}
        self._lock = asyncio.Lock()
        self._last_adaptation = time.time()
        self._recent_response_times: deque = deque(maxlen=100)
        self._recent_error_rate: deque = deque(maxlen=100)
        self._adaptation_task: Optional[asyncio.Task] = None
    
    def record_response_time(self, response_time: float):
        """Registrar tiempo de respuesta para adaptación"""
        self._recent_response_times.append(response_time)
    
    def record_error(self, is_error: bool):
        """Registrar error para adaptación"""
        self._recent_error_rate.append(1.0 if is_error else 0.0)
    
    def _calculate_load(self) -> float:
        """Calcular carga actual del sistema"""
        if not self._recent_response_times:
            return 0.5
        
        avg_response_time = sum(self._recent_response_times) / len(self._recent_response_times)
        error_rate = sum(self._recent_error_rate) / len(self._recent_error_rate) if self._recent_error_rate else 0.0
        
        normalized_response_time = min(avg_response_time / 1.0, 1.0)
        load = (normalized_response_time * 0.7) + (error_rate * 0.3)
        
        return min(max(load, 0.0), 1.0)
    
    async def _adapt(self):
        """Adaptar límites según la carga"""
        while True:
            try:
                await asyncio.sleep(self.config.check_interval)
                
                load = self._calculate_load()
                
                if load > self.config.load_threshold_high:
                    new_max = int(self.max_requests * (1 - self.config.adaptation_factor))
                    self.max_requests = max(new_max, self.config.min_max_requests)
                    logger.info(f"High load detected ({load:.2f}), reducing rate limit to {self.max_requests}")
                
                elif load < self.config.load_threshold_low:
                    new_max = int(self.max_requests * (1 + self.config.adaptation_factor))
                    self.max_requests = min(new_max, self.config.max_max_requests)
                    logger.info(f"Low load detected ({load:.2f}), increasing rate limit to {self.max_requests}")
                
                self._last_adaptation = time.time()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in adaptive rate limiter: {e}", exc_info=True)
    
    async def start(self):
        """Iniciar adaptación automática"""
        if self._adaptation_task is None or self._adaptation_task.done():
            self._adaptation_task = asyncio.create_task(self._adapt())
    
    async def stop(self):
        """Detener adaptación automática"""
        if self._adaptation_task:
            self._adaptation_task.cancel()
            try:
                await self._adaptation_task
            except asyncio.CancelledError:
                pass
    
    def is_allowed(self, client_id: str = "default") -> bool:
        """Verificar si el request está permitido"""
        now = time.time()
        
        if client_id not in self._requests:
            self._requests[client_id] = deque()
        
        requests = self._requests[client_id]
        
        while requests and now - requests[0] > self.window_seconds:
            requests.popleft()
        
        if len(requests) >= self.max_requests:
            return False
        
        requests.append(now)
        return True
    
    def get_current_limit(self) -> int:
        """Obtener límite actual"""
        return self.max_requests
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas"""
        load = self._calculate_load()
        avg_response_time = sum(self._recent_response_times) / len(self._recent_response_times) if self._recent_response_times else 0.0
        error_rate = sum(self._recent_error_rate) / len(self._recent_error_rate) if self._recent_error_rate else 0.0
        
        return {
            "current_limit": self.max_requests,
            "base_limit": self.config.base_max_requests,
            "window_seconds": self.window_seconds,
            "current_load": round(load, 4),
            "avg_response_time": round(avg_response_time, 4),
            "error_rate": round(error_rate, 4),
            "last_adaptation": datetime.fromtimestamp(self._last_adaptation).isoformat(),
            "active_clients": len(self._requests)
        }

