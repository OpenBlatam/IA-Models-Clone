"""
API Throttling - Throttling Avanzado de API
==========================================

Throttling avanzado:
- Adaptive throttling
- Priority-based throttling
- Burst handling
- Throttle policies
- Queue management
"""

import logging
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class ThrottlePolicy(str, Enum):
    """Políticas de throttling"""
    FIXED = "fixed"
    ADAPTIVE = "adaptive"
    BURST = "burst"
    PRIORITY = "priority"


class RequestPriority(str, Enum):
    """Prioridades de request"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class APIThrottler:
    """
    Throttler de API.
    """
    
    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: int = 60,
        policy: ThrottlePolicy = ThrottlePolicy.ADAPTIVE
    ) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.policy = policy
        self.request_history: deque = deque(maxlen=max_requests * 2)
        self.priority_queues: Dict[RequestPriority, asyncio.Queue] = {
            RequestPriority.CRITICAL: asyncio.Queue(),
            RequestPriority.HIGH: asyncio.Queue(),
            RequestPriority.NORMAL: asyncio.Queue(),
            RequestPriority.LOW: asyncio.Queue()
        }
        self.current_rate = max_requests
        self.adaptive_factor = 1.0
    
    async def throttle(
        self,
        request_id: str,
        priority: RequestPriority = RequestPriority.NORMAL
    ) -> bool:
        """Aplica throttling"""
        now = datetime.now()
        
        # Limpiar requests antiguos
        cutoff = now - timedelta(seconds=self.window_seconds)
        self.request_history = deque(
            [r for r in self.request_history if r["timestamp"] > cutoff],
            maxlen=self.max_requests * 2
        )
        
        # Contar requests en ventana
        current_count = len(self.request_history)
        
        if self.policy == ThrottlePolicy.ADAPTIVE:
            # Throttling adaptativo basado en carga
            if current_count >= self.current_rate:
                # Ajustar rate dinámicamente
                if current_count > self.max_requests * 1.2:
                    self.adaptive_factor *= 0.9  # Reducir rate
                elif current_count < self.max_requests * 0.8:
                    self.adaptive_factor = min(1.0, self.adaptive_factor * 1.1)  # Aumentar rate
                
                self.current_rate = int(self.max_requests * self.adaptive_factor)
                return False
        elif self.policy == ThrottlePolicy.PRIORITY:
            # Throttling basado en prioridad
            if current_count >= self.max_requests:
                # Permitir solo requests de alta prioridad
                if priority in [RequestPriority.CRITICAL, RequestPriority.HIGH]:
                    return True
                else:
                    # Agregar a cola
                    await self.priority_queues[priority].put(request_id)
                    return False
        elif self.policy == ThrottlePolicy.BURST:
            # Permitir bursts pero limitar promedio
            if current_count >= self.max_requests * 1.5:
                return False
        else:
            # Fixed policy
            if current_count >= self.max_requests:
                return False
        
        # Registrar request
        self.request_history.append({
            "request_id": request_id,
            "priority": priority.value,
            "timestamp": now
        })
        
        return True
    
    def get_throttle_status(self) -> Dict[str, Any]:
        """Obtiene estado de throttling"""
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.window_seconds)
        recent_requests = [
            r for r in self.request_history
            if r["timestamp"] > cutoff
        ]
        
        return {
            "current_requests": len(recent_requests),
            "max_requests": self.max_requests,
            "window_seconds": self.window_seconds,
            "policy": self.policy.value,
            "adaptive_factor": self.adaptive_factor if self.policy == ThrottlePolicy.ADAPTIVE else None,
            "queue_sizes": {
                priority.value: queue.qsize()
                for priority, queue in self.priority_queues.items()
            }
        }
    
    async def process_priority_queue(self, priority: RequestPriority) -> List[str]:
        """Procesa cola de prioridad"""
        processed = []
        queue = self.priority_queues[priority]
        
        while not queue.empty() and len(processed) < self.max_requests:
            try:
                request_id = await asyncio.wait_for(queue.get(), timeout=0.1)
                if await self.throttle(request_id, priority):
                    processed.append(request_id)
            except asyncio.TimeoutError:
                break
        
        return processed


def get_api_throttler(
    max_requests: int = 100,
    window_seconds: int = 60,
    policy: ThrottlePolicy = ThrottlePolicy.ADAPTIVE
) -> APIThrottler:
    """Obtiene throttler de API"""
    return APIThrottler(max_requests=max_requests, window_seconds=window_seconds, policy=policy)















