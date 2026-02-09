"""
Rate Limiter - Limitador de tasa de solicitudes
================================================

Controla la tasa de solicitudes para prevenir sobrecarga.
"""

import time
import asyncio
import logging
from typing import Dict, Optional
from collections import deque
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RateLimiter:
    """Limitador de tasa de solicitudes"""
    
    def __init__(
        self,
        max_requests: int = 100,
        time_window: float = 60.0,  # segundos
        per_user: bool = False
    ):
        self.max_requests = max_requests
        self.time_window = time_window
        self.per_user = per_user
        
        # Almacenar requests por usuario o global
        self.requests: Dict[Optional[str], deque] = {}
        self.locks: Dict[Optional[str], asyncio.Lock] = {}
    
    async def is_allowed(self, user_id: Optional[str] = None) -> bool:
        """Verificar si se permite la solicitud"""
        key = user_id if self.per_user else None
        
        # Obtener o crear lock para esta clave
        if key not in self.locks:
            self.locks[key] = asyncio.Lock()
        
        async with self.locks[key]:
            # Obtener o crear deque para esta clave
            if key not in self.requests:
                self.requests[key] = deque()
            
            now = time.time()
            request_times = self.requests[key]
            
            # Eliminar requests antiguos
            while request_times and request_times[0] < now - self.time_window:
                request_times.popleft()
            
            # Verificar si hay espacio
            if len(request_times) < self.max_requests:
                request_times.append(now)
                return True
            
            return False
    
    async def wait_for_slot(self, user_id: Optional[str] = None, timeout: float = 30.0) -> bool:
        """Esperar hasta que haya un slot disponible"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if await self.is_allowed(user_id):
                return True
            await asyncio.sleep(0.1)
        
        return False
    
    def get_remaining(self, user_id: Optional[str] = None) -> int:
        """Obtener número de solicitudes restantes"""
        key = user_id if self.per_user else None
        
        if key not in self.requests:
            return self.max_requests
        
        now = time.time()
        request_times = self.requests[key]
        
        # Eliminar requests antiguos
        while request_times and request_times[0] < now - self.time_window:
            request_times.popleft()
        
        return max(0, self.max_requests - len(request_times))
    
    def get_reset_time(self, user_id: Optional[str] = None) -> datetime:
        """Obtener tiempo de reset del rate limit"""
        key = user_id if self.per_user else None
        
        if key not in self.requests or not self.requests[key]:
            return datetime.now()
        
        oldest_request = self.requests[key][0]
        reset_time = datetime.fromtimestamp(oldest_request + self.time_window)
        return reset_time
    
    def reset(self, user_id: Optional[str] = None):
        """Resetear rate limit para un usuario"""
        key = user_id if self.per_user else None
        
        if key in self.requests:
            self.requests[key].clear()
    
    def reset_all(self):
        """Resetear todos los rate limits"""
        self.requests.clear()
        self.locks.clear()


class TaskRateLimiter:
    """Rate limiter específico para tareas"""
    
    def __init__(
        self,
        max_tasks_per_minute: int = 60,
        max_concurrent_tasks: int = 10
    ):
        self.task_limiter = RateLimiter(
            max_requests=max_tasks_per_minute,
            time_window=60.0
        )
        self.max_concurrent = max_concurrent_tasks
        self.active_tasks = 0
        self._lock = asyncio.Lock()
    
    async def can_add_task(self) -> bool:
        """Verificar si se puede agregar una tarea"""
        async with self._lock:
            # Verificar límite de concurrentes
            if self.active_tasks >= self.max_concurrent:
                return False
            
            # Verificar límite de tasa
            return await self.task_limiter.is_allowed()
    
    async def add_task(self):
        """Registrar que se agregó una tarea"""
        async with self._lock:
            self.active_tasks += 1
    
    async def complete_task(self):
        """Registrar que se completó una tarea"""
        async with self._lock:
            self.active_tasks = max(0, self.active_tasks - 1)
    
    def get_stats(self) -> Dict[str, int]:
        """Obtener estadísticas"""
        return {
            "active_tasks": self.active_tasks,
            "max_concurrent": self.max_concurrent,
            "remaining_this_minute": self.task_limiter.get_remaining()
        }


