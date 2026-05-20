"""
Queue Helpers
=============
Utilidades para manejo de colas.
"""

from typing import Any, Optional, Callable
import asyncio
from collections import deque
from datetime import datetime


class AsyncQueue:
    """Cola asíncrona thread-safe."""
    
    def __init__(self, maxsize: int = 0):
        """
        Inicializar cola.
        
        Args:
            maxsize: Tamaño máximo (0 = ilimitado)
        """
        self._queue = asyncio.Queue(maxsize=maxsize)
        self._stats = {
            "put_count": 0,
            "get_count": 0,
            "created_at": datetime.now()
        }
    
    async def put(self, item: Any):
        """Agregar item a la cola."""
        await self._queue.put(item)
        self._stats["put_count"] += 1
    
    async def get(self) -> Any:
        """Obtener item de la cola."""
        item = await self._queue.get()
        self._stats["get_count"] += 1
        return item
    
    def get_nowait(self) -> Any:
        """Obtener item sin esperar."""
        return self._queue.get_nowait()
    
    def put_nowait(self, item: Any):
        """Agregar item sin esperar."""
        self._queue.put_nowait(item)
        self._stats["put_count"] += 1
    
    def qsize(self) -> int:
        """Obtener tamaño de la cola."""
        return self._queue.qsize()
    
    def empty(self) -> bool:
        """Verificar si la cola está vacía."""
        return self._queue.empty()
    
    def full(self) -> bool:
        """Verificar si la cola está llena."""
        return self._queue.full()
    
    def get_stats(self) -> dict:
        """Obtener estadísticas de la cola."""
        return {
            **self._stats,
            "current_size": self.qsize(),
            "is_empty": self.empty(),
            "is_full": self.full()
        }


class PriorityQueue:
    """Cola con prioridad."""
    
    def __init__(self):
        self._items = []
    
    async def put(self, item: Any, priority: int = 0):
        """Agregar item con prioridad."""
        self._items.append((priority, datetime.now(), item))
        self._items.sort(key=lambda x: (-x[0], x[1]))  # Ordenar por prioridad (mayor primero), luego por tiempo
    
    async def get(self) -> Any:
        """Obtener item de mayor prioridad."""
        if not self._items:
            raise asyncio.QueueEmpty()
        
        _, _, item = self._items.pop(0)
        return item
    
    def qsize(self) -> int:
        """Obtener tamaño."""
        return len(self._items)
    
    def empty(self) -> bool:
        """Verificar si está vacía."""
        return len(self._items) == 0

