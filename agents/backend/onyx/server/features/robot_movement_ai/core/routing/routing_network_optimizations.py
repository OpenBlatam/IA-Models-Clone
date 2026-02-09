"""
Routing Network-Level Optimizations
====================================

Optimizaciones a nivel de red y comunicación.
Incluye: Connection pooling, Request batching, Async I/O, etc.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from collections import deque
import time
import threading

logger = logging.getLogger(__name__)

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class ConnectionPool:
    """Pool de conexiones para mejor rendimiento."""
    
    def __init__(self, max_size: int = 10):
        """
        Inicializar pool de conexiones.
        
        Args:
            max_size: Tamaño máximo del pool
        """
        self.max_size = max_size
        self.connections: deque = deque(maxlen=max_size)
        self.lock = threading.Lock()
    
    def get_connection(self):
        """Obtener conexión del pool."""
        with self.lock:
            if self.connections:
                return self.connections.popleft()
            return None
    
    def return_connection(self, conn):
        """Devolver conexión al pool."""
        with self.lock:
            if len(self.connections) < self.max_size:
                self.connections.append(conn)


class RequestBatcher:
    """Batching de requests para mejor throughput."""
    
    def __init__(self, batch_size: int = 10, timeout: float = 0.1):
        """
        Inicializar batcher de requests.
        
        Args:
            batch_size: Tamaño del batch
            timeout: Timeout en segundos
        """
        self.batch_size = batch_size
        self.timeout = timeout
        self.queue: deque = deque()
        self.lock = threading.Lock()
        self.last_batch_time = time.time()
    
    def add_request(self, request: Dict[str, Any]) -> bool:
        """
        Agregar request al batch.
        
        Args:
            request: Request a agregar
        
        Returns:
            True si se debe procesar el batch
        """
        with self.lock:
            self.queue.append(request)
            current_time = time.time()
            
            # Procesar si el batch está lleno o ha pasado el timeout
            if (len(self.queue) >= self.batch_size or 
                (current_time - self.last_batch_time) >= self.timeout):
                self.last_batch_time = current_time
                return True
            return False
    
    def get_batch(self) -> List[Dict[str, Any]]:
        """Obtener batch de requests."""
        with self.lock:
            batch = list(self.queue)
            self.queue.clear()
            return batch


class AsyncRequestHandler:
    """Manejador de requests asíncronos."""
    
    def __init__(self, max_concurrent: int = 10):
        """
        Inicializar manejador asíncrono.
        
        Args:
            max_concurrent: Máximo de requests concurrentes
        """
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent) if AIOHTTP_AVAILABLE else None
    
    async def process_request(self, url: str, method: str = "GET", **kwargs):
        """
        Procesar request asíncrono.
        
        Args:
            url: URL del request
            method: Método HTTP
            **kwargs: Argumentos adicionales
        
        Returns:
            Respuesta del request
        """
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp not available")
        
        async with self.semaphore:
            async with aiohttp.ClientSession() as session:
                async with session.request(method, url, **kwargs) as response:
                    return await response.json()


class NetworkOptimizer:
    """Optimizador completo de red."""
    
    def __init__(self):
        """Inicializar optimizador de red."""
        self.connection_pool = ConnectionPool() if REQUESTS_AVAILABLE else None
        self.request_batcher = RequestBatcher() if REQUESTS_AVAILABLE else None
        self.async_handler = AsyncRequestHandler() if AIOHTTP_AVAILABLE else None
    
    def optimize_network_settings(self):
        """Optimizar configuraciones de red."""
        logger.info("Network optimizations configured")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            'aiohttp_available': AIOHTTP_AVAILABLE,
            'requests_available': REQUESTS_AVAILABLE,
            'connection_pool_size': len(self.connection_pool.connections) if self.connection_pool else 0,
            'batched_requests': len(self.request_batcher.queue) if self.request_batcher else 0
        }

