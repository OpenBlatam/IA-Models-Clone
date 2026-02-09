"""
Optimizations - Optimizaciones de rendimiento avanzadas
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import functools

logger = logging.getLogger(__name__)


class AsyncBatchProcessor:
    """Procesador batch asíncrono optimizado"""

    def __init__(self, max_workers: int = 10, batch_size: int = 100):
        """
        Inicializar procesador batch.

        Args:
            max_workers: Número máximo de workers
            batch_size: Tamaño del batch
        """
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def process_batch(
        self,
        items: List[Any],
        processor: Callable,
        *args,
        **kwargs
    ) -> List[Any]:
        """
        Procesar items en batch de forma asíncrona.

        Args:
            items: Lista de items a procesar
            processor: Función procesadora
            *args: Argumentos adicionales
            **kwargs: Argumentos con nombre

        Returns:
            Lista de resultados
        """
        # Dividir en batches
        batches = [
            items[i:i + self.batch_size]
            for i in range(0, len(items), self.batch_size)
        ]
        
        # Procesar batches en paralelo
        tasks = []
        for batch in batches:
            task = asyncio.create_task(
                self._process_single_batch(batch, processor, *args, **kwargs)
            )
            tasks.append(task)
        
        # Esperar todos los batches
        results = await asyncio.gather(*tasks)
        
        # Aplanar resultados
        return [item for batch_result in results for item in batch_result]

    async def _process_single_batch(
        self,
        batch: List[Any],
        processor: Callable,
        *args,
        **kwargs
    ) -> List[Any]:
        """Procesar un batch individual"""
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            self.executor,
            lambda: [processor(item, *args, **kwargs) for item in batch]
        )
        return results


class ContentCache:
    """Cache optimizado para contenido"""

    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """
        Inicializar cache.

        Args:
            max_size: Tamaño máximo
            ttl: Tiempo de vida en segundos
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, tuple[Any, float]] = {}
        self.access_times: Dict[str, float] = {}

    def get(self, key: str) -> Optional[Any]:
        """
        Obtener del cache.

        Args:
            key: Clave

        Returns:
            Valor o None
        """
        import time
        if key not in self.cache:
            return None
        
        value, timestamp = self.cache[key]
        
        # Verificar TTL
        if time.time() - timestamp > self.ttl:
            del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
            return None
        
        # Actualizar tiempo de acceso
        self.access_times[key] = time.time()
        return value

    def set(self, key: str, value: Any):
        """
        Almacenar en cache.

        Args:
            key: Clave
            value: Valor
        """
        import time
        
        # Si el cache está lleno, eliminar el menos usado
        if len(self.cache) >= self.max_size and key not in self.cache:
            # Encontrar clave menos usada
            if self.access_times:
                lru_key = min(self.access_times.items(), key=lambda x: x[1])[0]
                del self.cache[lru_key]
                del self.access_times[lru_key]
        
        self.cache[key] = (value, time.time())
        self.access_times[key] = time.time()

    def clear(self):
        """Limpiar cache"""
        self.cache.clear()
        self.access_times.clear()


def memoize_async(ttl: int = 3600):
    """
    Decorador para memoización asíncrona.

    Args:
        ttl: Tiempo de vida en segundos
    """
    cache = ContentCache(max_size=1000, ttl=ttl)
    
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Generar clave del cache
            import hashlib
            import json
            key_data = json.dumps({
                "args": str(args),
                "kwargs": str(sorted(kwargs.items()))
            }, sort_keys=True)
            key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Verificar cache
            cached = cache.get(key)
            if cached is not None:
                return cached
            
            # Ejecutar función
            result = await func(*args, **kwargs)
            
            # Almacenar en cache
            cache.set(key, result)
            
            return result
        
        return wrapper
    return decorator


class ConnectionPool:
    """Pool de conexiones para optimización"""

    def __init__(self, max_connections: int = 10):
        """
        Inicializar pool.

        Args:
            max_connections: Número máximo de conexiones
        """
        self.max_connections = max_connections
        self.connections: List[Any] = []
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Adquirir conexión del pool"""
        async with self.lock:
            if self.connections:
                return self.connections.pop()
            # Crear nueva conexión si es necesario
            # (implementación específica según el tipo de conexión)
            return None

    async def release(self, connection: Any):
        """Liberar conexión al pool"""
        async with self.lock:
            if len(self.connections) < self.max_connections:
                self.connections.append(connection)






