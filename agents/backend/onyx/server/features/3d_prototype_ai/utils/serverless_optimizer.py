"""
Serverless Optimizer - Optimizaciones para entornos serverless
===============================================================

Incluye:
- Reducción de cold start times
- Optimización de imports
- Lazy loading
- Connection pooling
- Memory optimization
"""

import logging
import importlib
from typing import Optional, Callable, Any
from functools import lru_cache, wraps
import time

logger = logging.getLogger(__name__)


class LazyImport:
    """Lazy import para reducir cold start times"""
    
    def __init__(self, module_name: str):
        self.module_name = module_name
        self._module = None
    
    def __getattr__(self, name: str):
        if self._module is None:
            self._module = importlib.import_module(self.module_name)
        return getattr(self._module, name)


class ConnectionPool:
    """Pool de conexiones para reutilizar conexiones"""
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.pool: list = []
        self.active: int = 0
    
    def get_connection(self, factory: Callable) -> Any:
        """Obtiene una conexión del pool o crea una nueva"""
        if self.pool:
            return self.pool.pop()
        
        if self.active < self.max_size:
            self.active += 1
            return factory()
        
        raise Exception("Connection pool exhausted")
    
    def return_connection(self, connection: Any):
        """Devuelve una conexión al pool"""
        if len(self.pool) < self.max_size:
            self.pool.append(connection)
        else:
            # Cerrar conexión si el pool está lleno
            if hasattr(connection, 'close'):
                connection.close()
            self.active -= 1


class ColdStartOptimizer:
    """Optimizador para reducir cold start times"""
    
    def __init__(self):
        self.initialized = False
        self.init_time = None
    
    def initialize(self):
        """Inicializa componentes críticos de forma lazy"""
        if self.initialized:
            return
        
        start_time = time.time()
        
        # Pre-warm critical components
        self._prewarm_components()
        
        self.initialized = True
        self.init_time = time.time() - start_time
        
        logger.info(f"Cold start optimization completed in {self.init_time:.3f}s")
    
    def _prewarm_components(self):
        """Pre-calienta componentes críticos"""
        # Importar módulos críticos
        try:
            import json
            import asyncio
            # Pre-compilar regex patterns si es necesario
            # Pre-cargar modelos pequeños
        except Exception as e:
            logger.warning(f"Error in prewarming: {e}")


def optimize_for_serverless(func: Callable) -> Callable:
    """Decorator para optimizar funciones para serverless"""
    
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Lazy initialization
        optimizer = ColdStartOptimizer()
        optimizer.initialize()
        
        return await func(*args, **kwargs)
    
    return wrapper


class MemoryOptimizer:
    """Optimizador de memoria para entornos serverless"""
    
    @staticmethod
    def clear_cache():
        """Limpia caches para liberar memoria"""
        import gc
        gc.collect()
    
    @staticmethod
    def get_memory_usage() -> float:
        """Obtiene el uso de memoria actual en MB"""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    @staticmethod
    def optimize_imports():
        """Optimiza imports para reducir memoria"""
        # Remover imports no usados
        # Usar lazy imports donde sea posible
        pass


class ServerlessConfig:
    """Configuración para entornos serverless"""
    
    def __init__(self,
                 max_memory_mb: int = 512,
                 timeout_seconds: int = 30,
                 enable_connection_pooling: bool = True,
                 enable_lazy_loading: bool = True):
        self.max_memory_mb = max_memory_mb
        self.timeout_seconds = timeout_seconds
        self.enable_connection_pooling = enable_connection_pooling
        self.enable_lazy_loading = enable_lazy_loading
        
        self.connection_pool = ConnectionPool() if enable_connection_pooling else None
        self.cold_start_optimizer = ColdStartOptimizer()
        self.memory_optimizer = MemoryOptimizer()
    
    def initialize(self):
        """Inicializa la configuración serverless"""
        self.cold_start_optimizer.initialize()
        logger.info("Serverless configuration initialized")
    
    def check_memory_limit(self) -> bool:
        """Verifica si estamos dentro del límite de memoria"""
        usage = self.memory_optimizer.get_memory_usage()
        return usage < self.max_memory_mb
    
    def optimize_if_needed(self):
        """Optimiza si es necesario"""
        if not self.check_memory_limit():
            self.memory_optimizer.clear_cache()
            logger.warning("Memory limit approaching, cache cleared")


# Singleton para configuración serverless
_serverless_config: Optional[ServerlessConfig] = None


def get_serverless_config() -> ServerlessConfig:
    """Obtiene la configuración serverless global"""
    global _serverless_config
    if _serverless_config is None:
        _serverless_config = ServerlessConfig()
        _serverless_config.initialize()
    return _serverless_config


def serverless_handler(func: Callable) -> Callable:
    """Decorator para handlers serverless que aplica optimizaciones"""
    
    @wraps(func)
    async def wrapper(*args, **kwargs):
        config = get_serverless_config()
        
        # Verificar memoria
        config.optimize_if_needed()
        
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            # Limpiar después de la ejecución
            if config.enable_connection_pooling:
                config.memory_optimizer.clear_cache()
    
    return wrapper




