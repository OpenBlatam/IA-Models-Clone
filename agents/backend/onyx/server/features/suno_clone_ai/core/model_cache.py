"""
Model Cache
Cache inteligente de modelos ML para reducir carga
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Callable
from collections import OrderedDict
import time

logger = logging.getLogger(__name__)


class ModelCache:
    """Cache de modelos ML"""
    
    def __init__(self, max_models: int = 3, max_idle_time: float = 3600.0):
        self.max_models = max_models
        self.max_idle_time = max_idle_time
        self._models: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._access_times: Dict[str, float] = {}
        self._lock = asyncio.Lock()
    
    async def get_model(
        self,
        model_key: str,
        loader: Callable[[], Any],
        *args,
        **kwargs
    ) -> Any:
        """
        Obtiene modelo del cache o lo carga
        
        Args:
            model_key: Clave única del modelo
            loader: Función que carga el modelo
            *args, **kwargs: Argumentos para loader
            
        Returns:
            Modelo cargado
        """
        async with self._lock:
            # Verificar si está en cache
            if model_key in self._models:
                model_data = self._models[model_key]
                
                # Verificar si no ha expirado
                if time.time() - self._access_times[model_key] < self.max_idle_time:
                    # Mover al final (LRU)
                    self._models.move_to_end(model_key)
                    self._access_times[model_key] = time.time()
                    logger.debug(f"Model cache hit: {model_key}")
                    return model_data["model"]
                else:
                    # Expirar
                    del self._models[model_key]
                    del self._access_times[model_key]
            
            # Cargar modelo
            logger.info(f"Loading model: {model_key}")
            model = await self._load_model(loader, *args, **kwargs)
            
            # Agregar al cache
            self._add_to_cache(model_key, model)
            
            return model
    
    async def _load_model(self, loader: Callable, *args, **kwargs) -> Any:
        """Carga modelo (puede ser async o sync)"""
        if asyncio.iscoroutinefunction(loader):
            return await loader(*args, **kwargs)
        else:
            # Ejecutar en thread pool si es sync
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, loader, *args, **kwargs)
    
    def _add_to_cache(self, model_key: str, model: Any):
        """Agrega modelo al cache"""
        # Limpiar si está lleno
        if len(self._models) >= self.max_models:
            # Eliminar el menos usado (LRU)
            oldest_key = next(iter(self._models))
            del self._models[oldest_key]
            if oldest_key in self._access_times:
                del self._access_times[oldest_key]
            logger.debug(f"Evicted model from cache: {oldest_key}")
        
        self._models[model_key] = {
            "model": model,
            "loaded_at": time.time()
        }
        self._access_times[model_key] = time.time()
    
    async def preload_model(self, model_key: str, loader: Callable, *args, **kwargs):
        """Pre-carga modelo en background"""
        asyncio.create_task(
            self.get_model(model_key, loader, *args, **kwargs)
        )
    
    async def unload_model(self, model_key: str):
        """Descarga modelo del cache"""
        async with self._lock:
            if model_key in self._models:
                # Limpiar recursos del modelo si es necesario
                model = self._models[model_key]["model"]
                if hasattr(model, 'cpu'):
                    model.cpu()  # Mover a CPU para liberar GPU
                del self._models[model_key]
                if model_key in self._access_times:
                    del self._access_times[model_key]
                logger.info(f"Unloaded model: {model_key}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del cache"""
        return {
            "cached_models": len(self._models),
            "max_models": self.max_models,
            "models": list(self._models.keys())
        }


# Instancia global
_model_cache: Optional[ModelCache] = None


def get_model_cache() -> ModelCache:
    """Obtiene el cache de modelos"""
    global _model_cache
    if _model_cache is None:
        _model_cache = ModelCache()
    return _model_cache















