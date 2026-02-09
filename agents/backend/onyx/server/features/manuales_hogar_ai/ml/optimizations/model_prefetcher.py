"""
Prefetcher de Modelos
=====================

Carga y mantiene modelos en memoria para acceso rápido.
"""

import logging
import asyncio
from typing import Dict, Optional, Any, Callable
import weakref

logger = logging.getLogger(__name__)


class ModelPrefetcher:
    """Prefetcher para mantener modelos en memoria."""
    
    def __init__(self, max_models: int = 3):
        """
        Inicializar prefetcher.
        
        Args:
            max_models: Número máximo de modelos en memoria
        """
        self.max_models = max_models
        self._models: Dict[str, Any] = {}
        self._access_times: Dict[str, float] = {}
        self._load_functions: Dict[str, Callable] = {}
        self._logger = logger
    
    def register_model(
        self,
        model_id: str,
        load_function: Callable,
        preload: bool = False
    ):
        """
        Registrar modelo para prefetching.
        
        Args:
            model_id: ID del modelo
            load_function: Función para cargar el modelo
            preload: Precargar inmediatamente
        """
        self._load_functions[model_id] = load_function
        
        if preload:
            asyncio.create_task(self._load_model(model_id))
    
    async def _load_model(self, model_id: str):
        """Cargar modelo."""
        try:
            if model_id in self._models:
                return self._models[model_id]
            
            # Si hay demasiados modelos, eliminar el menos usado
            if len(self._models) >= self.max_models:
                await self._evict_oldest()
            
            # Cargar modelo
            load_fn = self._load_functions.get(model_id)
            if load_fn:
                model = await asyncio.to_thread(load_fn)
                self._models[model_id] = model
                self._access_times[model_id] = asyncio.get_event_loop().time()
                self._logger.info(f"Modelo {model_id} cargado en memoria")
                return model
        
        except Exception as e:
            self._logger.error(f"Error cargando modelo {model_id}: {str(e)}")
            return None
    
    async def get_model(self, model_id: str) -> Optional[Any]:
        """
        Obtener modelo (cargar si no está en memoria).
        
        Args:
            model_id: ID del modelo
        
        Returns:
            Modelo o None
        """
        if model_id in self._models:
            self._access_times[model_id] = asyncio.get_event_loop().time()
            return self._models[model_id]
        
        return await self._load_model(model_id)
    
    async def _evict_oldest(self):
        """Eliminar el modelo menos usado recientemente."""
        if not self._access_times:
            return
        
        oldest_id = min(self._access_times.items(), key=lambda x: x[1])[0]
        del self._models[oldest_id]
        del self._access_times[oldest_id]
        self._logger.info(f"Modelo {oldest_id} eliminado de memoria")
    
    def clear(self):
        """Limpiar todos los modelos."""
        self._models.clear()
        self._access_times.clear()
        self._logger.info("Todos los modelos eliminados de memoria")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            "loaded_models": len(self._models),
            "max_models": self.max_models,
            "model_ids": list(self._models.keys())
        }




