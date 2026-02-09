"""
Intelligent Prefetching
=======================

Prefetching inteligente basado en predicción.
"""

import logging
import asyncio
from typing import List, Optional, Callable, Any, Dict
from collections import deque
import time

logger = logging.getLogger(__name__)


class IntelligentPrefetcher:
    """Prefetcher inteligente con predicción."""
    
    def __init__(
        self,
        prefetch_fn: Callable,
        prediction_window: int = 5,
        prefetch_count: int = 2
    ):
        """
        Inicializar prefetcher.
        
        Args:
            prefetch_fn: Función de prefetch
            prediction_window: Ventana de predicción
            prefetch_count: Número de items a prefetch
        """
        self.prefetch_fn = prefetch_fn
        self.prediction_window = prediction_window
        self.prefetch_count = prefetch_count
        
        self.access_pattern: deque = deque(maxlen=prediction_window)
        self.prefetched_items: Dict[str, Any] = {}
        self.prefetch_tasks: List[asyncio.Task] = []
        self._logger = logger
    
    def record_access(self, item_id: str):
        """
        Registrar acceso a item.
        
        Args:
            item_id: ID del item
        """
        self.access_pattern.append(item_id)
        
        # Predecir siguiente acceso
        if len(self.access_pattern) >= 2:
            self._predict_and_prefetch()
    
    def _predict_and_prefetch(self):
        """Predecir y prefetch items."""
        if len(self.access_pattern) < 2:
            return
        
        # Patrón simple: siguiente item probable
        recent = list(self.access_pattern)[-2:]
        
        # Prefetch basado en patrón
        predicted_items = self._predict_next(recent)
        
        for item_id in predicted_items[:self.prefetch_count]:
            if item_id not in self.prefetched_items:
                task = asyncio.create_task(self._prefetch_item(item_id))
                self.prefetch_tasks.append(task)
    
    def _predict_next(self, recent: List[str]) -> List[str]:
        """
        Predecir siguientes items.
        
        Args:
            recent: Items recientes
        
        Returns:
            Items predichos
        """
        # Lógica simple: siguiente en secuencia o similar
        # En implementación real, usar ML para predicción
        if len(recent) >= 2:
            # Patrón: siguiente ID
            last_id = recent[-1]
            try:
                # Asumir IDs numéricos
                next_id = str(int(last_id) + 1)
                return [next_id]
            except:
                pass
        
        return []
    
    async def _prefetch_item(self, item_id: str):
        """Prefetch item."""
        try:
            result = await asyncio.to_thread(self.prefetch_fn, item_id)
            self.prefetched_items[item_id] = result
            self._logger.debug(f"Item prefetched: {item_id}")
        except Exception as e:
            self._logger.warning(f"Error prefetching {item_id}: {str(e)}")
    
    async def get_item(self, item_id: str) -> Any:
        """
        Obtener item (usar prefetched si disponible).
        
        Args:
            item_id: ID del item
        
        Returns:
            Item
        """
        # Registrar acceso
        self.record_access(item_id)
        
        # Verificar si está prefetched
        if item_id in self.prefetched_items:
            item = self.prefetched_items.pop(item_id)
            self._logger.debug(f"Item obtenido de cache prefetch: {item_id}")
            return item
        
        # Obtener normalmente
        return await asyncio.to_thread(self.prefetch_fn, item_id)
    
    def clear_prefetched(self):
        """Limpiar items prefetched."""
        self.prefetched_items.clear()
        for task in self.prefetch_tasks:
            task.cancel()
        self.prefetch_tasks.clear()




