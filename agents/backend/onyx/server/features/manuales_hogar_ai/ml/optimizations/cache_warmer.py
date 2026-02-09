"""
Cache Warmer
============

Pre-calentar caches para acceso inmediato.
"""

import logging
import asyncio
from typing import List, Optional, Callable, Any, Dict
import time

logger = logging.getLogger(__name__)


class CacheWarmer:
    """Pre-calentador de caches."""
    
    def __init__(
        self,
        warmup_items: List[Any],
        warmup_fn: Callable,
        batch_size: int = 8
    ):
        """
        Inicializar cache warmer.
        
        Args:
            warmup_items: Items a pre-calentar
            warmup_fn: Función de warmup
            batch_size: Tamaño de batch
        """
        self.warmup_items = warmup_items
        self.warmup_fn = warmup_fn
        self.batch_size = batch_size
        self.warmed = False
        self._logger = logger
    
    async def warmup(self):
        """Pre-calentar cache."""
        if self.warmed:
            return
        
        try:
            self._logger.info(f"Pre-calentando cache con {len(self.warmup_items)} items...")
            
            # Procesar en batches
            for i in range(0, len(self.warmup_items), self.batch_size):
                batch = self.warmup_items[i:i + self.batch_size]
                await asyncio.to_thread(self.warmup_fn, batch)
            
            self.warmed = True
            self._logger.info("Cache pre-calentado exitosamente")
        
        except Exception as e:
            self._logger.error(f"Error pre-calentando cache: {str(e)}")
    
    def warmup_sync(self):
        """Pre-calentar de forma síncrona."""
        if self.warmed:
            return
        
        try:
            self._logger.info(f"Pre-calentando cache (sync) con {len(self.warmup_items)} items...")
            
            for i in range(0, len(self.warmup_items), self.batch_size):
                batch = self.warmup_items[i:i + self.batch_size]
                self.warmup_fn(batch)
            
            self.warmed = True
            self._logger.info("Cache pre-calentado exitosamente")
        
        except Exception as e:
            self._logger.error(f"Error pre-calentando cache: {str(e)}")




