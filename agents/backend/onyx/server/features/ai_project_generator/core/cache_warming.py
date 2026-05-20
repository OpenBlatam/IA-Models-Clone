"""
Cache Warming - Cache Warming
=============================

Sistema de cache warming:
- Preload strategies
- Predictive caching
- Cache warming on startup
- Background cache refresh
"""

import logging
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class WarmingStrategy(str, Enum):
    """Estrategias de cache warming"""
    ON_STARTUP = "on_startup"
    SCHEDULED = "scheduled"
    PREDICTIVE = "predictive"
    ON_DEMAND = "on_demand"


class CacheWarmer:
    """
    Sistema de cache warming.
    """
    
    def __init__(
        self,
        cache_service: Optional[Any] = None,
        strategy: WarmingStrategy = WarmingStrategy.ON_STARTUP
    ) -> None:
        self.cache_service = cache_service
        self.strategy = strategy
        self.warming_tasks: List[Callable] = []
        self.warmed_keys: List[str] = []
    
    def register_warming_task(
        self,
        cache_key: str,
        loader: Callable,
        ttl: Optional[int] = None
    ) -> None:
        """Registra tarea de warming"""
        self.warming_tasks.append({
            "key": cache_key,
            "loader": loader,
            "ttl": ttl
        })
        logger.info(f"Cache warming task registered: {cache_key}")
    
    async def warm_cache(self) -> Dict[str, Any]:
        """Ejecuta cache warming"""
        results = {
            "warmed": 0,
            "failed": 0,
            "keys": []
        }
        
        for task in self.warming_tasks:
            try:
                # Ejecutar loader
                if asyncio.iscoroutinefunction(task["loader"]):
                    value = await task["loader"]()
                else:
                    value = task["loader"]()
                
                # Guardar en cache
                if self.cache_service:
                    await self.cache_service.set(
                        task["key"],
                        value,
                        ttl=task.get("ttl")
                    )
                
                results["warmed"] += 1
                results["keys"].append(task["key"])
                self.warmed_keys.append(task["key"])
                logger.info(f"Cache warmed: {task['key']}")
                
            except Exception as e:
                results["failed"] += 1
                logger.error(f"Cache warming failed for {task['key']}: {e}")
        
        return results
    
    async def warm_specific_keys(self, keys: List[str]) -> Dict[str, Any]:
        """Warm keys específicos"""
        tasks = [t for t in self.warming_tasks if t["key"] in keys]
        original_tasks = self.warming_tasks
        self.warming_tasks = tasks
        
        try:
            results = await self.warm_cache()
        finally:
            self.warming_tasks = original_tasks
        
        return results
    
    def get_warmed_keys(self) -> List[str]:
        """Obtiene keys que han sido warmed"""
        return self.warmed_keys.copy()


import asyncio


def get_cache_warmer(
    cache_service: Optional[Any] = None,
    strategy: WarmingStrategy = WarmingStrategy.ON_STARTUP
) -> CacheWarmer:
    """Obtiene cache warmer"""
    return CacheWarmer(cache_service=cache_service, strategy=strategy)















