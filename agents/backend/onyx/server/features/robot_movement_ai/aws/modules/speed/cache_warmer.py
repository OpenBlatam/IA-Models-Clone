"""
Cache Warmer
============

Pre-warm cache for faster responses.
"""

import logging
import asyncio
from typing import Dict, Any, List, Callable, Optional
from aws.modules.ports.cache_port import CachePort

logger = logging.getLogger(__name__)


class CacheWarmer:
    """Cache warmer for pre-loading frequently accessed data."""
    
    def __init__(self, cache: CachePort):
        self.cache = cache
        self._warm_up_tasks: List[Dict[str, Any]] = []
    
    def register_warm_up(
        self,
        key: str,
        loader: Callable,
        ttl: Optional[int] = None,
        priority: int = 0
    ):
        """Register cache warm-up task."""
        self._warm_up_tasks.append({
            "key": key,
            "loader": loader,
            "ttl": ttl,
            "priority": priority
        })
        logger.debug(f"Registered cache warm-up: {key}")
    
    async def warm_up(self, parallel: bool = True):
        """Execute all warm-up tasks."""
        # Sort by priority
        tasks = sorted(self._warm_up_tasks, key=lambda x: x["priority"], reverse=True)
        
        if parallel:
            # Execute in parallel
            warm_up_coros = [self._warm_single(task) for task in tasks]
            results = await asyncio.gather(*warm_up_coros, return_exceptions=True)
        else:
            # Execute sequentially
            results = []
            for task in tasks:
                result = await self._warm_single(task)
                results.append(result)
        
        success_count = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
        logger.info(f"Cache warm-up completed: {success_count}/{len(tasks)} successful")
        
        return results
    
    async def _warm_single(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Warm up single cache entry."""
        try:
            loader = task["loader"]
            
            # Load data
            if asyncio.iscoroutinefunction(loader):
                data = await loader()
            else:
                data = loader()
            
            # Cache data
            await self.cache.set(
                task["key"],
                data,
                ttl=task.get("ttl")
            )
            
            logger.debug(f"Warmed cache: {task['key']}")
            return {"success": True, "key": task["key"]}
        
        except Exception as e:
            logger.error(f"Cache warm-up failed for {task['key']}: {e}")
            return {"success": False, "key": task["key"], "error": str(e)}
    
    async def warm_up_pattern(self, pattern: str, loader: Callable):
        """Warm up cache entries matching pattern."""
        # In production, implement pattern-based warm-up
        logger.info(f"Warming cache pattern: {pattern}")
        return True
    
    def get_warm_up_stats(self) -> Dict[str, Any]:
        """Get warm-up statistics."""
        return {
            "registered_tasks": len(self._warm_up_tasks),
            "tasks": [
                {
                    "key": task["key"],
                    "priority": task["priority"],
                    "has_ttl": task.get("ttl") is not None
                }
                for task in self._warm_up_tasks
            ]
        }















