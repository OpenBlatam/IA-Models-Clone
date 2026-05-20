"""
Cache Warming System for Color Grading AI
==========================================

Intelligent cache warming and preloading system.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class WarmingStrategy(Enum):
    """Cache warming strategies."""
    POPULAR = "popular"  # Warm popular items
    RECENT = "recent"  # Warm recent items
    PREDICTIVE = "predictive"  # Predictive warming
    MANUAL = "manual"  # Manual selection
    FULL = "full"  # Warm all items


@dataclass
class WarmingTask:
    """Cache warming task."""
    key: str
    value: Any
    priority: int = 0
    ttl: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WarmingResult:
    """Cache warming result."""
    success: bool
    items_warmed: int
    errors: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class CacheWarmingSystem:
    """
    Cache warming system.
    
    Features:
    - Multiple warming strategies
    - Priority-based warming
    - Predictive warming
    - Background warming
    - Warming statistics
    """
    
    def __init__(self, cache_system: Any):
        """
        Initialize cache warming system.
        
        Args:
            cache_system: Cache system instance
        """
        self.cache_system = cache_system
        self._warming_history: List[WarmingResult] = []
        self._popular_items: Dict[str, int] = {}  # key -> access count
    
    async def warm_cache(
        self,
        items: List[WarmingTask],
        strategy: WarmingStrategy = WarmingStrategy.MANUAL
    ) -> WarmingResult:
        """
        Warm cache with items.
        
        Args:
            items: List of warming tasks
            strategy: Warming strategy
            
        Returns:
            Warming result
        """
        start_time = datetime.now()
        warmed_count = 0
        errors = []
        
        # Sort by priority if needed
        if strategy == WarmingStrategy.POPULAR:
            items = sorted(items, key=lambda x: self._popular_items.get(x.key, 0), reverse=True)
        elif strategy == WarmingStrategy.RECENT:
            items = sorted(items, key=lambda x: x.metadata.get("timestamp", 0), reverse=True)
        
        # Warm items
        for task in items:
            try:
                await self.cache_system.set(
                    task.key,
                    task.value,
                    ttl=task.ttl
                )
                warmed_count += 1
                logger.debug(f"Warmed cache key: {task.key}")
            except Exception as e:
                errors.append(f"Error warming {task.key}: {str(e)}")
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        result = WarmingResult(
            success=len(errors) == 0,
            items_warmed=warmed_count,
            errors=errors,
            execution_time=execution_time
        )
        
        self._warming_history.append(result)
        logger.info(f"Cache warming completed: {warmed_count} items warmed")
        
        return result
    
    def track_access(self, key: str):
        """
        Track cache access for popularity.
        
        Args:
            key: Cache key
        """
        self._popular_items[key] = self._popular_items.get(key, 0) + 1
    
    def get_popular_items(self, limit: int = 100) -> List[str]:
        """
        Get popular cache items.
        
        Args:
            limit: Result limit
            
        Returns:
            List of popular keys
        """
        sorted_items = sorted(
            self._popular_items.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [key for key, _ in sorted_items[:limit]]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get warming statistics."""
        if not self._warming_history:
            return {
                "total_warming_operations": 0,
            }
        
        total_items = sum(r.items_warmed for r in self._warming_history)
        return {
            "total_warming_operations": len(self._warming_history),
            "total_items_warmed": total_items,
            "avg_items_per_warming": total_items / len(self._warming_history) if self._warming_history else 0,
            "popular_items_count": len(self._popular_items),
        }


