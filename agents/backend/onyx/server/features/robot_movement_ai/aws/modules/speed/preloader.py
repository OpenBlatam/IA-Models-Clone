"""
Preloader
=========

Pre-load resources for faster access.
"""

import logging
import asyncio
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PreloadTask:
    """Preload task definition."""
    name: str
    loader: Callable
    priority: int = 0
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class Preloader:
    """Pre-loader for resources."""
    
    def __init__(self):
        self._tasks: Dict[str, PreloadTask] = {}
        self._loaded: Dict[str, Any] = {}
        self._loading: Dict[str, asyncio.Task] = {}
    
    def register(
        self,
        name: str,
        loader: Callable,
        priority: int = 0,
        dependencies: Optional[List[str]] = None
    ):
        """Register preload task."""
        self._tasks[name] = PreloadTask(
            name=name,
            loader=loader,
            priority=priority,
            dependencies=dependencies or []
        )
        logger.debug(f"Registered preload task: {name}")
    
    async def preload_all(self, parallel: bool = True):
        """Preload all registered resources."""
        # Sort by priority
        tasks = sorted(
            self._tasks.values(),
            key=lambda x: x.priority,
            reverse=True
        )
        
        if parallel:
            # Load in parallel (respecting dependencies)
            await self._preload_parallel(tasks)
        else:
            # Load sequentially
            await self._preload_sequential(tasks)
        
        logger.info(f"Preloaded {len(self._loaded)} resources")
    
    async def _preload_parallel(self, tasks: List[PreloadTask]):
        """Preload in parallel."""
        # Group by dependency level
        levels = self._group_by_dependencies(tasks)
        
        for level in levels:
            # Load all tasks in this level in parallel
            coros = [self._load_single(task) for task in level]
            await asyncio.gather(*coros, return_exceptions=True)
    
    async def _preload_sequential(self, tasks: List[PreloadTask]):
        """Preload sequentially."""
        for task in tasks:
            await self._load_single(task)
    
    async def _load_single(self, task: PreloadTask):
        """Load single resource."""
        # Check dependencies
        for dep in task.dependencies:
            if dep not in self._loaded:
                logger.warning(f"Task {task.name} depends on {dep} which is not loaded")
        
        try:
            loader = task.loader
            
            if asyncio.iscoroutinefunction(loader):
                result = await loader()
            else:
                result = loader()
            
            self._loaded[task.name] = result
            logger.debug(f"Preloaded: {task.name}")
        
        except Exception as e:
            logger.error(f"Preload failed for {task.name}: {e}")
    
    def _group_by_dependencies(self, tasks: List[PreloadTask]) -> List[List[PreloadTask]]:
        """Group tasks by dependency level."""
        levels = []
        remaining = set(tasks)
        loaded = set()
        
        while remaining:
            # Find tasks with no unloaded dependencies
            level = []
            for task in list(remaining):
                if all(dep in loaded for dep in task.dependencies):
                    level.append(task)
            
            if not level:
                # Circular dependency or missing dependency
                logger.warning("Circular or missing dependencies detected")
                level = list(remaining)
            
            levels.append(level)
            for task in level:
                remaining.remove(task)
                loaded.add(task.name)
        
        return levels
    
    def get(self, name: str) -> Optional[Any]:
        """Get preloaded resource."""
        return self._loaded.get(name)
    
    def is_loaded(self, name: str) -> bool:
        """Check if resource is loaded."""
        return name in self._loaded
    
    def get_stats(self) -> Dict[str, Any]:
        """Get preload statistics."""
        return {
            "registered": len(self._tasks),
            "loaded": len(self._loaded),
            "loading": len(self._loading),
            "resources": list(self._loaded.keys())
        }















