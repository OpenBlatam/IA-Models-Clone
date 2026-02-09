"""
Warm Up Manager
===============

Manage Lambda warm-up requests.
"""

import logging
from typing import Dict, Any, Optional
import asyncio

logger = logging.getLogger(__name__)


class WarmUpManager:
    """Manage Lambda warm-up."""
    
    def __init__(self):
        self._warm_up_tasks: Dict[str, Any] = {}
    
    def is_warm_up_request(self, event: Dict[str, Any]) -> bool:
        """Check if request is a warm-up request."""
        return (
            event.get("source") == "aws.events" or
            event.get("warmup") is True or
            event.get("httpMethod") == "WARMUP"
        )
    
    async def warm_up(self, tasks: Optional[Dict[str, callable]] = None):
        """Execute warm-up tasks."""
        if not tasks:
            tasks = {}
        
        warm_up_results = {}
        
        for name, task in tasks.items():
            try:
                if asyncio.iscoroutinefunction(task):
                    result = await task()
                else:
                    result = task()
                warm_up_results[name] = {"status": "success", "result": result}
            except Exception as e:
                warm_up_results[name] = {"status": "error", "error": str(e)}
                logger.error(f"Warm-up task {name} failed: {e}")
        
        logger.info(f"Warm-up completed: {warm_up_results}")
        return warm_up_results
    
    def register_warm_up_task(self, name: str, task: callable):
        """Register warm-up task."""
        self._warm_up_tasks[name] = task
        logger.debug(f"Registered warm-up task: {name}")
    
    async def execute_warm_up(self):
        """Execute all registered warm-up tasks."""
        return await self.warm_up(self._warm_up_tasks)















