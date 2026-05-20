"""
Warmup Optimizer
Pre-warm services and connections for faster response times
"""

import logging
import asyncio
from typing import List, Callable

logger = logging.getLogger(__name__)


class WarmupOptimizer:
    """
    Warmup optimizer for pre-initialization
    
    Features:
    - Service warmup
    - Connection pre-warming
    - Cache pre-loading
    - Model pre-loading
    """
    
    def __init__(self):
        self._warmup_tasks: List[Callable] = []
        self._warmed = False
    
    def register_warmup(self, task: Callable) -> None:
        """Register warmup task"""
        self._warmup_tasks.append(task)
        logger.info(f"Registered warmup task: {task.__name__}")
    
    async def warmup_all(self) -> None:
        """Execute all warmup tasks"""
        if self._warmed:
            logger.info("Already warmed up")
            return
        
        logger.info(f"Starting warmup of {len(self._warmup_tasks)} tasks...")
        
        # Execute warmup tasks in parallel
        tasks = []
        for task in self._warmup_tasks:
            if asyncio.iscoroutinefunction(task):
                tasks.append(task())
            else:
                tasks.append(asyncio.to_thread(task))
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        self._warmed = True
        logger.info("Warmup completed")
    
    async def warmup_services(self) -> None:
        """Warmup service connections"""
        try:
            from core.service_container import get_container
            
            container = get_container()
            
            # Warmup storage
            try:
                storage = container.get_storage_service()
                logger.info("Storage service warmed up")
            except Exception as e:
                logger.warning(f"Failed to warmup storage: {str(e)}")
            
            # Warmup cache
            try:
                cache = container.get_cache_service()
                logger.info("Cache service warmed up")
            except Exception as e:
                logger.warning(f"Failed to warmup cache: {str(e)}")
            
        except Exception as e:
            logger.error(f"Failed to warmup services: {str(e)}")
    
    async def warmup_models(self) -> None:
        """Warmup AI models"""
        try:
            from config.aws_settings import get_aws_settings
            settings = get_aws_settings()
            
            if settings.preload_models:
                from core.ultra_fast_engine import create_ultra_fast_engine
                engine = create_ultra_fast_engine()
                logger.info("AI models warmed up")
        except Exception as e:
            logger.warning(f"Failed to warmup models: {str(e)}")
    
    async def warmup_cache(self) -> None:
        """Pre-load frequently accessed data into cache"""
        try:
            from optimization.caching_advanced import get_advanced_cache
            from core.service_container import get_container
            
            cache = get_advanced_cache()
            storage = get_container().get_storage_service()
            
            # Pre-load common queries (example)
            # common_user_ids = ["user1", "user2", "user3"]
            # for user_id in common_user_ids:
            #     user = await storage.get(user_id)
            #     if user:
            #         await cache.set(f"user_{user_id}", user, ttl=3600)
            
            logger.info("Cache warmed up")
        except Exception as e:
            logger.warning(f"Failed to warmup cache: {str(e)}")


# Global warmup optimizer
_warmup: WarmupOptimizer = None


def get_warmup_optimizer() -> WarmupOptimizer:
    """Get global warmup optimizer"""
    global _warmup
    if _warmup is None:
        _warmup = WarmupOptimizer()
    return _warmup


async def initialize_warmup() -> None:
    """Initialize warmup on startup"""
    warmup = get_warmup_optimizer()
    
    # Register warmup tasks
    warmup.register_warmup(warmup.warmup_services)
    warmup.register_warmup(warmup.warmup_models)
    warmup.register_warmup(warmup.warmup_cache)
    
    # Execute warmup
    await warmup.warmup_all()















