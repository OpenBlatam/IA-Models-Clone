"""
Manager Mixins for Piel Mejorador AI SAM3
=========================================

Common mixins for managers to reduce duplication.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class StatsMixin:
    """Mixin for statistics tracking."""
    
    def __init__(self, *args, **kwargs):
        """Initialize stats mixin."""
        super().__init__(*args, **kwargs)
        self._stats: Dict[str, Any] = {}
        self._init_stats()
    
    def _init_stats(self):
        """Initialize statistics dictionary."""
        self._stats = {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "created_at": datetime.now().isoformat(),
        }
    
    def record_success(self):
        """Record successful operation."""
        self._stats["total"] += 1
        self._stats["successful"] += 1
    
    def record_failure(self):
        """Record failed operation."""
        self._stats["total"] += 1
        self._stats["failed"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        stats = self._stats.copy()
        if stats["total"] > 0:
            stats["success_rate"] = stats["successful"] / stats["total"]
        else:
            stats["success_rate"] = 0.0
        return stats


class LockMixin:
    """Mixin for async locking."""
    
    def __init__(self, *args, **kwargs):
        """Initialize lock mixin."""
        super().__init__(*args, **kwargs)
        self._lock = asyncio.Lock()
    
    async def _acquire_lock(self):
        """Acquire lock."""
        return await self._lock.acquire()
    
    def _release_lock(self):
        """Release lock."""
        if self._lock.locked():
            self._lock.release()


class ClientMixin:
    """Mixin for HTTP client management."""
    
    def __init__(self, *args, **kwargs):
        """Initialize client mixin."""
        super().__init__(*args, **kwargs)
        self._client: Optional[Any] = None
        self._client_lock = asyncio.Lock()
    
    async def _get_client(self):
        """Get or create client (to be implemented by subclass)."""
        raise NotImplementedError
    
    async def _close_client(self):
        """Close client."""
        if self._client:
            try:
                if hasattr(self._client, "aclose"):
                    await self._client.aclose()
                elif hasattr(self._client, "close"):
                    self._client.close()
            except Exception as e:
                logger.warning(f"Error closing client: {e}")
            finally:
                self._client = None


class LifecycleMixin:
    """Mixin for lifecycle management."""
    
    def __init__(self, *args, **kwargs):
        """Initialize lifecycle mixin."""
        super().__init__(*args, **kwargs)
        self._running = False
        self._started_at: Optional[datetime] = None
        self._stopped_at: Optional[datetime] = None
    
    async def start(self):
        """Start the component."""
        if self._running:
            logger.warning(f"{self.__class__.__name__} is already running")
            return
        
        self._running = True
        self._started_at = datetime.now()
        logger.info(f"{self.__class__.__name__} started")
    
    async def stop(self):
        """Stop the component."""
        if not self._running:
            return
        
        self._running = False
        self._stopped_at = datetime.now()
        logger.info(f"{self.__class__.__name__} stopped")
    
    def is_running(self) -> bool:
        """Check if component is running."""
        return self._running
    
    def get_uptime(self) -> Optional[float]:
        """Get uptime in seconds."""
        if not self._running or not self._started_at:
            return None
        
        return (datetime.now() - self._started_at).total_seconds()




