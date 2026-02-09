"""
Config Reloader
===============

Configuration reload management.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ReloadEvent:
    """Configuration reload event."""
    source: str
    timestamp: datetime
    success: bool
    changes: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.changes is None:
            self.changes = {}


class ConfigReloader:
    """Configuration reloader."""
    
    def __init__(self, config_manager: Any):
        self.config_manager = config_manager
        self._reload_handlers: List[Callable] = []
        self._reload_history: List[ReloadEvent] = []
        self._auto_reload = False
        self._reload_interval = 60.0
    
    def register_reload_handler(self, handler: Callable):
        """Register reload handler."""
        self._reload_handlers.append(handler)
        logger.info("Registered config reload handler")
    
    async def reload(self, source: str = "manual") -> ReloadEvent:
        """Reload configuration."""
        try:
            # In production, load from actual source
            # This is a placeholder
            changes = {}
            
            # Notify handlers
            for handler in self._reload_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(changes)
                    else:
                        handler(changes)
                except Exception as e:
                    logger.error(f"Reload handler failed: {e}")
            
            event = ReloadEvent(
                source=source,
                success=True,
                changes=changes
            )
            
            self._reload_history.append(event)
            logger.info(f"Configuration reloaded from {source}")
            return event
        
        except Exception as e:
            event = ReloadEvent(
                source=source,
                success=False
            )
            
            self._reload_history.append(event)
            logger.error(f"Configuration reload failed: {e}")
            return event
    
    def start_auto_reload(self, interval: float = 60.0):
        """Start automatic reload."""
        self._auto_reload = True
        self._reload_interval = interval
        
        async def auto_reload_loop():
            while self._auto_reload:
                await self.reload(source="auto")
                await asyncio.sleep(interval)
        
        asyncio.create_task(auto_reload_loop())
        logger.info(f"Started auto-reload with interval {interval}s")
    
    def stop_auto_reload(self):
        """Stop automatic reload."""
        self._auto_reload = False
        logger.info("Stopped auto-reload")
    
    def get_reload_history(self, limit: int = 100) -> List[ReloadEvent]:
        """Get reload history."""
        return self._reload_history[-limit:]
    
    def get_reload_stats(self) -> Dict[str, Any]:
        """Get reload statistics."""
        return {
            "total_reloads": len(self._reload_history),
            "successful": sum(1 for e in self._reload_history if e.success),
            "failed": sum(1 for e in self._reload_history if not e.success),
            "auto_reload": self._auto_reload
        }















