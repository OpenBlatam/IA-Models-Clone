"""
Dynamic Config
==============

Dynamic configuration management.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ConfigChange:
    """Configuration change."""
    key: str
    old_value: Any
    new_value: Any
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class DynamicConfig:
    """Dynamic configuration manager."""
    
    def __init__(self):
        self._config: Dict[str, Any] = {}
        self._watchers: Dict[str, List[Callable]] = {}
        self._change_history: List[ConfigChange] = []
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        old_value = self._config.get(key)
        
        if old_value != value:
            change = ConfigChange(
                key=key,
                old_value=old_value,
                new_value=value
            )
            
            self._change_history.append(change)
            self._config[key] = value
            
            # Notify watchers
            if key in self._watchers:
                for watcher in self._watchers[key]:
                    try:
                        if asyncio.iscoroutinefunction(watcher):
                            asyncio.create_task(watcher(key, value, old_value))
                        else:
                            watcher(key, value, old_value)
                    except Exception as e:
                        logger.error(f"Config watcher failed: {e}")
            
            logger.info(f"Config changed: {key}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)
    
    def watch(self, key: str, callback: Callable):
        """Watch configuration key for changes."""
        if key not in self._watchers:
            self._watchers[key] = []
        
        self._watchers[key].append(callback)
        logger.info(f"Added watcher for config key: {key}")
    
    def unwatch(self, key: str, callback: Callable):
        """Remove watcher."""
        if key in self._watchers:
            self._watchers[key] = [
                w for w in self._watchers[key]
                if w != callback
            ]
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration."""
        return self._config.copy()
    
    def get_change_history(self, key: Optional[str] = None, limit: int = 100) -> List[ConfigChange]:
        """Get configuration change history."""
        changes = self._change_history
        
        if key:
            changes = [c for c in changes if c.key == key]
        
        return changes[-limit:]
    
    def get_config_stats(self) -> Dict[str, Any]:
        """Get configuration statistics."""
        return {
            "total_keys": len(self._config),
            "total_watchers": sum(len(watchers) for watchers in self._watchers.values()),
            "total_changes": len(self._change_history)
        }















