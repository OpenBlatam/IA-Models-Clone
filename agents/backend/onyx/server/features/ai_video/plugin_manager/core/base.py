from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

from typing import Any, Dict
from abc import ABC, abstractmethod
from onyx.utils.logger import setup_logger
from onyx.utils.threadpool_concurrency import ThreadSafeDict
from onyx.utils.telemetry import TelemetryLogger
from onyx.utils.gpu_utils import is_gpu_available
from .models import OnyxPluginInfo, OnyxPluginContext
from ..core.exceptions import PluginError
                from datetime import datetime, timedelta
        from datetime import datetime
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Onyx Plugin Manager - Base Plugin Class

Base class for Onyx plugins providing common functionality and integration
with Onyx's infrastructure for plugin development.
"""


# Onyx imports

# Local imports


class OnyxPluginBase(ABC):
    """
    Base class for Onyx plugins.
    
    Provides common functionality and integration with Onyx's
    infrastructure for plugin development.
    """
    
    def __init__(self, config: Dict[str, Any]):
        
    """__init__ function."""
self.config = config
        self.logger = setup_logger(f"onyx_plugin.{self.__class__.__name__}")
        self.telemetry = TelemetryLogger()
        self.cache: ThreadSafeDict[str, Any] = ThreadSafeDict()
    
    async def initialize(self) -> None:
        """Initialize the plugin."""
        try:
            self.logger.info(f"Initializing plugin: {self.__class__.__name__}")
            
            # Check GPU requirement
            if getattr(self, 'gpu_required', False) and not is_gpu_available():
                raise PluginError("GPU required but not available")
            
            # Initialize plugin-specific resources
            await self._initialize_plugin()
            
            self.logger.info(f"Plugin initialized: {self.__class__.__name__}")
            
        except Exception as e:
            self.logger.error(f"Plugin initialization failed: {self.__class__.__name__} - {e}")
            raise PluginError(f"Plugin initialization failed: {e}")
    
    async def _initialize_plugin(self) -> None:
        """Initialize plugin-specific resources. Override in subclasses."""
        pass
    
    @abstractmethod
    async def process(self, context: OnyxPluginContext) -> Dict[str, Any]:
        """Process data using the plugin. Must be implemented in subclasses."""
        pass
    
    async def cleanup(self) -> None:
        """Cleanup plugin resources."""
        try:
            self.cache.clear()
            await self._cleanup_plugin()
            self.logger.info(f"Plugin cleanup completed: {self.__class__.__name__}")
        except Exception as e:
            self.logger.error(f"Plugin cleanup failed: {self.__class__.__name__} - {e}")
    
    async def _cleanup_plugin(self) -> None:
        """Cleanup plugin-specific resources. Override in subclasses."""
        pass
    
    def get_info(self) -> OnyxPluginInfo:
        """Get plugin information."""
        return OnyxPluginInfo(
            name=self.__class__.__name__,
            version=getattr(self, 'version', '1.0.0'),
            description=getattr(self, 'description', ''),
            author=getattr(self, 'author', ''),
            category=getattr(self, 'category', 'general'),
            gpu_required=getattr(self, 'gpu_required', False),
            timeout=getattr(self, 'timeout', 60),
            max_workers=getattr(self, 'max_workers', 1)
        )
    
    def validate_config(self) -> bool:
        """Validate plugin configuration."""
        try:
            # Basic validation
            if not hasattr(self, 'version'):
                self.logger.warning("Plugin missing version attribute")
            
            if not hasattr(self, 'description'):
                self.logger.warning("Plugin missing description attribute")
            
            if not hasattr(self, 'author'):
                self.logger.warning("Plugin missing author attribute")
            
            if not hasattr(self, 'category'):
                self.logger.warning("Plugin missing category attribute")
            
            # Custom validation
            await self._validate_plugin_config()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Plugin validation failed: {self.__class__.__name__} - {e}")
            return False
    
    async def _validate_plugin_config(self) -> None:
        """Validate plugin-specific configuration. Override in subclasses."""
        pass
    
    def get_cache_key(self, context: OnyxPluginContext) -> str:
        """Generate cache key for plugin execution."""
        return f"{self.__class__.__name__}_{context.request.request_id}"
    
    def is_cache_valid(self, cache_key: str, max_age: int = 3600) -> bool:
        """Check if cached result is still valid."""
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if 'timestamp' in cached_data:
                cache_time = datetime.fromisoformat(cached_data['timestamp'])
                return datetime.now() - cache_time < timedelta(seconds=max_age)
        return False
    
    def update_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Update plugin cache with new data."""
        data['timestamp'] = datetime.now().isoformat()
        self.cache[cache_key] = data 