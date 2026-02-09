"""
Feature Manager
===============

Feature management and feature flags.
"""

import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from aws.modules.ports.cache_port import CachePort

logger = logging.getLogger(__name__)


class FeatureFlag:
    """Feature flag."""
    
    def __init__(
        self,
        name: str,
        enabled: bool = False,
        description: str = "",
        rollout_percentage: float = 0.0
    ):
        self.name = name
        self.enabled = enabled
        self.description = description
        self.rollout_percentage = rollout_percentage
    
    def is_enabled(self, user_id: Optional[str] = None) -> bool:
        """Check if feature is enabled."""
        if not self.enabled:
            return False
        
        if self.rollout_percentage >= 100.0:
            return True
        
        if self.rollout_percentage <= 0.0:
            return False
        
        # Simple hash-based rollout
        if user_id:
            hash_value = hash(user_id) % 100
            return hash_value < self.rollout_percentage
        
        return False


class FeatureManager:
    """Feature manager with feature flags."""
    
    def __init__(self, cache: Optional[CachePort] = None):
        self.cache = cache
        self._features: Dict[str, FeatureFlag] = {}
        self._handlers: Dict[str, Callable] = {}
    
    def register_feature(
        self,
        name: str,
        enabled: bool = False,
        description: str = "",
        rollout_percentage: float = 0.0
    ):
        """Register feature flag."""
        self._features[name] = FeatureFlag(
            name=name,
            enabled=enabled,
            description=description,
            rollout_percentage=rollout_percentage
        )
        logger.info(f"Registered feature: {name} (enabled={enabled})")
    
    def enable_feature(self, name: str):
        """Enable feature."""
        if name in self._features:
            self._features[name].enabled = True
            logger.info(f"Enabled feature: {name}")
    
    def disable_feature(self, name: str):
        """Disable feature."""
        if name in self._features:
            self._features[name].enabled = False
            logger.info(f"Disabled feature: {name}")
    
    async def is_feature_enabled(self, name: str, user_id: Optional[str] = None) -> bool:
        """Check if feature is enabled."""
        if name not in self._features:
            return False
        
        # Check cache first
        if self.cache:
            cache_key = f"feature:{name}:{user_id or 'global'}"
            cached = await self.cache.get(cache_key)
            if cached is not None:
                return cached
        
        enabled = self._features[name].is_enabled(user_id)
        
        # Cache result
        if self.cache:
            await self.cache.set(cache_key, enabled, ttl=300)
        
        return enabled
    
    def register_handler(self, feature_name: str, handler: Callable):
        """Register feature handler."""
        self._handlers[feature_name] = handler
    
    async def execute_feature(self, feature_name: str, *args, **kwargs) -> Any:
        """Execute feature if enabled."""
        if not await self.is_feature_enabled(feature_name):
            raise ValueError(f"Feature {feature_name} is not enabled")
        
        if feature_name in self._handlers:
            handler = self._handlers[feature_name]
            if asyncio.iscoroutinefunction(handler):
                return await handler(*args, **kwargs)
            return handler(*args, **kwargs)
        
        return None


# Import asyncio
import asyncio

