"""
Feature Flags System
Allows enabling/disabling features without code deployment
"""

import os
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FeatureFlagType(str, Enum):
    """Feature flag types"""
    BOOLEAN = "boolean"
    PERCENTAGE = "percentage"
    USER_LIST = "user_list"
    CUSTOM = "custom"


@dataclass
class FeatureFlag:
    """Feature flag definition"""
    name: str
    enabled: bool = False
    flag_type: FeatureFlagType = FeatureFlagType.BOOLEAN
    percentage: int = 0  # For percentage rollout
    user_list: list[str] = None  # For user-specific flags
    custom_check: Optional[Callable] = None  # For custom logic
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.user_list is None:
            self.user_list = []
        if self.metadata is None:
            self.metadata = {}


class FeatureFlagManager:
    """
    Manages feature flags.
    Supports multiple flag types and dynamic updates.
    """
    
    def __init__(self):
        self.flags: Dict[str, FeatureFlag] = {}
        self._load_from_env()
    
    def _load_from_env(self):
        """Load feature flags from environment variables"""
        # Format: FEATURE_FLAG_<NAME>=true/false
        for key, value in os.environ.items():
            if key.startswith("FEATURE_FLAG_"):
                flag_name = key.replace("FEATURE_FLAG_", "").lower()
                enabled = value.lower() in ("true", "1", "yes")
                self.register_flag(FeatureFlag(name=flag_name, enabled=enabled))
    
    def register_flag(self, flag: FeatureFlag):
        """Register a feature flag"""
        self.flags[flag.name] = flag
        logger.info(f"Registered feature flag: {flag.name} (enabled: {flag.enabled})")
    
    def is_enabled(
        self,
        flag_name: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if feature flag is enabled
        
        Args:
            flag_name: Name of feature flag
            user_id: Optional user ID for user-specific flags
            context: Optional context for custom checks
            
        Returns:
            True if feature is enabled
        """
        flag = self.flags.get(flag_name)
        
        if not flag:
            # Default to disabled if flag not found
            logger.debug(f"Feature flag {flag_name} not found, defaulting to disabled")
            return False
        
        if not flag.enabled:
            return False
        
        # Check flag type
        if flag.flag_type == FeatureFlagType.BOOLEAN:
            return flag.enabled
        
        elif flag.flag_type == FeatureFlagType.PERCENTAGE:
            if user_id:
                # Deterministic based on user_id
                import hashlib
                hash_value = int(hashlib.md5(f"{flag_name}:{user_id}".encode()).hexdigest(), 16)
                user_percentage = (hash_value % 100) + 1
                return user_percentage <= flag.percentage
            return False
        
        elif flag.flag_type == FeatureFlagType.USER_LIST:
            return user_id in flag.user_list if user_id else False
        
        elif flag.flag_type == FeatureFlagType.CUSTOM:
            if flag.custom_check:
                try:
                    return flag.custom_check(user_id, context or {})
                except Exception as e:
                    logger.error(f"Custom flag check failed for {flag_name}: {e}")
                    return False
        
        return False
    
    def get_flag(self, flag_name: str) -> Optional[FeatureFlag]:
        """Get feature flag by name"""
        return self.flags.get(flag_name)
    
    def list_flags(self) -> Dict[str, FeatureFlag]:
        """List all feature flags"""
        return self.flags.copy()
    
    def update_flag(self, flag_name: str, enabled: bool):
        """Update feature flag (runtime update)"""
        if flag_name in self.flags:
            self.flags[flag_name].enabled = enabled
            logger.info(f"Updated feature flag {flag_name}: enabled={enabled}")
        else:
            logger.warning(f"Feature flag {flag_name} not found")


# Global feature flag manager
_feature_flag_manager: Optional[FeatureFlagManager] = None


def get_feature_flag_manager() -> FeatureFlagManager:
    """Get or create global feature flag manager"""
    global _feature_flag_manager
    if _feature_flag_manager is None:
        _feature_flag_manager = FeatureFlagManager()
    return _feature_flag_manager


def feature_flag(flag_name: str, user_id: Optional[str] = None):
    """
    Decorator to check feature flag before executing function
    
    Usage:
        @feature_flag("new_analysis_algorithm", user_id="user123")
        async def analyze_image(...):
            ...
    """
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            # Extract user_id from kwargs or args if available
            user_id_value = user_id or kwargs.get("user_id") or kwargs.get("current_user", {}).get("sub")
            
            manager = get_feature_flag_manager()
            if not manager.is_enabled(flag_name, user_id_value):
                raise Exception(f"Feature {flag_name} is not enabled")
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator















