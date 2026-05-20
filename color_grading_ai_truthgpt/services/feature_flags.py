"""
Feature Flags for Color Grading AI
===================================

Feature flag management for gradual rollouts and A/B testing.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class FeatureFlagType(Enum):
    """Feature flag types."""
    BOOLEAN = "boolean"
    PERCENTAGE = "percentage"
    USER_LIST = "user_list"
    CUSTOM = "custom"


@dataclass
class FeatureFlag:
    """Feature flag definition."""
    name: str
    enabled: bool
    flag_type: FeatureFlagType
    percentage: float = 0.0  # 0-100
    user_list: List[str] = field(default_factory=list)
    custom_rule: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class FeatureFlagManager:
    """
    Feature flag manager.
    
    Features:
    - Boolean flags
    - Percentage rollouts
    - User-based flags
    - Custom rules
    - A/B testing support
    """
    
    def __init__(self):
        """Initialize feature flag manager."""
        self._flags: Dict[str, FeatureFlag] = {}
    
    def register_flag(
        self,
        name: str,
        enabled: bool = False,
        flag_type: FeatureFlagType = FeatureFlagType.BOOLEAN,
        percentage: float = 0.0,
        user_list: Optional[List[str]] = None,
        custom_rule: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Register a feature flag.
        
        Args:
            name: Flag name
            enabled: Whether flag is enabled
            flag_type: Flag type
            percentage: Percentage for percentage-based flags (0-100)
            user_list: List of user IDs for user-based flags
            custom_rule: Custom rule function
            metadata: Optional metadata
        """
        flag = FeatureFlag(
            name=name,
            enabled=enabled,
            flag_type=flag_type,
            percentage=percentage,
            user_list=user_list or [],
            custom_rule=custom_rule,
            metadata=metadata or {}
        )
        
        self._flags[name] = flag
        logger.info(f"Registered feature flag: {name} (enabled: {enabled})")
    
    def is_enabled(
        self,
        flag_name: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if feature flag is enabled.
        
        Args:
            flag_name: Flag name
            user_id: Optional user ID
            context: Optional context for custom rules
            
        Returns:
            True if enabled
        """
        flag = self._flags.get(flag_name)
        if not flag:
            return False
        
        if not flag.enabled:
            return False
        
        if flag.flag_type == FeatureFlagType.BOOLEAN:
            return True
        
        elif flag.flag_type == FeatureFlagType.PERCENTAGE:
            # Percentage-based rollout
            import hashlib
            if user_id:
                # Deterministic based on user ID
                hash_value = int(hashlib.md5(f"{flag_name}:{user_id}".encode()).hexdigest(), 16)
                user_percentage = (hash_value % 100) + 1
                return user_percentage <= flag.percentage
            else:
                # Random for non-user requests
                import random
                return random.random() * 100 <= flag.percentage
        
        elif flag.flag_type == FeatureFlagType.USER_LIST:
            return user_id in flag.user_list if user_id else False
        
        elif flag.flag_type == FeatureFlagType.CUSTOM:
            if flag.custom_rule:
                try:
                    return flag.custom_rule(user_id=user_id, context=context or {})
                except Exception as e:
                    logger.error(f"Error in custom rule for {flag_name}: {e}")
                    return False
        
        return False
    
    def enable_flag(self, flag_name: str):
        """Enable a feature flag."""
        if flag_name in self._flags:
            self._flags[flag_name].enabled = True
            logger.info(f"Enabled feature flag: {flag_name}")
    
    def disable_flag(self, flag_name: str):
        """Disable a feature flag."""
        if flag_name in self._flags:
            self._flags[flag_name].enabled = False
            logger.info(f"Disabled feature flag: {flag_name}")
    
    def update_flag_percentage(self, flag_name: str, percentage: float):
        """Update flag percentage."""
        if flag_name in self._flags:
            self._flags[flag_name].percentage = max(0, min(100, percentage))
            logger.info(f"Updated {flag_name} percentage to {percentage}%")
    
    def add_user_to_flag(self, flag_name: str, user_id: str):
        """Add user to flag user list."""
        if flag_name in self._flags:
            flag = self._flags[flag_name]
            if user_id not in flag.user_list:
                flag.user_list.append(user_id)
                logger.info(f"Added user {user_id} to flag {flag_name}")
    
    def remove_user_from_flag(self, flag_name: str, user_id: str):
        """Remove user from flag user list."""
        if flag_name in self._flags:
            flag = self._flags[flag_name]
            if user_id in flag.user_list:
                flag.user_list.remove(user_id)
                logger.info(f"Removed user {user_id} from flag {flag_name}")
    
    def list_flags(self) -> List[Dict[str, Any]]:
        """List all feature flags."""
        return [
            {
                "name": flag.name,
                "enabled": flag.enabled,
                "type": flag.flag_type.value,
                "percentage": flag.percentage,
                "user_count": len(flag.user_list),
                "metadata": flag.metadata,
            }
            for flag in self._flags.values()
        ]
    
    def get_flag(self, flag_name: str) -> Optional[Dict[str, Any]]:
        """Get feature flag details."""
        flag = self._flags.get(flag_name)
        if not flag:
            return None
        
        return {
            "name": flag.name,
            "enabled": flag.enabled,
            "type": flag.flag_type.value,
            "percentage": flag.percentage,
            "user_list": flag.user_list.copy(),
            "metadata": flag.metadata.copy(),
        }


def feature_flag(flag_name: str, default: bool = False):
    """
    Decorator for feature flag gating.
    
    Args:
        flag_name: Feature flag name
        default: Default value if flag not found
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Check if manager has feature flag manager
            if hasattr(self, 'feature_flags'):
                if not self.feature_flags.is_enabled(flag_name):
                    # Return default or raise
                    if default:
                        return None
                    raise FeatureFlagDisabledError(f"Feature flag {flag_name} is disabled")
            
            return await func(self, *args, **kwargs)
        return wrapper
    return decorator


class FeatureFlagDisabledError(Exception):
    """Feature flag is disabled."""
    pass




