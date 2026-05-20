"""
Feature Flags System
====================

System for feature flags and gradual rollouts.
"""

import logging
import json
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import asyncio

logger = logging.getLogger(__name__)


class FeatureFlagType(Enum):
    """Feature flag type."""
    BOOLEAN = "boolean"
    PERCENTAGE = "percentage"
    USER_LIST = "user_list"
    CUSTOM = "custom"


@dataclass
class FeatureFlag:
    """Feature flag definition."""
    name: str
    enabled: bool
    flag_type: FeatureFlagType = FeatureFlagType.BOOLEAN
    percentage: float = 0.0  # For percentage-based flags
    user_list: List[str] = field(default_factory=list)  # For user-based flags
    custom_check: Optional[Callable[[Dict[str, Any]], bool]] = None  # For custom flags
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "flag_type": self.flag_type.value,
            "percentage": self.percentage,
            "user_list": self.user_list,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureFlag":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            enabled=data["enabled"],
            flag_type=FeatureFlagType(data.get("flag_type", "boolean")),
            percentage=data.get("percentage", 0.0),
            user_list=data.get("user_list", []),
            metadata=data.get("metadata", {})
        )


class FeatureFlagManager:
    """Feature flag manager."""
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize feature flag manager.
        
        Args:
            config_file: Optional path to feature flags config file
        """
        self.config_file = config_file
        self.flags: Dict[str, FeatureFlag] = {}
        self._load_flags()
    
    def _load_flags(self):
        """Load feature flags from file."""
        if self.config_file and self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for flag_data in data.get("flags", []):
                        flag = FeatureFlag.from_dict(flag_data)
                        self.flags[flag.name] = flag
                logger.info(f"Loaded {len(self.flags)} feature flags from {self.config_file}")
            except Exception as e:
                logger.warning(f"Error loading feature flags: {e}")
    
    def _save_flags(self):
        """Save feature flags to file."""
        if not self.config_file:
            return
        
        try:
            data = {
                "flags": [flag.to_dict() for flag in self.flags.values()]
            }
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving feature flags: {e}")
    
    def register(self, flag: FeatureFlag):
        """
        Register a feature flag.
        
        Args:
            flag: Feature flag to register
        """
        self.flags[flag.name] = flag
        self._save_flags()
        logger.info(f"Registered feature flag: {flag.name}")
    
    def is_enabled(
        self,
        flag_name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if a feature flag is enabled.
        
        Args:
            flag_name: Flag name
            context: Optional context (user_id, etc.)
            
        Returns:
            True if enabled
        """
        if flag_name not in self.flags:
            logger.warning(f"Feature flag not found: {flag_name}")
            return False
        
        flag = self.flags[flag_name]
        
        if not flag.enabled:
            return False
        
        # Check based on flag type
        if flag.flag_type == FeatureFlagType.BOOLEAN:
            return True
        
        elif flag.flag_type == FeatureFlagType.PERCENTAGE:
            if not context or "user_id" not in context:
                return False
            # Simple hash-based percentage
            user_id = str(context["user_id"])
            hash_value = hash(user_id) % 100
            return hash_value < flag.percentage
        
        elif flag.flag_type == FeatureFlagType.USER_LIST:
            if not context or "user_id" not in context:
                return False
            return str(context["user_id"]) in flag.user_list
        
        elif flag.flag_type == FeatureFlagType.CUSTOM:
            if not flag.custom_check:
                return False
            return flag.custom_check(context or {})
        
        return False
    
    def enable(self, flag_name: str):
        """Enable a feature flag."""
        if flag_name in self.flags:
            self.flags[flag_name].enabled = True
            self._save_flags()
    
    def disable(self, flag_name: str):
        """Disable a feature flag."""
        if flag_name in self.flags:
            self.flags[flag_name].enabled = False
            self._save_flags()
    
    def get_flag(self, flag_name: str) -> Optional[FeatureFlag]:
        """Get feature flag by name."""
        return self.flags.get(flag_name)
    
    def get_all_flags(self) -> Dict[str, FeatureFlag]:
        """Get all feature flags."""
        return self.flags.copy()


def feature_flag(flag_name: str, context_provider: Optional[Callable] = None):
    """
    Decorator to check feature flag before executing function.
    
    Args:
        flag_name: Feature flag name
        context_provider: Optional function to provide context
        
    Usage:
        @feature_flag("new_feature", lambda: {"user_id": current_user.id})
        def my_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        from .feature_flags import FeatureFlagManager
        
        # Get global feature flag manager (would be injected in real app)
        manager = FeatureFlagManager()
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            context = None
            if context_provider:
                context = context_provider()
            
            if not manager.is_enabled(flag_name, context):
                raise Exception(f"Feature flag '{flag_name}' is not enabled")
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator

