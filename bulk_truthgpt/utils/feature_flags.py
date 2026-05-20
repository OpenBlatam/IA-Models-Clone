"""
Feature Flags Manager
=====================

Advanced feature flags with A/B testing and gradual rollout.
"""

import asyncio
import logging
import random
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class FeatureFlagType(str, Enum):
    """Feature flag types."""
    BOOLEAN = "boolean"
    PERCENTAGE = "percentage"
    CONDITIONAL = "conditional"
    A_B_TEST = "a_b_test"

class FeatureFlag:
    """Feature flag definition."""
    
    def __init__(
        self,
        name: str,
        flag_type: FeatureFlagType,
        enabled: bool = False,
        percentage: float = 0.0,
        conditions: Optional[Dict[str, Any]] = None,
        variants: Optional[Dict[str, float]] = None
    ):
        self.name = name
        self.flag_type = flag_type
        self.enabled = enabled
        self.percentage = percentage
        self.conditions = conditions or {}
        self.variants = variants or {}
        self.stats = {
            "checks": 0,
            "enabled": 0,
            "disabled": 0
        }

class FeatureFlagsManager:
    """Advanced feature flags manager."""
    
    def __init__(self):
        self.flags: Dict[str, FeatureFlag] = {}
        self.user_context: Dict[str, Any] = {}
    
    def register_flag(
        self,
        name: str,
        flag_type: FeatureFlagType = FeatureFlagType.BOOLEAN,
        **kwargs
    ) -> FeatureFlag:
        """Register a feature flag."""
        flag = FeatureFlag(name=name, flag_type=flag_type, **kwargs)
        self.flags[name] = flag
        logger.info(f"Feature flag registered: {name} ({flag_type.value})")
        return flag
    
    def is_enabled(
        self,
        flag_name: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Check if feature flag is enabled."""
        if flag_name not in self.flags:
            logger.warning(f"Feature flag not found: {flag_name}")
            return False
        
        flag = self.flags[flag_name]
        flag.stats["checks"] += 1
        
        # Merge context
        check_context = {**self.user_context}
        if context:
            check_context.update(context)
        if user_id:
            check_context["user_id"] = user_id
        
        result = False
        
        if flag.flag_type == FeatureFlagType.BOOLEAN:
            result = flag.enabled
        
        elif flag.flag_type == FeatureFlagType.PERCENTAGE:
            if flag.enabled:
                # Use user_id or random for percentage
                seed = user_id or str(random.random())
                hash_value = hash(seed) % 100
                result = hash_value < flag.percentage
        
        elif flag.flag_type == FeatureFlagType.CONDITIONAL:
            if flag.enabled:
                result = self._check_conditions(flag.conditions, check_context)
        
        elif flag.flag_type == FeatureFlagType.A_B_TEST:
            if flag.enabled and flag.variants:
                # Select variant based on user_id
                seed = user_id or str(random.random())
                hash_value = abs(hash(seed)) % 100
                
                cumulative = 0
                for variant, percentage in flag.variants.items():
                    cumulative += percentage
                    if hash_value < cumulative:
                        result = variant == "enabled" or variant == "A"
                        break
        
        if result:
            flag.stats["enabled"] += 1
        else:
            flag.stats["disabled"] += 1
        
        return result
    
    def _check_conditions(self, conditions: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check if conditions are met."""
        for key, expected_value in conditions.items():
            actual_value = context.get(key)
            
            if isinstance(expected_value, dict):
                # Complex condition
                op = expected_value.get("op", "eq")
                value = expected_value.get("value")
                
                if op == "eq":
                    if actual_value != value:
                        return False
                elif op == "gt":
                    if not (actual_value and actual_value > value):
                        return False
                elif op == "lt":
                    if not (actual_value and actual_value < value):
                        return False
                elif op == "in":
                    if actual_value not in value:
                        return False
            else:
                # Simple equality
                if actual_value != expected_value:
                    return False
        
        return True
    
    def get_variant(self, flag_name: str, user_id: Optional[str] = None) -> Optional[str]:
        """Get A/B test variant."""
        if flag_name not in self.flags:
            return None
        
        flag = self.flags[flag_name]
        if flag.flag_type != FeatureFlagType.A_B_TEST:
            return None
        
        if not flag.enabled or not flag.variants:
            return None
        
        seed = user_id or str(random.random())
        hash_value = abs(hash(seed)) % 100
        
        cumulative = 0
        for variant, percentage in flag.variants.items():
            cumulative += percentage
            if hash_value < cumulative:
                return variant
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get feature flags statistics."""
        return {
            "total_flags": len(self.flags),
            "flags": {
                name: {
                    "type": flag.flag_type.value,
                    "enabled": flag.enabled,
                    "stats": flag.stats
                }
                for name, flag in self.flags.items()
            }
        }
    
    def set_user_context(self, context: Dict[str, Any]):
        """Set user context for feature flags."""
        self.user_context = context

# Global instance
feature_flags = FeatureFlagsManager()
































