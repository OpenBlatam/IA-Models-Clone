"""
Feature Flags Service
=====================
Service for managing feature flags and toggles
"""

import logging
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class FeatureFlagType(Enum):
    """Feature flag types"""
    BOOLEAN = "boolean"  # Simple on/off
    PERCENTAGE = "percentage"  # Percentage rollout
    TARGETED = "targeted"  # Targeted to specific users/conditions


@dataclass
class FeatureFlag:
    """Feature flag definition"""
    name: str
    flag_type: FeatureFlagType
    enabled: bool = False
    percentage: float = 0.0  # 0-100 for percentage rollout
    target_users: List[str] = field(default_factory=list)
    target_conditions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class FeatureFlagsService:
    """
    Service for managing feature flags.
    
    Features:
    - Boolean flags (on/off)
    - Percentage rollouts
    - Targeted flags (user/condition-based)
    - Flag metadata
    - Statistics
    """
    
    def __init__(self):
        """Initialize feature flags service"""
        self._flags: Dict[str, FeatureFlag] = {}
        self._stats = {
            'flag_checks': 0,
            'enabled_checks': 0,
            'disabled_checks': 0
        }
    
    def create_flag(
        self,
        name: str,
        flag_type: FeatureFlagType = FeatureFlagType.BOOLEAN,
        enabled: bool = False,
        percentage: float = 0.0,
        target_users: Optional[List[str]] = None,
        target_conditions: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FeatureFlag:
        """
        Create a feature flag.
        
        Args:
            name: Flag name
            flag_type: Flag type
            enabled: Whether flag is enabled (for boolean)
            percentage: Percentage rollout (0-100)
            target_users: List of user IDs for targeted flags
            target_conditions: Conditions for targeted flags
            metadata: Optional metadata
        
        Returns:
            FeatureFlag
        """
        flag = FeatureFlag(
            name=name,
            flag_type=flag_type,
            enabled=enabled,
            percentage=percentage,
            target_users=target_users or [],
            target_conditions=target_conditions or {},
            metadata=metadata or {}
        )
        
        self._flags[name] = flag
        logger.info(f"Created feature flag '{name}' (type: {flag_type.value}, enabled: {enabled})")
        
        return flag
    
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
            user_id: Optional user ID for targeted flags
            context: Optional context for condition evaluation
        
        Returns:
            True if flag is enabled for this check
        """
        self._stats['flag_checks'] += 1
        
        flag = self._flags.get(flag_name)
        if not flag:
            logger.warning(f"Feature flag '{flag_name}' not found")
            return False
        
        result = False
        
        if flag.flag_type == FeatureFlagType.BOOLEAN:
            result = flag.enabled
        
        elif flag.flag_type == FeatureFlagType.PERCENTAGE:
            # Simple hash-based percentage rollout
            if user_id:
                # Use user ID for consistent rollout
                hash_value = hash(f"{flag_name}:{user_id}") % 100
                result = hash_value < flag.percentage
            else:
                # Random for non-user requests
                import random
                result = random.random() * 100 < flag.percentage
        
        elif flag.flag_type == FeatureFlagType.TARGETED:
            # Check if user is in target list
            if user_id and user_id in flag.target_users:
                result = True
            # Check conditions
            elif context and flag.target_conditions:
                result = self._evaluate_conditions(flag.target_conditions, context)
        
        if result:
            self._stats['enabled_checks'] += 1
        else:
            self._stats['disabled_checks'] += 1
        
        return result
    
    def _evaluate_conditions(
        self,
        conditions: Dict[str, Any],
        context: Dict[str, Any]
    ) -> bool:
        """Evaluate target conditions against context"""
        for key, expected_value in conditions.items():
            actual_value = context.get(key)
            if actual_value != expected_value:
                return False
        return True
    
    def update_flag(
        self,
        flag_name: str,
        enabled: Optional[bool] = None,
        percentage: Optional[float] = None,
        target_users: Optional[List[str]] = None,
        target_conditions: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update feature flag.
        
        Args:
            flag_name: Flag name
            enabled: New enabled state
            percentage: New percentage
            target_users: New target users
            target_conditions: New target conditions
            metadata: New metadata
        
        Returns:
            True if flag was updated
        """
        flag = self._flags.get(flag_name)
        if not flag:
            return False
        
        if enabled is not None:
            flag.enabled = enabled
        if percentage is not None:
            flag.percentage = max(0.0, min(100.0, percentage))
        if target_users is not None:
            flag.target_users = target_users
        if target_conditions is not None:
            flag.target_conditions = target_conditions
        if metadata is not None:
            flag.metadata.update(metadata)
        
        flag.updated_at = datetime.now()
        logger.info(f"Updated feature flag '{flag_name}'")
        
        return True
    
    def delete_flag(self, flag_name: str) -> bool:
        """Delete feature flag"""
        if flag_name in self._flags:
            del self._flags[flag_name]
            logger.info(f"Deleted feature flag '{flag_name}'")
            return True
        return False
    
    def get_flag(self, flag_name: str) -> Optional[FeatureFlag]:
        """Get feature flag by name"""
        return self._flags.get(flag_name)
    
    def list_flags(self) -> List[FeatureFlag]:
        """List all feature flags"""
        return list(self._flags.values())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get feature flags statistics"""
        total_checks = self._stats['flag_checks']
        enabled_rate = (
            self._stats['enabled_checks'] / total_checks * 100
            if total_checks > 0 else 0
        )
        
        return {
            'total_flags': len(self._flags),
            'enabled_flags': len([f for f in self._flags.values() if f.enabled]),
            'total_checks': total_checks,
            'enabled_checks': self._stats['enabled_checks'],
            'disabled_checks': self._stats['disabled_checks'],
            'enabled_rate': round(enabled_rate, 2)
        }


# Global feature flags service instance
_feature_flags_service: Optional[FeatureFlagsService] = None


def get_feature_flags_service() -> FeatureFlagsService:
    """Get or create feature flags service instance"""
    global _feature_flags_service
    if _feature_flags_service is None:
        _feature_flags_service = FeatureFlagsService()
    return _feature_flags_service

