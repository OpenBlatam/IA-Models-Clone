"""
Feature Flags System for Piel Mejorador AI SAM3
================================================

Feature flag management for gradual rollouts.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class FlagStatus(Enum):
    """Feature flag status."""
    ENABLED = "enabled"
    DISABLED = "disabled"
    ROLLOUT = "rollout"  # Gradual rollout


@dataclass
class FeatureFlag:
    """Feature flag definition."""
    name: str
    status: FlagStatus
    description: str = ""
    rollout_percentage: float = 0.0  # 0-100
    target_users: List[str] = field(default_factory=list)
    target_groups: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class FeatureFlagManager:
    """
    Manages feature flags.
    
    Features:
    - Enable/disable features
    - Gradual rollouts
    - User targeting
    - Group targeting
    - A/B testing support
    """
    
    def __init__(self):
        """Initialize feature flag manager."""
        self._flags: Dict[str, FeatureFlag] = {}
        self._evaluations: Dict[str, int] = {
            "enabled": 0,
            "disabled": 0,
        }
    
    def create_flag(
        self,
        name: str,
        status: FlagStatus = FlagStatus.DISABLED,
        description: str = "",
        **kwargs
    ) -> FeatureFlag:
        """
        Create a feature flag.
        
        Args:
            name: Flag name
            status: Initial status
            description: Flag description
            **kwargs: Additional flag properties
            
        Returns:
            Created feature flag
        """
        flag = FeatureFlag(
            name=name,
            status=status,
            description=description,
            **kwargs
        )
        
        self._flags[name] = flag
        logger.info(f"Feature flag created: {name} ({status.value})")
        return flag
    
    def is_enabled(
        self,
        flag_name: str,
        user_id: Optional[str] = None,
        user_groups: Optional[List[str]] = None
    ) -> bool:
        """
        Check if a feature flag is enabled.
        
        Args:
            flag_name: Flag name
            user_id: Optional user ID for targeting
            user_groups: Optional user groups for targeting
            
        Returns:
            True if enabled
        """
        if flag_name not in self._flags:
            return False
        
        flag = self._flags[flag_name]
        
        if flag.status == FlagStatus.DISABLED:
            self._evaluations["disabled"] += 1
            return False
        
        if flag.status == FlagStatus.ENABLED:
            self._evaluations["enabled"] += 1
            return True
        
        # Rollout logic
        if flag.status == FlagStatus.ROLLOUT:
            # Check user targeting
            if user_id and flag.target_users:
                if user_id in flag.target_users:
                    self._evaluations["enabled"] += 1
                    return True
            
            # Check group targeting
            if user_groups and flag.target_groups:
                if any(g in flag.target_groups for g in user_groups):
                    self._evaluations["enabled"] += 1
                    return True
            
            # Percentage rollout (simplified - uses hash of user_id)
            if user_id:
                import hashlib
                hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
                percentage = (hash_value % 100) + 1
                
                if percentage <= flag.rollout_percentage:
                    self._evaluations["enabled"] += 1
                    return True
            
            self._evaluations["disabled"] += 1
            return False
    
    def update_flag(
        self,
        flag_name: str,
        status: Optional[FlagStatus] = None,
        rollout_percentage: Optional[float] = None,
        **kwargs
    ) -> bool:
        """
        Update a feature flag.
        
        Args:
            flag_name: Flag name
            status: Optional new status
            rollout_percentage: Optional rollout percentage
            **kwargs: Additional updates
            
        Returns:
            True if updated
        """
        if flag_name not in self._flags:
            return False
        
        flag = self._flags[flag_name]
        
        if status is not None:
            flag.status = status
        if rollout_percentage is not None:
            flag.rollout_percentage = rollout_percentage
        
        for key, value in kwargs.items():
            if hasattr(flag, key):
                setattr(flag, key, value)
        
        flag.updated_at = datetime.now()
        logger.info(f"Feature flag updated: {flag_name}")
        return True
    
    def get_flag(self, flag_name: str) -> Optional[FeatureFlag]:
        """Get feature flag."""
        return self._flags.get(flag_name)
    
    def list_flags(self) -> List[FeatureFlag]:
        """List all feature flags."""
        return list(self._flags.values())
    
    def delete_flag(self, flag_name: str) -> bool:
        """Delete a feature flag."""
        if flag_name in self._flags:
            del self._flags[flag_name]
            logger.info(f"Feature flag deleted: {flag_name}")
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get feature flag statistics."""
        return {
            "total_flags": len(self._flags),
            "enabled_flags": len([f for f in self._flags.values() if f.status == FlagStatus.ENABLED]),
            "disabled_flags": len([f for f in self._flags.values() if f.status == FlagStatus.DISABLED]),
            "rollout_flags": len([f for f in self._flags.values() if f.status == FlagStatus.ROLLOUT]),
            "evaluations": self._evaluations,
        }




