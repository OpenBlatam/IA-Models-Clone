"""
Feature Flags System for Flux2 Clothing Changer
================================================

Feature flag management and A/B testing.
"""

import time
import random
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import json

logger = logging.getLogger(__name__)


class FeatureFlagType(Enum):
    """Feature flag type."""
    BOOLEAN = "boolean"
    PERCENTAGE = "percentage"
    USER_LIST = "user_list"
    A_B_TEST = "a_b_test"


@dataclass
class FeatureFlag:
    """Feature flag configuration."""
    flag_name: str
    flag_type: FeatureFlagType
    enabled: bool = True
    percentage: float = 100.0  # For percentage rollout
    user_list: List[str] = None  # For user list
    variants: Dict[str, float] = None  # For A/B testing
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.user_list is None:
            self.user_list = []
        if self.variants is None:
            self.variants = {}
        if self.metadata is None:
            self.metadata = {}


class FeatureFlags:
    """Feature flag management system."""
    
    def __init__(
        self,
        enable_persistence: bool = True,
        persistence_path: Optional[str] = None,
    ):
        """
        Initialize feature flags system.
        
        Args:
            enable_persistence: Enable persistence to disk
            persistence_path: Path for persistence
        """
        self.enable_persistence = enable_persistence
        self.persistence_path = persistence_path or "feature_flags.json"
        
        self.flags: Dict[str, FeatureFlag] = {}
        self.usage_tracking: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        if enable_persistence:
            self._load_flags()
    
    def create_flag(
        self,
        flag_name: str,
        flag_type: FeatureFlagType,
        enabled: bool = True,
        percentage: float = 100.0,
        user_list: Optional[List[str]] = None,
        variants: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FeatureFlag:
        """
        Create a feature flag.
        
        Args:
            flag_name: Flag name
            flag_type: Flag type
            enabled: Whether flag is enabled
            percentage: Percentage for rollout
            user_list: User list for targeting
            variants: Variants for A/B testing
            metadata: Optional metadata
            
        Returns:
            Created feature flag
        """
        flag = FeatureFlag(
            flag_name=flag_name,
            flag_type=flag_type,
            enabled=enabled,
            percentage=percentage,
            user_list=user_list or [],
            variants=variants or {},
            metadata=metadata or {},
        )
        
        self.flags[flag_name] = flag
        self._save_flags()
        
        logger.info(f"Created feature flag: {flag_name}")
        return flag
    
    def is_enabled(
        self,
        flag_name: str,
        user_id: Optional[str] = None,
        default: bool = False,
    ) -> bool:
        """
        Check if feature flag is enabled.
        
        Args:
            flag_name: Flag name
            user_id: Optional user ID
            default: Default value if flag not found
            
        Returns:
            True if enabled
        """
        if flag_name not in self.flags:
            return default
        
        flag = self.flags[flag_name]
        
        if not flag.enabled:
            return False
        
        if flag.flag_type == FeatureFlagType.BOOLEAN:
            return True
        
        elif flag.flag_type == FeatureFlagType.PERCENTAGE:
            # Hash user_id for consistent assignment
            if user_id:
                hash_value = hash(user_id) % 100
                return hash_value < flag.percentage
            else:
                return random.random() * 100 < flag.percentage
        
        elif flag.flag_type == FeatureFlagType.USER_LIST:
            return user_id in flag.user_list if user_id else False
        
        elif flag.flag_type == FeatureFlagType.A_B_TEST:
            # A/B test - return True if user should see feature
            return self._get_ab_variant(flag_name, user_id) is not None
        
        return False
    
    def get_variant(
        self,
        flag_name: str,
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Get A/B test variant for user.
        
        Args:
            flag_name: Flag name
            user_id: Optional user ID
            
        Returns:
            Variant name or None
        """
        if flag_name not in self.flags:
            return None
        
        flag = self.flags[flag_name]
        
        if flag.flag_type != FeatureFlagType.A_B_TEST:
            return None
        
        return self._get_ab_variant(flag_name, user_id)
    
    def _get_ab_variant(
        self,
        flag_name: str,
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        """Get A/B test variant."""
        flag = self.flags[flag_name]
        
        if not flag.variants:
            return None
        
        # Consistent assignment based on user_id
        if user_id:
            hash_value = hash(f"{flag_name}:{user_id}") % 100
        else:
            hash_value = random.randint(0, 99)
        
        # Assign to variant based on percentages
        cumulative = 0.0
        for variant, percentage in flag.variants.items():
            cumulative += percentage
            if hash_value < cumulative:
                # Track usage
                self.usage_tracking[flag_name][variant] += 1
                return variant
        
        return None
    
    def toggle_flag(self, flag_name: str, enabled: Optional[bool] = None) -> bool:
        """
        Toggle feature flag.
        
        Args:
            flag_name: Flag name
            enabled: Optional explicit enabled state
            
        Returns:
            New enabled state
        """
        if flag_name not in self.flags:
            return False
        
        if enabled is None:
            self.flags[flag_name].enabled = not self.flags[flag_name].enabled
        else:
            self.flags[flag_name].enabled = enabled
        
        self._save_flags()
        logger.info(f"Toggled flag {flag_name} to {self.flags[flag_name].enabled}")
        
        return self.flags[flag_name].enabled
    
    def get_flag_statistics(self, flag_name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a feature flag."""
        if flag_name not in self.flags:
            return None
        
        flag = self.flags[flag_name]
        usage = self.usage_tracking.get(flag_name, {})
        
        return {
            "flag_name": flag_name,
            "enabled": flag.enabled,
            "type": flag.flag_type.value,
            "usage": dict(usage),
            "total_usage": sum(usage.values()),
        }
    
    def _save_flags(self) -> None:
        """Save flags to disk."""
        if not self.enable_persistence:
            return
        
        try:
            flags_data = {
                flag_name: {
                    "flag_name": flag.flag_name,
                    "flag_type": flag.flag_type.value,
                    "enabled": flag.enabled,
                    "percentage": flag.percentage,
                    "user_list": flag.user_list,
                    "variants": flag.variants,
                    "metadata": flag.metadata,
                }
                for flag_name, flag in self.flags.items()
            }
            
            with open(self.persistence_path, "w", encoding="utf-8") as f:
                json.dump(flags_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save feature flags: {e}")
    
    def _load_flags(self) -> None:
        """Load flags from disk."""
        try:
            with open(self.persistence_path, "r", encoding="utf-8") as f:
                flags_data = json.load(f)
            
            for flag_name, data in flags_data.items():
                self.flags[flag_name] = FeatureFlag(
                    flag_name=data["flag_name"],
                    flag_type=FeatureFlagType(data["flag_type"]),
                    enabled=data.get("enabled", True),
                    percentage=data.get("percentage", 100.0),
                    user_list=data.get("user_list", []),
                    variants=data.get("variants", {}),
                    metadata=data.get("metadata", {}),
                )
            
            logger.info(f"Loaded {len(self.flags)} feature flags")
        except FileNotFoundError:
            logger.info("No existing feature flags file found")
        except Exception as e:
            logger.warning(f"Failed to load feature flags: {e}")
    
    def get_all_flags(self) -> Dict[str, FeatureFlag]:
        """Get all feature flags."""
        return self.flags.copy()


