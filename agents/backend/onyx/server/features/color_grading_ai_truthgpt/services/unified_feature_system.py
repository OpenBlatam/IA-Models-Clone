"""
Unified Feature System for Color Grading AI
============================================

Consolidates feature management services:
- FeatureFlagManager (feature flags)
- FeatureToggleSystem (feature toggles)

Features:
- Unified feature management
- Feature flags
- Feature toggles
- A/B testing
- Percentage rollouts
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .feature_flags import FeatureFlagManager, FeatureFlag, FeatureFlagType
from .feature_toggle import FeatureToggleSystem, FeatureToggle, ToggleType

logger = logging.getLogger(__name__)


class FeatureSystemMode(Enum):
    """Feature system modes."""
    FLAGS_ONLY = "flags_only"  # Use only feature flags
    TOGGLES_ONLY = "toggles_only"  # Use only feature toggles
    UNIFIED = "unified"  # Use both systems


@dataclass
class FeatureCheckResult:
    """Feature check result."""
    enabled: bool
    source: str  # "flag" or "toggle"
    metadata: Dict[str, Any] = field(default_factory=dict)


class UnifiedFeatureSystem:
    """
    Unified feature system.
    
    Consolidates:
    - FeatureFlagManager: Feature flags
    - FeatureToggleSystem: Feature toggles
    
    Features:
    - Unified feature management
    - Feature flags and toggles
    - A/B testing support
    """
    
    def __init__(
        self,
        mode: FeatureSystemMode = FeatureSystemMode.UNIFIED
    ):
        """
        Initialize unified feature system.
        
        Args:
            mode: Feature system mode
        """
        self.mode = mode
        
        # Initialize components
        self.feature_flags = FeatureFlagManager()
        self.feature_toggle = FeatureToggleSystem()
        
        logger.info(f"Initialized UnifiedFeatureSystem (mode={mode.value})")
    
    def is_enabled(
        self,
        feature_name: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> FeatureCheckResult:
        """
        Check if feature is enabled.
        
        Args:
            feature_name: Feature name
            user_id: Optional user ID
            context: Optional context
            
        Returns:
            Feature check result
        """
        # Check feature flags first
        if self.mode in [FeatureSystemMode.FLAGS_ONLY, FeatureSystemMode.UNIFIED]:
            try:
                flag = self.feature_flags.get_flag(feature_name)
                if flag:
                    enabled = self.feature_flags.is_enabled(feature_name, user_id, context)
                    return FeatureCheckResult(
                        enabled=enabled,
                        source="flag",
                        metadata={"flag_type": flag.flag_type.value}
                    )
            except Exception:
                pass
        
        # Check feature toggles
        if self.mode in [FeatureSystemMode.TOGGLES_ONLY, FeatureSystemMode.UNIFIED]:
            try:
                enabled = self.feature_toggle.is_enabled(feature_name, user_id, context)
                return FeatureCheckResult(
                    enabled=enabled,
                    source="toggle"
                )
            except Exception:
                pass
        
        # Default to disabled
        return FeatureCheckResult(
            enabled=False,
            source="none"
        )
    
    def register_flag(
        self,
        name: str,
        flag_type: FeatureFlagType,
        enabled: bool = False,
        **kwargs
    ):
        """
        Register feature flag.
        
        Args:
            name: Flag name
            flag_type: Flag type
            enabled: Whether enabled
            **kwargs: Additional parameters
        """
        flag = FeatureFlag(
            name=name,
            flag_type=flag_type,
            enabled=enabled,
            **kwargs
        )
        self.feature_flags.register_flag(flag)
    
    def register_toggle(
        self,
        name: str,
        toggle_type: ToggleType,
        enabled: bool = False,
        **kwargs
    ):
        """
        Register feature toggle.
        
        Args:
            name: Toggle name
            toggle_type: Toggle type
            enabled: Whether enabled
            **kwargs: Additional parameters
        """
        self.feature_toggle.register_toggle(
            name=name,
            toggle_type=toggle_type,
            enabled=enabled,
            **kwargs
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get feature system statistics."""
        return {
            "mode": self.mode.value,
            "feature_flags": self.feature_flags.get_statistics(),
            "feature_toggles": self.feature_toggle.get_statistics(),
        }


