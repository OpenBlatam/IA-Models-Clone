"""
Feature Toggle System for Color Grading AI
===========================================

Advanced feature toggle and A/B testing system.
"""

import logging
import random
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ToggleType(Enum):
    """Toggle types."""
    BOOLEAN = "boolean"  # Simple on/off
    PERCENTAGE = "percentage"  # Percentage rollout
    USER_BASED = "user_based"  # User-specific
    TIME_BASED = "time_based"  # Time-based
    CUSTOM = "custom"  # Custom logic


@dataclass
class FeatureToggle:
    """Feature toggle definition."""
    name: str
    toggle_type: ToggleType
    enabled: bool = False
    percentage: float = 0.0  # 0.0 - 100.0
    user_list: List[str] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    custom_logic: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class FeatureToggleSystem:
    """
    Feature toggle system.
    
    Features:
    - Multiple toggle types
    - A/B testing support
    - Percentage rollouts
    - User-based toggles
    - Time-based toggles
    - Custom logic
    """
    
    def __init__(self):
        """Initialize feature toggle system."""
        self._toggles: Dict[str, FeatureToggle] = {}
    
    def register_toggle(
        self,
        name: str,
        toggle_type: ToggleType,
        enabled: bool = False,
        **kwargs
    ):
        """
        Register a feature toggle.
        
        Args:
            name: Toggle name
            toggle_type: Toggle type
            enabled: Whether enabled by default
            **kwargs: Additional parameters
        """
        toggle = FeatureToggle(
            name=name,
            toggle_type=toggle_type,
            enabled=enabled,
            percentage=kwargs.get("percentage", 0.0),
            user_list=kwargs.get("user_list", []),
            start_time=kwargs.get("start_time"),
            end_time=kwargs.get("end_time"),
            custom_logic=kwargs.get("custom_logic"),
            metadata=kwargs.get("metadata", {})
        )
        
        self._toggles[name] = toggle
        logger.info(f"Registered feature toggle: {name} ({toggle_type.value})")
    
    def is_enabled(
        self,
        toggle_name: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if feature toggle is enabled.
        
        Args:
            toggle_name: Toggle name
            user_id: Optional user ID
            context: Optional context
            
        Returns:
            True if enabled
        """
        toggle = self._toggles.get(toggle_name)
        if not toggle:
            return False
        
        if not toggle.enabled:
            return False
        
        # Check based on toggle type
        if toggle.toggle_type == ToggleType.BOOLEAN:
            return toggle.enabled
        
        elif toggle.toggle_type == ToggleType.PERCENTAGE:
            random_value = random.random() * 100.0
            return random_value <= toggle.percentage
        
        elif toggle.toggle_type == ToggleType.USER_BASED:
            if user_id:
                return user_id in toggle.user_list
            return False
        
        elif toggle.toggle_type == ToggleType.TIME_BASED:
            now = datetime.now()
            if toggle.start_time and now < toggle.start_time:
                return False
            if toggle.end_time and now > toggle.end_time:
                return False
            return True
        
        elif toggle.toggle_type == ToggleType.CUSTOM:
            if toggle.custom_logic:
                try:
                    return toggle.custom_logic(user_id, context or {})
                except Exception as e:
                    logger.error(f"Custom toggle logic error: {e}")
                    return False
        
        return False
    
    def enable(self, toggle_name: str):
        """Enable a toggle."""
        toggle = self._toggles.get(toggle_name)
        if toggle:
            toggle.enabled = True
            logger.info(f"Enabled toggle: {toggle_name}")
    
    def disable(self, toggle_name: str):
        """Disable a toggle."""
        toggle = self._toggles.get(toggle_name)
        if toggle:
            toggle.enabled = False
            logger.info(f"Disabled toggle: {toggle_name}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get toggle statistics."""
        enabled_count = sum(1 for t in self._toggles.values() if t.enabled)
        return {
            "total_toggles": len(self._toggles),
            "enabled_toggles": enabled_count,
            "disabled_toggles": len(self._toggles) - enabled_count,
        }


