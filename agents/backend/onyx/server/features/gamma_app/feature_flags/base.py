"""
Feature Flags Base Classes and Interfaces
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
from uuid import uuid4
from enum import Enum


class FlagCondition:
    """Feature flag condition"""
    
    def __init__(
        self,
        condition_type: str,
        value: Any,
        operator: str = "equals"
    ):
        self.condition_type = condition_type
        self.value = value
        self.operator = operator


class FeatureFlag:
    """Feature flag definition"""
    
    def __init__(
        self,
        name: str,
        enabled: bool = False,
        rollout_percentage: int = 0,
        conditions: Optional[List[FlagCondition]] = None
    ):
        self.id = str(uuid4())
        self.name = name
        self.enabled = enabled
        self.rollout_percentage = rollout_percentage
        self.conditions = conditions or []
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()


class FeatureFlagBase(ABC):
    """Base interface for feature flags"""
    
    @abstractmethod
    async def is_enabled(
        self,
        flag_name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Check if feature flag is enabled"""
        pass
    
    @abstractmethod
    async def get_flag(self, flag_name: str) -> Optional[FeatureFlag]:
        """Get feature flag"""
        pass
    
    @abstractmethod
    async def set_flag(
        self,
        flag: FeatureFlag
    ) -> bool:
        """Set feature flag"""
        pass

