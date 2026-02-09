"""
Feature Flags Module - Feature flag management.

Provides:
- Feature flag system
- A/B testing support
- Gradual rollouts
- Environment-based flags
"""

import logging
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class FlagType(str, Enum):
    """Feature flag types."""
    BOOLEAN = "boolean"
    PERCENTAGE = "percentage"
    USER_LIST = "user_list"
    ENVIRONMENT = "environment"


@dataclass
class FeatureFlag:
    """Feature flag definition."""
    name: str
    flag_type: FlagType
    enabled: bool = False
    percentage: float = 0.0  # 0-100 for percentage rollout
    user_list: List[str] = field(default_factory=list)
    environments: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "flag_type": self.flag_type.value,
            "enabled": self.enabled,
            "percentage": self.percentage,
            "user_list": self.user_list,
            "environments": self.environments,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class FeatureFlagManager:
    """Feature flag manager."""
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize feature flag manager.
        
        Args:
            storage_path: Path to storage file
        """
        self.storage_path = Path(storage_path) if storage_path else Path("feature_flags.json")
        self.flags: Dict[str, FeatureFlag] = {}
        self.current_environment = "production"
        self._load_flags()
    
    def _load_flags(self) -> None:
        """Load flags from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    for flag_data in data:
                        flag = FeatureFlag(
                            name=flag_data["name"],
                            flag_type=FlagType(flag_data["flag_type"]),
                            enabled=flag_data.get("enabled", False),
                            percentage=flag_data.get("percentage", 0.0),
                            user_list=flag_data.get("user_list", []),
                            environments=flag_data.get("environments", []),
                            metadata=flag_data.get("metadata", {}),
                            created_at=flag_data.get("created_at", datetime.now().isoformat()),
                            updated_at=flag_data.get("updated_at", datetime.now().isoformat()),
                        )
                        self.flags[flag.name] = flag
            except Exception as e:
                logger.error(f"Error loading feature flags: {e}")
    
    def _save_flags(self) -> None:
        """Save flags to storage."""
        data = [flag.to_dict() for flag in self.flags.values()]
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def create_flag(
        self,
        name: str,
        flag_type: FlagType = FlagType.BOOLEAN,
        enabled: bool = False,
    ) -> FeatureFlag:
        """
        Create a feature flag.
        
        Args:
            name: Flag name
            flag_type: Flag type
            enabled: Initial enabled state
            
        Returns:
            Created flag
        """
        flag = FeatureFlag(
            name=name,
            flag_type=flag_type,
            enabled=enabled,
        )
        self.flags[name] = flag
        self._save_flags()
        logger.info(f"Created feature flag: {name}")
        return flag
    
    def is_enabled(
        self,
        flag_name: str,
        user_id: Optional[str] = None,
        environment: Optional[str] = None,
    ) -> bool:
        """
        Check if feature flag is enabled.
        
        Args:
            flag_name: Flag name
            user_id: Optional user ID for user-based flags
            environment: Optional environment override
            
        Returns:
            True if enabled
        """
        flag = self.flags.get(flag_name)
        if not flag:
            return False
        
        env = environment or self.current_environment
        
        # Environment check
        if flag.environments and env not in flag.environments:
            return False
        
        # Boolean flag
        if flag.flag_type == FlagType.BOOLEAN:
            return flag.enabled
        
        # Percentage rollout
        if flag.flag_type == FlagType.PERCENTAGE:
            if not flag.enabled:
                return False
            if user_id:
                # Deterministic based on user_id
                import hashlib
                hash_value = int(hashlib.md5(f"{flag_name}:{user_id}".encode()).hexdigest(), 16)
                return (hash_value % 100) < flag.percentage
            return False
        
        # User list
        if flag.flag_type == FlagType.USER_LIST:
            return flag.enabled and (user_id in flag.user_list if user_id else False)
        
        return False
    
    def enable_flag(self, flag_name: str) -> None:
        """Enable a feature flag."""
        flag = self.flags.get(flag_name)
        if flag:
            flag.enabled = True
            flag.updated_at = datetime.now().isoformat()
            self._save_flags()
            logger.info(f"Enabled feature flag: {flag_name}")
    
    def disable_flag(self, flag_name: str) -> None:
        """Disable a feature flag."""
        flag = self.flags.get(flag_name)
        if flag:
            flag.enabled = False
            flag.updated_at = datetime.now().isoformat()
            self._save_flags()
            logger.info(f"Disabled feature flag: {flag_name}")
    
    def set_percentage(self, flag_name: str, percentage: float) -> None:
        """
        Set percentage rollout.
        
        Args:
            flag_name: Flag name
            percentage: Percentage (0-100)
        """
        flag = self.flags.get(flag_name)
        if flag:
            flag.percentage = max(0.0, min(100.0, percentage))
            flag.updated_at = datetime.now().isoformat()
            self._save_flags()
            logger.info(f"Set percentage for {flag_name}: {percentage}%")
    
    def set_environment(self, environment: str) -> None:
        """Set current environment."""
        self.current_environment = environment
        logger.info(f"Set environment: {environment}")












