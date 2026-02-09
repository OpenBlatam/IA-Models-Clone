"""
Feature Flags Service
Manage feature flags for gradual rollouts
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class FeatureFlag:
    """Represents a feature flag"""
    
    def __init__(
        self,
        name: str,
        enabled: bool = False,
        rollout_percentage: float = 0.0,
        user_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.enabled = enabled
        self.rollout_percentage = rollout_percentage
        self.user_ids = user_ids or []
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "rollout_percentage": self.rollout_percentage,
            "user_ids": self.user_ids,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


class FeatureFlagsService:
    """Manages feature flags"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = Path(config_file) if config_file else Path("/tmp/faceless_video/feature_flags.json")
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        self.flags: Dict[str, FeatureFlag] = {}
        self.load_flags()
    
    def load_flags(self):
        """Load feature flags from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    for flag_data in data.get("flags", []):
                        flag = FeatureFlag(
                            name=flag_data["name"],
                            enabled=flag_data.get("enabled", False),
                            rollout_percentage=flag_data.get("rollout_percentage", 0.0),
                            user_ids=flag_data.get("user_ids", []),
                            metadata=flag_data.get("metadata", {})
                        )
                        self.flags[flag.name] = flag
                logger.info(f"Loaded {len(self.flags)} feature flags")
            except Exception as e:
                logger.warning(f"Failed to load feature flags: {str(e)}")
    
    def save_flags(self):
        """Save feature flags to file"""
        try:
            data = {
                "flags": [flag.to_dict() for flag in self.flags.values()]
            }
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info("Feature flags saved")
        except Exception as e:
            logger.error(f"Failed to save feature flags: {str(e)}")
    
    def is_enabled(self, flag_name: str, user_id: Optional[str] = None) -> bool:
        """
        Check if feature flag is enabled
        
        Args:
            flag_name: Feature flag name
            user_id: User ID (optional, for user-specific flags)
            
        Returns:
            True if enabled
        """
        flag = self.flags.get(flag_name)
        if not flag:
            return False
        
        if not flag.enabled:
            return False
        
        # Check user-specific list
        if flag.user_ids and user_id:
            return user_id in flag.user_ids
        
        # Check rollout percentage
        if flag.rollout_percentage > 0:
            import hashlib
            # Deterministic hash based on user_id or flag name
            hash_input = f"{flag_name}:{user_id or 'default'}"
            hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
            percentage = (hash_value % 100) / 100.0
            return percentage < flag.rollout_percentage / 100.0
        
        return True
    
    def create_flag(
        self,
        name: str,
        enabled: bool = False,
        rollout_percentage: float = 0.0,
        user_ids: Optional[List[str]] = None
    ) -> FeatureFlag:
        """Create new feature flag"""
        flag = FeatureFlag(
            name=name,
            enabled=enabled,
            rollout_percentage=rollout_percentage,
            user_ids=user_ids
        )
        self.flags[name] = flag
        self.save_flags()
        logger.info(f"Created feature flag: {name}")
        return flag
    
    def update_flag(
        self,
        name: str,
        enabled: Optional[bool] = None,
        rollout_percentage: Optional[float] = None,
        user_ids: Optional[List[str]] = None
    ) -> FeatureFlag:
        """Update feature flag"""
        flag = self.flags.get(name)
        if not flag:
            raise ValueError(f"Feature flag {name} not found")
        
        if enabled is not None:
            flag.enabled = enabled
        if rollout_percentage is not None:
            flag.rollout_percentage = rollout_percentage
        if user_ids is not None:
            flag.user_ids = user_ids
        
        self.save_flags()
        logger.info(f"Updated feature flag: {name}")
        return flag
    
    def list_flags(self) -> List[Dict[str, Any]]:
        """List all feature flags"""
        return [flag.to_dict() for flag in self.flags.values()]


_feature_flags_service: Optional[FeatureFlagsService] = None


def get_feature_flags_service(config_file: Optional[str] = None) -> FeatureFlagsService:
    """Get feature flags service instance (singleton)"""
    global _feature_flags_service
    if _feature_flags_service is None:
        _feature_flags_service = FeatureFlagsService(config_file=config_file)
    return _feature_flags_service

