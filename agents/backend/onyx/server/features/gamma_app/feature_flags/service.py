"""
Feature Flag Service Implementation
"""

from typing import Dict, Any, Optional
import logging
import hashlib
from datetime import datetime

from .base import FeatureFlagBase, FeatureFlag, FlagCondition

logger = logging.getLogger(__name__)


class FeatureFlagService(FeatureFlagBase):
    """Feature flag service implementation"""
    
    def __init__(self, db=None, redis_client=None, config_service=None):
        """Initialize feature flag service"""
        self.db = db
        self.redis_client = redis_client
        self.config_service = config_service
        self._flags: dict = {}
    
    async def is_enabled(
        self,
        flag_name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Check if feature flag is enabled"""
        try:
            flag = await self.get_flag(flag_name)
            if not flag:
                return False
            
            if not flag.enabled:
                return False
            
            # Check rollout percentage
            if flag.rollout_percentage < 100:
                user_id = context.get("user_id", "") if context else ""
                if user_id:
                    # Consistent hashing for rollout
                    hash_value = int(
                        hashlib.md5(f"{flag_name}:{user_id}".encode()).hexdigest(),
                        16
                    )
                    percentage = (hash_value % 100) + 1
                    if percentage > flag.rollout_percentage:
                        return False
            
            # Check conditions
            if flag.conditions and context:
                for condition in flag.conditions:
                    context_value = context.get(condition.condition_type)
                    if not self._evaluate_condition(condition, context_value):
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking feature flag: {e}")
            return False
    
    async def get_flag(self, flag_name: str) -> Optional[FeatureFlag]:
        """Get feature flag"""
        return self._flags.get(flag_name)
    
    async def set_flag(self, flag: FeatureFlag) -> bool:
        """Set feature flag"""
        try:
            flag.updated_at = datetime.utcnow()
            self._flags[flag.name] = flag
            
            # Cache in Redis if available
            if self.redis_client:
                await self.redis_client.setex(
                    f"flag:{flag.name}",
                    3600,
                    str(flag.enabled)
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting feature flag: {e}")
            return False
    
    def _evaluate_condition(
        self,
        condition: FlagCondition,
        context_value: Any
    ) -> bool:
        """Evaluate condition"""
        if condition.operator == "equals":
            return context_value == condition.value
        elif condition.operator == "contains":
            return condition.value in str(context_value)
        # Add more operators as needed
        return False
