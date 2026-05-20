"""
Feature Flags - Feature Flags y A/B Testing
==========================================

Sistema de feature flags y A/B testing:
- Feature toggles
- A/B testing
- Gradual rollouts
- Feature analytics
"""

import logging
import hashlib
from typing import Optional, Dict, Any, List, Callable
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class FeatureFlagType(str, Enum):
    """Tipos de feature flags"""
    BOOLEAN = "boolean"
    PERCENTAGE = "percentage"
    USER_LIST = "user_list"
    A_B_TEST = "a_b_test"


class FeatureFlag:
    """Feature flag"""
    
    def __init__(
        self,
        name: str,
        flag_type: FeatureFlagType,
        enabled: bool = False,
        **kwargs: Any
    ) -> None:
        self.name = name
        self.flag_type = flag_type
        self.enabled = enabled
        self.config = kwargs
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def is_enabled_for_user(self, user_id: Optional[str] = None) -> bool:
        """Verifica si está habilitado para un usuario"""
        if not self.enabled:
            return False
        
        if self.flag_type == FeatureFlagType.BOOLEAN:
            return True
        
        elif self.flag_type == FeatureFlagType.PERCENTAGE:
            percentage = self.config.get("percentage", 0)
            if user_id:
                # Hash user_id para distribución consistente
                hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
                user_percentage = (hash_value % 100)
                return user_percentage < percentage
            return False
        
        elif self.flag_type == FeatureFlagType.USER_LIST:
            allowed_users = self.config.get("users", [])
            return user_id in allowed_users if user_id else False
        
        elif self.flag_type == FeatureFlagType.A_B_TEST:
            variant = self.get_ab_variant(user_id)
            return variant is not None
        
        return False
    
    def get_ab_variant(self, user_id: Optional[str] = None) -> Optional[str]:
        """Obtiene variante A/B para un usuario"""
        if self.flag_type != FeatureFlagType.A_B_TEST:
            return None
        
        variants = self.config.get("variants", {})
        if not variants or not user_id:
            return None
        
        # Distribuir usuarios consistentemente entre variantes
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        total_weight = sum(v.get("weight", 0) for v in variants.values())
        
        if total_weight == 0:
            return None
        
        user_value = hash_value % total_weight
        current = 0
        
        for variant_name, variant_config in variants.items():
            weight = variant_config.get("weight", 0)
            current += weight
            if user_value < current:
                return variant_name
        
        return None


class FeatureFlagManager:
    """
    Gestor de feature flags.
    """
    
    def __init__(self) -> None:
        self.flags: Dict[str, FeatureFlag] = {}
        self.analytics: Dict[str, List[Dict[str, Any]]] = {}
    
    def register_flag(
        self,
        name: str,
        flag_type: FeatureFlagType,
        enabled: bool = False,
        **kwargs: Any
    ) -> None:
        """Registra un feature flag"""
        flag = FeatureFlag(name, flag_type, enabled, **kwargs)
        self.flags[name] = flag
        logger.info(f"Feature flag registered: {name}")
    
    def is_enabled(
        self,
        flag_name: str,
        user_id: Optional[str] = None
    ) -> bool:
        """Verifica si un flag está habilitado"""
        flag = self.flags.get(flag_name)
        if not flag:
            return False
        
        result = flag.is_enabled_for_user(user_id)
        
        # Track analytics
        self._track_usage(flag_name, user_id, result)
        
        return result
    
    def get_variant(
        self,
        flag_name: str,
        user_id: Optional[str] = None
    ) -> Optional[str]:
        """Obtiene variante A/B"""
        flag = self.flags.get(flag_name)
        if not flag:
            return None
        
        variant = flag.get_ab_variant(user_id)
        self._track_variant(flag_name, user_id, variant)
        return variant
    
    def _track_usage(
        self,
        flag_name: str,
        user_id: Optional[str],
        enabled: bool
    ) -> None:
        """Track uso de feature flag"""
        if flag_name not in self.analytics:
            self.analytics[flag_name] = []
        
        self.analytics[flag_name].append({
            "user_id": user_id,
            "enabled": enabled,
            "timestamp": datetime.now().isoformat()
        })
    
    def _track_variant(
        self,
        flag_name: str,
        user_id: Optional[str],
        variant: Optional[str]
    ) -> None:
        """Track variante A/B"""
        if flag_name not in self.analytics:
            self.analytics[flag_name] = []
        
        self.analytics[flag_name].append({
            "user_id": user_id,
            "variant": variant,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_analytics(self, flag_name: str) -> Dict[str, Any]:
        """Obtiene analytics de un flag"""
        events = self.analytics.get(flag_name, [])
        
        if not events:
            return {
                "flag": flag_name,
                "total_checks": 0,
                "enabled_count": 0,
                "disabled_count": 0
            }
        
        enabled_count = sum(1 for e in events if e.get("enabled"))
        disabled_count = len(events) - enabled_count
        
        # Analytics de variantes
        variants = {}
        for event in events:
            variant = event.get("variant")
            if variant:
                variants[variant] = variants.get(variant, 0) + 1
        
        return {
            "flag": flag_name,
            "total_checks": len(events),
            "enabled_count": enabled_count,
            "disabled_count": disabled_count,
            "enabled_percentage": (enabled_count / len(events) * 100) if events else 0,
            "variants": variants
        }
    
    def enable_flag(self, flag_name: str) -> None:
        """Habilita un flag"""
        if flag_name in self.flags:
            self.flags[flag_name].enabled = True
            self.flags[flag_name].updated_at = datetime.now()
            logger.info(f"Feature flag enabled: {flag_name}")
    
    def disable_flag(self, flag_name: str) -> None:
        """Deshabilita un flag"""
        if flag_name in self.flags:
            self.flags[flag_name].enabled = False
            self.flags[flag_name].updated_at = datetime.now()
            logger.info(f"Feature flag disabled: {flag_name}")


def get_feature_flag_manager() -> FeatureFlagManager:
    """Obtiene gestor de feature flags"""
    return FeatureFlagManager()















