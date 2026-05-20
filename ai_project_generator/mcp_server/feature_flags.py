"""
MCP Feature Flags - Sistema de feature flags
============================================
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime

logger = logging.getLogger(__name__)


class FeatureFlagStatus(str, Enum):
    """Estados de feature flag"""
    ENABLED = "enabled"
    DISABLED = "disabled"
    ROLLOUT = "rollout"  # Rollout gradual


class FeatureFlag(BaseModel):
    """Feature flag"""
    flag_id: str = Field(..., description="ID único del flag")
    name: str = Field(..., description="Nombre del flag")
    status: FeatureFlagStatus = Field(default=FeatureFlagStatus.DISABLED)
    rollout_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    enabled_for_users: List[str] = Field(default_factory=list)
    disabled_for_users: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class FeatureFlagManager:
    """
    Gestor de feature flags
    
    Permite habilitar/deshabilitar funcionalidades dinámicamente.
    """
    
    def __init__(self):
        self._flags: Dict[str, FeatureFlag] = {}
    
    def register_flag(self, flag: FeatureFlag):
        """
        Registra un feature flag
        
        Args:
            flag: Feature flag
        """
        self._flags[flag.flag_id] = flag
        logger.info(f"Registered feature flag: {flag.name}")
    
    def is_enabled(
        self,
        flag_id: str,
        user_id: Optional[str] = None,
    ) -> bool:
        """
        Verifica si un feature flag está habilitado
        
        Args:
            flag_id: ID del flag
            user_id: ID del usuario (opcional)
            
        Returns:
            True si está habilitado
        """
        flag = self._flags.get(flag_id)
        
        if not flag:
            return False
        
        if flag.status == FeatureFlagStatus.DISABLED:
            return False
        
        if flag.status == FeatureFlagStatus.ENABLED:
            # Verificar exclusiones
            if user_id and user_id in flag.disabled_for_users:
                return False
            return True
        
        if flag.status == FeatureFlagStatus.ROLLOUT:
            # Verificar inclusiones explícitas
            if user_id and user_id in flag.enabled_for_users:
                return True
            
            # Verificar exclusiones
            if user_id and user_id in flag.disabled_for_users:
                return False
            
            # Rollout basado en porcentaje
            if user_id:
                import hashlib
                hash_value = int(hashlib.md5(f"{flag_id}:{user_id}".encode()).hexdigest(), 16)
                user_percentage = (hash_value % 100) + 1
                return user_percentage <= flag.rollout_percentage
            
            return False
        
        return False
    
    def enable_flag(self, flag_id: str):
        """Habilita un flag"""
        if flag_id in self._flags:
            self._flags[flag_id].status = FeatureFlagStatus.ENABLED
            logger.info(f"Enabled feature flag: {flag_id}")
    
    def disable_flag(self, flag_id: str):
        """Deshabilita un flag"""
        if flag_id in self._flags:
            self._flags[flag_id].status = FeatureFlagStatus.DISABLED
            logger.info(f"Disabled feature flag: {flag_id}")
    
    def set_rollout(self, flag_id: str, percentage: float):
        """
        Configura rollout gradual
        
        Args:
            flag_id: ID del flag
            percentage: Porcentaje (0-100)
        """
        if flag_id in self._flags:
            self._flags[flag_id].status = FeatureFlagStatus.ROLLOUT
            self._flags[flag_id].rollout_percentage = percentage
            logger.info(f"Set rollout for {flag_id}: {percentage}%")
    
    def get_flag(self, flag_id: str) -> Optional[FeatureFlag]:
        """Obtiene un flag"""
        return self._flags.get(flag_id)
    
    def list_flags(self) -> List[FeatureFlag]:
        """Lista todos los flags"""
        return list(self._flags.values())


def feature_flag(flag_id: str, manager: FeatureFlagManager):
    """
    Decorador para feature flags
    
    Args:
        flag_id: ID del flag
        manager: FeatureFlagManager
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            user_id = kwargs.get("user_id") or (args[0].user_id if args else None)
            
            if not manager.is_enabled(flag_id, user_id):
                raise Exception(f"Feature flag {flag_id} is not enabled")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

