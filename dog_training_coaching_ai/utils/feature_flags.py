"""
Feature Flags
=============
Sistema de feature flags para control de funcionalidades.
"""

from typing import Dict, Any, Optional, Callable
from enum import Enum
from datetime import datetime
import asyncio


class FeatureFlagStatus(str, Enum):
    """Estados de feature flag."""
    ENABLED = "enabled"
    DISABLED = "disabled"
    ROLLOUT = "rollout"  # Gradual rollout


class FeatureFlag:
    """Representación de un feature flag."""
    
    def __init__(
        self,
        name: str,
        status: FeatureFlagStatus = FeatureFlagStatus.DISABLED,
        rollout_percentage: float = 0.0,
        conditions: Optional[Dict[str, Any]] = None
    ):
        """
        Inicializar feature flag.
        
        Args:
            name: Nombre del flag
            status: Estado del flag
            rollout_percentage: Porcentaje de rollout (0-100)
            conditions: Condiciones adicionales
        """
        self.name = name
        self.status = status
        self.rollout_percentage = rollout_percentage
        self.conditions = conditions or {}
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def is_enabled(self, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Verificar si el flag está habilitado.
        
        Args:
            context: Contexto adicional para evaluación
            
        Returns:
            True si está habilitado
        """
        if self.status == FeatureFlagStatus.DISABLED:
            return False
        
        if self.status == FeatureFlagStatus.ENABLED:
            return self._check_conditions(context)
        
        if self.status == FeatureFlagStatus.ROLLOUT:
            # Evaluar rollout basado en hash del contexto
            if context:
                user_id = context.get("user_id") or context.get("ip") or "default"
                hash_value = hash(user_id) % 100
                if hash_value < self.rollout_percentage:
                    return self._check_conditions(context)
            return False
        
        return False
    
    def _check_conditions(self, context: Optional[Dict[str, Any]]) -> bool:
        """Verificar condiciones adicionales."""
        if not self.conditions:
            return True
        
        if not context:
            return True
        
        # Verificar condiciones simples
        for key, expected_value in self.conditions.items():
            actual_value = context.get(key)
            if actual_value != expected_value:
                return False
        
        return True


class FeatureFlagManager:
    """Manager para feature flags."""
    
    def __init__(self):
        self.flags: Dict[str, FeatureFlag] = {}
        self._lock = asyncio.Lock()
    
    def register(self, flag: FeatureFlag):
        """Registrar feature flag."""
        self.flags[flag.name] = flag
    
    def get_flag(self, name: str) -> Optional[FeatureFlag]:
        """Obtener feature flag."""
        return self.flags.get(name)
    
    def is_enabled(self, name: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Verificar si un flag está habilitado.
        
        Args:
            name: Nombre del flag
            context: Contexto para evaluación
            
        Returns:
            True si está habilitado
        """
        flag = self.flags.get(name)
        if not flag:
            return False
        
        return flag.is_enabled(context)
    
    def update_flag(
        self,
        name: str,
        status: Optional[FeatureFlagStatus] = None,
        rollout_percentage: Optional[float] = None,
        conditions: Optional[Dict[str, Any]] = None
    ):
        """
        Actualizar feature flag.
        
        Args:
            name: Nombre del flag
            status: Nuevo estado
            rollout_percentage: Nuevo porcentaje
            conditions: Nuevas condiciones
        """
        flag = self.flags.get(name)
        if not flag:
            raise ValueError(f"Feature flag '{name}' not found")
        
        if status is not None:
            flag.status = status
        if rollout_percentage is not None:
            flag.rollout_percentage = rollout_percentage
        if conditions is not None:
            flag.conditions = conditions
        
        flag.updated_at = datetime.now()
    
    def list_flags(self) -> list:
        """Listar todos los flags."""
        return [
            {
                "name": flag.name,
                "status": flag.status.value,
                "rollout_percentage": flag.rollout_percentage,
                "conditions": flag.conditions,
                "created_at": flag.created_at.isoformat(),
                "updated_at": flag.updated_at.isoformat()
            }
            for flag in self.flags.values()
        ]


# Instancia global
_feature_flag_manager: Optional[FeatureFlagManager] = None


def get_feature_flag_manager() -> FeatureFlagManager:
    """Obtener instancia global del manager."""
    global _feature_flag_manager
    if _feature_flag_manager is None:
        _feature_flag_manager = FeatureFlagManager()
    return _feature_flag_manager

