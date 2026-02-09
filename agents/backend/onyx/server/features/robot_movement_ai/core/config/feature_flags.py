"""
Feature Flags System
====================

Sistema de feature flags (banderas de características).
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class FeatureStatus(Enum):
    """Estado de feature."""
    ENABLED = "enabled"
    DISABLED = "disabled"
    EXPERIMENTAL = "experimental"
    DEPRECATED = "deprecated"


@dataclass
class FeatureFlag:
    """Feature flag."""
    flag_id: str
    name: str
    description: str
    status: FeatureStatus
    enabled: bool = True
    rollout_percentage: float = 100.0  # 0-100
    conditions: List[Callable[[Dict[str, Any]], bool]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


class FeatureFlagManager:
    """
    Gestor de feature flags.
    
    Gestiona feature flags del sistema.
    """
    
    def __init__(self):
        """Inicializar gestor de feature flags."""
        self.flags: Dict[str, FeatureFlag] = {}
        self.usage_history: List[Dict[str, Any]] = []
    
    def create_flag(
        self,
        flag_id: str,
        name: str,
        description: str,
        status: FeatureStatus = FeatureStatus.ENABLED,
        enabled: bool = True,
        rollout_percentage: float = 100.0,
        conditions: Optional[List[Callable]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FeatureFlag:
        """
        Crear feature flag.
        
        Args:
            flag_id: ID único del flag
            name: Nombre del flag
            description: Descripción
            status: Estado del flag
            enabled: Si está habilitado
            rollout_percentage: Porcentaje de rollout (0-100)
            conditions: Condiciones adicionales
            metadata: Metadata
            
        Returns:
            Feature flag creado
        """
        flag = FeatureFlag(
            flag_id=flag_id,
            name=name,
            description=description,
            status=status,
            enabled=enabled,
            rollout_percentage=rollout_percentage,
            conditions=conditions or [],
            metadata=metadata or {}
        )
        
        self.flags[flag_id] = flag
        logger.info(f"Created feature flag: {name} ({flag_id})")
        
        return flag
    
    def is_enabled(
        self,
        flag_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Verificar si feature flag está habilitado.
        
        Args:
            flag_id: ID del flag
            context: Contexto adicional
            
        Returns:
            True si está habilitado
        """
        if flag_id not in self.flags:
            return False
        
        flag = self.flags[flag_id]
        
        # Verificar estado básico
        if not flag.enabled or flag.status == FeatureStatus.DISABLED:
            return False
        
        # Verificar rollout percentage
        import random
        if random.random() * 100 > flag.rollout_percentage:
            return False
        
        # Verificar condiciones
        if context and flag.conditions:
            for condition in flag.conditions:
                try:
                    if not condition(context):
                        return False
                except Exception as e:
                    logger.error(f"Error in feature flag condition: {e}")
                    return False
        
        # Registrar uso
        self._record_usage(flag_id, True, context)
        
        return True
    
    def _record_usage(
        self,
        flag_id: str,
        enabled: bool,
        context: Optional[Dict[str, Any]]
    ) -> None:
        """Registrar uso de feature flag."""
        self.usage_history.append({
            "flag_id": flag_id,
            "enabled": enabled,
            "context": context or {},
            "timestamp": datetime.now().isoformat()
        })
    
    def enable_flag(self, flag_id: str) -> bool:
        """Habilitar feature flag."""
        if flag_id in self.flags:
            self.flags[flag_id].enabled = True
            self.flags[flag_id].updated_at = datetime.now().isoformat()
            return True
        return False
    
    def disable_flag(self, flag_id: str) -> bool:
        """Deshabilitar feature flag."""
        if flag_id in self.flags:
            self.flags[flag_id].enabled = False
            self.flags[flag_id].updated_at = datetime.now().isoformat()
            return True
        return False
    
    def get_flag(self, flag_id: str) -> Optional[FeatureFlag]:
        """Obtener feature flag."""
        return self.flags.get(flag_id)
    
    def list_flags(self) -> List[FeatureFlag]:
        """Listar todos los feature flags."""
        return list(self.flags.values())
    
    def get_usage_statistics(self, flag_id: Optional[str] = None) -> Dict[str, Any]:
        """Obtener estadísticas de uso."""
        if flag_id:
            usage = [u for u in self.usage_history if u["flag_id"] == flag_id]
        else:
            usage = self.usage_history
        
        if not usage:
            return {
                "total_checks": 0,
                "enabled_count": 0,
                "disabled_count": 0
            }
        
        enabled_count = sum(1 for u in usage if u["enabled"])
        
        return {
            "total_checks": len(usage),
            "enabled_count": enabled_count,
            "disabled_count": len(usage) - enabled_count,
            "enabled_rate": enabled_count / len(usage) if usage else 0.0
        }


# Instancia global
_feature_flag_manager: Optional[FeatureFlagManager] = None


def get_feature_flag_manager() -> FeatureFlagManager:
    """Obtener instancia global del gestor de feature flags."""
    global _feature_flag_manager
    if _feature_flag_manager is None:
        _feature_flag_manager = FeatureFlagManager()
    return _feature_flag_manager






