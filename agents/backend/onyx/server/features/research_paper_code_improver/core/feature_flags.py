"""
Feature Flags - Sistema de feature flags para control de funcionalidades
==========================================================================
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class FlagType(Enum):
    """Tipos de feature flags"""
    BOOLEAN = "boolean"
    PERCENTAGE = "percentage"
    TARGETED = "targeted"
    EXPERIMENT = "experiment"


@dataclass
class FeatureFlag:
    """Feature flag individual"""
    key: str
    name: str
    description: str
    type: FlagType
    enabled: bool = False
    percentage: int = 0  # 0-100 para percentage rollout
    target_users: List[str] = field(default_factory=list)
    target_organizations: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "key": self.key,
            "name": self.name,
            "description": self.description,
            "type": self.type.value,
            "enabled": self.enabled,
            "percentage": self.percentage,
            "target_users": self.target_users,
            "target_organizations": self.target_organizations,
            "conditions": self.conditions,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }


class FeatureFlags:
    """Sistema de feature flags"""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.flags: Dict[str, FeatureFlag] = {}
        self.storage_path = storage_path
        self._load_flags()
    
    def _load_flags(self):
        """Carga flags desde almacenamiento"""
        if not self.storage_path:
            return
        
        try:
            import os
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    for flag_data in data:
                        flag = FeatureFlag(
                            key=flag_data["key"],
                            name=flag_data["name"],
                            description=flag_data["description"],
                            type=FlagType(flag_data["type"]),
                            enabled=flag_data.get("enabled", False),
                            percentage=flag_data.get("percentage", 0),
                            target_users=flag_data.get("target_users", []),
                            target_organizations=flag_data.get("target_organizations", []),
                            conditions=flag_data.get("conditions", {}),
                            created_at=datetime.fromisoformat(flag_data.get("created_at", datetime.now().isoformat())),
                            updated_at=datetime.fromisoformat(flag_data.get("updated_at", datetime.now().isoformat())),
                            metadata=flag_data.get("metadata", {})
                        )
                        self.flags[flag.key] = flag
        except Exception as e:
            logger.error(f"Error cargando feature flags: {e}")
    
    def _save_flags(self):
        """Guarda flags en almacenamiento"""
        if not self.storage_path:
            return
        
        try:
            import os
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump([f.to_dict() for f in self.flags.values()], f, indent=2)
        except Exception as e:
            logger.error(f"Error guardando feature flags: {e}")
    
    def create_flag(
        self,
        key: str,
        name: str,
        description: str,
        flag_type: FlagType = FlagType.BOOLEAN,
        enabled: bool = False,
        **kwargs
    ) -> FeatureFlag:
        """Crea un nuevo feature flag"""
        if key in self.flags:
            raise ValueError(f"Feature flag {key} ya existe")
        
        flag = FeatureFlag(
            key=key,
            name=name,
            description=description,
            type=flag_type,
            enabled=enabled,
            percentage=kwargs.get("percentage", 0),
            target_users=kwargs.get("target_users", []),
            target_organizations=kwargs.get("target_organizations", []),
            conditions=kwargs.get("conditions", {}),
            metadata=kwargs.get("metadata", {})
        )
        
        self.flags[key] = flag
        self._save_flags()
        logger.info(f"Feature flag {key} creado")
        return flag
    
    def update_flag(
        self,
        key: str,
        enabled: Optional[bool] = None,
        percentage: Optional[int] = None,
        target_users: Optional[List[str]] = None,
        target_organizations: Optional[List[str]] = None,
        conditions: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> bool:
        """Actualiza un feature flag"""
        if key not in self.flags:
            return False
        
        flag = self.flags[key]
        
        if enabled is not None:
            flag.enabled = enabled
        if percentage is not None:
            flag.percentage = max(0, min(100, percentage))
        if target_users is not None:
            flag.target_users = target_users
        if target_organizations is not None:
            flag.target_organizations = target_organizations
        if conditions is not None:
            flag.conditions = conditions
        
        flag.updated_at = datetime.now()
        self._save_flags()
        logger.info(f"Feature flag {key} actualizado")
        return True
    
    def is_enabled(
        self,
        key: str,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Verifica si un feature flag está habilitado"""
        if key not in self.flags:
            return False
        
        flag = self.flags[key]
        
        # Si está deshabilitado globalmente
        if not flag.enabled:
            return False
        
        # Boolean flag: siempre habilitado si enabled=True
        if flag.type == FlagType.BOOLEAN:
            return True
        
        # Percentage rollout
        if flag.type == FlagType.PERCENTAGE:
            if user_id:
                # Hash del user_id para consistencia
                hash_value = hash(f"{key}:{user_id}") % 100
                return hash_value < flag.percentage
            return False
        
        # Targeted flag
        if flag.type == FlagType.TARGETED:
            if user_id and user_id in flag.target_users:
                return True
            if organization_id and organization_id in flag.target_organizations:
                return True
            return False
        
        # Experiment flag: evalúa condiciones
        if flag.type == FlagType.EXPERIMENT:
            if not context:
                return False
            
            # Evaluar condiciones
            for condition_key, condition_value in flag.conditions.items():
                if condition_key not in context:
                    return False
                if context[condition_key] != condition_value:
                    return False
            
            return True
        
        return False
    
    def delete_flag(self, key: str) -> bool:
        """Elimina un feature flag"""
        if key not in self.flags:
            return False
        
        del self.flags[key]
        self._save_flags()
        logger.info(f"Feature flag {key} eliminado")
        return True
    
    def get_flag(self, key: str) -> Optional[FeatureFlag]:
        """Obtiene un feature flag"""
        return self.flags.get(key)
    
    def list_flags(self) -> List[Dict[str, Any]]:
        """Lista todos los feature flags"""
        return [f.to_dict() for f in self.flags.values()]
    
    def get_enabled_flags(
        self,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Obtiene todos los feature flags habilitados para un contexto"""
        enabled = []
        for key in self.flags:
            if self.is_enabled(key, user_id, organization_id, context):
                enabled.append(key)
        return enabled




