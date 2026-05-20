"""
Advanced Feature Flags - Sistema de feature flags avanzado
==========================================================
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class FeatureFlagType(str, Enum):
    """Tipos de feature flags"""
    BOOLEAN = "boolean"
    PERCENTAGE = "percentage"
    USER_LIST = "user_list"
    CUSTOM = "custom"


class AdvancedFeatureFlags:
    """Sistema avanzado de feature flags"""
    
    def __init__(self):
        self.flags: Dict[str, Dict[str, Any]] = {}
        self.user_overrides: Dict[str, Dict[str, Any]] = {}
        self.environment_flags: Dict[str, Dict[str, bool]] = {}
    
    def create_flag(self, flag_name: str, flag_type: FeatureFlagType,
                   default_value: Any, description: str = "",
                   enabled: bool = True) -> Dict[str, Any]:
        """Crea un feature flag"""
        flag = {
            "name": flag_name,
            "type": flag_type.value,
            "default_value": default_value,
            "description": description,
            "enabled": enabled,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": {}
        }
        
        self.flags[flag_name] = flag
        
        logger.info(f"Feature flag creado: {flag_name}")
        return flag
    
    def get_flag_value(self, flag_name: str, user_id: Optional[str] = None,
                      environment: Optional[str] = None) -> Any:
        """Obtiene valor de un feature flag"""
        flag = self.flags.get(flag_name)
        if not flag:
            return None
        
        if not flag["enabled"]:
            return flag["default_value"]
        
        # Verificar override de usuario
        if user_id and user_id in self.user_overrides.get(flag_name, {}):
            return self.user_overrides[flag_name][user_id]
        
        # Verificar override de ambiente
        if environment and environment in self.environment_flags.get(flag_name, {}):
            return self.environment_flags[flag_name][environment]
        
        # Aplicar lógica según tipo
        if flag["type"] == FeatureFlagType.BOOLEAN.value:
            return flag["default_value"]
        
        elif flag["type"] == FeatureFlagType.PERCENTAGE.value:
            # Rollout por porcentaje (simplificado)
            percentage = flag["default_value"]
            # En producción, usar hash del user_id para consistencia
            return percentage > 50  # Simplificado
        
        elif flag["type"] == FeatureFlagType.USER_LIST.value:
            # Lista de usuarios
            user_list = flag.get("metadata", {}).get("user_list", [])
            return user_id in user_list if user_id else False
        
        return flag["default_value"]
    
    def set_user_override(self, flag_name: str, user_id: str, value: Any):
        """Establece override para un usuario"""
        if flag_name not in self.user_overrides:
            self.user_overrides[flag_name] = {}
        
        self.user_overrides[flag_name][user_id] = value
        logger.info(f"Override establecido: {flag_name} para usuario {user_id}")
    
    def set_environment_override(self, flag_name: str, environment: str, value: bool):
        """Establece override para un ambiente"""
        if flag_name not in self.environment_flags:
            self.environment_flags[flag_name] = {}
        
        self.environment_flags[flag_name][environment] = value
        logger.info(f"Override establecido: {flag_name} para ambiente {environment}")
    
    def enable_flag(self, flag_name: str):
        """Habilita un feature flag"""
        if flag_name in self.flags:
            self.flags[flag_name]["enabled"] = True
            self.flags[flag_name]["updated_at"] = datetime.now().isoformat()
            logger.info(f"Feature flag habilitado: {flag_name}")
    
    def disable_flag(self, flag_name: str):
        """Deshabilita un feature flag"""
        if flag_name in self.flags:
            self.flags[flag_name]["enabled"] = False
            self.flags[flag_name]["updated_at"] = datetime.now().isoformat()
            logger.info(f"Feature flag deshabilitado: {flag_name}")
    
    def list_flags(self, enabled_only: bool = False) -> List[Dict[str, Any]]:
        """Lista feature flags"""
        flags = list(self.flags.values())
        
        if enabled_only:
            flags = [f for f in flags if f["enabled"]]
        
        return flags
    
    def get_flag_stats(self, flag_name: str) -> Optional[Dict[str, Any]]:
        """Obtiene estadísticas de un feature flag"""
        flag = self.flags.get(flag_name)
        if not flag:
            return None
        
        user_overrides_count = len(self.user_overrides.get(flag_name, {}))
        environment_overrides_count = len(self.environment_flags.get(flag_name, {}))
        
        return {
            "flag": flag,
            "user_overrides": user_overrides_count,
            "environment_overrides": environment_overrides_count
        }




