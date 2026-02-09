"""
Sistema de feature flags para Robot Movement AI v2.0
Control de características mediante flags
"""

from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class FeatureFlagType(str, Enum):
    """Tipos de feature flags"""
    BOOLEAN = "boolean"
    PERCENTAGE = "percentage"
    USER_LIST = "user_list"
    CUSTOM = "custom"


@dataclass
class FeatureFlag:
    """Representa un feature flag"""
    name: str
    enabled: bool = True
    flag_type: FeatureFlagType = FeatureFlagType.BOOLEAN
    percentage: int = 100  # Para tipo PERCENTAGE
    user_list: list = field(default_factory=list)  # Para tipo USER_LIST
    custom_check: Optional[Callable] = None  # Para tipo CUSTOM
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class FeatureFlagManager:
    """Gestor de feature flags"""
    
    def __init__(self):
        """Inicializar gestor"""
        self.flags: Dict[str, FeatureFlag] = {}
    
    def register(self, flag: FeatureFlag):
        """Registrar feature flag"""
        self.flags[flag.name] = flag
    
    def is_enabled(
        self,
        flag_name: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Verificar si un feature flag está habilitado
        
        Args:
            flag_name: Nombre del flag
            user_id: ID de usuario (opcional)
            context: Contexto adicional (opcional)
            
        Returns:
            True si está habilitado
        """
        flag = self.flags.get(flag_name)
        
        if not flag:
            return False  # Por defecto, deshabilitado si no existe
        
        if not flag.enabled:
            return False
        
        # Verificar según tipo
        if flag.flag_type == FeatureFlagType.BOOLEAN:
            return flag.enabled
        
        elif flag.flag_type == FeatureFlagType.PERCENTAGE:
            # Rollout por porcentaje (usando hash de user_id o contexto)
            if user_id:
                hash_value = hash(user_id) % 100
                return hash_value < flag.percentage
            return False
        
        elif flag.flag_type == FeatureFlagType.USER_LIST:
            return user_id in flag.user_list if user_id else False
        
        elif flag.flag_type == FeatureFlagType.CUSTOM:
            if flag.custom_check:
                return flag.custom_check(user_id, context or {})
            return False
        
        return False
    
    def enable(self, flag_name: str):
        """Habilitar feature flag"""
        if flag_name in self.flags:
            self.flags[flag_name].enabled = True
            self.flags[flag_name].updated_at = datetime.now()
    
    def disable(self, flag_name: str):
        """Deshabilitar feature flag"""
        if flag_name in self.flags:
            self.flags[flag_name].enabled = False
            self.flags[flag_name].updated_at = datetime.now()
    
    def set_percentage(self, flag_name: str, percentage: int):
        """Establecer porcentaje para rollout gradual"""
        if flag_name in self.flags:
            self.flags[flag_name].percentage = max(0, min(100, percentage))
            self.flags[flag_name].updated_at = datetime.now()
    
    def get_all_flags(self) -> Dict[str, FeatureFlag]:
        """Obtener todos los flags"""
        return self.flags.copy()


# Instancia global
_feature_flag_manager: Optional[FeatureFlagManager] = None


def get_feature_flag_manager() -> FeatureFlagManager:
    """Obtener instancia global del gestor de feature flags"""
    global _feature_flag_manager
    if _feature_flag_manager is None:
        _feature_flag_manager = FeatureFlagManager()
    return _feature_flag_manager


def feature_flag(flag_name: str, user_id: Optional[str] = None):
    """
    Decorator para proteger funciones con feature flag
    
    Usage:
        @feature_flag("new_feature", user_id="user-1")
        async def new_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        import functools
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            manager = get_feature_flag_manager()
            if not manager.is_enabled(flag_name, user_id):
                raise Exception(f"Feature flag '{flag_name}' is not enabled")
            return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            manager = get_feature_flag_manager()
            if not manager.is_enabled(flag_name, user_id):
                raise Exception(f"Feature flag '{flag_name}' is not enabled")
            return func(*args, **kwargs)
        
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator




