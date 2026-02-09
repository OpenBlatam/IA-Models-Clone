"""
Sistema de Feature Flags

Proporciona:
- Feature flags por usuario
- Feature flags por porcentaje
- Feature flags por atributos
- A/B testing
- Rollout gradual
"""

import logging
import hashlib
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class FlagType(Enum):
    """Tipos de feature flags"""
    BOOLEAN = "boolean"
    PERCENTAGE = "percentage"
    USER_LIST = "user_list"
    ATTRIBUTE = "attribute"


@dataclass
class FeatureFlag:
    """Representa un feature flag"""
    name: str
    flag_type: FlagType
    enabled: bool = True
    description: Optional[str] = None
    percentage: int = 100  # Para flags de porcentaje (0-100)
    user_list: List[str] = field(default_factory=list)  # Para flags de lista de usuarios
    attributes: Dict[str, Any] = field(default_factory=dict)  # Para flags por atributos
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class FeatureFlagService:
    """Servicio de feature flags"""
    
    def __init__(self):
        self.flags: Dict[str, FeatureFlag] = {}
        self._setup_default_flags()
        logger.info("FeatureFlagService initialized")
    
    def _setup_default_flags(self):
        """Configura flags por defecto"""
        # Ejemplo de flags por defecto
        self.create_flag(
            name="advanced_audio_processing",
            flag_type=FlagType.BOOLEAN,
            enabled=True,
            description="Enable advanced audio processing features"
        )
        
        self.create_flag(
            name="new_generation_model",
            flag_type=FlagType.PERCENTAGE,
            enabled=True,
            percentage=10,
            description="Rollout new generation model to 10% of users"
        )
    
    def create_flag(
        self,
        name: str,
        flag_type: FlagType,
        enabled: bool = True,
        description: Optional[str] = None,
        percentage: int = 100,
        user_list: Optional[List[str]] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> FeatureFlag:
        """
        Crea un nuevo feature flag
        
        Args:
            name: Nombre del flag
            flag_type: Tipo de flag
            enabled: Si está habilitado
            description: Descripción
            percentage: Porcentaje para flags de porcentaje
            user_list: Lista de usuarios para flags de lista
            attributes: Atributos para flags por atributos
        
        Returns:
            FeatureFlag creado
        """
        flag = FeatureFlag(
            name=name,
            flag_type=flag_type,
            enabled=enabled,
            description=description,
            percentage=percentage,
            user_list=user_list or [],
            attributes=attributes or {}
        )
        
        self.flags[name] = flag
        logger.info(f"Feature flag created: {name}")
        return flag
    
    def is_enabled(
        self,
        flag_name: str,
        user_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Verifica si un feature flag está habilitado
        
        Args:
            flag_name: Nombre del flag
            user_id: ID del usuario (opcional)
            attributes: Atributos del usuario (opcional)
        
        Returns:
            True si el flag está habilitado para el usuario
        """
        flag = self.flags.get(flag_name)
        
        if not flag:
            return False
        
        if not flag.enabled:
            return False
        
        # Verificar según el tipo de flag
        if flag.flag_type == FlagType.BOOLEAN:
            return True
        
        elif flag.flag_type == FlagType.PERCENTAGE:
            if not user_id:
                return False
            
            # Calcular hash del user_id para distribución consistente
            hash_value = int(
                hashlib.md5(f"{flag_name}:{user_id}".encode()).hexdigest(),
                16
            )
            user_percentage = (hash_value % 100) + 1
            return user_percentage <= flag.percentage
        
        elif flag.flag_type == FlagType.USER_LIST:
            return user_id in flag.user_list if user_id else False
        
        elif flag.flag_type == FlagType.ATTRIBUTE:
            if not attributes:
                return False
            
            # Verificar que todos los atributos coincidan
            for key, value in flag.attributes.items():
                if attributes.get(key) != value:
                    return False
            return True
        
        return False
    
    def update_flag(
        self,
        flag_name: str,
        enabled: Optional[bool] = None,
        percentage: Optional[int] = None,
        user_list: Optional[List[str]] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Actualiza un feature flag
        
        Args:
            flag_name: Nombre del flag
            enabled: Nuevo estado (opcional)
            percentage: Nuevo porcentaje (opcional)
            user_list: Nueva lista de usuarios (opcional)
            attributes: Nuevos atributos (opcional)
        
        Returns:
            True si se actualizó correctamente
        """
        flag = self.flags.get(flag_name)
        if not flag:
            return False
        
        if enabled is not None:
            flag.enabled = enabled
        
        if percentage is not None:
            flag.percentage = max(0, min(100, percentage))
        
        if user_list is not None:
            flag.user_list = user_list
        
        if attributes is not None:
            flag.attributes = attributes
        
        flag.updated_at = datetime.now()
        logger.info(f"Feature flag updated: {flag_name}")
        return True
    
    def delete_flag(self, flag_name: str) -> bool:
        """Elimina un feature flag"""
        if flag_name in self.flags:
            del self.flags[flag_name]
            logger.info(f"Feature flag deleted: {flag_name}")
            return True
        return False
    
    def get_flag(self, flag_name: str) -> Optional[FeatureFlag]:
        """Obtiene un feature flag"""
        return self.flags.get(flag_name)
    
    def list_flags(self) -> List[Dict[str, Any]]:
        """Lista todos los feature flags"""
        return [
            {
                "name": flag.name,
                "type": flag.flag_type.value,
                "enabled": flag.enabled,
                "description": flag.description,
                "percentage": flag.percentage,
                "user_list_count": len(flag.user_list),
                "attributes": flag.attributes,
                "created_at": flag.created_at.isoformat(),
                "updated_at": flag.updated_at.isoformat()
            }
            for flag in self.flags.values()
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de feature flags"""
        total_flags = len(self.flags)
        enabled_flags = sum(1 for f in self.flags.values() if f.enabled)
        
        by_type = {}
        for flag in self.flags.values():
            flag_type = flag.flag_type.value
            by_type[flag_type] = by_type.get(flag_type, 0) + 1
        
        return {
            "total_flags": total_flags,
            "enabled_flags": enabled_flags,
            "disabled_flags": total_flags - enabled_flags,
            "flags_by_type": by_type
        }


# Instancia global
_feature_flag_service: Optional[FeatureFlagService] = None


def get_feature_flag_service() -> FeatureFlagService:
    """Obtiene la instancia global del servicio de feature flags"""
    global _feature_flag_service
    if _feature_flag_service is None:
        _feature_flag_service = FeatureFlagService()
    return _feature_flag_service


def feature_flag(flag_name: str):
    """
    Decorator para habilitar/deshabilitar funciones basado en feature flags
    
    Usage:
        @feature_flag("new_feature")
        async def new_feature_endpoint():
            # ...
    """
    def decorator(func: Callable) -> Callable:
        async def async_wrapper(*args, **kwargs):
            service = get_feature_flag_service()
            user_id = kwargs.get("user_id") or (args[0] if args else None)
            
            if not service.is_enabled(flag_name, user_id=user_id):
                from fastapi import HTTPException, status
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Feature '{flag_name}' is not enabled"
                )
            
            return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            service = get_feature_flag_service()
            user_id = kwargs.get("user_id") or (args[0] if args else None)
            
            if not service.is_enabled(flag_name, user_id=user_id):
                from fastapi import HTTPException, status
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Feature '{flag_name}' is not enabled"
                )
            
            return func(*args, **kwargs)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator

