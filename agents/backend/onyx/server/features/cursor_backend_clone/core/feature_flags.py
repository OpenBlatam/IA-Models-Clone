"""
Feature Flags - Sistema de Feature Flags
========================================

Sistema para habilitar/deshabilitar features dinámicamente.
"""

import logging
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class FeatureFlagType(Enum):
    """Tipos de feature flags"""
    BOOLEAN = "boolean"
    PERCENTAGE = "percentage"
    USER_LIST = "user_list"
    CONDITIONAL = "conditional"


@dataclass
class FeatureFlag:
    """Feature flag"""
    name: str
    enabled: bool
    flag_type: FeatureFlagType = FeatureFlagType.BOOLEAN
    percentage: float = 100.0  # Para flags de porcentaje
    user_list: List[str] = field(default_factory=list)  # Para flags de lista de usuarios
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None  # Para flags condicionales
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def is_enabled_for(self, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Verificar si el flag está habilitado para un contexto.
        
        Args:
            context: Contexto (usuario, IP, etc.)
            
        Returns:
            True si está habilitado
        """
        if not self.enabled:
            return False
        
        context = context or {}
        
        if self.flag_type == FeatureFlagType.BOOLEAN:
            return self.enabled
        
        elif self.flag_type == FeatureFlagType.PERCENTAGE:
            # Usar hash del contexto para determinismo
            import hashlib
            context_str = str(context.get("user_id", "") or context.get("ip", ""))
            hash_value = int(hashlib.md5(context_str.encode()).hexdigest(), 16)
            percentage = (hash_value % 100) + 1
            return percentage <= self.percentage
        
        elif self.flag_type == FeatureFlagType.USER_LIST:
            user_id = context.get("user_id") or context.get("username")
            return user_id in self.user_list if user_id else False
        
        elif self.flag_type == FeatureFlagType.CONDITIONAL:
            if self.condition:
                try:
                    return self.condition(context)
                except Exception as e:
                    logger.error(f"Error evaluating feature flag condition: {e}")
                    return False
        
        return False


class FeatureFlagManager:
    """
    Gestor de feature flags.
    
    Permite habilitar/deshabilitar features dinámicamente.
    """
    
    def __init__(self):
        self.flags: Dict[str, FeatureFlag] = {}
        self.default_enabled = False
    
    def register(
        self,
        name: str,
        enabled: bool = False,
        flag_type: FeatureFlagType = FeatureFlagType.BOOLEAN,
        **kwargs
    ) -> FeatureFlag:
        """
        Registrar feature flag.
        
        Args:
            name: Nombre del flag
            enabled: Si está habilitado por defecto
            flag_type: Tipo de flag
            **kwargs: Argumentos adicionales (percentage, user_list, condition, metadata)
            
        Returns:
            Feature flag creado
        """
        flag = FeatureFlag(
            name=name,
            enabled=enabled,
            flag_type=flag_type,
            percentage=kwargs.get("percentage", 100.0),
            user_list=kwargs.get("user_list", []),
            condition=kwargs.get("condition"),
            metadata=kwargs.get("metadata", {})
        )
        
        self.flags[name] = flag
        logger.info(f"🚩 Feature flag registered: {name} (enabled={enabled})")
        return flag
    
    def enable(self, name: str) -> bool:
        """
        Habilitar feature flag.
        
        Args:
            name: Nombre del flag
            
        Returns:
            True si se habilitó, False si no existe
        """
        if name in self.flags:
            self.flags[name].enabled = True
            self.flags[name].updated_at = datetime.now()
            logger.info(f"🚩 Feature flag enabled: {name}")
            return True
        return False
    
    def disable(self, name: str) -> bool:
        """
        Deshabilitar feature flag.
        
        Args:
            name: Nombre del flag
            
        Returns:
            True si se deshabilitó, False si no existe
        """
        if name in self.flags:
            self.flags[name].enabled = False
            self.flags[name].updated_at = datetime.now()
            logger.info(f"🚩 Feature flag disabled: {name}")
            return True
        return False
    
    def is_enabled(
        self,
        name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Verificar si un feature flag está habilitado.
        
        Args:
            name: Nombre del flag
            context: Contexto opcional
            
        Returns:
            True si está habilitado
        """
        if name not in self.flags:
            return self.default_enabled
        
        return self.flags[name].is_enabled_for(context)
    
    def get_flag(self, name: str) -> Optional[FeatureFlag]:
        """Obtener feature flag"""
        return self.flags.get(name)
    
    def get_all_flags(self) -> Dict[str, FeatureFlag]:
        """Obtener todos los flags"""
        return self.flags.copy()
    
    def update_percentage(self, name: str, percentage: float) -> bool:
        """
        Actualizar porcentaje de un flag.
        
        Args:
            name: Nombre del flag
            percentage: Nuevo porcentaje (0-100)
            
        Returns:
            True si se actualizó
        """
        if name in self.flags and self.flags[name].flag_type == FeatureFlagType.PERCENTAGE:
            self.flags[name].percentage = max(0.0, min(100.0, percentage))
            self.flags[name].updated_at = datetime.now()
            logger.info(f"🚩 Feature flag percentage updated: {name} = {percentage}%")
            return True
        return False
    
    def add_user_to_list(self, name: str, user_id: str) -> bool:
        """
        Agregar usuario a lista de flag.
        
        Args:
            name: Nombre del flag
            user_id: ID de usuario
            
        Returns:
            True si se agregó
        """
        if name in self.flags and self.flags[name].flag_type == FeatureFlagType.USER_LIST:
            if user_id not in self.flags[name].user_list:
                self.flags[name].user_list.append(user_id)
                self.flags[name].updated_at = datetime.now()
                logger.info(f"🚩 User added to feature flag: {name} - {user_id}")
                return True
        return False
    
    def remove_user_from_list(self, name: str, user_id: str) -> bool:
        """
        Remover usuario de lista de flag.
        
        Args:
            name: Nombre del flag
            user_id: ID de usuario
            
        Returns:
            True si se removió
        """
        if name in self.flags and self.flags[name].flag_type == FeatureFlagType.USER_LIST:
            if user_id in self.flags[name].user_list:
                self.flags[name].user_list.remove(user_id)
                self.flags[name].updated_at = datetime.now()
                logger.info(f"🚩 User removed from feature flag: {name} - {user_id}")
                return True
        return False




