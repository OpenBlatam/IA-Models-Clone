"""
Sistema de Feature Flags para habilitar/deshabilitar funcionalidades.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class FeatureFlag:
    """
    Representa un feature flag con validaciones.
    
    Attributes:
        name: Nombre del flag
        enabled: Si está habilitado
        description: Descripción
        rollout_percentage: Porcentaje de rollout (0-100)
        target_users: Lista de usuarios objetivo
        metadata: Metadata adicional
        created_at: Fecha de creación
        updated_at: Fecha de última actualización
    """
    
    def __init__(
        self,
        name: str,
        enabled: bool = False,
        description: Optional[str] = None,
        rollout_percentage: int = 100,
        target_users: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Inicializar feature flag con validaciones.
        
        Args:
            name: Nombre del flag (debe ser string no vacío)
            enabled: Si está habilitado (debe ser bool)
            description: Descripción (opcional, debe ser string si se proporciona)
            rollout_percentage: Porcentaje de rollout (0-100, debe ser entero)
            target_users: Lista de usuarios objetivo (opcional, debe ser lista de strings)
            metadata: Metadata adicional (opcional, debe ser diccionario)
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        # Validaciones
        if not name or not isinstance(name, str) or not name.strip():
            raise ValueError(f"name debe ser un string no vacío, recibido: {name}")
        
        if not isinstance(enabled, bool):
            raise ValueError(f"enabled debe ser un bool, recibido: {type(enabled)}")
        
        if description is not None:
            if not isinstance(description, str):
                raise ValueError(f"description debe ser un string si se proporciona, recibido: {type(description)}")
            description = description.strip() if description else None
        
        if not isinstance(rollout_percentage, int):
            raise ValueError(f"rollout_percentage debe ser un entero, recibido: {type(rollout_percentage)}")
        
        if target_users is not None:
            if not isinstance(target_users, list):
                raise ValueError(f"target_users debe ser una lista si se proporciona, recibido: {type(target_users)}")
            for user_id in target_users:
                if not isinstance(user_id, str) or not user_id.strip():
                    raise ValueError(f"Todos los elementos de target_users deben ser strings no vacíos, recibido: {user_id}")
            target_users = [u.strip() for u in target_users if u.strip()]
        
        if metadata is not None:
            if not isinstance(metadata, dict):
                raise ValueError(f"metadata debe ser un diccionario si se proporciona, recibido: {type(metadata)}")
        
        self.name = name.strip()
        self.enabled = enabled
        self.description = description
        self.rollout_percentage = max(0, min(100, rollout_percentage))
        self.target_users = target_users or []
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        
        logger.debug(f"FeatureFlag creado: {self.name} (enabled: {enabled}, rollout: {self.rollout_percentage}%)")
    
    def is_enabled_for(self, user_id: Optional[str] = None) -> bool:
        """
        Verificar si el flag está habilitado para un usuario.
        
        Args:
            user_id: ID del usuario (opcional)
            
        Returns:
            True si está habilitado
        """
        if not self.enabled:
            return False
        
        # Si hay usuarios objetivo, verificar
        if self.target_users:
            return user_id in self.target_users
        
        # Rollout porcentual
        if self.rollout_percentage < 100:
            if user_id:
                # Hash del user_id para consistencia
                import hashlib
                hash_value = int(hashlib.md5(f"{self.name}:{user_id}".encode()).hexdigest(), 16)
                return (hash_value % 100) < self.rollout_percentage
            else:
                # Sin user_id, usar porcentaje
                import random
                return random.randint(0, 100) < self.rollout_percentage
        
        return True


class FeatureFlagService:
    """
    Servicio de feature flags con mejoras.
    
    Attributes:
        flags: Diccionario de feature flags por nombre
    """
    
    def __init__(self):
        """Inicializar servicio con logging."""
        self.flags: Dict[str, FeatureFlag] = {}
        self._initialize_default_flags()
        logger.info(f"✅ FeatureFlagService inicializado: {len(self.flags)} flags por defecto")
    
    def _initialize_default_flags(self):
        """Inicializar flags por defecto."""
        default_flags = [
            FeatureFlag("llm_service", enabled=settings.LLM_ENABLED, description="LLM Service"),
            FeatureFlag("batch_operations", enabled=True, description="Batch operations"),
            FeatureFlag("websocket_updates", enabled=True, description="WebSocket real-time updates"),
            FeatureFlag("audit_logging", enabled=True, description="Audit logging"),
            FeatureFlag("monitoring", enabled=True, description="Monitoring service"),
            FeatureFlag("cache_warming", enabled=True, description="Cache warming"),
        ]
        
        for flag in default_flags:
            self.flags[flag.name] = flag
    
    def register_flag(self, flag: FeatureFlag) -> None:
        """
        Registrar feature flag con validaciones.
        
        Args:
            flag: Feature flag a registrar (debe ser FeatureFlag)
            
        Raises:
            ValueError: Si flag no es FeatureFlag o ya existe
        """
        # Validación
        if not isinstance(flag, FeatureFlag):
            raise ValueError(f"flag debe ser un FeatureFlag, recibido: {type(flag)}")
        
        if flag.name in self.flags:
            logger.warning(f"Feature flag {flag.name} ya existe, será reemplazado")
        
        self.flags[flag.name] = flag
        logger.info(
            f"✅ Feature flag registrado: {flag.name} "
            f"(enabled: {flag.enabled}, rollout: {flag.rollout_percentage}%, "
            f"target_users: {len(flag.target_users)})"
        )
    
    def is_enabled(self, flag_name: str, user_id: Optional[str] = None) -> bool:
        """
        Verificar si un flag está habilitado con validaciones.
        
        Args:
            flag_name: Nombre del flag (debe ser string no vacío)
            user_id: ID del usuario (opcional, debe ser string si se proporciona)
            
        Returns:
            True si está habilitado, False si no existe o está deshabilitado
            
        Raises:
            ValueError: Si flag_name es inválido
        """
        # Validación
        if not flag_name or not isinstance(flag_name, str) or not flag_name.strip():
            raise ValueError(f"flag_name debe ser un string no vacío, recibido: {flag_name}")
        
        if user_id is not None:
            if not isinstance(user_id, str) or not user_id.strip():
                raise ValueError(f"user_id debe ser un string no vacío si se proporciona, recibido: {user_id}")
            user_id = user_id.strip()
        
        flag_name = flag_name.strip()
        
        flag = self.flags.get(flag_name)
        if not flag:
            logger.debug(f"Feature flag '{flag_name}' no encontrado")
            return False
        
        result = flag.is_enabled_for(user_id)
        logger.debug(f"Feature flag '{flag_name}' para usuario '{user_id or 'anonymous'}': {result}")
        return result
    
    def enable(self, flag_name: str) -> bool:
        """
        Habilitar feature flag.
        
        Args:
            flag_name: Nombre del flag
            
        Returns:
            True si se habilitó
        """
        flag = self.flags.get(flag_name)
        if flag:
            flag.enabled = True
            flag.updated_at = datetime.now()
            logger.info(f"Feature flag habilitado: {flag_name}")
            return True
        return False
    
    def disable(self, flag_name: str) -> bool:
        """
        Deshabilitar feature flag.
        
        Args:
            flag_name: Nombre del flag
            
        Returns:
            True si se deshabilitó
        """
        flag = self.flags.get(flag_name)
        if flag:
            flag.enabled = False
            flag.updated_at = datetime.now()
            logger.info(f"Feature flag deshabilitado: {flag_name}")
            return True
        return False
    
    def set_rollout(self, flag_name: str, percentage: int) -> bool:
        """
        Establecer porcentaje de rollout con validaciones.
        
        Args:
            flag_name: Nombre del flag (debe ser string no vacío)
            percentage: Porcentaje (0-100, debe ser entero)
            
        Returns:
            True si se actualizó, False si el flag no existe
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        # Validaciones
        if not flag_name or not isinstance(flag_name, str) or not flag_name.strip():
            raise ValueError(f"flag_name debe ser un string no vacío, recibido: {flag_name}")
        
        if not isinstance(percentage, int):
            raise ValueError(f"percentage debe ser un entero, recibido: {type(percentage)}")
        
        if percentage < 0 or percentage > 100:
            raise ValueError(f"percentage debe estar entre 0 y 100, recibido: {percentage}")
        
        flag_name = flag_name.strip()
        
        flag = self.flags.get(flag_name)
        if not flag:
            logger.warning(f"Feature flag '{flag_name}' no encontrado para actualizar rollout")
            return False
        
        old_percentage = flag.rollout_percentage
        flag.rollout_percentage = max(0, min(100, percentage))
        flag.updated_at = datetime.now()
        logger.info(
            f"✅ Rollout actualizado para {flag_name}: {old_percentage}% -> {flag.rollout_percentage}%"
        )
        return True
    
    def get_flag(self, flag_name: str) -> Optional[FeatureFlag]:
        """
        Obtener feature flag.
        
        Args:
            flag_name: Nombre del flag
            
        Returns:
            FeatureFlag o None
        """
        return self.flags.get(flag_name)
    
    def list_flags(self) -> List[FeatureFlag]:
        """
        Listar todos los feature flags.
        
        Returns:
            Lista de feature flags
        """
        return list(self.flags.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        enabled_count = sum(1 for f in self.flags.values() if f.enabled)
        return {
            "total_flags": len(self.flags),
            "enabled_flags": enabled_count,
            "disabled_flags": len(self.flags) - enabled_count,
            "flags": {
                name: {
                    "enabled": flag.enabled,
                    "rollout_percentage": flag.rollout_percentage,
                    "target_users_count": len(flag.target_users)
                }
                for name, flag in self.flags.items()
            }
        }

