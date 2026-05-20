"""
Feature Flags - Sistema de Feature Flags
=========================================

Sistema para activar/desactivar características dinámicamente.
"""

import asyncio
import logging
from typing import Dict, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class FeatureStatus(Enum):
    """Estado de feature flag."""
    ENABLED = "enabled"
    DISABLED = "disabled"
    CONDITIONAL = "conditional"  # Habilitado bajo condiciones


@dataclass
class FeatureFlag:
    """Feature flag."""
    name: str
    status: FeatureStatus
    description: str = ""
    enabled_for: Optional[list] = None  # Lista de usuarios/roles habilitados
    condition: Optional[Callable] = None  # Función de condición
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class FeatureFlagManager:
    """Gestor de feature flags."""
    
    def __init__(self):
        self.flags: Dict[str, FeatureFlag] = {}
        self._lock = asyncio.Lock()
        self._load_default_flags()
    
    def _load_default_flags(self):
        """Cargar feature flags por defecto."""
        default_flags = [
            FeatureFlag(
                name="chat_continuous",
                status=FeatureStatus.ENABLED,
                description="Chat continuo proactivo",
            ),
            FeatureFlag(
                name="cache_enabled",
                status=FeatureStatus.ENABLED,
                description="Cache de respuestas",
            ),
            FeatureFlag(
                name="webhooks",
                status=FeatureStatus.ENABLED,
                description="Sistema de webhooks",
            ),
            FeatureFlag(
                name="analytics",
                status=FeatureStatus.ENABLED,
                description="Análisis de conversaciones",
            ),
            FeatureFlag(
                name="graphql",
                status=FeatureStatus.ENABLED,
                description="API GraphQL",
            ),
        ]
        
        for flag in default_flags:
            self.flags[flag.name] = flag
    
    def register_flag(self, flag: FeatureFlag):
        """Registrar un feature flag."""
        self.flags[flag.name] = flag
        logger.info(f"Registered feature flag: {flag.name}")
    
    async def is_enabled(
        self,
        flag_name: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Verificar si un feature flag está habilitado.
        
        Args:
            flag_name: Nombre del flag
            user_id: ID de usuario (opcional)
            context: Contexto adicional (opcional)
        
        Returns:
            True si está habilitado, False en caso contrario
        """
        flag = self.flags.get(flag_name)
        
        if not flag:
            return False  # Por defecto, deshabilitado si no existe
        
        if flag.status == FeatureStatus.DISABLED:
            return False
        
        if flag.status == FeatureStatus.ENABLED:
            # Verificar si está habilitado para usuarios específicos
            if flag.enabled_for:
                return user_id in flag.enabled_for
            return True
        
        if flag.status == FeatureStatus.CONDITIONAL:
            # Evaluar condición
            if flag.condition:
                try:
                    if asyncio.iscoroutinefunction(flag.condition):
                        return await flag.condition(user_id, context or {})
                    else:
                        return flag.condition(user_id, context or {})
                except Exception as e:
                    logger.error(f"Error evaluating condition for {flag_name}: {e}")
                    return False
        
        return False
    
    async def enable(self, flag_name: str):
        """Habilitar feature flag."""
        async with self._lock:
            if flag_name in self.flags:
                self.flags[flag_name].status = FeatureStatus.ENABLED
                self.flags[flag_name].updated_at = datetime.now()
                logger.info(f"Enabled feature flag: {flag_name}")
    
    async def disable(self, flag_name: str):
        """Deshabilitar feature flag."""
        async with self._lock:
            if flag_name in self.flags:
                self.flags[flag_name].status = FeatureStatus.DISABLED
                self.flags[flag_name].updated_at = datetime.now()
                logger.info(f"Disabled feature flag: {flag_name}")
    
    def get_flag(self, flag_name: str) -> Optional[FeatureFlag]:
        """Obtener feature flag."""
        return self.flags.get(flag_name)
    
    def list_flags(self) -> Dict[str, Dict[str, Any]]:
        """Listar todos los feature flags."""
        return {
            name: {
                "name": flag.name,
                "status": flag.status.value,
                "description": flag.description,
                "enabled_for": flag.enabled_for,
            }
            for name, flag in self.flags.items()
        }
































