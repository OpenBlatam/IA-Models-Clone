"""
Config Manager - Gestor de Configuraciones
==========================================

Sistema avanzado de gestión de configuraciones con validación, versionado y hot-reload.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict
import json
import yaml

logger = logging.getLogger(__name__)

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning("yaml not available, YAML parsing will be limited")


class ConfigFormat(Enum):
    """Formato de configuración."""
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    INI = "ini"
    ENV = "env"


class ConfigStatus(Enum):
    """Estado de configuración."""
    ACTIVE = "active"
    DRAFT = "draft"
    DEPRECATED = "deprecated"
    INVALID = "invalid"


@dataclass
class ConfigEntry:
    """Entrada de configuración."""
    config_id: str
    name: str
    format: ConfigFormat
    data: Dict[str, Any]
    version: int = 1
    status: ConfigStatus = ConfigStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    validator: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfigChange:
    """Cambio de configuración."""
    change_id: str
    config_id: str
    old_version: int
    new_version: int
    changes: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    changed_by: str = "system"
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConfigManager:
    """Gestor de configuraciones."""
    
    def __init__(self):
        self.configs: Dict[str, ConfigEntry] = {}
        self.config_history: Dict[str, List[ConfigEntry]] = defaultdict(list)
        self.config_changes: List[ConfigChange] = []
        self.change_listeners: Dict[str, List[Callable]] = defaultdict(list)
        self._lock = asyncio.Lock()
    
    def register_config(
        self,
        config_id: str,
        name: str,
        data: Dict[str, Any],
        format: ConfigFormat = ConfigFormat.JSON,
        validator: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Registrar configuración."""
        # Validar si hay validator
        if validator:
            try:
                validator(data)
            except Exception as e:
                logger.error(f"Config validation failed for {config_id}: {e}")
                status = ConfigStatus.INVALID
            else:
                status = ConfigStatus.ACTIVE
        else:
            status = ConfigStatus.ACTIVE
        
        config_entry = ConfigEntry(
            config_id=config_id,
            name=name,
            format=format,
            data=data,
            validator=validator,
            metadata=metadata or {},
            status=status,
        )
        
        async def save_config():
            async with self._lock:
                # Guardar versión anterior en historial
                existing = self.configs.get(config_id)
                if existing:
                    self.config_history[config_id].append(existing)
                    config_entry.version = existing.version + 1
                    
                    # Registrar cambio
                    change = ConfigChange(
                        change_id=f"change_{config_id}_{datetime.now().timestamp()}",
                        config_id=config_id,
                        old_version=existing.version,
                        new_version=config_entry.version,
                        changes=self._calculate_diff(existing.data, data),
                    )
                    self.config_changes.append(change)
                    
                    # Notificar listeners
                    for listener in self.change_listeners.get(config_id, []):
                        try:
                            if asyncio.iscoroutinefunction(listener):
                                await listener(config_entry, existing)
                            else:
                                listener(config_entry, existing)
                        except Exception as e:
                            logger.error(f"Error in config change listener: {e}")
                
                self.configs[config_id] = config_entry
        
        asyncio.create_task(save_config())
        
        logger.info(f"Registered config: {config_id} - {name}")
        return config_id
    
    def _calculate_diff(self, old_data: Dict[str, Any], new_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calcular diferencias entre configuraciones."""
        changes = {}
        
        all_keys = set(old_data.keys()) | set(new_data.keys())
        
        for key in all_keys:
            old_value = old_data.get(key)
            new_value = new_data.get(key)
            
            if old_value != new_value:
                changes[key] = {
                    "old": old_value,
                    "new": new_value,
                }
        
        return changes
    
    def get_config(self, config_id: str, version: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Obtener configuración."""
        if version:
            # Buscar en historial
            history = self.config_history.get(config_id, [])
            config = next((c for c in history if c.version == version), None)
            if config:
                return {
                    "config_id": config.config_id,
                    "name": config.name,
                    "format": config.format.value,
                    "data": config.data,
                    "version": config.version,
                    "status": config.status.value,
                    "created_at": config.created_at.isoformat(),
                    "updated_at": config.updated_at.isoformat(),
                }
            return None
        
        # Obtener versión actual
        config = self.configs.get(config_id)
        if not config:
            return None
        
        return {
            "config_id": config.config_id,
            "name": config.name,
            "format": config.format.value,
            "data": config.data,
            "version": config.version,
            "status": config.status.value,
            "created_at": config.created_at.isoformat(),
            "updated_at": config.updated_at.isoformat(),
        }
    
    def subscribe_to_changes(self, config_id: str, listener: Callable):
        """Suscribirse a cambios de configuración."""
        self.change_listeners[config_id].append(listener)
        logger.info(f"Subscribed to changes for config: {config_id}")
    
    def get_config_history(self, config_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Obtener historial de configuración."""
        history = self.config_history.get(config_id, [])
        
        return [
            {
                "config_id": c.config_id,
                "version": c.version,
                "status": c.status.value,
                "created_at": c.created_at.isoformat(),
                "updated_at": c.updated_at.isoformat(),
            }
            for c in history[-limit:]
        ]
    
    def get_config_changes(
        self,
        config_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Obtener cambios de configuración."""
        changes = self.config_changes
        
        if config_id:
            changes = [c for c in changes if c.config_id == config_id]
        
        changes.sort(key=lambda c: c.timestamp, reverse=True)
        
        return [
            {
                "change_id": c.change_id,
                "config_id": c.config_id,
                "old_version": c.old_version,
                "new_version": c.new_version,
                "changes": c.changes,
                "timestamp": c.timestamp.isoformat(),
                "changed_by": c.changed_by,
            }
            for c in changes[:limit]
        ]
    
    def rollback_config(self, config_id: str, target_version: int) -> bool:
        """Revertir configuración a versión anterior."""
        history = self.config_history.get(config_id, [])
        target_config = next((c for c in history if c.version == target_version), None)
        
        if not target_config:
            return False
        
        # Restaurar configuración
        current = self.configs.get(config_id)
        if current:
            self.config_history[config_id].append(current)
        
        target_config.status = ConfigStatus.ACTIVE
        target_config.updated_at = datetime.now()
        self.configs[config_id] = target_config
        
        logger.info(f"Rolled back config {config_id} to version {target_version}")
        return True
    
    def get_config_manager_summary(self) -> Dict[str, Any]:
        """Obtener resumen del gestor."""
        by_status: Dict[str, int] = defaultdict(int)
        
        for config in self.configs.values():
            by_status[config.status.value] += 1
        
        return {
            "total_configs": len(self.configs),
            "configs_by_status": dict(by_status),
            "total_changes": len(self.config_changes),
            "total_listeners": sum(len(listeners) for listeners in self.change_listeners.values()),
        }















