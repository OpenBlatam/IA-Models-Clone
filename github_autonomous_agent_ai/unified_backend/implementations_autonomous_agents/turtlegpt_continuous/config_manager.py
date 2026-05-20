"""
Configuration Manager Module
============================

Gestión centralizada de configuración del agente.
Proporciona una interfaz estructurada para cargar, validar, actualizar y persistir configuración.
"""

import logging
import json
from typing import Dict, Any, Optional, List, Set
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime

from .config_validator import ConfigValidator
from .constants import (
    DEFAULT_REACT_ENABLED, DEFAULT_LATS_ENABLED, DEFAULT_TOT_ENABLED,
    DEFAULT_TOM_ENABLED, DEFAULT_PERSONALITY_ENABLED, DEFAULT_TOOLFORMER_ENABLED,
    DEFAULT_REFLECTION_INTERVAL, DEFAULT_PLANNING_INTERVAL,
    DEFAULT_MAX_EPISODIC_MEMORIES, DEFAULT_MAX_SEMANTIC_MEMORIES
)

logger = logging.getLogger(__name__)


@dataclass
class ConfigSnapshot:
    """Snapshot de configuración en un momento dado."""
    timestamp: datetime
    config: Dict[str, Any]
    version: str = "1.0"
    source: str = "runtime"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "config": self.config,
            "version": self.version,
            "source": self.source
        }


class ConfigManager:
    """
    Gestor centralizado de configuración.
    
    Proporciona funcionalidades para:
    - Cargar configuración desde múltiples fuentes
    - Validar y normalizar configuración
    - Actualizar configuración en tiempo de ejecución
    - Persistir y restaurar configuración
    - Historial de cambios
    - Rollback a configuraciones anteriores
    """
    
    def __init__(
        self,
        initial_config: Optional[Dict[str, Any]] = None,
        config_file: Optional[Path] = None,
        enable_persistence: bool = True,
        enable_history: bool = True
    ):
        """
        Inicializar gestor de configuración.
        
        Args:
            initial_config: Configuración inicial
            config_file: Archivo de configuración persistente
            enable_persistence: Habilitar persistencia
            enable_history: Habilitar historial de cambios
        """
        self.config_file = config_file
        self.enable_persistence = enable_persistence
        self.enable_history = enable_history
        
        # Historial de configuraciones
        self._history: List[ConfigSnapshot] = []
        self._max_history_size = 50
        
        # Configuración actual
        self._config: Dict[str, Any] = {}
        
        # Cargar configuración inicial
        if initial_config:
            self.update_config(initial_config, source="initialization")
        elif config_file and config_file.exists():
            self.load_from_file(config_file)
        else:
            self._config = self._get_default_config()
            if self.enable_history:
                self._add_to_history(self._config, source="default")
    
    def get_config(self, key: Optional[str] = None, default: Any = None) -> Any:
        """
        Obtener configuración.
        
        Args:
            key: Clave de configuración (None para obtener todo)
            default: Valor por defecto si no existe
            
        Returns:
            Valor de configuración o diccionario completo
        """
        if key is None:
            return self._config.copy()
        
        return self._config.get(key, default)
    
    def update_config(
        self,
        updates: Dict[str, Any],
        source: str = "runtime",
        validate: bool = True,
        persist: bool = True
    ) -> bool:
        """
        Actualizar configuración.
        
        Args:
            updates: Diccionario con actualizaciones
            source: Origen de la actualización
            validate: Validar antes de aplicar
            persist: Persistir después de actualizar
            
        Returns:
            True si se actualizó exitosamente
        """
        try:
            # Validar si es necesario
            if validate:
                validated_updates = ConfigValidator.merge_with_defaults(updates)
            else:
                validated_updates = updates
            
            # Crear snapshot antes de actualizar
            if self.enable_history:
                self._add_to_history(self._config.copy(), source="before_update")
            
            # Aplicar actualizaciones
            self._config.update(validated_updates)
            
            # Agregar al historial
            if self.enable_history:
                self._add_to_history(self._config.copy(), source=source)
            
            # Persistir si es necesario
            if persist and self.enable_persistence:
                self.save_to_file()
            
            logger.info(f"Configuration updated from {source}: {len(updates)} keys")
            return True
            
        except Exception as e:
            logger.error(f"Error updating configuration: {e}", exc_info=True)
            return False
    
    def set_config(self, key: str, value: Any, source: str = "runtime") -> bool:
        """
        Establecer un valor de configuración específico.
        
        Args:
            key: Clave de configuración
            value: Valor a establecer
            source: Origen del cambio
            
        Returns:
            True si se estableció exitosamente
        """
        return self.update_config({key: value}, source=source)
    
    def remove_config(self, key: str, source: str = "runtime") -> bool:
        """
        Remover una clave de configuración.
        
        Args:
            key: Clave a remover
            source: Origen del cambio
            
        Returns:
            True si se removió exitosamente
        """
        if key in self._config:
            # Crear snapshot antes de remover
            if self.enable_history:
                self._add_to_history(self._config.copy(), source="before_remove")
            
            del self._config[key]
            
            # Agregar al historial
            if self.enable_history:
                self._add_to_history(self._config.copy(), source=source)
            
            # Persistir
            if self.enable_persistence:
                self.save_to_file()
            
            logger.info(f"Configuration key '{key}' removed")
            return True
        
        return False
    
    def reset_to_defaults(self) -> bool:
        """
        Resetear configuración a valores por defecto.
        
        Returns:
            True si se reseteó exitosamente
        """
        default_config = self._get_default_config()
        return self.update_config(default_config, source="reset", persist=True)
    
    def load_from_file(self, config_file: Optional[Path] = None) -> bool:
        """
        Cargar configuración desde archivo.
        
        Args:
            config_file: Archivo a cargar (None para usar self.config_file)
            
        Returns:
            True si se cargó exitosamente
        """
        file_path = config_file or self.config_file
        if not file_path:
            logger.warning("No config file specified")
            return False
        
        try:
            if not file_path.exists():
                logger.warning(f"Config file not found: {file_path}")
                return False
            
            with open(file_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
            
            # Validar y normalizar
            validated_config = ConfigValidator.merge_with_defaults(loaded_config)
            
            # Actualizar configuración
            self._config = validated_config
            
            # Agregar al historial
            if self.enable_history:
                self._add_to_history(self._config.copy(), source="file_load")
            
            logger.info(f"Configuration loaded from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading configuration from file: {e}", exc_info=True)
            return False
    
    def save_to_file(self, config_file: Optional[Path] = None) -> bool:
        """
        Guardar configuración a archivo.
        
        Args:
            config_file: Archivo donde guardar (None para usar self.config_file)
            
        Returns:
            True si se guardó exitosamente
        """
        file_path = config_file or self.config_file
        if not file_path:
            logger.warning("No config file specified for saving")
            return False
        
        try:
            # Crear directorio si no existe
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Guardar configuración
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuration saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration to file: {e}", exc_info=True)
            return False
    
    def get_history(self, limit: Optional[int] = None) -> List[ConfigSnapshot]:
        """
        Obtener historial de configuraciones.
        
        Args:
            limit: Límite de entradas a retornar
            
        Returns:
            Lista de snapshots de configuración
        """
        if limit:
            return self._history[-limit:]
        return self._history.copy()
    
    def rollback(self, index: int = -1) -> bool:
        """
        Hacer rollback a una configuración anterior.
        
        Args:
            index: Índice en el historial (-1 para la anterior)
            
        Returns:
            True si se hizo rollback exitosamente
        """
        if not self._history:
            logger.warning("No configuration history available")
            return False
        
        try:
            snapshot = self._history[index]
            self._config = snapshot.config.copy()
            
            # Agregar al historial
            if self.enable_history:
                self._add_to_history(self._config.copy(), source="rollback")
            
            # Persistir
            if self.enable_persistence:
                self.save_to_file()
            
            logger.info(f"Configuration rolled back to {snapshot.timestamp}")
            return True
            
        except IndexError:
            logger.error(f"Invalid history index: {index}")
            return False
    
    def get_diff(self, other_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Obtener diferencias entre configuración actual y otra.
        
        Args:
            other_config: Otra configuración para comparar
            
        Returns:
            Dict con diferencias: {added, removed, changed}
        """
        current_keys = set(self._config.keys())
        other_keys = set(other_config.keys())
        
        added = {k: other_config[k] for k in other_keys - current_keys}
        removed = {k: self._config[k] for k in current_keys - other_keys}
        changed = {
            k: {"old": self._config[k], "new": other_config[k]}
            for k in current_keys & other_keys
            if self._config[k] != other_config[k]
        }
        
        return {
            "added": added,
            "removed": removed,
            "changed": changed
        }
    
    def validate_config(self, config: Optional[Dict[str, Any]] = None) -> tuple[bool, Optional[str]]:
        """
        Validar configuración.
        
        Args:
            config: Configuración a validar (None para usar la actual)
            
        Returns:
            Tuple (is_valid, error_message)
        """
        config_to_validate = config or self._config
        
        try:
            ConfigValidator.merge_with_defaults(config_to_validate)
            return True, None
        except Exception as e:
            return False, str(e)
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Obtener resumen de configuración.
        
        Returns:
            Dict con resumen de configuración
        """
        return {
            "total_keys": len(self._config),
            "history_size": len(self._history),
            "persistence_enabled": self.enable_persistence,
            "history_enabled": self.enable_history,
            "config_file": str(self.config_file) if self.config_file else None,
            "keys": list(self._config.keys())
        }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Obtener configuración por defecto."""
        return {
            "react_enabled": DEFAULT_REACT_ENABLED,
            "lats_enabled": DEFAULT_LATS_ENABLED,
            "tot_enabled": DEFAULT_TOT_ENABLED,
            "tom_enabled": DEFAULT_TOM_ENABLED,
            "personality_enabled": DEFAULT_PERSONALITY_ENABLED,
            "toolformer_enabled": DEFAULT_TOOLFORMER_ENABLED,
            "reflection_interval": DEFAULT_REFLECTION_INTERVAL,
            "planning_interval": DEFAULT_PLANNING_INTERVAL,
            "max_episodic_memories": DEFAULT_MAX_EPISODIC_MEMORIES,
            "max_semantic_memories": DEFAULT_MAX_SEMANTIC_MEMORIES
        }
    
    def _add_to_history(self, config: Dict[str, Any], source: str = "unknown") -> None:
        """Agregar snapshot al historial."""
        snapshot = ConfigSnapshot(
            timestamp=datetime.now(),
            config=config.copy(),
            source=source
        )
        
        self._history.append(snapshot)
        
        # Limitar tamaño del historial
        if len(self._history) > self._max_history_size:
            self._history = self._history[-self._max_history_size:]


def create_config_manager(
    initial_config: Optional[Dict[str, Any]] = None,
    config_file: Optional[Path] = None,
    enable_persistence: bool = True,
    enable_history: bool = True
) -> ConfigManager:
    """
    Factory function para crear ConfigManager.
    
    Args:
        initial_config: Configuración inicial
        config_file: Archivo de configuración persistente
        enable_persistence: Habilitar persistencia
        enable_history: Habilitar historial de cambios
        
    Returns:
        Instancia de ConfigManager
    """
    return ConfigManager(
        initial_config=initial_config,
        config_file=config_file,
        enable_persistence=enable_persistence,
        enable_history=enable_history
    )
