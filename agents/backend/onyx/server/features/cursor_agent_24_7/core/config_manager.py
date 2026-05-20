"""
Config Manager - Gestor de configuración avanzado
==================================================

Gestión avanzada de configuración con validación, hot-reload, y soporte
para dot notation. Incluye sistema de watchers para cambios en tiempo real.
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional, Callable, List, Union
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Gestor de configuración avanzado.
    
    Proporciona:
    - Carga/guardado de configuración desde archivo JSON
    - Soporte para dot notation (ej: "agent.check_interval")
    - Sistema de watchers para cambios en tiempo real
    - Valores por defecto
    - Hot-reload de configuración
    """
    
    def __init__(self, config_file: str = "./data/config.json") -> None:
        """
        Inicializar gestor de configuración.
        
        Args:
            config_file: Ruta del archivo de configuración (default: "./data/config.json").
        
        Raises:
            ValueError: Si config_file está vacío.
            RuntimeError: Si hay error al cargar la configuración.
        """
        if not config_file or not config_file.strip():
            raise ValueError("Config file path cannot be empty")
        
        self.config_file: Path = Path(config_file.strip())
        self.config: Dict[str, Any] = {}
        self.defaults: Dict[str, Any] = {}
        self.watchers: List[Callable[[str, Any], Union[None, Any]]] = []
        
        try:
            self._load_defaults()
            self.load()
        except Exception as e:
            logger.error(f"Error initializing ConfigManager: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize ConfigManager: {e}") from e
    
    def _load_defaults(self) -> None:
        """Cargar valores por defecto de configuración"""
        self.defaults = {
            "agent": {
                "check_interval": 1.0,
                "max_concurrent_tasks": 5,
                "task_timeout": 300.0,
                "auto_restart": True,
                "persistent_storage": True
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8024,
                "cors_origins": ["*"]
            },
            "cache": {
                "enabled": True,
                "max_size": 500,
                "ttl": 3600
            },
            "rate_limiting": {
                "enabled": True,
                "max_tasks_per_minute": 60,
                "max_concurrent": 10
            },
            "backup": {
                "enabled": True,
                "interval_seconds": 3600,
                "max_backups": 10
            },
            "scheduler": {
                "enabled": True
            },
            "notifications": {
                "enabled": True,
                "max_notifications": 1000
            },
            "metrics": {
                "enabled": True,
                "max_history": 10000
            }
        }
    
    def load(self) -> None:
        """
        Cargar configuración desde archivo.
        
        Si el archivo no existe, usa valores por defecto y lo crea.
        
        Raises:
            RuntimeError: Si hay error al leer o parsear el archivo.
        """
        try:
            if self.config_file.exists():
                # Usar orjson si está disponible para mejor performance
                try:
                    import orjson
                    self.config = orjson.loads(self.config_file.read_bytes())
                    logger.debug(f"Configuration loaded using orjson from {self.config_file}")
                except ImportError:
                    with open(self.config_file, 'r', encoding='utf-8') as f:
                        self.config = json.load(f)
                    logger.debug(f"Configuration loaded using json from {self.config_file}")
                
                # Validar que la configuración sea un diccionario
                if not isinstance(self.config, dict):
                    raise ValueError("Configuration must be a dictionary")
                
                logger.info(f"📂 Configuration loaded from {self.config_file}")
            else:
                # Usar defaults y guardar
                self.config = self.defaults.copy()
                self.save()
                logger.info("📂 Using default configuration (file created)")
        
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}", exc_info=True)
            logger.warning("Using default configuration due to parse error")
            self.config = self.defaults.copy()
            self.save()
        
        except Exception as e:
            logger.error(f"Error loading configuration: {e}", exc_info=True)
            logger.warning("Using default configuration due to error")
            self.config = self.defaults.copy()
    
    def save(self) -> None:
        """
        Guardar configuración a archivo.
        
        Raises:
            RuntimeError: Si hay error al escribir el archivo.
        """
        try:
            # Crear directorio si no existe
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Usar orjson si está disponible
            try:
                import orjson
                self.config_file.write_bytes(
                    orjson.dumps(
                        self.config,
                        option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS
                    )
                )
                logger.debug("Configuration saved using orjson")
            except ImportError:
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(
                        self.config,
                        f,
                        indent=2,
                        ensure_ascii=False,
                        sort_keys=True
                    )
                logger.debug("Configuration saved using json")
            
            logger.info(f"💾 Configuration saved to {self.config_file}")
        
        except PermissionError as e:
            logger.error(f"Permission denied writing config file: {e}", exc_info=True)
            raise RuntimeError(f"Permission denied writing config file: {e}") from e
        
        except Exception as e:
            logger.error(f"Error saving configuration: {e}", exc_info=True)
            raise RuntimeError(f"Failed to save configuration: {e}") from e
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Obtener valor de configuración (soporta dot notation).
        
        Args:
            key: Clave de configuración (ej: "agent.check_interval").
            default: Valor por defecto si no se encuentra (default: None).
        
        Returns:
            Valor de configuración o default si no se encuentra.
        
        Raises:
            ValueError: Si key está vacío.
        """
        if not key or not key.strip():
            raise ValueError("Key cannot be empty")
        
        keys = key.strip().split('.')
        value: Any = self.config
        
        # Buscar en configuración actual
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                # Intentar en defaults
                value = self.defaults
                for k2 in keys:
                    if isinstance(value, dict) and k2 in value:
                        value = value[k2]
                    else:
                        return default
                break
        
        return value if value is not None else default
    
    def set(self, key: str, value: Any) -> None:
        """
        Establecer valor de configuración (soporta dot notation).
        
        Args:
            key: Clave de configuración (ej: "agent.check_interval").
            value: Valor a establecer.
        
        Raises:
            ValueError: Si key está vacío.
            RuntimeError: Si hay error al guardar.
        """
        if not key or not key.strip():
            raise ValueError("Key cannot be empty")
        
        keys = key.strip().split('.')
        config = self.config
        
        # Crear estructura anidada si es necesario
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            elif not isinstance(config[k], dict):
                # Si existe pero no es dict, reemplazarlo
                config[k] = {}
            config = config[k]
        
        # Establecer valor
        config[keys[-1]] = value
        
        # Guardar y notificar watchers
        try:
            self.save()
            self._notify_watchers(key, value)
        except Exception as e:
            logger.error(f"Error saving config after set: {e}", exc_info=True)
            raise
    
    def watch(
        self,
        callback: Callable[[str, Any], Union[None, Any]]
    ) -> None:
        """
        Registrar callback para cambios de configuración.
        
        Args:
            callback: Función o coroutine a llamar cuando cambia la configuración.
                Recibe (key: str, value: Any).
        
        Raises:
            ValueError: Si callback no es callable.
        """
        if not callable(callback):
            raise ValueError("Callback must be callable")
        self.watchers.append(callback)
        logger.debug(f"Watcher registered (total: {len(self.watchers)})")
    
    def _notify_watchers(self, key: str, value: Any) -> None:
        """
        Notificar watchers de cambios en configuración.
        
        Args:
            key: Clave que cambió.
            value: Nuevo valor.
        """
        if not self.watchers:
            return
        
        for watcher in self.watchers:
            try:
                if asyncio.iscoroutinefunction(watcher):
                    # Crear tarea para callback async
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.create_task(watcher(key, value))
                        else:
                            loop.run_until_complete(watcher(key, value))
                    except RuntimeError:
                        # No hay loop, crear uno nuevo
                        asyncio.run(watcher(key, value))
                else:
                    # Ejecutar callback síncrono
                    watcher(key, value)
            
            except Exception as e:
                logger.error(f"Error in config watcher: {e}", exc_info=True)
    
    def reset(self, section: Optional[str] = None) -> None:
        """
        Resetear configuración a valores por defecto.
        
        Args:
            section: Sección específica a resetear (opcional). Si es None,
                resetea toda la configuración.
        
        Raises:
            ValueError: Si section no existe en defaults.
        """
        try:
            if section:
                if not section.strip():
                    raise ValueError("Section cannot be empty")
                
                section = section.strip()
                if section not in self.defaults:
                    raise ValueError(f"Section '{section}' not found in defaults")
                
                self.config[section] = self.defaults[section].copy()
                logger.info(f"🔄 Configuration reset: {section}")
            else:
                self.config = self.defaults.copy()
                logger.info("🔄 Configuration reset: all")
            
            self.save()
        
        except Exception as e:
            logger.error(f"Error resetting configuration: {e}", exc_info=True)
            raise
