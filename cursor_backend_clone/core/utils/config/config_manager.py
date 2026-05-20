"""
Config Manager - Gestor de configuración avanzado
==================================================

Gestión avanzada de configuración con validación y hot-reload.
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional, Callable
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class ConfigManager:
    """Gestor de configuración avanzado"""
    
    def __init__(self, config_file: str = "./data/config.json"):
        self.config_file = Path(config_file)
        self.config: Dict[str, Any] = {}
        self.defaults: Dict[str, Any] = {}
        self.watchers: List[Callable] = []
        self._load_defaults()
        self.load()
    
    def _load_defaults(self):
        """Cargar valores por defecto"""
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
    
    def load(self):
        """Cargar configuración desde archivo"""
        try:
            if self.config_file.exists():
                # Usar orjson si está disponible
                try:
                    import orjson
                    self.config = orjson.loads(self.config_file.read_bytes())
                except ImportError:
                    with open(self.config_file, 'r', encoding='utf-8') as f:
                        self.config = json.load(f)
                
                logger.info(f"📂 Configuration loaded from {self.config_file}")
            else:
                # Usar defaults y guardar
                self.config = self.defaults.copy()
                self.save()
                logger.info("📂 Using default configuration")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self.config = self.defaults.copy()
    
    def save(self):
        """Guardar configuración a archivo"""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Usar orjson si está disponible
            try:
                import orjson
                self.config_file.write_bytes(
                    orjson.dumps(self.config, option=orjson.OPT_INDENT_2)
                )
            except ImportError:
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Obtener valor de configuración (soporta dot notation)"""
        keys = key.split('.')
        value = self.config
        
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
    
    def set(self, key: str, value: Any):
        """Establecer valor de configuración (soporta dot notation)"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        self.save()
        
        # Notificar watchers
        self._notify_watchers(key, value)
    
    def watch(self, callback: Callable):
        """Registrar callback para cambios de configuración"""
        self.watchers.append(callback)
    
    def _notify_watchers(self, key: str, value: Any):
        """Notificar watchers de cambios"""
        for watcher in self.watchers:
            try:
                if asyncio.iscoroutinefunction(watcher):
                    asyncio.create_task(watcher(key, value))
                else:
                    watcher(key, value)
            except Exception as e:
                logger.error(f"Error in config watcher: {e}")
    
    def reset(self, section: Optional[str] = None):
        """Resetear configuración"""
        if section:
            if section in self.defaults:
                self.config[section] = self.defaults[section].copy()
        else:
            self.config = self.defaults.copy()
        
        self.save()
        logger.info(f"🔄 Configuration reset: {section or 'all'}")


