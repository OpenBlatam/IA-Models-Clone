"""
Routing Configuration Optimizations
====================================

Optimizaciones de configuración dinámica.
Incluye: Hot reload, Configuration validation, Environment variables, etc.
"""

import logging
import os
import json
import yaml
from typing import Dict, Any, Optional, List, Callable, Tuple
from pathlib import Path
import threading
import time

logger = logging.getLogger(__name__)

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class ConfigurationValidator:
    """Validador de configuración."""
    
    def __init__(self):
        """Inicializar validador."""
        self.validation_schema: Dict[str, Any] = {}
    
    def add_schema(self, key: str, schema: Dict[str, Any]):
        """
        Agregar esquema de validación.
        
        Args:
            key: Clave de configuración
            schema: Esquema de validación
        """
        self.validation_schema[key] = schema
    
    def validate(self, config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validar configuración.
        
        Args:
            config: Configuración a validar
        
        Returns:
            (is_valid, error_message)
        """
        for key, schema in self.validation_schema.items():
            if key not in config:
                if schema.get('required', False):
                    return False, f"Required configuration key '{key}' is missing"
                continue
            
            value = config[key]
            
            # Validar tipo
            expected_type = schema.get('type')
            if expected_type and not isinstance(value, expected_type):
                return False, f"Configuration key '{key}' must be of type {expected_type.__name__}"
            
            # Validar valores permitidos
            allowed_values = schema.get('allowed_values')
            if allowed_values and value not in allowed_values:
                return False, f"Configuration key '{key}' must be one of {allowed_values}"
            
            # Validar rango
            if isinstance(value, (int, float)):
                if 'min' in schema and value < schema['min']:
                    return False, f"Configuration key '{key}' must be >= {schema['min']}"
                if 'max' in schema and value > schema['max']:
                    return False, f"Configuration key '{key}' must be <= {schema['max']}"
        
        return True, None


class HotReloadManager:
    """Gestor de hot reload de configuración."""
    
    def __init__(self, config_file: str, reload_interval: float = 5.0):
        """
        Inicializar gestor de hot reload.
        
        Args:
            config_file: Archivo de configuración
            reload_interval: Intervalo de verificación en segundos
        """
        self.config_file = Path(config_file)
        self.reload_interval = reload_interval
        self.last_modified = None
        self.config_callbacks: List[Callable] = []
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
    
    def register_callback(self, callback: Callable):
        """
        Registrar callback para cambios de configuración.
        
        Args:
            callback: Función a llamar cuando cambie la configuración
        """
        with self.lock:
            self.config_callbacks.append(callback)
    
    def start_monitoring(self):
        """Iniciar monitoreo de cambios."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Hot reload monitoring started")
    
    def stop_monitoring(self):
        """Detener monitoreo."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Hot reload monitoring stopped")
    
    def _monitor_loop(self):
        """Loop de monitoreo."""
        while self.monitoring:
            try:
                if self.config_file.exists():
                    current_modified = self.config_file.stat().st_mtime
                    
                    if self.last_modified is None:
                        self.last_modified = current_modified
                    elif current_modified > self.last_modified:
                        # Configuración cambió
                        logger.info("Configuration file changed, reloading...")
                        self._reload_config()
                        self.last_modified = current_modified
                
                time.sleep(self.reload_interval)
            except Exception as e:
                logger.error(f"Error in hot reload monitoring: {e}")
                time.sleep(self.reload_interval)
    
    def _reload_config(self):
        """Recargar configuración."""
        try:
            with self.lock:
                for callback in self.config_callbacks:
                    try:
                        callback()
                    except Exception as e:
                        logger.error(f"Error in config reload callback: {e}")
        except Exception as e:
            logger.error(f"Error reloading configuration: {e}")


class EnvironmentConfigLoader:
    """Cargador de configuración desde variables de entorno."""
    
    def __init__(self, prefix: str = "ROUTING_"):
        """
        Inicializar cargador.
        
        Args:
            prefix: Prefijo para variables de entorno
        """
        self.prefix = prefix
    
    def load_config(self) -> Dict[str, Any]:
        """Cargar configuración desde variables de entorno."""
        config = {}
        
        for key, value in os.environ.items():
            if key.startswith(self.prefix):
                config_key = key[len(self.prefix):].lower()
                
                # Intentar convertir tipos
                if value.lower() in ('true', 'false'):
                    config[config_key] = value.lower() == 'true'
                elif value.isdigit():
                    config[config_key] = int(value)
                else:
                    try:
                        config[config_key] = float(value)
                    except ValueError:
                        config[config_key] = value
        
        return config


class ConfigurationOptimizer:
    """Optimizador completo de configuración."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Inicializar optimizador de configuración.
        
        Args:
            config_file: Archivo de configuración (opcional)
        """
        self.validator = ConfigurationValidator()
        self.hot_reload = HotReloadManager(config_file) if config_file else None
        self.env_loader = EnvironmentConfigLoader()
    
    def load_from_file(self, config_file: str) -> Dict[str, Any]:
        """Cargar configuración desde archivo."""
        config_path = Path(config_file)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        if config_path.suffix == '.json':
            with open(config_path, 'r') as f:
                return json.load(f)
        elif config_path.suffix in ['.yaml', '.yml']:
            if not YAML_AVAILABLE:
                raise ImportError("PyYAML not available")
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    
    def load_from_env(self) -> Dict[str, Any]:
        """Cargar configuración desde variables de entorno."""
        return self.env_loader.load_config()
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validar configuración."""
        return self.validator.validate(config)
    
    def enable_hot_reload(self, callback: Callable):
        """Habilitar hot reload."""
        if self.hot_reload:
            self.hot_reload.register_callback(callback)
            self.hot_reload.start_monitoring()
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            'hot_reload_enabled': self.hot_reload is not None and self.hot_reload.monitoring,
            'validation_schema_count': len(self.validator.validation_schema)
        }

