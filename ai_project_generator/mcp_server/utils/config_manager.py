"""
Configuration Manager - Gestor avanzado de configuración
========================================================

Gestor de configuración con validación, caché, y utilidades avanzadas.
"""

import logging
import json
import yaml
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from functools import lru_cache

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Gestor avanzado de configuración.
    
    Proporciona carga, validación, y gestión de configuración
    con soporte para múltiples formatos y fuentes.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializar gestor de configuración.
        
        Args:
            config_path: Ruta al archivo de configuración (opcional)
        """
        self.config_path = config_path
        self._config: Optional[Dict[str, Any]] = None
        self._cache: Dict[str, Any] = {}
    
    def load(self, path: Optional[str] = None) -> Dict[str, Any]:
        """
        Cargar configuración desde archivo.
        
        Args:
            path: Ruta al archivo (usa self.config_path si no se proporciona)
        
        Returns:
            Diccionario con configuración
        
        Raises:
            FileNotFoundError: Si el archivo no existe
            ValueError: Si el formato no es soportado
        """
        file_path = path or self.config_path
        if not file_path:
            raise ValueError("No config path provided")
        
        path_obj = Path(file_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
        
        suffix = path_obj.suffix.lower()
        
        if suffix in ['.json']:
            with open(path_obj, 'r', encoding='utf-8') as f:
                self._config = json.load(f)
        elif suffix in ['.yaml', '.yml']:
            try:
                import yaml
                with open(path_obj, 'r', encoding='utf-8') as f:
                    self._config = yaml.safe_load(f)
            except ImportError:
                raise ValueError("PyYAML not installed. Install with: pip install pyyaml")
        else:
            raise ValueError(f"Unsupported config format: {suffix}")
        
        logger.info(f"Configuration loaded from {file_path}")
        return self._config
    
    def save(self, path: Optional[str] = None, format: str = 'json') -> None:
        """
        Guardar configuración en archivo.
        
        Args:
            path: Ruta al archivo (usa self.config_path si no se proporciona)
            format: Formato ('json' o 'yaml')
        
        Raises:
            ValueError: Si no hay configuración cargada o formato inválido
        """
        if self._config is None:
            raise ValueError("No configuration loaded")
        
        file_path = path or self.config_path
        if not file_path:
            raise ValueError("No config path provided")
        
        path_obj = Path(file_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            with open(path_obj, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
        elif format in ['yaml', 'yml']:
            try:
                import yaml
                with open(path_obj, 'w', encoding='utf-8') as f:
                    yaml.safe_dump(self._config, f, default_flow_style=False)
            except ImportError:
                raise ValueError("PyYAML not installed. Install with: pip install pyyaml")
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Configuration saved to {file_path}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Obtener valor de configuración usando notación de punto.
        
        Args:
            key: Clave en notación de punto (ej: "server.host")
            default: Valor por defecto
        
        Returns:
            Valor de configuración o default
        
        Example:
            host = config.get("server.host", "localhost")
        """
        if self._config is None:
            return default
        
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Establecer valor de configuración usando notación de punto.
        
        Args:
            key: Clave en notación de punto (ej: "server.host")
            value: Valor a establecer
        
        Example:
            config.set("server.host", "0.0.0.0")
        """
        if self._config is None:
            self._config = {}
        
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any], deep: bool = True) -> None:
        """
        Actualizar configuración con diccionario.
        
        Args:
            updates: Diccionario con actualizaciones
            deep: Si True, hace merge profundo
        """
        if self._config is None:
            self._config = {}
        
        if deep:
            self._deep_merge(self._config, updates)
        else:
            self._config.update(updates)
    
    def _deep_merge(self, base: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """Merge profundo de diccionarios."""
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def validate(self, schema: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Validar configuración.
        
        Args:
            schema: Esquema de validación (opcional)
        
        Returns:
            Lista de errores (vacía si es válida)
        """
        errors = []
        
        if self._config is None:
            errors.append("No configuration loaded")
            return errors
        
        if schema:
            errors.extend(self._validate_schema(self._config, schema))
        
        return errors
    
    def _validate_schema(self, config: Any, schema: Dict[str, Any], path: str = "") -> List[str]:
        """Validar configuración contra esquema."""
        errors = []
        
        for key, rule in schema.items():
            current_path = f"{path}.{key}" if path else key
            value = config.get(key) if isinstance(config, dict) else None
            
            if rule.get('required', False) and value is None:
                errors.append(f"{current_path} is required")
            
            if value is not None:
                value_type = rule.get('type')
                if value_type and not isinstance(value, value_type):
                    errors.append(f"{current_path} must be {value_type.__name__}")
                
                if 'min' in rule and value < rule['min']:
                    errors.append(f"{current_path} must be >= {rule['min']}")
                
                if 'max' in rule and value > rule['max']:
                    errors.append(f"{current_path} must be <= {rule['max']}")
        
        return errors
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Obtener sección completa de configuración.
        
        Args:
            section: Nombre de la sección
        
        Returns:
            Diccionario con la sección o vacío si no existe
        """
        if self._config is None:
            return {}
        
        return self._config.get(section, {})
    
    def has_section(self, section: str) -> bool:
        """
        Verificar si existe una sección.
        
        Args:
            section: Nombre de la sección
        
        Returns:
            True si existe, False en caso contrario
        """
        if self._config is None:
            return False
        
        return section in self._config
    
    def list_sections(self) -> List[str]:
        """
        Listar todas las secciones.
        
        Returns:
            Lista de nombres de secciones
        """
        if self._config is None:
            return []
        
        return list(self._config.keys())
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Obtener configuración como diccionario.
        
        Returns:
            Diccionario con configuración completa
        """
        if self._config is None:
            return {}
        
        return self._config.copy()
    
    def clear_cache(self) -> None:
        """Limpiar caché."""
        self._cache.clear()


@lru_cache(maxsize=1)
def get_default_config() -> Dict[str, Any]:
    """
    Obtener configuración por defecto.
    
    Returns:
        Diccionario con configuración por defecto
    """
    return {
        "server": {
            "host": "0.0.0.0",
            "port": 8020,
            "debug": False,
            "reload": False,
            "workers": 1,
        },
        "security": {
            "secret_key": "change-me-in-production",
            "token_expire_minutes": 30,
            "algorithm": "HS256",
            "require_https": False,
        },
        "rate_limiting": {
            "enabled": True,
            "max_requests": 100,
            "window_seconds": 60,
        },
        "cache": {
            "enabled": True,
            "ttl_seconds": 3600,
        },
        "cors": {
            "enabled": True,
            "origins": ["*"],
        },
        "observability": {
            "tracing": True,
            "metrics": True,
            "logging": True,
        },
    }


def create_config_template(output_path: str, format: str = 'json') -> None:
    """
    Crear plantilla de configuración.
    
    Args:
        output_path: Ruta donde guardar la plantilla
        format: Formato ('json' o 'yaml')
    """
    manager = ConfigManager()
    manager._config = get_default_config()
    manager.save(output_path, format=format)
    logger.info(f"Config template created at {output_path}")


__all__ = [
    "ConfigManager",
    "get_default_config",
    "create_config_template",
]

