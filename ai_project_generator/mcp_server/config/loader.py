"""
Configuration Loader - Cargador de configuración
=================================================

Utilidades para cargar configuración desde diferentes fuentes:
- Archivos JSON/YAML
- Variables de entorno
- Configuración por defecto
"""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    import json
    JSON_AVAILABLE = True
except ImportError:
    JSON_AVAILABLE = False

from .settings import MCPSettings

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Cargador de configuración con soporte para múltiples fuentes.
    
    Permite cargar configuración desde:
    - Archivos JSON/YAML
    - Variables de entorno
    - Valores por defecto
    """
    
    @staticmethod
    def load_from_file(file_path: str) -> Dict[str, Any]:
        """
        Cargar configuración desde archivo.
        
        Args:
            file_path: Ruta al archivo (JSON o YAML)
            
        Returns:
            Diccionario con configuración
            
        Raises:
            FileNotFoundError: Si el archivo no existe
            ValueError: Si el formato no es soportado
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
        
        with open(path, "r", encoding="utf-8") as f:
            if path.suffix.lower() in [".yaml", ".yml"]:
                if not YAML_AVAILABLE:
                    raise ImportError("YAML support not available. Install PyYAML: pip install pyyaml")
                try:
                    data = yaml.safe_load(f)
                except yaml.YAMLError as e:
                    raise ValueError(f"Invalid YAML file: {e}") from e
            elif path.suffix.lower() == ".json":
                if not JSON_AVAILABLE:
                    raise ImportError("JSON support not available")
                try:
                    import json
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON file: {e}") from e
            else:
                raise ValueError(f"Unsupported config file format: {path.suffix}")
        
        if not isinstance(data, dict):
            raise ValueError("Config file must contain a dictionary/object")
        
        return data
    
    @staticmethod
    def load_from_env(prefix: str = "MCP_") -> Dict[str, Any]:
        """
        Cargar configuración desde variables de entorno.
        
        Args:
            prefix: Prefijo para variables de entorno (default: "MCP_")
            
        Returns:
            Diccionario con configuración desde env
        """
        config: Dict[str, Any] = {}
        
        # Mapear variables de entorno a estructura de configuración
        env_mappings = {
            f"{prefix}HOST": ("server", "host"),
            f"{prefix}PORT": ("server", "port"),
            f"{prefix}DEBUG": ("server", "debug"),
            f"{prefix}SECRET_KEY": ("security", "secret_key"),
            f"{prefix}TOKEN_EXPIRE_MINUTES": ("security", "token_expire_minutes"),
            f"{prefix}RATE_LIMITING_ENABLED": ("rate_limiting", "enabled"),
            f"{prefix}CACHE_ENABLED": ("cache", "enabled"),
            f"{prefix}CORS_ENABLED": ("cors", "enabled"),
            f"{prefix}OBSERVABILITY_ENABLED": ("observability", "enabled"),
            f"{prefix}MANIFESTS_PATH": ("manifests_path",),
        }
        
        for env_var, path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Navegar y establecer valor en estructura anidada
                current = config
                for key in path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                
                # Convertir tipos según sea necesario
                final_key = path[-1]
                if final_key in ["port", "token_expire_minutes", "workers"]:
                    current[final_key] = int(value)
                elif final_key in ["debug", "enabled", "require_https"]:
                    current[final_key] = value.lower() in ("true", "1", "yes")
                else:
                    current[final_key] = value
        
        return config
    
    @staticmethod
    def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combinar múltiples configuraciones (última tiene prioridad).
        
        Args:
            *configs: Configuraciones a combinar
            
        Returns:
            Configuración combinada
        """
        def deep_merge(base: dict, update: dict) -> dict:
            """Merge profundo de diccionarios"""
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    base[key] = deep_merge(base[key], value)
                else:
                    base[key] = value
            return base
        
        result: Dict[str, Any] = {}
        for config in configs:
            if config:
                result = deep_merge(result, config)
        
        return result


def load_config_from_file(file_path: str) -> MCPSettings:
    """
    Cargar configuración desde archivo.
    
    Args:
        file_path: Ruta al archivo de configuración
        
    Returns:
        MCPSettings cargado
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si la configuración es inválida
    """
    loader = ConfigLoader()
    data = loader.load_from_file(file_path)
    
    try:
        return MCPSettings(**data)
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}") from e


def load_config_from_env(prefix: str = "MCP_") -> MCPSettings:
    """
    Cargar configuración desde variables de entorno.
    
    Args:
        prefix: Prefijo para variables de entorno
        
    Returns:
        MCPSettings cargado desde env
    """
    from .settings import get_settings
    return get_settings()

