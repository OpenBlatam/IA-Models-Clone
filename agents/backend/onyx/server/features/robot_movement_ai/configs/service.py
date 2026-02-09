"""
Config Service - Servicio de configuración
"""
from typing import Any, Dict
from .env_loader import EnvLoader
from .yaml_loader import YAMLLoader
from .hot_reload import HotReloadManager


class ConfigService:
    """Servicio centralizado de configuración"""
    
    def __init__(self):
        self.env_loader = EnvLoader()
        self.yaml_loader = YAMLLoader()
        self.hot_reload = HotReloadManager()
        self._config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self):
        """Carga la configuración desde múltiples fuentes"""
        env_config = self.env_loader.load()
        yaml_config = self.yaml_loader.load()
        self._config = {**env_config, **yaml_config}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Obtiene un valor de configuración"""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> bool:
        """Establece un valor de configuración"""
        self._config[key] = value
        return True
    
    def reload(self):
        """Recarga la configuración"""
        self._load_config()

