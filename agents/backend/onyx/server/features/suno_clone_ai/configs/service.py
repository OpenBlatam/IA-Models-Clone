"""
Config Service - Servicio de configuraciones
"""

from typing import Any, Dict, Optional
from .base import BaseConfig
from .settings import Settings


class ConfigService:
    """Servicio para gestionar configuraciones del sistema"""

    def __init__(self, settings: Optional[Settings] = None):
        """Inicializa el servicio de configuraciones"""
        self.settings = settings or Settings()
        self._configs: Dict[str, BaseConfig] = {}

    def register_config(self, name: str, config: BaseConfig) -> None:
        """Registra una configuración"""
        self._configs[name] = config

    def get_config(self, name: str) -> Optional[BaseConfig]:
        """Obtiene una configuración por nombre"""
        return self._configs.get(name)

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Obtiene un setting del sistema"""
        return getattr(self.settings, key, default)

    def reload(self) -> None:
        """Recarga las configuraciones"""
        self.settings = Settings()

