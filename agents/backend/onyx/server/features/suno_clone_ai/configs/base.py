"""
Base Configuration - Clase base para configuraciones
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseConfig(ABC):
    """Clase base abstracta para configuraciones"""

    def __init__(self, **kwargs):
        """Inicializa la configuración con parámetros opcionales"""
        self._config: Dict[str, Any] = {}
        self._load_defaults()
        self._update_from_kwargs(kwargs)

    @abstractmethod
    def _load_defaults(self) -> None:
        """Carga valores por defecto de la configuración"""
        pass

    def _update_from_kwargs(self, kwargs: Dict[str, Any]) -> None:
        """Actualiza configuración desde kwargs"""
        for key, value in kwargs.items():
            if hasattr(self, key) or key in self._config:
                setattr(self, key, value)
                self._config[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Obtiene un valor de configuración"""
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Establece un valor de configuración"""
        self._config[key] = value
        setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convierte la configuración a diccionario"""
        return self._config.copy()

    def validate(self) -> bool:
        """Valida la configuración"""
        return True

