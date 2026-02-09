"""
Base Config - Clase base para configuración
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseConfig(ABC):
    """Clase base abstracta para configuradores"""
    
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Obtiene un valor de configuración"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any) -> bool:
        """Establece un valor de configuración"""
        pass
    
    @abstractmethod
    def load(self) -> Dict[str, Any]:
        """Carga la configuración"""
        pass

