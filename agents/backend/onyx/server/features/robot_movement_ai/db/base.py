"""
Base Model - Clase base para modelos de base de datos
"""
from abc import ABC
from typing import Any, Dict, Optional


class BaseModel(ABC):
    """Clase base abstracta para modelos de base de datos"""
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el modelo a diccionario"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Crea una instancia desde un diccionario"""
        return cls(**data)

