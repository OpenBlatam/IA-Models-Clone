"""
Base Models
===========

Modelos base con funcionalidades comunes.
"""

from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class BaseModel:
    """Modelo base."""
    id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "id": self.id
        }
    
    def to_json(self) -> str:
        """Convertir a JSON."""
        import json
        return json.dumps(self.to_dict(), default=str, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Crear desde diccionario."""
        return cls(**data)


@dataclass
class TimestampMixin:
    """Mixin para timestamps."""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    
    def touch(self):
        """Actualizar timestamp."""
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Agregar timestamps a dict."""
        return {
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }




