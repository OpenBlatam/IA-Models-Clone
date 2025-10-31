"""
History Entry Entity
===================

Entidad que representa una entrada de historial de IA.
Responsabilidad única: Gestionar datos de una entrada de historial.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional
import uuid


@dataclass
class HistoryEntry:
    """
    Entidad que representa una entrada de historial de IA.
    
    Responsabilidad única: Gestionar los datos y comportamiento
    de una entrada de historial de IA.
    """
    id: str
    content: str
    model: str
    timestamp: datetime
    quality: float
    words: int
    readability: float
    sentiment: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validar datos después de la inicialización."""
        if not self.content or not self.content.strip():
            raise ValueError("Content cannot be empty")
        
        if not self.model or not self.model.strip():
            raise ValueError("Model cannot be empty")
        
        if not (0.0 <= self.quality <= 1.0):
            raise ValueError("Quality must be between 0.0 and 1.0")
        
        if self.words < 0:
            raise ValueError("Word count cannot be negative")
    
    @classmethod
    def create(
        cls,
        content: str,
        model: str,
        quality: float,
        words: int,
        readability: float,
        sentiment: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'HistoryEntry':
        """
        Factory method para crear una nueva entrada de historial.
        
        Args:
            content: Contenido de la entrada
            model: Modelo de IA utilizado
            quality: Puntuación de calidad
            words: Número de palabras
            readability: Puntuación de legibilidad
            sentiment: Puntuación de sentimiento
            metadata: Metadatos opcionales
            
        Returns:
            Nueva instancia de HistoryEntry
        """
        return cls(
            id=str(uuid.uuid4()),
            content=content,
            model=model,
            timestamp=datetime.utcnow(),
            quality=quality,
            words=words,
            readability=readability,
            sentiment=sentiment,
            metadata=metadata or {}
        )
    
    def is_high_quality(self, threshold: float = 0.7) -> bool:
        """
        Verificar si la entrada es de alta calidad.
        
        Args:
            threshold: Umbral de calidad
            
        Returns:
            True si la calidad es alta
        """
        return self.quality >= threshold
    
    def is_recent(self, days: int = 7) -> bool:
        """
        Verificar si la entrada es reciente.
        
        Args:
            days: Número de días
            
        Returns:
            True si la entrada es reciente
        """
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(days=days)
        return self.timestamp >= cutoff
    
    def get_content_length(self) -> int:
        """Obtener longitud del contenido."""
        return len(self.content)
    
    def has_metadata(self, key: str) -> bool:
        """
        Verificar si existe una clave en los metadatos.
        
        Args:
            key: Clave a verificar
            
        Returns:
            True si la clave existe
        """
        return key in self.metadata
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Obtener valor de metadatos por clave.
        
        Args:
            key: Clave de metadatos
            default: Valor por defecto
            
        Returns:
            Valor de metadatos o valor por defecto
        """
        return self.metadata.get(key, default)
    
    def update_metadata(self, key: str, value: Any) -> None:
        """
        Actualizar valor de metadatos.
        
        Args:
            key: Clave de metadatos
            value: Nuevo valor
        """
        self.metadata[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertir a diccionario.
        
        Returns:
            Representación en diccionario
        """
        return {
            "id": self.id,
            "content": self.content,
            "model": self.model,
            "timestamp": self.timestamp.isoformat(),
            "quality": self.quality,
            "words": self.words,
            "readability": self.readability,
            "sentiment": self.sentiment,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HistoryEntry':
        """
        Crear desde diccionario.
        
        Args:
            data: Datos del diccionario
            
        Returns:
            Instancia de HistoryEntry
        """
        return cls(
            id=data["id"],
            content=data["content"],
            model=data["model"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            quality=data["quality"],
            words=data["words"],
            readability=data["readability"],
            sentiment=data["sentiment"],
            metadata=data.get("metadata", {})
        )
    
    def __str__(self) -> str:
        """Representación de cadena."""
        return f"HistoryEntry(id={self.id}, model={self.model}, quality={self.quality:.2f})"
    
    def __repr__(self) -> str:
        """Representación detallada."""
        return f"HistoryEntry(id='{self.id}', model='{self.model}', quality={self.quality:.2f})"




