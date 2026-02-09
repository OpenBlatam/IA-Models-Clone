"""
Content Entity - Entidad de Contenido
Entidad de dominio para representar contenido
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
import hashlib

class ContentType(str, Enum):
    """Tipos de contenido"""
    TEXT = "text"
    DOCUMENT = "document"
    CODE = "code"
    MARKDOWN = "markdown"
    HTML = "html"

class ContentStatus(str, Enum):
    """Estados del contenido"""
    DRAFT = "draft"
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"

@dataclass
class Content:
    """Entidad de contenido"""
    
    # Identificadores
    id: str
    content: str
    content_hash: str = field(init=False)
    
    # Metadatos
    title: Optional[str] = None
    description: Optional[str] = None
    content_type: ContentType = ContentType.TEXT
    status: ContentStatus = ContentStatus.DRAFT
    
    # Información del modelo
    model_version: Optional[str] = None
    model_provider: Optional[str] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Metadatos adicionales
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    # Propiedades calculadas
    word_count: int = field(init=False)
    character_count: int = field(init=False)
    
    def __post_init__(self):
        """Inicialización post-construcción"""
        self.content_hash = self._calculate_hash()
        self.word_count = len(self.content.split())
        self.character_count = len(self.content)
    
    def _calculate_hash(self) -> str:
        """Calcular hash del contenido"""
        return hashlib.md5(self.content.encode('utf-8')).hexdigest()
    
    def update_content(self, new_content: str) -> None:
        """Actualizar contenido"""
        self.content = new_content
        self.content_hash = self._calculate_hash()
        self.word_count = len(self.content.split())
        self.character_count = len(self.content)
        self.updated_at = datetime.utcnow()
    
    def add_tag(self, tag: str) -> None:
        """Agregar tag"""
        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.utcnow()
    
    def remove_tag(self, tag: str) -> None:
        """Remover tag"""
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.utcnow()
    
    def update_metadata(self, key: str, value: Any) -> None:
        """Actualizar metadatos"""
        self.metadata[key] = value
        self.updated_at = datetime.utcnow()
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Obtener metadato"""
        return self.metadata.get(key, default)
    
    def change_status(self, new_status: ContentStatus) -> None:
        """Cambiar estado"""
        self.status = new_status
        self.updated_at = datetime.utcnow()
    
    def is_processed(self) -> bool:
        """Verificar si está procesado"""
        return self.status == ContentStatus.COMPLETED
    
    def is_failed(self) -> bool:
        """Verificar si falló"""
        return self.status == ContentStatus.FAILED
    
    def is_processing(self) -> bool:
        """Verificar si está procesando"""
        return self.status == ContentStatus.PROCESSING
    
    def get_summary(self, max_length: int = 100) -> str:
        """Obtener resumen del contenido"""
        if len(self.content) <= max_length:
            return self.content
        return self.content[:max_length] + "..."
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "id": self.id,
            "content": self.content,
            "content_hash": self.content_hash,
            "title": self.title,
            "description": self.description,
            "content_type": self.content_type.value,
            "status": self.status.value,
            "model_version": self.model_version,
            "model_provider": self.model_provider,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
            "tags": self.tags,
            "word_count": self.word_count,
            "character_count": self.character_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Content':
        """Crear desde diccionario"""
        return cls(
            id=data["id"],
            content=data["content"],
            title=data.get("title"),
            description=data.get("description"),
            content_type=ContentType(data.get("content_type", "text")),
            status=ContentStatus(data.get("status", "draft")),
            model_version=data.get("model_version"),
            model_provider=data.get("model_provider"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", [])
        )
    
    def __str__(self) -> str:
        """Representación string"""
        return f"Content(id={self.id}, type={self.content_type}, status={self.status})"
    
    def __repr__(self) -> str:
        """Representación para debugging"""
        return f"Content(id='{self.id}', content_type='{self.content_type}', status='{self.status}', word_count={self.word_count})"







