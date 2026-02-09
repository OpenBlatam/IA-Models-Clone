"""
Tags - Sistema de etiquetas y categorización
"""

import logging
from typing import Dict, Any, Optional, List, Set
from datetime import datetime
from dataclasses import dataclass, field
import uuid

logger = logging.getLogger(__name__)


@dataclass
class Tag:
    """Etiqueta"""
    id: str
    name: str
    color: Optional[str] = None
    description: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    usage_count: int = 0


class TagManager:
    """Gestor de etiquetas"""

    def __init__(self):
        """Inicializar gestor de etiquetas"""
        self.tags: Dict[str, Tag] = {}
        self.content_tags: Dict[str, Set[str]] = {}  # content_id -> {tag_ids}

    def create_tag(
        self,
        name: str,
        color: Optional[str] = None,
        description: Optional[str] = None
    ) -> Tag:
        """
        Crear una etiqueta.

        Args:
            name: Nombre de la etiqueta
            color: Color (opcional)
            description: Descripción (opcional)

        Returns:
            Etiqueta creada
        """
        tag_id = str(uuid.uuid4())
        tag = Tag(
            id=tag_id,
            name=name,
            color=color,
            description=description
        )
        self.tags[tag_id] = tag
        logger.info(f"Etiqueta creada: {name}")
        return tag

    def tag_content(self, content_id: str, tag_id: str):
        """
        Etiquetar contenido.

        Args:
            content_id: ID del contenido
            tag_id: ID de la etiqueta
        """
        if tag_id not in self.tags:
            logger.warning(f"Etiqueta no encontrada: {tag_id}")
            return
        
        if content_id not in self.content_tags:
            self.content_tags[content_id] = set()
        
        self.content_tags[content_id].add(tag_id)
        self.tags[tag_id].usage_count += 1
        logger.info(f"Contenido {content_id} etiquetado con {tag_id}")

    def untag_content(self, content_id: str, tag_id: str):
        """
        Remover etiqueta de contenido.

        Args:
            content_id: ID del contenido
            tag_id: ID de la etiqueta
        """
        if content_id in self.content_tags:
            self.content_tags[content_id].discard(tag_id)
            if tag_id in self.tags:
                self.tags[tag_id].usage_count = max(0, self.tags[tag_id].usage_count - 1)

    def get_content_tags(self, content_id: str) -> List[Dict[str, Any]]:
        """
        Obtener etiquetas de un contenido.

        Args:
            content_id: ID del contenido

        Returns:
            Lista de etiquetas
        """
        tag_ids = self.content_tags.get(content_id, set())
        return [
            {
                "id": tag.id,
                "name": tag.name,
                "color": tag.color,
                "description": tag.description
            }
            for tag_id in tag_ids
            if tag_id in self.tags
            for tag in [self.tags[tag_id]]
        ]

    def search_by_tags(self, tag_ids: List[str]) -> List[str]:
        """
        Buscar contenidos por etiquetas.

        Args:
            tag_ids: IDs de etiquetas

        Returns:
            Lista de IDs de contenido
        """
        tag_set = set(tag_ids)
        matching_content = []
        
        for content_id, content_tag_ids in self.content_tags.items():
            if tag_set.issubset(content_tag_ids):
                matching_content.append(content_id)
        
        return matching_content

    def get_popular_tags(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Obtener etiquetas más populares.

        Args:
            limit: Número de etiquetas

        Returns:
            Lista de etiquetas
        """
        sorted_tags = sorted(
            self.tags.values(),
            key=lambda t: t.usage_count,
            reverse=True
        )
        
        return [
            {
                "id": tag.id,
                "name": tag.name,
                "color": tag.color,
                "usage_count": tag.usage_count
            }
            for tag in sorted_tags[:limit]
        ]

    def get_all_tags(self) -> List[Dict[str, Any]]:
        """
        Obtener todas las etiquetas.

        Returns:
            Lista de etiquetas
        """
        return [
            {
                "id": tag.id,
                "name": tag.name,
                "color": tag.color,
                "description": tag.description,
                "usage_count": tag.usage_count,
                "created_at": tag.created_at.isoformat()
            }
            for tag in self.tags.values()
        ]






