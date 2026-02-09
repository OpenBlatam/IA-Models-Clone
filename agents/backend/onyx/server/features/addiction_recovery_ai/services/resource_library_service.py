"""
Servicio de Biblioteca de Recursos Educativos - Sistema completo de recursos educativos
"""

from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum


class ResourceType(str, Enum):
    """Tipos de recursos"""
    ARTICLE = "article"
    VIDEO = "video"
    PODCAST = "podcast"
    BOOK = "book"
    COURSE = "course"
    WORKSHOP = "workshop"
    TOOL = "tool"


class ResourceLibraryService:
    """Servicio de biblioteca de recursos educativos"""
    
    def __init__(self):
        """Inicializa el servicio de biblioteca"""
        self.resources = self._load_resources()
    
    def get_resources(
        self,
        resource_type: Optional[str] = None,
        topic: Optional[str] = None,
        difficulty: Optional[str] = None
    ) -> List[Dict]:
        """
        Obtiene recursos educativos
        
        Args:
            resource_type: Tipo de recurso (opcional)
            topic: Tema (opcional)
            difficulty: Nivel de dificultad (opcional)
        
        Returns:
            Lista de recursos
        """
        resources = self.resources.copy()
        
        if resource_type:
            resources = [r for r in resources if r.get("type") == resource_type]
        
        if topic:
            resources = [r for r in resources if topic.lower() in r.get("topics", [])]
        
        if difficulty:
            resources = [r for r in resources if r.get("difficulty") == difficulty]
        
        return resources
    
    def get_resource_details(
        self,
        resource_id: str
    ) -> Dict:
        """
        Obtiene detalles de un recurso
        
        Args:
            resource_id: ID del recurso
        
        Returns:
            Detalles del recurso
        """
        resource = next((r for r in self.resources if r.get("id") == resource_id), None)
        
        if not resource:
            return {
                "error": "Resource not found"
            }
        
        return {
            **resource,
            "views": 0,
            "rating": 0.0,
            "reviews_count": 0,
            "related_resources": self._get_related_resources(resource_id)
        }
    
    def search_resources(
        self,
        query: str,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Busca recursos
        
        Args:
            query: Término de búsqueda
            filters: Filtros adicionales (opcional)
        
        Returns:
            Lista de recursos encontrados
        """
        query_lower = query.lower()
        results = []
        
        for resource in self.resources:
            if (query_lower in resource.get("title", "").lower() or
                query_lower in resource.get("description", "").lower() or
                any(query_lower in topic.lower() for topic in resource.get("topics", []))):
                results.append(resource)
        
        return results
    
    def get_curated_collections(
        self,
        collection_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Obtiene colecciones curadas
        
        Args:
            collection_type: Tipo de colección (opcional)
        
        Returns:
            Lista de colecciones
        """
        collections = [
            {
                "id": "collection_1",
                "name": "Fundamentos de la Recuperación",
                "description": "Recursos esenciales para comenzar tu viaje",
                "resources_count": 10,
                "type": "beginner"
            },
            {
                "id": "collection_2",
                "name": "Manejo de Cravings",
                "description": "Técnicas y estrategias para manejar deseos intensos",
                "resources_count": 8,
                "type": "intermediate"
            },
            {
                "id": "collection_3",
                "name": "Mantenimiento a Largo Plazo",
                "description": "Recursos para mantener tu recuperación",
                "resources_count": 12,
                "type": "advanced"
            }
        ]
        
        if collection_type:
            collections = [c for c in collections if c.get("type") == collection_type]
        
        return collections
    
    def _load_resources(self) -> List[Dict]:
        """Carga recursos educativos"""
        return [
            {
                "id": "resource_1",
                "type": ResourceType.ARTICLE,
                "title": "Entendiendo la Adicción",
                "description": "Guía completa sobre cómo funciona la adicción",
                "topics": ["addiction", "education", "basics"],
                "difficulty": "beginner",
                "duration_minutes": 15,
                "url": "/resources/articles/understanding-addiction"
            },
            {
                "id": "resource_2",
                "type": ResourceType.VIDEO,
                "title": "Técnicas de Prevención de Recaídas",
                "description": "Video educativo sobre estrategias de prevención",
                "topics": ["relapse", "prevention", "strategies"],
                "difficulty": "intermediate",
                "duration_minutes": 25,
                "url": "/resources/videos/relapse-prevention"
            },
            {
                "id": "resource_3",
                "type": ResourceType.COURSE,
                "title": "Curso de Mindfulness para la Recuperación",
                "description": "Curso completo de 4 semanas",
                "topics": ["mindfulness", "meditation", "recovery"],
                "difficulty": "beginner",
                "duration_minutes": 480,
                "url": "/resources/courses/mindfulness-recovery"
            }
        ]
    
    def _get_related_resources(self, resource_id: str) -> List[str]:
        """Obtiene recursos relacionados"""
        # En implementación real, esto buscaría recursos con temas similares
        return []

