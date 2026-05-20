"""
Resources Service - Sistema de recursos y biblioteca
====================================================

Biblioteca de recursos educativos: artículos, videos, cursos, plantillas, etc.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class ResourceType(str, Enum):
    """Tipos de recursos"""
    ARTICLE = "article"
    VIDEO = "video"
    COURSE = "course"
    TEMPLATE = "template"
    TOOL = "tool"
    EBOOK = "ebook"
    PODCAST = "podcast"
    CHECKLIST = "checklist"


class ResourceCategory(str, Enum):
    """Categorías de recursos"""
    CV_RESUME = "cv_resume"
    INTERVIEW = "interview"
    NETWORKING = "networking"
    SKILL_DEVELOPMENT = "skill_development"
    CAREER_TRANSITION = "career_transition"
    MOTIVATION = "motivation"
    SALARY_NEGOTIATION = "salary_negotiation"


@dataclass
class Resource:
    """Recurso educativo"""
    id: str
    title: str
    description: str
    resource_type: ResourceType
    category: ResourceCategory
    url: str
    author: Optional[str] = None
    duration: Optional[str] = None  # Para videos/cursos
    difficulty_level: Optional[str] = None  # beginner, intermediate, advanced
    tags: List[str] = field(default_factory=list)
    views: int = 0
    likes: int = 0
    rating: float = 0.0
    ratings_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    featured: bool = False


class ResourcesService:
    """Servicio de recursos"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.resources: Dict[str, Resource] = {}
        self.user_bookmarks: Dict[str, List[str]] = {}  # user_id -> [resource_ids]
        self._initialize_default_resources()
        logger.info("ResourcesService initialized")
    
    def _initialize_default_resources(self):
        """Inicializar recursos por defecto"""
        default_resources = [
            Resource(
                id="res_1",
                title="Cómo escribir un CV que destaque",
                description="Guía completa para crear un CV efectivo",
                resource_type=ResourceType.ARTICLE,
                category=ResourceCategory.CV_RESUME,
                url="#",
                author="Career Experts",
                difficulty_level="beginner",
                tags=["CV", "resume", "job search"],
                featured=True,
            ),
            Resource(
                id="res_2",
                title="Plantilla de CV profesional",
                description="Plantilla descargable para CV",
                resource_type=ResourceType.TEMPLATE,
                category=ResourceCategory.CV_RESUME,
                url="#",
                tags=["template", "CV"],
            ),
            Resource(
                id="res_3",
                title="Preparación para entrevistas técnicas",
                description="Curso completo de preparación",
                resource_type=ResourceType.COURSE,
                category=ResourceCategory.INTERVIEW,
                url="#",
                duration="4 horas",
                difficulty_level="intermediate",
                tags=["interview", "technical"],
            ),
        ]
        
        for resource in default_resources:
            self.resources[resource.id] = resource
    
    def get_resources(
        self,
        resource_type: Optional[ResourceType] = None,
        category: Optional[ResourceCategory] = None,
        tags: Optional[List[str]] = None,
        limit: int = 20
    ) -> List[Resource]:
        """Obtener recursos"""
        resources = list(self.resources.values())
        
        if resource_type:
            resources = [r for r in resources if r.resource_type == resource_type]
        
        if category:
            resources = [r for r in resources if r.category == category]
        
        if tags:
            resources = [
                r for r in resources
                if any(tag in r.tags for tag in tags)
            ]
        
        # Ordenar por featured primero, luego por rating
        resources.sort(
            key=lambda x: (not x.featured, -x.rating),
        )
        
        return resources[:limit]
    
    def get_resource(self, resource_id: str) -> Optional[Resource]:
        """Obtener recurso específico"""
        resource = self.resources.get(resource_id)
        if resource:
            resource.views += 1
        return resource
    
    def bookmark_resource(self, user_id: str, resource_id: str) -> bool:
        """Guardar recurso en favoritos"""
        if resource_id not in self.resources:
            return False
        
        if user_id not in self.user_bookmarks:
            self.user_bookmarks[user_id] = []
        
        if resource_id not in self.user_bookmarks[user_id]:
            self.user_bookmarks[user_id].append(resource_id)
            return True
        
        return False
    
    def get_user_bookmarks(self, user_id: str) -> List[Resource]:
        """Obtener recursos guardados del usuario"""
        resource_ids = self.user_bookmarks.get(user_id, [])
        return [
            self.resources[rid] for rid in resource_ids
            if rid in self.resources
        ]
    
    def rate_resource(
        self,
        user_id: str,
        resource_id: str,
        rating: float
    ) -> Resource:
        """Calificar recurso"""
        resource = self.resources.get(resource_id)
        if not resource:
            raise ValueError(f"Resource {resource_id} not found")
        
        # Actualizar rating promedio
        total_rating = resource.rating * resource.ratings_count
        resource.ratings_count += 1
        resource.rating = (total_rating + rating) / resource.ratings_count
        
        return resource
    
    def get_featured_resources(self, limit: int = 10) -> List[Resource]:
        """Obtener recursos destacados"""
        featured = [r for r in self.resources.values() if r.featured]
        featured.sort(key=lambda x: x.rating, reverse=True)
        return featured[:limit]




