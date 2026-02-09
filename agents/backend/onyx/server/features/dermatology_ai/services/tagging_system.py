"""
Sistema de tags y categorización
"""

from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict


@dataclass
class Tag:
    """Tag"""
    name: str
    category: Optional[str] = None
    color: Optional[str] = None
    description: Optional[str] = None
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "name": self.name,
            "category": self.category,
            "color": self.color,
            "description": self.description,
            "created_at": self.created_at
        }


class TaggingSystem:
    """Sistema de tags"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.tags: Dict[str, Tag] = {}  # tag_name -> Tag
        self.resource_tags: Dict[str, Set[str]] = {}  # resource_id -> {tag_names}
        self.tag_resources: Dict[str, Set[str]] = {}  # tag_name -> {resource_ids}
    
    def create_tag(self, name: str, category: Optional[str] = None,
                   color: Optional[str] = None,
                   description: Optional[str] = None) -> Tag:
        """
        Crea un tag
        
        Args:
            name: Nombre del tag
            category: Categoría
            color: Color
            description: Descripción
            
        Returns:
            Tag creado
        """
        tag = Tag(
            name=name,
            category=category,
            color=color,
            description=description
        )
        
        self.tags[name] = tag
        return tag
    
    def tag_resource(self, resource_id: str, tag_names: List[str]):
        """
        Etiqueta un recurso
        
        Args:
            resource_id: ID del recurso
            tag_names: Lista de nombres de tags
        """
        if resource_id not in self.resource_tags:
            self.resource_tags[resource_id] = set()
        
        for tag_name in tag_names:
            # Crear tag si no existe
            if tag_name not in self.tags:
                self.create_tag(tag_name)
            
            self.resource_tags[resource_id].add(tag_name)
            
            if tag_name not in self.tag_resources:
                self.tag_resources[tag_name] = set()
            self.tag_resources[tag_name].add(resource_id)
    
    def untag_resource(self, resource_id: str, tag_names: List[str]):
        """
        Elimina tags de un recurso
        
        Args:
            resource_id: ID del recurso
            tag_names: Lista de nombres de tags
        """
        if resource_id not in self.resource_tags:
            return
        
        for tag_name in tag_names:
            self.resource_tags[resource_id].discard(tag_name)
            
            if tag_name in self.tag_resources:
                self.tag_resources[tag_name].discard(resource_id)
    
    def get_resource_tags(self, resource_id: str) -> List[Tag]:
        """Obtiene tags de un recurso"""
        tag_names = self.resource_tags.get(resource_id, set())
        return [self.tags[name] for name in tag_names if name in self.tags]
    
    def get_resources_by_tag(self, tag_name: str) -> List[str]:
        """Obtiene recursos por tag"""
        return list(self.tag_resources.get(tag_name, set()))
    
    def get_all_tags(self) -> List[Tag]:
        """Obtiene todos los tags"""
        return list(self.tags.values())
    
    def get_tag_stats(self) -> Dict:
        """Obtiene estadísticas de tags"""
        return {
            "total_tags": len(self.tags),
            "total_resources_tagged": len(self.resource_tags),
            "most_used_tags": self._get_most_used_tags(limit=10)
        }
    
    def _get_most_used_tags(self, limit: int = 10) -> List[Dict]:
        """Obtiene tags más usados"""
        tag_counts = {
            tag_name: len(resources)
            for tag_name, resources in self.tag_resources.items()
        }
        
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {
                "tag": tag_name,
                "count": count,
                "tag_info": self.tags.get(tag_name).to_dict() if tag_name in self.tags else None
            }
            for tag_name, count in sorted_tags[:limit]
        ]






