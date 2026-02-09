"""
Clothing Templates System
=========================
Sistema de plantillas de ropa predefinidas
"""

import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class ClothingCategory(Enum):
    """Categorías de ropa"""
    CASUAL = "casual"
    FORMAL = "formal"
    SPORTY = "sporty"
    VINTAGE = "vintage"
    MODERN = "modern"
    ELEGANT = "elegant"
    FANTASY = "fantasy"
    COSTUME = "costume"


class ClothingType(Enum):
    """Tipos de ropa"""
    TOP = "top"
    BOTTOM = "bottom"
    DRESS = "dress"
    OUTERWEAR = "outerwear"
    SHOES = "shoes"
    ACCESSORIES = "accessories"
    FULL_OUTFIT = "full_outfit"


@dataclass
class ClothingTemplate:
    """Plantilla de ropa"""
    id: str
    name: str
    description: str
    category: ClothingCategory
    clothing_type: ClothingType
    prompt: str
    negative_prompt: str
    tags: List[str]
    colors: List[str]
    styles: List[str]
    preview_image: Optional[str] = None
    usage_count: int = 0
    rating: float = 0.0


class ClothingTemplateManager:
    """
    Gestor de plantillas de ropa
    """
    
    def __init__(self):
        self.templates: Dict[str, ClothingTemplate] = {}
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Cargar plantillas por defecto"""
        default_templates = [
            {
                'id': 'casual_tshirt',
                'name': 'Camiseta Casual',
                'description': 'Camiseta casual cómoda y moderna',
                'category': ClothingCategory.CASUAL,
                'clothing_type': ClothingType.TOP,
                'prompt': 'a casual t-shirt, comfortable, modern style, high quality',
                'negative_prompt': 'formal, elegant, vintage, blurry',
                'tags': ['casual', 'comfortable', 'modern'],
                'colors': ['white', 'black', 'blue', 'gray'],
                'styles': ['casual', 'streetwear']
            },
            {
                'id': 'elegant_dress',
                'name': 'Vestido Elegante',
                'description': 'Vestido elegante para ocasiones formales',
                'category': ClothingCategory.ELEGANT,
                'clothing_type': ClothingType.DRESS,
                'prompt': 'an elegant dress, formal, sophisticated, high quality, detailed',
                'negative_prompt': 'casual, sporty, informal, low quality',
                'tags': ['elegant', 'formal', 'sophisticated'],
                'colors': ['black', 'navy', 'burgundy', 'emerald'],
                'styles': ['elegant', 'formal', 'classic']
            },
            {
                'id': 'sporty_jacket',
                'name': 'Chaqueta Deportiva',
                'description': 'Chaqueta deportiva funcional y moderna',
                'category': ClothingCategory.SPORTY,
                'clothing_type': ClothingType.OUTERWEAR,
                'prompt': 'a sporty jacket, athletic, functional, modern design, high quality',
                'negative_prompt': 'formal, elegant, vintage, delicate',
                'tags': ['sporty', 'athletic', 'functional'],
                'colors': ['black', 'navy', 'red', 'gray'],
                'styles': ['sporty', 'athletic', 'modern']
            },
            {
                'id': 'vintage_jeans',
                'name': 'Jeans Vintage',
                'description': 'Pantalones jeans con estilo vintage',
                'category': ClothingCategory.VINTAGE,
                'clothing_type': ClothingType.BOTTOM,
                'prompt': 'vintage jeans, retro style, classic design, high quality denim',
                'negative_prompt': 'modern, futuristic, formal, elegant',
                'tags': ['vintage', 'retro', 'classic'],
                'colors': ['blue', 'indigo', 'black'],
                'styles': ['vintage', 'retro', 'classic']
            },
            {
                'id': 'fantasy_armor',
                'name': 'Armadura Fantástica',
                'description': 'Armadura de estilo fantástico',
                'category': ClothingCategory.FANTASY,
                'clothing_type': ClothingType.FULL_OUTFIT,
                'prompt': 'fantasy armor, epic design, detailed, high quality, magical',
                'negative_prompt': 'casual, modern, realistic, simple',
                'tags': ['fantasy', 'epic', 'magical'],
                'colors': ['silver', 'gold', 'bronze', 'steel'],
                'styles': ['fantasy', 'epic', 'adventure']
            }
        ]
        
        for template_data in default_templates:
            template = ClothingTemplate(**template_data)
            self.templates[template.id] = template
    
    def create_template(
        self,
        name: str,
        description: str,
        category: ClothingCategory,
        clothing_type: ClothingType,
        prompt: str,
        negative_prompt: str,
        tags: List[str],
        colors: List[str],
        styles: List[str],
        preview_image: Optional[str] = None
    ) -> ClothingTemplate:
        """Crear nueva plantilla"""
        template_id = f"{category.value}_{name.lower().replace(' ', '_')}"
        
        template = ClothingTemplate(
            id=template_id,
            name=name,
            description=description,
            category=category,
            clothing_type=clothing_type,
            prompt=prompt,
            negative_prompt=negative_prompt,
            tags=tags,
            colors=colors,
            styles=styles,
            preview_image=preview_image
        )
        
        self.templates[template_id] = template
        return template
    
    def get_template(self, template_id: str) -> Optional[ClothingTemplate]:
        """Obtener plantilla por ID"""
        return self.templates.get(template_id)
    
    def list_templates(
        self,
        category: Optional[ClothingCategory] = None,
        clothing_type: Optional[ClothingType] = None,
        search: Optional[str] = None
    ) -> List[ClothingTemplate]:
        """
        Listar plantillas con filtros
        
        Args:
            category: Filtrar por categoría
            clothing_type: Filtrar por tipo
            search: Buscar en nombre/descripción
        """
        templates = list(self.templates.values())
        
        if category:
            templates = [t for t in templates if t.category == category]
        
        if clothing_type:
            templates = [t for t in templates if t.clothing_type == clothing_type]
        
        if search:
            search_lower = search.lower()
            templates = [
                t for t in templates
                if search_lower in t.name.lower() or search_lower in t.description.lower()
            ]
        
        return templates
    
    def use_template(self, template_id: str):
        """Marcar plantilla como usada"""
        if template_id in self.templates:
            self.templates[template_id].usage_count += 1
    
    def rate_template(self, template_id: str, rating: float):
        """Calificar plantilla"""
        if template_id in self.templates:
            template = self.templates[template_id]
            # Promedio móvil simple
            if template.rating == 0:
                template.rating = rating
            else:
                template.rating = (template.rating + rating) / 2
    
    def get_popular_templates(self, limit: int = 10) -> List[ClothingTemplate]:
        """Obtener plantillas más populares"""
        templates = sorted(
            self.templates.values(),
            key=lambda t: (t.usage_count, t.rating),
            reverse=True
        )
        return templates[:limit]
    
    def get_recommendations(
        self,
        category: Optional[ClothingCategory] = None,
        exclude_ids: Optional[List[str]] = None
    ) -> List[ClothingTemplate]:
        """Obtener recomendaciones basadas en uso y rating"""
        templates = self.list_templates(category=category)
        
        if exclude_ids:
            templates = [t for t in templates if t.id not in exclude_ids]
        
        # Ordenar por score combinado
        templates.sort(
            key=lambda t: (t.rating * 0.6 + min(t.usage_count / 10, 1) * 0.4),
            reverse=True
        )
        
        return templates[:5]  # Top 5 recomendaciones
    
    def export_template(self, template_id: str) -> Dict:
        """Exportar plantilla como diccionario"""
        if template_id in self.templates:
            template = self.templates[template_id]
            return asdict(template)
        return {}
    
    def import_template(self, template_data: Dict) -> ClothingTemplate:
        """Importar plantilla desde diccionario"""
        template = ClothingTemplate(**template_data)
        self.templates[template.id] = template
        return template


# Instancia global
clothing_template_manager = ClothingTemplateManager()

