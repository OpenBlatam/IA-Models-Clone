"""
Sistema de recetas personalizadas
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid


@dataclass
class CustomRecipe:
    """Receta personalizada"""
    id: str
    user_id: str
    name: str
    description: str
    ingredients: List[Dict]  # [{"name": "...", "amount": "...", "notes": "..."}]
    instructions: List[str]
    skin_type_target: List[str]
    benefits: List[str]
    preparation_time: int  # minutos
    shelf_life: str  # "1 week", "2 weeks", etc.
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "description": self.description,
            "ingredients": self.ingredients,
            "instructions": self.instructions,
            "skin_type_target": self.skin_type_target,
            "benefits": self.benefits,
            "preparation_time": self.preparation_time,
            "shelf_life": self.shelf_life,
            "created_at": self.created_at
        }


class CustomRecipesManager:
    """Gestor de recetas personalizadas"""
    
    def __init__(self):
        """Inicializa el gestor"""
        self.recipes: Dict[str, List[CustomRecipe]] = {}  # user_id -> [recipes]
        self.templates: List[Dict] = []
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Inicializa plantillas de recetas"""
        self.templates = [
            {
                "name": "Mascarilla Hidratante",
                "description": "Mascarilla casera para hidratación intensa",
                "ingredients": [
                    {"name": "Miel", "amount": "1 cucharada"},
                    {"name": "Yogur", "amount": "1 cucharada"},
                    {"name": "Aceite de coco", "amount": "1 cucharadita"}
                ],
                "instructions": [
                    "Mezclar todos los ingredientes",
                    "Aplicar sobre el rostro limpio",
                    "Dejar actuar 15-20 minutos",
                    "Enjuagar con agua tibia"
                ],
                "skin_type_target": ["dry", "combination"],
                "benefits": ["Hidratación", "Suavidad"],
                "preparation_time": 5
            }
        ]
    
    def create_recipe(self, user_id: str, name: str, description: str,
                     ingredients: List[Dict], instructions: List[str],
                     skin_type_target: List[str], benefits: List[str],
                     preparation_time: int = 10,
                     shelf_life: str = "1 week") -> CustomRecipe:
        """Crea una nueva receta"""
        recipe = CustomRecipe(
            id=str(uuid.uuid4()),
            user_id=user_id,
            name=name,
            description=description,
            ingredients=ingredients,
            instructions=instructions,
            skin_type_target=skin_type_target,
            benefits=benefits,
            preparation_time=preparation_time,
            shelf_life=shelf_life
        )
        
        if user_id not in self.recipes:
            self.recipes[user_id] = []
        
        self.recipes[user_id].append(recipe)
        return recipe
    
    def create_from_template(self, user_id: str, template_name: str,
                            modifications: Optional[Dict] = None) -> CustomRecipe:
        """Crea receta desde plantilla"""
        template = next((t for t in self.templates if t["name"] == template_name), None)
        
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        # Aplicar modificaciones si existen
        ingredients = template["ingredients"].copy()
        instructions = template["instructions"].copy()
        
        if modifications:
            if "ingredients" in modifications:
                ingredients = modifications["ingredients"]
            if "instructions" in modifications:
                instructions = modifications["instructions"]
        
        return self.create_recipe(
            user_id=user_id,
            name=template["name"],
            description=template["description"],
            ingredients=ingredients,
            instructions=instructions,
            skin_type_target=template["skin_type_target"],
            benefits=template["benefits"],
            preparation_time=template.get("preparation_time", 10)
        )
    
    def get_user_recipes(self, user_id: str) -> List[CustomRecipe]:
        """Obtiene recetas del usuario"""
        return self.recipes.get(user_id, [])
    
    def get_recipe(self, user_id: str, recipe_id: str) -> Optional[CustomRecipe]:
        """Obtiene una receta específica"""
        user_recipes = self.recipes.get(user_id, [])
        
        for recipe in user_recipes:
            if recipe.id == recipe_id:
                return recipe
        
        return None
    
    def get_templates(self) -> List[Dict]:
        """Obtiene plantillas disponibles"""
        return self.templates
    
    def recommend_recipes(self, user_id: str, skin_type: str,
                         desired_benefits: List[str]) -> List[CustomRecipe]:
        """Recomienda recetas basadas en perfil"""
        user_recipes = self.recipes.get(user_id, [])
        recommended = []
        
        for recipe in user_recipes:
            # Verificar tipo de piel
            if skin_type in recipe.skin_type_target:
                # Verificar beneficios deseados
                matching_benefits = [
                    b for b in recipe.benefits
                    if any(db.lower() in b.lower() for db in desired_benefits)
                ]
                
                if matching_benefits or not desired_benefits:
                    recommended.append(recipe)
        
        # Ordenar por número de beneficios coincidentes
        recommended.sort(
            key=lambda r: len([b for b in r.benefits if b in desired_benefits]),
            reverse=True
        )
        
        return recommended






