"""
Sistema de análisis de ingredientes
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Ingredient:
    """Ingrediente"""
    name: str
    category: str  # "active", "preservative", "emollient", "humectant", etc.
    benefits: List[str]
    concerns: List[str]
    skin_type_compatibility: List[str]  # "oily", "dry", "sensitive", "combination"
    concentration_range: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "name": self.name,
            "category": self.category,
            "benefits": self.benefits,
            "concerns": self.concerns,
            "skin_type_compatibility": self.skin_type_compatibility,
            "concentration_range": self.concentration_range
        }


@dataclass
class ProductIngredientAnalysis:
    """Análisis de ingredientes de producto"""
    product_name: str
    ingredients: List[str]
    analysis: Dict
    compatibility_score: float
    recommendations: List[str]
    warnings: List[str]
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "product_name": self.product_name,
            "ingredients": self.ingredients,
            "analysis": self.analysis,
            "compatibility_score": self.compatibility_score,
            "recommendations": self.recommendations,
            "warnings": self.warnings,
            "created_at": self.created_at
        }


class IngredientAnalyzer:
    """Sistema de análisis de ingredientes"""
    
    def __init__(self):
        """Inicializa el analizador"""
        self.ingredient_database: Dict[str, Ingredient] = {}
        self._initialize_database()
    
    def _initialize_database(self):
        """Inicializa base de datos de ingredientes"""
        # Ingredientes comunes
        self.ingredient_database = {
            "hyaluronic_acid": Ingredient(
                name="Ácido Hialurónico",
                category="humectant",
                benefits=["Hidratación", "Relleno de arrugas", "Reparación de barrera"],
                concerns=[],
                skin_type_compatibility=["oily", "dry", "sensitive", "combination"]
            ),
            "retinol": Ingredient(
                name="Retinol",
                category="active",
                benefits=["Anti-envejecimiento", "Renovación celular", "Reducción de arrugas"],
                concerns=["Sensibilidad al sol", "Irritación inicial"],
                skin_type_compatibility=["oily", "combination"],
                concentration_range="0.1-1.0%"
            ),
            "niacinamide": Ingredient(
                name="Niacinamida",
                category="active",
                benefits=["Control de sebo", "Reducción de poros", "Unificación de tono"],
                concerns=[],
                skin_type_compatibility=["oily", "dry", "sensitive", "combination"]
            ),
            "salicylic_acid": Ingredient(
                name="Ácido Salicílico",
                category="active",
                benefits=["Exfoliación", "Tratamiento de acné", "Limpieza de poros"],
                concerns=["Puede secar la piel", "Sensibilidad"],
                skin_type_compatibility=["oily", "combination"],
                concentration_range="0.5-2.0%"
            ),
            "vitamin_c": Ingredient(
                name="Vitamina C",
                category="active",
                benefits=["Antioxidante", "Iluminación", "Protección UV"],
                concerns=["Inestabilidad", "Puede irritar"],
                skin_type_compatibility=["oily", "dry", "combination"]
            )
        }
    
    def analyze_product_ingredients(self, product_name: str, ingredients: List[str],
                                   user_skin_type: Optional[str] = None) -> ProductIngredientAnalysis:
        """Analiza ingredientes de un producto"""
        # Identificar ingredientes conocidos
        identified_ingredients = []
        benefits = []
        concerns = []
        warnings = []
        
        for ingredient in ingredients:
            ingredient_lower = ingredient.lower().replace(" ", "_")
            
            # Buscar en base de datos
            for key, ing_data in self.ingredient_database.items():
                if key in ingredient_lower or ing_data.name.lower() in ingredient.lower():
                    identified_ingredients.append(ing_data.name)
                    benefits.extend(ing_data.benefits)
                    concerns.extend(ing_data.concerns)
                    
                    # Verificar compatibilidad con tipo de piel
                    if user_skin_type and user_skin_type not in ing_data.skin_type_compatibility:
                        warnings.append(
                            f"{ing_data.name} puede no ser compatible con piel {user_skin_type}"
                        )
                    break
        
        # Calcular score de compatibilidad
        compatibility_score = self._calculate_compatibility_score(
            identified_ingredients, user_skin_type
        )
        
        # Generar recomendaciones
        recommendations = self._generate_recommendations(
            identified_ingredients, benefits, concerns, user_skin_type
        )
        
        return ProductIngredientAnalysis(
            product_name=product_name,
            ingredients=ingredients,
            analysis={
                "identified_ingredients": identified_ingredients,
                "benefits": list(set(benefits)),
                "concerns": list(set(concerns))
            },
            compatibility_score=compatibility_score,
            recommendations=recommendations,
            warnings=warnings
        )
    
    def _calculate_compatibility_score(self, ingredients: List[str],
                                      skin_type: Optional[str]) -> float:
        """Calcula score de compatibilidad"""
        if not ingredients:
            return 0.5
        
        # Score base
        score = 0.7
        
        # Bonus por ingredientes beneficiosos conocidos
        beneficial_count = len(ingredients)
        score += min(0.2, beneficial_count * 0.05)
        
        # Penalización si hay incompatibilidades
        if skin_type:
            # Lógica simplificada - en producción sería más compleja
            score = max(0.3, score)
        
        return min(1.0, score)
    
    def _generate_recommendations(self, ingredients: List[str], benefits: List[str],
                                 concerns: List[str], skin_type: Optional[str]) -> List[str]:
        """Genera recomendaciones"""
        recommendations = []
        
        if benefits:
            recommendations.append(f"Beneficios principales: {', '.join(benefits[:3])}")
        
        if concerns:
            recommendations.append(f"Consideraciones: {', '.join(concerns[:2])}")
        
        if not ingredients:
            recommendations.append("No se identificaron ingredientes activos conocidos")
        
        return recommendations
    
    def get_ingredient_info(self, ingredient_name: str) -> Optional[Ingredient]:
        """Obtiene información de un ingrediente"""
        ingredient_lower = ingredient_name.lower().replace(" ", "_")
        
        for key, ingredient in self.ingredient_database.items():
            if key in ingredient_lower or ingredient.name.lower() in ingredient_name.lower():
                return ingredient
        
        return None
    
    def check_ingredient_compatibility(self, ingredient_name: str,
                                     skin_type: str) -> Dict:
        """Verifica compatibilidad de ingrediente con tipo de piel"""
        ingredient = self.get_ingredient_info(ingredient_name)
        
        if not ingredient:
            return {
                "compatible": None,
                "message": "Ingrediente no encontrado en base de datos"
            }
        
        compatible = skin_type in ingredient.skin_type_compatibility
        
        return {
            "compatible": compatible,
            "ingredient": ingredient.to_dict(),
            "message": "Compatible" if compatible else "Puede causar irritación"
        }






