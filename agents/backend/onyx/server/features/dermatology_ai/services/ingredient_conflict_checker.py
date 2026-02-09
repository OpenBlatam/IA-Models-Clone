"""
Sistema de verificación de conflictos de ingredientes
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class IngredientConflict:
    """Conflicto de ingredientes"""
    ingredient1: str
    ingredient2: str
    conflict_type: str  # "incompatible", "redundant", "overuse"
    severity: str  # "low", "medium", "high"
    description: str
    recommendation: str
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "ingredient1": self.ingredient1,
            "ingredient2": self.ingredient2,
            "conflict_type": self.conflict_type,
            "severity": self.severity,
            "description": self.description,
            "recommendation": self.recommendation
        }


@dataclass
class ProductCompatibilityCheck:
    """Verificación de compatibilidad de productos"""
    products: List[str]
    conflicts: List[IngredientConflict]
    is_compatible: bool
    recommendations: List[str]
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "products": self.products,
            "conflicts": [c.to_dict() for c in self.conflicts],
            "is_compatible": self.is_compatible,
            "recommendations": self.recommendations
        }


class IngredientConflictChecker:
    """Sistema de verificación de conflictos de ingredientes"""
    
    def __init__(self):
        """Inicializa el checker"""
        self.conflict_rules: Dict[tuple, Dict] = {
            ("retinol", "vitamin_c"): {
                "type": "incompatible",
                "severity": "medium",
                "description": "Retinol y Vitamina C pueden causar irritación cuando se usan juntos",
                "recommendation": "Usa uno en la mañana y otro en la noche"
            },
            ("retinol", "aha"): {
                "type": "incompatible",
                "severity": "high",
                "description": "Retinol y AHA juntos pueden causar irritación severa",
                "recommendation": "No uses ambos el mismo día. Alterna entre ellos"
            },
            ("vitamin_c", "niacinamide"): {
                "type": "incompatible",
                "severity": "low",
                "description": "Pueden neutralizarse cuando se usan juntos",
                "recommendation": "Usa en momentos diferentes del día"
            },
            ("salicylic_acid", "glycolic_acid"): {
                "type": "overuse",
                "severity": "medium",
                "description": "Múltiples exfoliantes pueden ser demasiado agresivos",
                "recommendation": "Usa solo uno a la vez o alterna"
            }
        }
    
    def check_product_compatibility(self, products: List[Dict]) -> ProductCompatibilityCheck:
        """Verifica compatibilidad de productos"""
        product_names = [p.get("name", "") for p in products]
        all_ingredients = []
        
        # Extraer ingredientes de todos los productos
        for product in products:
            ingredients = product.get("ingredients", [])
            all_ingredients.extend(ingredients)
        
        # Verificar conflictos
        conflicts = []
        for i, ing1 in enumerate(all_ingredients):
            for ing2 in all_ingredients[i+1:]:
                conflict = self._check_ingredient_conflict(ing1, ing2)
                if conflict:
                    conflicts.append(conflict)
        
        # Generar recomendaciones
        recommendations = self._generate_recommendations(conflicts, products)
        
        is_compatible = len([c for c in conflicts if c.severity in ["high", "medium"]]) == 0
        
        return ProductCompatibilityCheck(
            products=product_names,
            conflicts=conflicts,
            is_compatible=is_compatible,
            recommendations=recommendations
        )
    
    def _check_ingredient_conflict(self, ingredient1: str, ingredient2: str) -> Optional[IngredientConflict]:
        """Verifica conflicto entre dos ingredientes"""
        ing1_lower = ingredient1.lower().replace(" ", "_")
        ing2_lower = ingredient2.lower().replace(" ", "_")
        
        # Verificar reglas de conflicto
        for (rule_ing1, rule_ing2), conflict_info in self.conflict_rules.items():
            if (rule_ing1 in ing1_lower and rule_ing2 in ing2_lower) or \
               (rule_ing1 in ing2_lower and rule_ing2 in ing1_lower):
                return IngredientConflict(
                    ingredient1=ingredient1,
                    ingredient2=ingredient2,
                    conflict_type=conflict_info["type"],
                    severity=conflict_info["severity"],
                    description=conflict_info["description"],
                    recommendation=conflict_info["recommendation"]
                )
        
        return None
    
    def _generate_recommendations(self, conflicts: List[IngredientConflict],
                                 products: List[Dict]) -> List[str]:
        """Genera recomendaciones basadas en conflictos"""
        recommendations = []
        
        if not conflicts:
            recommendations.append("Los productos son compatibles. Puedes usarlos juntos.")
            return recommendations
        
        high_severity = [c for c in conflicts if c.severity == "high"]
        if high_severity:
            recommendations.append("⚠️ ADVERTENCIA: Conflictos de alta severidad detectados.")
            recommendations.append("No uses estos productos juntos sin consultar a un dermatólogo.")
        
        medium_severity = [c for c in conflicts if c.severity == "medium"]
        if medium_severity:
            recommendations.append("Considera usar estos productos en momentos diferentes del día.")
        
        # Recomendaciones específicas
        for conflict in conflicts[:3]:  # Top 3 conflictos
            recommendations.append(conflict.recommendation)
        
        return recommendations






