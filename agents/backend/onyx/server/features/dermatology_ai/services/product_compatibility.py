"""
Sistema de análisis de compatibilidad de productos
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid


@dataclass
class ProductIngredient:
    """Ingrediente de producto"""
    name: str
    concentration: Optional[str] = None  # "low", "medium", "high"
    category: str = "unknown"  # "active", "preservative", "emollient", etc.
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "name": self.name,
            "concentration": self.concentration,
            "category": self.category
        }


@dataclass
class Product:
    """Producto"""
    product_id: str
    name: str
    category: str
    ingredients: List[ProductIngredient]
    ph_level: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "product_id": self.product_id,
            "name": self.name,
            "category": self.category,
            "ingredients": [i.to_dict() for i in self.ingredients],
            "ph_level": self.ph_level
        }


@dataclass
class CompatibilityIssue:
    """Problema de compatibilidad"""
    issue_type: str  # "conflict", "ph_mismatch", "overuse", "sensitivity"
    severity: str  # "low", "medium", "high"
    description: str
    affected_products: List[str]
    recommendation: str
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "issue_type": self.issue_type,
            "severity": self.severity,
            "description": self.description,
            "affected_products": self.affected_products,
            "recommendation": self.recommendation
        }


@dataclass
class CompatibilityReport:
    """Reporte de compatibilidad"""
    id: str
    user_id: str
    products: List[Product]
    is_compatible: bool
    issues: List[CompatibilityIssue]
    recommendations: List[str]
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "products": [p.to_dict() for p in self.products],
            "is_compatible": self.is_compatible,
            "issues": [i.to_dict() for i in self.issues],
            "recommendations": self.recommendations,
            "created_at": self.created_at
        }


class ProductCompatibility:
    """Sistema de análisis de compatibilidad de productos"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.reports: Dict[str, List[CompatibilityReport]] = {}
        self.product_db: Dict[str, Product] = {}
        self._load_conflict_rules()
    
    def _load_conflict_rules(self):
        """Carga reglas de conflicto conocidas"""
        self.conflict_rules = {
            # Ingredientes que no deben usarse juntos
            ("retinol", "vitamin_c"): {
                "issue": "pH incompatibilidad",
                "severity": "high",
                "recommendation": "Usa en momentos diferentes del día"
            },
            ("retinol", "aha"): {
                "issue": "Sobre-exfoliación",
                "severity": "high",
                "recommendation": "No uses juntos, alterna días"
            },
            ("benzoyl_peroxide", "retinol"): {
                "issue": "Inactivación mutua",
                "severity": "high",
                "recommendation": "No uses juntos"
            },
            ("niacinamide", "vitamin_c"): {
                "issue": "Posible irritación",
                "severity": "medium",
                "recommendation": "Usa con precaución, separa por tiempo"
            },
            ("aha", "bha"): {
                "issue": "Sobre-exfoliación",
                "severity": "medium",
                "recommendation": "Alterna días o usa uno por la mañana y otro por la noche"
            }
        }
    
    def register_product(self, product_id: str, name: str, category: str,
                        ingredients: List[Dict], ph_level: Optional[float] = None) -> Product:
        """Registra producto en la base de datos"""
        product_ingredients = []
        for ing_data in ingredients:
            ing = ProductIngredient(
                name=ing_data.get("name", "").lower(),
                concentration=ing_data.get("concentration"),
                category=ing_data.get("category", "unknown")
            )
            product_ingredients.append(ing)
        
        product = Product(
            product_id=product_id,
            name=name,
            category=category,
            ingredients=product_ingredients,
            ph_level=ph_level
        )
        
        self.product_db[product_id] = product
        return product
    
    def check_compatibility(self, user_id: str, product_ids: List[str]) -> CompatibilityReport:
        """Verifica compatibilidad de productos"""
        products = [self.product_db[pid] for pid in product_ids if pid in self.product_db]
        
        if len(products) != len(product_ids):
            raise ValueError("Algunos productos no están registrados")
        
        issues = []
        recommendations = []
        
        # Verificar conflictos de ingredientes
        for i, product1 in enumerate(products):
            for product2 in products[i+1:]:
                conflict = self._check_ingredient_conflicts(product1, product2)
                if conflict:
                    issues.append(conflict)
        
        # Verificar pH
        ph_issues = self._check_ph_compatibility(products)
        if ph_issues:
            issues.extend(ph_issues)
        
        # Verificar sobre-uso de activos
        overuse_issues = self._check_overuse(products)
        if overuse_issues:
            issues.extend(overuse_issues)
        
        # Generar recomendaciones
        if not issues:
            recommendations.append("Los productos son compatibles")
        else:
            high_severity = [i for i in issues if i.severity == "high"]
            if high_severity:
                recommendations.append("ADVERTENCIA: Conflictos de alta severidad detectados")
            recommendations.extend([i.recommendation for i in issues])
        
        is_compatible = len([i for i in issues if i.severity == "high"]) == 0
        
        report = CompatibilityReport(
            id=str(uuid.uuid4()),
            user_id=user_id,
            products=products,
            is_compatible=is_compatible,
            issues=issues,
            recommendations=recommendations
        )
        
        if user_id not in self.reports:
            self.reports[user_id] = []
        self.reports[user_id].append(report)
        
        return report
    
    def _check_ingredient_conflicts(self, product1: Product, product2: Product) -> Optional[CompatibilityIssue]:
        """Verifica conflictos de ingredientes"""
        ing1_names = {ing.name for ing in product1.ingredients}
        ing2_names = {ing.name for ing in product2.ingredients}
        
        for (ing1, ing2), rule in self.conflict_rules.items():
            if (ing1 in ing1_names and ing2 in ing2_names) or (ing1 in ing2_names and ing2 in ing1_names):
                return CompatibilityIssue(
                    issue_type="conflict",
                    severity=rule["severity"],
                    description=rule["issue"],
                    affected_products=[product1.name, product2.name],
                    recommendation=rule["recommendation"]
                )
        
        return None
    
    def _check_ph_compatibility(self, products: List[Product]) -> List[CompatibilityIssue]:
        """Verifica compatibilidad de pH"""
        issues = []
        ph_values = [p.ph_level for p in products if p.ph_level is not None]
        
        if len(ph_values) < 2:
            return issues
        
        # Productos con pH muy diferentes pueden no funcionar bien juntos
        min_ph = min(ph_values)
        max_ph = max(ph_values)
        
        if max_ph - min_ph > 2.0:
            issues.append(CompatibilityIssue(
                issue_type="ph_mismatch",
                severity="medium",
                description=f"Diferencia de pH significativa ({min_ph:.1f} vs {max_ph:.1f})",
                affected_products=[p.name for p in products if p.ph_level is not None],
                recommendation="Usa productos con pH similar o separa por tiempo"
            ))
        
        return issues
    
    def _check_overuse(self, products: List[Product]) -> List[CompatibilityIssue]:
        """Verifica sobre-uso de ingredientes activos"""
        issues = []
        active_counts = {}
        
        for product in products:
            for ing in product.ingredients:
                if ing.category == "active":
                    active_counts[ing.name] = active_counts.get(ing.name, 0) + 1
        
        # Si un activo aparece en muchos productos, puede ser sobre-uso
        for active, count in active_counts.items():
            if count > 2:
                issues.append(CompatibilityIssue(
                    issue_type="overuse",
                    severity="medium",
                    description=f"{active} aparece en {count} productos",
                    affected_products=[p.name for p in products],
                    recommendation=f"Considera reducir el número de productos con {active}"
                ))
        
        return issues


