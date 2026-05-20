"""
Sistema de recomendaciones de skincare basado en análisis de piel
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class SkinType(str, Enum):
    """Tipos de piel"""
    DRY = "dry"
    OILY = "oily"
    COMBINATION = "combination"
    SENSITIVE = "sensitive"
    NORMAL = "normal"


class ProductCategory(str, Enum):
    """Categorías de productos"""
    CLEANSER = "cleanser"
    MOISTURIZER = "moisturizer"
    SERUM = "serum"
    SUNSCREEN = "sunscreen"
    TONER = "toner"
    EXFOLIANT = "exfoliant"
    MASK = "mask"
    EYE_CREAM = "eye_cream"


@dataclass
class SkincareProduct:
    """Producto de skincare recomendado"""
    name: str
    category: ProductCategory
    description: str
    key_ingredients: List[str]
    usage_frequency: str
    priority: int  # 1-5, donde 1 es más prioritario


@dataclass
class SkincareRoutine:
    """Rutina de skincare recomendada"""
    morning_routine: List[SkincareProduct]
    evening_routine: List[SkincareProduct]
    weekly_treatments: List[SkincareProduct]
    tips: List[str]


class SkincareRecommender:
    """Genera recomendaciones personalizadas de skincare"""
    
    def __init__(self):
        """Inicializa el recomendador"""
        self.product_database = self._initialize_product_database()
        self.condition_recommendations = self._initialize_condition_recommendations()
    
    def generate_recommendations(self, analysis_result: Dict) -> Dict:
        """
        Genera recomendaciones basadas en análisis de piel
        
        Args:
            analysis_result: Resultado del análisis de piel
            
        Returns:
            Diccionario con recomendaciones completas
        """
        quality_scores = analysis_result.get("quality_scores", {})
        conditions = analysis_result.get("conditions", [])
        skin_type = analysis_result.get("skin_type", "normal")
        priorities = analysis_result.get("recommendations_priority", [])
        
        # Generar rutina personalizada
        routine = self._create_routine(
            skin_type=skin_type,
            quality_scores=quality_scores,
            conditions=conditions,
            priorities=priorities
        )
        
        # Generar recomendaciones específicas
        specific_recommendations = self._get_specific_recommendations(
            quality_scores=quality_scores,
            conditions=conditions,
            priorities=priorities
        )
        
        # Generar tips generales
        tips = self._generate_tips(skin_type, conditions, quality_scores)
        
        return {
            "routine": {
                "morning": [self._product_to_dict(p) for p in routine.morning_routine],
                "evening": [self._product_to_dict(p) for p in routine.evening_routine],
                "weekly": [self._product_to_dict(p) for p in routine.weekly_treatments]
            },
            "specific_recommendations": specific_recommendations,
            "tips": tips,
            "skin_type": skin_type,
            "priority_areas": priorities
        }
    
    def _create_routine(self, skin_type: str, quality_scores: Dict,
                      conditions: List[Dict], priorities: List[str]) -> SkincareRoutine:
        """Crea una rutina personalizada"""
        morning = []
        evening = []
        weekly = []
        
        # Productos básicos según tipo de piel
        morning.extend(self._get_basic_products(skin_type, "morning"))
        evening.extend(self._get_basic_products(skin_type, "evening"))
        
        # Productos para condiciones específicas
        for condition in conditions:
            if condition["severity"] in ["moderate", "severe"]:
                products = self._get_products_for_condition(condition["name"])
                evening.extend(products)
        
        # Productos para áreas prioritarias
        for priority in priorities[:3]:  # Top 3
            products = self._get_products_for_priority(priority)
            if priority in ["hydration", "texture"]:
                morning.extend(products)
            else:
                evening.extend(products)
        
        # Tratamientos semanales
        if "texture" in priorities or any(c["name"] == "acne" for c in conditions):
            weekly.append(self._create_product(
                "Exfoliante Químico AHA/BHA",
                ProductCategory.EXFOLIANT,
                "Exfoliante suave para mejorar textura y poros",
                ["AHA", "BHA", "Ácido Salicílico"],
                "2-3 veces por semana",
                2
            ))
        
        if quality_scores.get("hydration_score", 50) < 50:
            weekly.append(self._create_product(
                "Máscara Hidratante",
                ProductCategory.MASK,
                "Máscara intensiva para hidratación profunda",
                ["Ácido Hialurónico", "Glicerina", "Ceramidas"],
                "1-2 veces por semana",
                3
            ))
        
        tips = self._generate_tips(skin_type, conditions, quality_scores)
        
        return SkincareRoutine(
            morning_routine=morning[:5],  # Máximo 5 productos en la mañana
            evening_routine=evening[:6],  # Máximo 6 productos en la noche
            weekly_treatments=weekly[:3],  # Máximo 3 tratamientos semanales
            tips=tips
        )
    
    def _get_basic_products(self, skin_type: str, time: str) -> List[SkincareProduct]:
        """Obtiene productos básicos según tipo de piel"""
        products = []
        
        # Limpiador
        if skin_type == SkinType.DRY:
            products.append(self._create_product(
                "Limpiador Suave Sin Jabón",
                ProductCategory.CLEANSER,
                "Limpiador cremoso para piel seca",
                ["Glicerina", "Aceites Nutritivos"],
                "2 veces al día",
                1
            ))
        elif skin_type == SkinType.OILY:
            products.append(self._create_product(
                "Limpiador Gel Espumoso",
                ProductCategory.CLEANSER,
                "Limpiador para control de grasa",
                ["Ácido Salicílico", "Niacinamida"],
                "2 veces al día",
                1
            ))
        else:
            products.append(self._create_product(
                "Limpiador Suave",
                ProductCategory.CLEANSER,
                "Limpiador equilibrado",
                ["Glicerina", "Extractos Botánicos"],
                "2 veces al día",
                1
            ))
        
        # Hidratante
        if time == "morning":
            if skin_type == SkinType.DRY:
                products.append(self._create_product(
                    "Crema Hidratante Rica",
                    ProductCategory.MOISTURIZER,
                    "Hidratante intensivo para piel seca",
                    ["Ácido Hialurónico", "Ceramidas", "Aceites"],
                    "Cada mañana",
                    1
                ))
            elif skin_type == SkinType.OILY:
                products.append(self._create_product(
                    "Hidratante Gel Ligero",
                    ProductCategory.MOISTURIZER,
                    "Hidratante sin aceites",
                    ["Ácido Hialurónico", "Niacinamida"],
                    "Cada mañana",
                    1
                ))
            else:
                products.append(self._create_product(
                    "Hidratante Equilibrado",
                    ProductCategory.MOISTURIZER,
                    "Hidratante para todo tipo de piel",
                    ["Ácido Hialurónico", "Glicerina"],
                    "Cada mañana",
                    1
                ))
            
            # Protector solar (siempre en la mañana)
            products.append(self._create_product(
                "Protector Solar SPF 30+",
                ProductCategory.SUNSCREEN,
                "Protección UV diaria esencial",
                ["Óxido de Zinc", "Dióxido de Titanio"],
                "Cada mañana, reaplicar cada 2 horas",
                1
            ))
        
        else:  # evening
            if skin_type == SkinType.DRY:
                products.append(self._create_product(
                    "Crema Nutritiva Nocturna",
                    ProductCategory.MOISTURIZER,
                    "Hidratante intensivo nocturno",
                    ["Ceramidas", "Aceites", "Manteca de Karité"],
                    "Cada noche",
                    1
                ))
            else:
                products.append(self._create_product(
                    "Hidratante Nocturno",
                    ProductCategory.MOISTURIZER,
                    "Hidratante para uso nocturno",
                    ["Ácido Hialurónico", "Peptidos"],
                    "Cada noche",
                    1
                ))
        
        return products
    
    def _get_products_for_condition(self, condition: str) -> List[SkincareProduct]:
        """Obtiene productos para condiciones específicas"""
        products = []
        
        if condition == "acne":
            products.append(self._create_product(
                "Serum con Ácido Salicílico",
                ProductCategory.SERUM,
                "Tratamiento para acné",
                ["Ácido Salicílico", "Niacinamida", "Zinc"],
                "Cada noche",
                1
            ))
        
        elif condition == "rosacea":
            products.append(self._create_product(
                "Serum Calmante",
                ProductCategory.SERUM,
                "Tratamiento para rosácea",
                ["Niacinamida", "Centella Asiática", "Aloe Vera"],
                "Cada noche",
                1
            ))
        
        elif condition == "hyperpigmentation":
            products.append(self._create_product(
                "Serum Despigmentante",
                ProductCategory.SERUM,
                "Tratamiento para manchas",
                ["Vitamina C", "Ácido Kójico", "Niacinamida"],
                "Cada noche",
                1
            ))
        
        elif condition == "dryness":
            products.append(self._create_product(
                "Serum Hidratante Intensivo",
                ProductCategory.SERUM,
                "Hidratación profunda",
                ["Ácido Hialurónico", "Glicerina", "Ceramidas"],
                "Cada noche",
                1
            ))
        
        return products
    
    def _get_products_for_priority(self, priority: str) -> List[SkincareProduct]:
        """Obtiene productos para áreas prioritarias"""
        products = []
        
        if priority == "hydration":
            products.append(self._create_product(
                "Serum de Ácido Hialurónico",
                ProductCategory.SERUM,
                "Hidratación intensiva",
                ["Ácido Hialurónico", "Glicerina"],
                "Cada mañana y noche",
                1
            ))
        
        elif priority == "anti_aging":
            products.append(self._create_product(
                "Serum Anti-Edad",
                ProductCategory.SERUM,
                "Tratamiento anti-envejecimiento",
                ["Retinol", "Peptidos", "Vitamina C"],
                "Cada noche",
                1
            ))
            products.append(self._create_product(
                "Crema para Contorno de Ojos",
                ProductCategory.EYE_CREAM,
                "Tratamiento para arrugas alrededor de los ojos",
                ["Cafeína", "Peptidos", "Ácido Hialurónico"],
                "Cada noche",
                2
            ))
        
        elif priority == "pigmentation":
            products.append(self._create_product(
                "Serum Vitamina C",
                ProductCategory.SERUM,
                "Unificación del tono",
                ["Vitamina C", "Ácido Ferúlico"],
                "Cada mañana",
                1
            ))
        
        elif priority == "pore_care":
            products.append(self._create_product(
                "Tónico con Niacinamida",
                ProductCategory.TONER,
                "Cuidado de poros",
                ["Niacinamida", "Ácido Salicílico"],
                "Cada noche",
                2
            ))
        
        elif priority == "texture":
            products.append(self._create_product(
                "Serum Exfoliante Suave",
                ProductCategory.SERUM,
                "Mejora de textura",
                ["AHA", "Ácido Láctico"],
                "Cada noche",
                2
            ))
        
        return products
    
    def _get_specific_recommendations(self, quality_scores: Dict,
                                     conditions: List[Dict],
                                     priorities: List[str]) -> List[Dict]:
        """Genera recomendaciones específicas"""
        recommendations = []
        
        # Recomendaciones basadas en scores
        if quality_scores.get("hydration_score", 50) < 50:
            recommendations.append({
                "area": "Hidratación",
                "issue": "Piel deshidratada detectada",
                "recommendation": "Aumente la hidratación con productos con ácido hialurónico y beba más agua",
                "priority": "high"
            })
        
        if quality_scores.get("wrinkles_score", 50) < 50:
            recommendations.append({
                "area": "Anti-envejecimiento",
                "issue": "Presencia de arrugas",
                "recommendation": "Considere productos con retinol y protector solar diario",
                "priority": "medium"
            })
        
        if quality_scores.get("pigmentation_score", 50) < 50:
            recommendations.append({
                "area": "Pigmentación",
                "issue": "Pigmentación irregular",
                "recommendation": "Use productos con vitamina C y protector solar SPF 30+",
                "priority": "high"
            })
        
        # Recomendaciones basadas en condiciones
        for condition in conditions:
            if condition["severity"] == "severe":
                recommendations.append({
                    "area": condition["name"].title(),
                    "issue": f"{condition['description']} - Severidad alta",
                    "recommendation": f"Consulte con un dermatólogo. Use productos específicos para {condition['name']}",
                    "priority": "high"
                })
        
        return recommendations
    
    def _generate_tips(self, skin_type: str, conditions: List[Dict],
                     quality_scores: Dict) -> List[str]:
        """Genera tips generales"""
        tips = []
        
        # Tips según tipo de piel
        if skin_type == SkinType.DRY:
            tips.append("Use limpiadores suaves sin jabón para evitar resecar más la piel")
            tips.append("Aplique hidratante inmediatamente después de lavarse la cara")
            tips.append("Evite agua muy caliente al lavarse")
        
        elif skin_type == SkinType.OILY:
            tips.append("No se exceda con la limpieza (máximo 2 veces al día)")
            tips.append("Use productos oil-free pero no se salte la hidratación")
            tips.append("El protector solar es esencial incluso para piel grasa")
        
        elif skin_type == SkinType.SENSITIVE:
            tips.append("Pruebe productos nuevos en un área pequeña primero")
            tips.append("Evite fragancias y alcohol en los productos")
            tips.append("Use productos con ingredientes calmantes como niacinamida")
        
        # Tips generales
        tips.append("Use protector solar SPF 30+ todos los días, incluso en interiores")
        tips.append("Sea consistente con su rutina - los resultados toman tiempo")
        tips.append("Beba al menos 8 vasos de agua al día")
        tips.append("Duerma al menos 7-8 horas para permitir la regeneración de la piel")
        
        # Tips según condiciones
        if any(c["name"] == "acne" for c in conditions):
            tips.append("No toque ni apriete los granos - puede empeorar la inflamación")
            tips.append("Lave fundas de almohada regularmente")
        
        if quality_scores.get("hydration_score", 50) < 50:
            tips.append("Use un humidificador en su habitación por la noche")
        
        return tips[:8]  # Máximo 8 tips
    
    def _create_product(self, name: str, category: ProductCategory,
                       description: str, ingredients: List[str],
                       usage: str, priority: int) -> SkincareProduct:
        """Crea un producto de skincare"""
        return SkincareProduct(
            name=name,
            category=category,
            description=description,
            key_ingredients=ingredients,
            usage_frequency=usage,
            priority=priority
        )
    
    def _product_to_dict(self, product: SkincareProduct) -> Dict:
        """Convierte producto a diccionario"""
        return {
            "name": product.name,
            "category": product.category.value,
            "description": product.description,
            "key_ingredients": product.key_ingredients,
            "usage_frequency": product.usage_frequency,
            "priority": product.priority
        }
    
    def _initialize_product_database(self) -> Dict:
        """Inicializa base de datos de productos (placeholder)"""
        return {}
    
    def _initialize_condition_recommendations(self) -> Dict:
        """Inicializa recomendaciones por condición (placeholder)"""
        return {}






