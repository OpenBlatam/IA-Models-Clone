"""
Base de datos de productos de skincare
"""

import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path


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
    TREATMENT = "treatment"
    OIL = "oil"


class SkinType(str, Enum):
    """Tipos de piel compatibles"""
    DRY = "dry"
    OILY = "oily"
    COMBINATION = "combination"
    SENSITIVE = "sensitive"
    NORMAL = "normal"
    ALL = "all"


@dataclass
class Product:
    """Producto de skincare"""
    id: str
    name: str
    brand: str
    category: ProductCategory
    description: str
    key_ingredients: List[str]
    skin_types: List[SkinType]
    price_range: str  # "budget", "mid-range", "premium"
    rating: float  # 0-5
    reviews_count: int
    benefits: List[str]
    concerns_targeted: List[str]  # acne, dryness, aging, etc.
    usage_frequency: str
    spf: Optional[int] = None
    cruelty_free: bool = False
    vegan: bool = False
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return asdict(self)


class ProductDatabase:
    """Base de datos de productos de skincare"""
    
    def __init__(self, db_file: str = "products_database.json"):
        """
        Inicializa la base de datos de productos
        
        Args:
            db_file: Archivo JSON con productos
        """
        self.db_file = Path(db_file)
        self.products: Dict[str, Product] = {}
        self._load_database()
    
    def _load_database(self):
        """Carga la base de datos desde archivo o crea una por defecto"""
        if self.db_file.exists():
            try:
                with open(self.db_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for product_data in data.get("products", []):
                        product = Product(**product_data)
                        self.products[product.id] = product
            except Exception as e:
                print(f"Error cargando base de datos: {e}")
                self._create_default_database()
        else:
            self._create_default_database()
    
    def _create_default_database(self):
        """Crea base de datos por defecto con productos comunes"""
        default_products = [
            {
                "id": "prod_001",
                "name": "Limpiador Suave con Ácido Hialurónico",
                "brand": "Generic",
                "category": "cleanser",
                "description": "Limpiador suave con ácido hialurónico para todo tipo de piel",
                "key_ingredients": ["Ácido Hialurónico", "Glicerina", "Extractos Botánicos"],
                "skin_types": ["all"],
                "price_range": "mid-range",
                "rating": 4.5,
                "reviews_count": 1200,
                "benefits": ["Hidratación", "Limpieza Suave", "Sin Resecar"],
                "concerns_targeted": ["dryness", "sensitivity"],
                "usage_frequency": "2 veces al día",
                "cruelty_free": True,
                "vegan": True
            },
            {
                "id": "prod_002",
                "name": "Serum de Vitamina C",
                "brand": "Generic",
                "category": "serum",
                "description": "Serum antioxidante con vitamina C para unificación del tono",
                "key_ingredients": ["Vitamina C", "Ácido Ferúlico", "Vitamina E"],
                "skin_types": ["normal", "combination", "oily"],
                "price_range": "mid-range",
                "rating": 4.7,
                "reviews_count": 2500,
                "benefits": ["Unificación del Tono", "Antioxidante", "Brillo"],
                "concerns_targeted": ["hyperpigmentation", "dullness"],
                "usage_frequency": "Cada mañana",
                "cruelty_free": True,
                "vegan": True
            },
            {
                "id": "prod_003",
                "name": "Crema Hidratante con Retinol",
                "brand": "Generic",
                "category": "moisturizer",
                "description": "Crema hidratante nocturna con retinol para anti-envejecimiento",
                "key_ingredients": ["Retinol", "Ácido Hialurónico", "Ceramidas"],
                "skin_types": ["normal", "dry", "combination"],
                "price_range": "premium",
                "rating": 4.6,
                "reviews_count": 1800,
                "benefits": ["Anti-envejecimiento", "Hidratación", "Textura"],
                "concerns_targeted": ["aging", "wrinkles", "texture"],
                "usage_frequency": "Cada noche",
                "cruelty_free": True,
                "vegan": False
            },
            {
                "id": "prod_004",
                "name": "Protector Solar SPF 50",
                "brand": "Generic",
                "category": "sunscreen",
                "description": "Protector solar de amplio espectro SPF 50",
                "key_ingredients": ["Óxido de Zinc", "Dióxido de Titanio"],
                "skin_types": ["all"],
                "price_range": "mid-range",
                "rating": 4.8,
                "reviews_count": 3200,
                "benefits": ["Protección UV", "No Comedogénico", "Resistente al Agua"],
                "concerns_targeted": ["sun_damage", "aging"],
                "usage_frequency": "Cada mañana, reaplicar cada 2 horas",
                "spf": 50,
                "cruelty_free": True,
                "vegan": True
            },
            {
                "id": "prod_005",
                "name": "Serum con Niacinamida",
                "brand": "Generic",
                "category": "serum",
                "description": "Serum con niacinamida para control de grasa y poros",
                "key_ingredients": ["Niacinamida", "Ácido Salicílico", "Zinc"],
                "skin_types": ["oily", "combination"],
                "price_range": "budget",
                "rating": 4.4,
                "reviews_count": 1500,
                "benefits": ["Control de Grasa", "Minimiza Poros", "Calmante"],
                "concerns_targeted": ["acne", "large_pores", "oiliness"],
                "usage_frequency": "Cada noche",
                "cruelty_free": True,
                "vegan": True
            }
        ]
        
        for product_data in default_products:
            product = Product(**product_data)
            self.products[product.id] = product
        
        self._save_database()
    
    def _save_database(self):
        """Guarda la base de datos en archivo"""
        try:
            data = {
                "products": [product.to_dict() for product in self.products.values()]
            }
            with open(self.db_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error guardando base de datos: {e}")
    
    def search_products(self, 
                       category: Optional[ProductCategory] = None,
                       skin_type: Optional[SkinType] = None,
                       concern: Optional[str] = None,
                       price_range: Optional[str] = None,
                       min_rating: float = 0.0,
                       limit: int = 10) -> List[Product]:
        """
        Busca productos según criterios
        
        Args:
            category: Categoría del producto
            skin_type: Tipo de piel
            concern: Preocupación a tratar
            price_range: Rango de precio
            min_rating: Rating mínimo
            limit: Límite de resultados
            
        Returns:
            Lista de productos que coinciden
        """
        results = []
        
        for product in self.products.values():
            # Filtrar por categoría
            if category and product.category != category:
                continue
            
            # Filtrar por tipo de piel
            if skin_type and skin_type not in product.skin_types and SkinType.ALL not in product.skin_types:
                continue
            
            # Filtrar por preocupación
            if concern and concern not in product.concerns_targeted:
                continue
            
            # Filtrar por precio
            if price_range and product.price_range != price_range:
                continue
            
            # Filtrar por rating
            if product.rating < min_rating:
                continue
            
            results.append(product)
        
        # Ordenar por rating
        results.sort(key=lambda x: x.rating, reverse=True)
        
        return results[:limit]
    
    def get_product(self, product_id: str) -> Optional[Product]:
        """Obtiene un producto por ID"""
        return self.products.get(product_id)
    
    def get_recommended_products(self, skin_type: str, 
                                 concerns: List[str],
                                 priorities: List[str],
                                 limit: int = 5) -> List[Product]:
        """
        Obtiene productos recomendados basados en análisis
        
        Args:
            skin_type: Tipo de piel
            concerns: Preocupaciones detectadas
            priorities: Áreas prioritarias
            limit: Límite de productos
            
        Returns:
            Lista de productos recomendados
        """
        recommended = []
        
        # Buscar productos para cada preocupación
        for concern in concerns[:3]:  # Top 3 preocupaciones
            products = self.search_products(
                skin_type=SkinType(skin_type) if skin_type in [e.value for e in SkinType] else None,
                concern=concern,
                min_rating=4.0,
                limit=2
            )
            recommended.extend(products)
        
        # Buscar productos para áreas prioritarias
        priority_map = {
            "hydration": "moisturizer",
            "texture": "exfoliant",
            "pigmentation": "serum",
            "anti_aging": "serum",
            "pore_care": "toner"
        }
        
        for priority in priorities[:2]:  # Top 2 prioridades
            if priority in priority_map:
                category = ProductCategory(priority_map[priority])
                products = self.search_products(
                    category=category,
                    skin_type=SkinType(skin_type) if skin_type in [e.value for e in SkinType] else None,
                    min_rating=4.0,
                    limit=1
                )
                recommended.extend(products)
        
        # Eliminar duplicados y ordenar
        seen_ids = set()
        unique_products = []
        for product in recommended:
            if product.id not in seen_ids:
                seen_ids.add(product.id)
                unique_products.append(product)
        
        unique_products.sort(key=lambda x: x.rating, reverse=True)
        
        return unique_products[:limit]
    
    def compare_products(self, product_ids: List[str]) -> Dict:
        """
        Compara múltiples productos
        
        Args:
            product_ids: Lista de IDs de productos
            
        Returns:
            Diccionario con comparación
        """
        products = [self.get_product(pid) for pid in product_ids if self.get_product(pid)]
        
        if not products:
            return {"error": "No se encontraron productos"}
        
        comparison = {
            "products": [p.to_dict() for p in products],
            "common_ingredients": self._find_common_ingredients(products),
            "price_comparison": {
                "budget": len([p for p in products if p.price_range == "budget"]),
                "mid_range": len([p for p in products if p.price_range == "mid-range"]),
                "premium": len([p for p in products if p.price_range == "premium"])
            },
            "average_rating": sum(p.rating for p in products) / len(products) if products else 0,
            "best_for": {
                product.id: product.concerns_targeted
                for product in products
            }
        }
        
        return comparison
    
    def _find_common_ingredients(self, products: List[Product]) -> List[str]:
        """Encuentra ingredientes comunes entre productos"""
        if not products:
            return []
        
        common = set(products[0].key_ingredients)
        for product in products[1:]:
            common &= set(product.key_ingredients)
        
        return list(common)
    
    def add_product(self, product: Product) -> bool:
        """
        Agrega un nuevo producto a la base de datos
        
        Args:
            product: Producto a agregar
            
        Returns:
            True si se agregó correctamente
        """
        try:
            self.products[product.id] = product
            self._save_database()
            return True
        except Exception as e:
            print(f"Error agregando producto: {e}")
            return False






