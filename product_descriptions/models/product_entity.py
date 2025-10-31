from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

from __future__ import annotations
from typing import Dict, List, Optional, Set, Any
from datetime import datetime
from decimal import Decimal
from enum import Enum
from uuid import uuid4
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Product Entity - Enterprise Domain Model
========================================

Entidad de dominio para productos con Clean Architecture y funcionalidades avanzadas.
"""



class ProductStatus(str, Enum):
    """Estados del producto"""
    DRAFT = "draft"
    ACTIVE = "active"
    INACTIVE = "inactive"
    DISCONTINUED = "discontinued"
    OUT_OF_STOCK = "out_of_stock"


class ProductType(str, Enum):
    """Tipos de producto"""
    PHYSICAL = "physical"
    DIGITAL = "digital"
    SERVICE = "service"
    SUBSCRIPTION = "subscription"
    BUNDLE = "bundle"


class PriceType(str, Enum):
    """Tipos de precio"""
    FIXED = "fixed"
    VARIABLE = "variable"
    TIERED = "tiered"
    SUBSCRIPTION = "subscription"


class InventoryTracking(str, Enum):
    """Seguimiento de inventario"""
    TRACK = "track"
    NO_TRACK = "no_track"
    BACKORDER = "backorder"


@dataclass
class Money:
    """Value object para representar dinero"""
    amount: Decimal
    currency: str = "USD"
    
    def __post_init__(self) -> Any:
        if self.amount < 0:
            raise ValueError("El monto no puede ser negativo")
        if len(self.currency) != 3:
            raise ValueError("Código de moneda debe tener 3 caracteres")
    
    def to_dict(self) -> Dict[str, Any]:
        return {"amount": float(self.amount), "currency": self.currency}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Money:
        return cls(amount=Decimal(str(data["amount"])), currency=data["currency"])


@dataclass
class Dimensions:
    """Value object para dimensiones del producto"""
    length: float
    width: float
    height: float
    weight: float
    unit: str = "cm"
    weight_unit: str = "kg"
    
    def volume(self) -> float:
        return self.length * self.width * self.height
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "length": self.length,
            "width": self.width,
            "height": self.height,
            "weight": self.weight,
            "unit": self.unit,
            "weight_unit": self.weight_unit,
            "volume": self.volume()
        }


@dataclass
class SEOData:
    """Value object para datos SEO"""
    title: Optional[str] = None
    description: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    meta_title: Optional[str] = None
    meta_description: Optional[str] = None
    slug: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "keywords": self.keywords,
            "meta_title": self.meta_title,
            "meta_description": self.meta_description,
            "slug": self.slug
        }


@dataclass
class ProductVariant:
    """Variante de producto"""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    sku: str = ""
    price: Optional[Money] = None
    inventory_quantity: int = 0
    attributes: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "sku": self.sku,
            "price": self.price.to_dict() if self.price else None,
            "inventory_quantity": self.inventory_quantity,
            "attributes": self.attributes,
            "is_active": self.is_active
        }


class ProductEntity:
    """
    Entidad de dominio para Producto - Core Business Logic
    
    Representa un producto en el sistema con todas sus propiedades,
    comportamientos y reglas de negocio empresariales.
    """
    
    def __init__(
        self,
        id: Optional[str] = None,
        name: str = "",
        description: str = "",
        short_description: str = "",
        sku: str = "",
        product_type: ProductType = ProductType.PHYSICAL,
        status: ProductStatus = ProductStatus.DRAFT,
        brand_id: Optional[str] = None,
        category_id: Optional[str] = None
    ):
        
    """__init__ function."""
self.id = id or str(uuid4())
        self.name = name
        self.description = description
        self.short_description = short_description
        self.sku = sku
        self.product_type = product_type
        self.status = status
        self.brand_id = brand_id
        self.category_id = category_id
        
        # Pricing
        self.base_price: Optional[Money] = None
        self.sale_price: Optional[Money] = None
        self.cost_price: Optional[Money] = None
        self.price_type: PriceType = PriceType.FIXED
        
        # Inventory
        self.inventory_quantity: int = 0
        self.low_stock_threshold: int = 10
        self.inventory_tracking: InventoryTracking = InventoryTracking.TRACK
        self.allow_backorder: bool = False
        
        # Physical properties
        self.dimensions: Optional[Dimensions] = None
        self.requires_shipping: bool = True
        
        # Digital properties
        self.download_url: Optional[str] = None
        self.download_limit: Optional[int] = None
        
        # SEO and Marketing
        self.seo_data: SEOData = SEOData()
        self.tags: Set[str] = set()
        self.featured: bool = False
        
        # Variants and Options
        self.variants: List[ProductVariant] = []
        self.has_variants: bool = False
        
        # Attributes and Custom Fields
        self.attributes: Dict[str, Any] = {}
        self.custom_fields: Dict[str, Any] = {}
        
        # Media
        self.images: List[str] = []
        self.videos: List[str] = []
        self.documents: List[str] = []
        
        # Timestamps
        self.created_at: datetime = datetime.utcnow()
        self.updated_at: datetime = datetime.utcnow()
        self.published_at: Optional[datetime] = None
        
        # AI-Generated Content
        self.ai_generated_description: Optional[str] = None
        self.ai_confidence_score: Optional[float] = None
        self.ai_last_updated: Optional[datetime] = None
    
    # ============================================================================
    # BUSINESS LOGIC METHODS
    # ============================================================================
    
    def set_price(self, price: Money, price_type: PriceType = PriceType.FIXED) -> None:
        """Establece el precio base del producto"""
        if price.amount <= 0:
            raise ValueError("El precio debe ser mayor a 0")
        
        self.base_price = price
        self.price_type = price_type
        self.updated_at = datetime.utcnow()
    
    def set_sale_price(self, sale_price: Money, end_date: Optional[datetime] = None) -> None:
        """Establece precio de oferta"""
        if self.base_price and sale_price.amount >= self.base_price.amount:
            raise ValueError("El precio de oferta debe ser menor al precio base")
        
        self.sale_price = sale_price
        self.updated_at = datetime.utcnow()
    
    def get_effective_price(self) -> Optional[Money]:
        """Obtiene el precio efectivo (oferta o base)"""
        return self.sale_price or self.base_price
    
    def is_on_sale(self) -> bool:
        """Verifica si el producto está en oferta"""
        return self.sale_price is not None
    
    def calculate_discount_percentage(self) -> float:
        """Calcula el porcentaje de descuento"""
        if not self.is_on_sale() or not self.base_price:
            return 0.0
        
        discount = self.base_price.amount - self.sale_price.amount
        return float((discount / self.base_price.amount) * 100)
    
    def update_inventory(self, quantity: int, operation: str = "set") -> None:
        """Actualiza el inventario"""
        if self.inventory_tracking == InventoryTracking.NO_TRACK:
            return
        
        if operation == "set":
            self.inventory_quantity = max(0, quantity)
        elif operation == "add":
            self.inventory_quantity += quantity
        elif operation == "subtract":
            if not self.allow_backorder and quantity > self.inventory_quantity:
                raise ValueError("Cantidad insuficiente en inventario")
            self.inventory_quantity -= quantity
        
        self.updated_at = datetime.utcnow()
        
        # Actualizar estado si está agotado
        if self.inventory_quantity <= 0 and not self.allow_backorder:
            self.status = ProductStatus.OUT_OF_STOCK
    
    def is_low_stock(self) -> bool:
        """Verifica si el stock está bajo"""
        if self.inventory_tracking == InventoryTracking.NO_TRACK:
            return False
        return self.inventory_quantity <= self.low_stock_threshold
    
    def is_in_stock(self) -> bool:
        """Verifica si está en stock"""
        if self.inventory_tracking == InventoryTracking.NO_TRACK:
            return True
        return self.inventory_quantity > 0 or self.allow_backorder
    
    def add_variant(self, variant: ProductVariant) -> None:
        """Añade una variante al producto"""
        if not self.has_variants:
            self.has_variants = True
        
        self.variants.append(variant)
        self.updated_at = datetime.utcnow()
    
    def remove_variant(self, variant_id: str) -> bool:
        """Elimina una variante"""
        for i, variant in enumerate(self.variants):
            if variant.id == variant_id:
                del self.variants[i]
                self.updated_at = datetime.utcnow()
                return True
        return False
    
    def get_variant(self, variant_id: str) -> Optional[ProductVariant]:
        """Obtiene una variante específica"""
        for variant in self.variants:
            if variant.id == variant_id:
                return variant
        return None
    
    def add_tag(self, tag: str) -> None:
        """Añade una etiqueta"""
        self.tags.add(tag.lower().strip())
        self.updated_at = datetime.utcnow()
    
    def remove_tag(self, tag: str) -> None:
        """Elimina una etiqueta"""
        self.tags.discard(tag.lower().strip())
        self.updated_at = datetime.utcnow()
    
    def publish(self) -> None:
        """Publica el producto"""
        if self.status == ProductStatus.DRAFT:
            self.status = ProductStatus.ACTIVE
            self.published_at = datetime.utcnow()
            self.updated_at = datetime.utcnow()
    
    def unpublish(self) -> None:
        """Despublica el producto"""
        if self.status == ProductStatus.ACTIVE:
            self.status = ProductStatus.INACTIVE
            self.updated_at = datetime.utcnow()
    
    def discontinue(self) -> None:
        """Descontinúa el producto"""
        self.status = ProductStatus.DISCONTINUED
        self.updated_at = datetime.utcnow()
    
    def set_ai_description(self, description: str, confidence: float = 0.0) -> None:
        """Establece descripción generada por IA"""
        self.ai_generated_description = description
        self.ai_confidence_score = confidence
        self.ai_last_updated = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def calculate_profit_margin(self) -> Optional[float]:
        """Calcula el margen de ganancia"""
        effective_price = self.get_effective_price()
        if not effective_price or not self.cost_price:
            return None
        
        profit = effective_price.amount - self.cost_price.amount
        return float((profit / effective_price.amount) * 100)
    
    def get_total_value(self) -> Optional[Money]:
        """Calcula el valor total del inventario"""
        effective_price = self.get_effective_price()
        if not effective_price:
            return None
        
        total = effective_price.amount * self.inventory_quantity
        return Money(amount=total, currency=effective_price.currency)
    
    # ============================================================================
    # VALIDATION METHODS
    # ============================================================================
    
    def validate(self) -> List[str]:
        """Valida la entidad y retorna lista de errores"""
        errors = []
        
        if not self.name or len(self.name.strip()) < 2:
            errors.append("El nombre del producto debe tener al menos 2 caracteres")
        
        if not self.sku or len(self.sku.strip()) < 1:
            errors.append("El SKU es requerido")
        
        if self.product_type == ProductType.PHYSICAL and not self.requires_shipping:
            if not self.dimensions:
                errors.append("Productos físicos requieren dimensiones")
        
        if self.product_type == ProductType.DIGITAL:
            if not self.download_url and not self.description:
                errors.append("Productos digitales requieren URL de descarga o descripción")
        
        if self.base_price and self.base_price.amount <= 0:
            errors.append("El precio base debe ser mayor a 0")
        
        if self.sale_price and self.base_price:
            if self.sale_price.amount >= self.base_price.amount:
                errors.append("El precio de oferta debe ser menor al precio base")
        
        if self.inventory_quantity < 0:
            errors.append("La cantidad en inventario no puede ser negativa")
        
        return errors
    
    def is_valid(self) -> bool:
        """Verifica si la entidad es válida"""
        return len(self.validate()) == 0
    
    # ============================================================================
    # SERIALIZATION
    # ============================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte la entidad a diccionario"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "short_description": self.short_description,
            "sku": self.sku,
            "product_type": self.product_type.value,
            "status": self.status.value,
            "brand_id": self.brand_id,
            "category_id": self.category_id,
            "base_price": self.base_price.to_dict() if self.base_price else None,
            "sale_price": self.sale_price.to_dict() if self.sale_price else None,
            "cost_price": self.cost_price.to_dict() if self.cost_price else None,
            "price_type": self.price_type.value,
            "inventory_quantity": self.inventory_quantity,
            "low_stock_threshold": self.low_stock_threshold,
            "inventory_tracking": self.inventory_tracking.value,
            "allow_backorder": self.allow_backorder,
            "dimensions": self.dimensions.to_dict() if self.dimensions else None,
            "requires_shipping": self.requires_shipping,
            "download_url": self.download_url,
            "download_limit": self.download_limit,
            "seo_data": self.seo_data.to_dict(),
            "tags": list(self.tags),
            "featured": self.featured,
            "variants": [v.to_dict() for v in self.variants],
            "has_variants": self.has_variants,
            "attributes": self.attributes,
            "custom_fields": self.custom_fields,
            "images": self.images,
            "videos": self.videos,
            "documents": self.documents,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "ai_generated_description": self.ai_generated_description,
            "ai_confidence_score": self.ai_confidence_score,
            "ai_last_updated": self.ai_last_updated.isoformat() if self.ai_last_updated else None,
            "effective_price": self.get_effective_price().to_dict() if self.get_effective_price() else None,
            "is_on_sale": self.is_on_sale(),
            "discount_percentage": self.calculate_discount_percentage(),
            "is_low_stock": self.is_low_stock(),
            "is_in_stock": self.is_in_stock(),
            "profit_margin": self.calculate_profit_margin(),
            "total_value": self.get_total_value().to_dict() if self.get_total_value() else None
        } 