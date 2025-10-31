from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

from __future__ import annotations
from typing import Dict, List, Optional, Union, Any, Set, Tuple
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
import asyncio
from pathlib import Path
from pydantic import BaseModel, Field, validator, root_validator
from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, Text, ForeignKey, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from typing import Any, List, Dict, Optional
import logging
"""
Enhanced Product Model - Enterprise Architecture
===============================================

Modelo de producto empresarial con Clean Architecture, funcionalidades avanzadas,
y optimizaciones de rendimiento para aplicaciones de e-commerce de gran escala.
"""



# ============================================================================
# DOMAIN ENTITIES - Core Business Objects
# ============================================================================

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
    comportamientos y reglas de negocio.
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
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProductEntity:
        """Crea una entidad desde diccionario"""
        product = cls(
            id=data.get("id"),
            name=data.get("name", ""),
            description=data.get("description", ""),
            short_description=data.get("short_description", ""),
            sku=data.get("sku", ""),
            product_type=ProductType(data.get("product_type", ProductType.PHYSICAL)),
            status=ProductStatus(data.get("status", ProductStatus.DRAFT)),
            brand_id=data.get("brand_id"),
            category_id=data.get("category_id")
        )
        
        # Precios
        if data.get("base_price"):
            product.base_price = Money.from_dict(data["base_price"])
        if data.get("sale_price"):
            product.sale_price = Money.from_dict(data["sale_price"])
        if data.get("cost_price"):
            product.cost_price = Money.from_dict(data["cost_price"])
        
        product.price_type = PriceType(data.get("price_type", PriceType.FIXED))
        
        # Inventario
        product.inventory_quantity = data.get("inventory_quantity", 0)
        product.low_stock_threshold = data.get("low_stock_threshold", 10)
        product.inventory_tracking = InventoryTracking(data.get("inventory_tracking", InventoryTracking.TRACK))
        product.allow_backorder = data.get("allow_backorder", False)
        
        # Dimensiones
        if data.get("dimensions"):
            dims = data["dimensions"]
            product.dimensions = Dimensions(
                length=dims["length"],
                width=dims["width"],
                height=dims["height"],
                weight=dims["weight"],
                unit=dims.get("unit", "cm"),
                weight_unit=dims.get("weight_unit", "kg")
            )
        
        # Propiedades adicionales
        product.requires_shipping = data.get("requires_shipping", True)
        product.download_url = data.get("download_url")
        product.download_limit = data.get("download_limit")
        
        # SEO
        if data.get("seo_data"):
            seo = data["seo_data"]
            product.seo_data = SEOData(
                title=seo.get("title"),
                description=seo.get("description"),
                keywords=seo.get("keywords", []),
                meta_title=seo.get("meta_title"),
                meta_description=seo.get("meta_description"),
                slug=seo.get("slug")
            )
        
        # Tags y otros
        product.tags = set(data.get("tags", []))
        product.featured = data.get("featured", False)
        product.attributes = data.get("attributes", {})
        product.custom_fields = data.get("custom_fields", {})
        product.images = data.get("images", [])
        product.videos = data.get("videos", [])
        product.documents = data.get("documents", [])
        
        # Variantes
        product.has_variants = data.get("has_variants", False)
        for variant_data in data.get("variants", []):
            variant = ProductVariant(
                id=variant_data["id"],
                name=variant_data["name"],
                sku=variant_data["sku"],
                inventory_quantity=variant_data["inventory_quantity"],
                attributes=variant_data["attributes"],
                is_active=variant_data["is_active"]
            )
            if variant_data.get("price"):
                variant.price = Money.from_dict(variant_data["price"])
            product.variants.append(variant)
        
        # Timestamps
        if data.get("created_at"):
            product.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("updated_at"):
            product.updated_at = datetime.fromisoformat(data["updated_at"])
        if data.get("published_at"):
            product.published_at = datetime.fromisoformat(data["published_at"])
        
        # IA
        product.ai_generated_description = data.get("ai_generated_description")
        product.ai_confidence_score = data.get("ai_confidence_score")
        if data.get("ai_last_updated"):
            product.ai_last_updated = datetime.fromisoformat(data["ai_last_updated"])
        
        return product


# ============================================================================
# PYDANTIC MODELS - API Layer
# ============================================================================

class ProductCreateRequest(BaseModel):
    """Request para crear producto"""
    name: str = Field(..., min_length=2, max_length=200)
    description: str = Field("", max_length=5000)
    short_description: str = Field("", max_length=500)
    sku: str = Field(..., min_length=1, max_length=100)
    product_type: ProductType = ProductType.PHYSICAL
    brand_id: Optional[str] = None
    category_id: Optional[str] = None
    
    # Pricing
    base_price_amount: Optional[float] = Field(None, gt=0)
    base_price_currency: str = "USD"
    cost_price_amount: Optional[float] = Field(None, gt=0)
    
    # Inventory
    inventory_quantity: int = Field(0, ge=0)
    low_stock_threshold: int = Field(10, ge=0)
    inventory_tracking: InventoryTracking = InventoryTracking.TRACK
    allow_backorder: bool = False
    
    # Physical properties
    length: Optional[float] = Field(None, gt=0)
    width: Optional[float] = Field(None, gt=0)
    height: Optional[float] = Field(None, gt=0)
    weight: Optional[float] = Field(None, gt=0)
    requires_shipping: bool = True
    
    # Digital properties
    download_url: Optional[str] = None
    download_limit: Optional[int] = Field(None, gt=0)
    
    # SEO
    seo_title: Optional[str] = None
    seo_description: Optional[str] = None
    seo_keywords: List[str] = Field(default_factory=list)
    slug: Optional[str] = None
    
    # Tags and attributes
    tags: List[str] = Field(default_factory=list)
    attributes: Dict[str, Any] = Field(default_factory=dict)
    custom_fields: Dict[str, Any] = Field(default_factory=dict)
    
    # Media
    images: List[str] = Field(default_factory=list)
    videos: List[str] = Field(default_factory=list)
    
    # AI Generation
    auto_generate_description: bool = False
    
    @validator('sku')
    def validate_sku(cls, v) -> bool:
        if not v or not v.strip():
            raise ValueError('SKU no puede estar vacío')
        return v.strip().upper()
    
    @validator('tags')
    def validate_tags(cls, v) -> bool:
        return [tag.strip().lower() for tag in v if tag.strip()]
    
    @root_validator
    def validate_physical_product(cls, values) -> bool:
        product_type = values.get('product_type')
        if product_type == ProductType.PHYSICAL:
            requires_shipping = values.get('requires_shipping', True)
            if requires_shipping:
                dimensions = [values.get('length'), values.get('width'), 
                            values.get('height'), values.get('weight')]
                if not all(d is not None and d > 0 for d in dimensions):
                    raise ValueError('Productos físicos requieren dimensiones válidas')
        return values


class ProductUpdateRequest(BaseModel):
    """Request para actualizar producto"""
    name: Optional[str] = Field(None, min_length=2, max_length=200)
    description: Optional[str] = Field(None, max_length=5000)
    short_description: Optional[str] = Field(None, max_length=500)
    status: Optional[ProductStatus] = None
    
    # Pricing updates
    base_price_amount: Optional[float] = Field(None, gt=0)
    sale_price_amount: Optional[float] = Field(None, gt=0)
    
    # Inventory updates
    inventory_quantity: Optional[int] = Field(None, ge=0)
    low_stock_threshold: Optional[int] = Field(None, ge=0)
    
    # Other updates
    featured: Optional[bool] = None
    tags: Optional[List[str]] = None
    attributes: Optional[Dict[str, Any]] = None
    
    @validator('tags')
    def validate_tags(cls, v) -> bool:
        if v is not None:
            return [tag.strip().lower() for tag in v if tag.strip()]
        return v


class ProductResponse(BaseModel):
    """Response del producto"""
    id: str
    name: str
    description: str
    short_description: str
    sku: str
    product_type: ProductType
    status: ProductStatus
    brand_id: Optional[str]
    category_id: Optional[str]
    
    # Pricing
    base_price: Optional[Dict[str, Any]]
    sale_price: Optional[Dict[str, Any]]
    effective_price: Optional[Dict[str, Any]]
    is_on_sale: bool
    discount_percentage: float
    
    # Inventory
    inventory_quantity: int
    is_low_stock: bool
    is_in_stock: bool
    
    # Properties
    dimensions: Optional[Dict[str, Any]]
    requires_shipping: bool
    
    # SEO and metadata
    seo_data: Dict[str, Any]
    tags: List[str]
    featured: bool
    
    # Variants
    has_variants: bool
    variants: List[Dict[str, Any]]
    
    # Media
    images: List[str]
    videos: List[str]
    
    # AI
    ai_generated_description: Optional[str]
    ai_confidence_score: Optional[float]
    
    # Timestamps
    created_at: str
    updated_at: str
    published_at: Optional[str]
    
    # Calculated fields
    profit_margin: Optional[float]
    total_value: Optional[Dict[str, Any]]


class ProductListResponse(BaseModel):
    """Response para lista de productos"""
    products: List[ProductResponse]
    total: int
    page: int
    per_page: int
    has_next: bool
    has_prev: bool


class ProductSearchRequest(BaseModel):
    """Request para búsqueda de productos"""
    query: Optional[str] = None
    category_id: Optional[str] = None
    brand_id: Optional[str] = None
    status: Optional[ProductStatus] = None
    product_type: Optional[ProductType] = None
    min_price: Optional[float] = Field(None, ge=0)
    max_price: Optional[float] = Field(None, ge=0)
    tags: Optional[List[str]] = None
    featured: Optional[bool] = None
    in_stock: Optional[bool] = None
    low_stock: Optional[bool] = None
    
    # Pagination
    page: int = Field(1, ge=1)
    per_page: int = Field(20, ge=1, le=100)
    
    # Sorting
    sort_by: str = Field("updated_at", regex="^(name|created_at|updated_at|price|inventory_quantity)$")
    sort_order: str = Field("desc", regex="^(asc|desc)$")


# ============================================================================
# REPOSITORY INTERFACES - Data Access Layer
# ============================================================================

class IProductRepository(ABC):
    """Interface para repositorio de productos"""
    
    @abstractmethod
    async def create(self, product: ProductEntity) -> ProductEntity:
        pass
    
    @abstractmethod
    async def get_by_id(self, product_id: str) -> Optional[ProductEntity]:
        pass
    
    @abstractmethod
    async def get_by_sku(self, sku: str) -> Optional[ProductEntity]:
        pass
    
    @abstractmethod
    async def update(self, product: ProductEntity) -> ProductEntity:
        pass
    
    @abstractmethod
    async def delete(self, product_id: str) -> bool:
        pass
    
    @abstractmethod
    async def search(self, criteria: ProductSearchRequest) -> Tuple[List[ProductEntity], int]:
        pass
    
    @abstractmethod
    async def get_by_category(self, category_id: str) -> List[ProductEntity]:
        pass
    
    @abstractmethod
    async def get_by_brand(self, brand_id: str) -> List[ProductEntity]:
        pass
    
    @abstractmethod
    async def get_featured(self, limit: int = 10) -> List[ProductEntity]:
        pass
    
    @abstractmethod
    async def get_low_stock(self, threshold: Optional[int] = None) -> List[ProductEntity]:
        pass


# ============================================================================
# USE CASES - Application Layer
# ============================================================================

class CreateProductUseCase:
    """Caso de uso para crear productos"""
    
    def __init__(self, repository: IProductRepository):
        
    """__init__ function."""
self.repository = repository
    
    async def execute(self, request: ProductCreateRequest) -> ProductEntity:
        """Ejecuta la creación del producto"""
        
        # Verificar SKU único
        existing = await self.repository.get_by_sku(request.sku)
        if existing:
            raise ValueError(f"Ya existe un producto con SKU: {request.sku}")
        
        # Crear entidad
        product = ProductEntity(
            name=request.name,
            description=request.description,
            short_description=request.short_description,
            sku=request.sku,
            product_type=request.product_type,
            brand_id=request.brand_id,
            category_id=request.category_id
        )
        
        # Configurar precio
        if request.base_price_amount:
            product.set_price(Money(
                amount=Decimal(str(request.base_price_amount)),
                currency=request.base_price_currency
            ))
        
        if request.cost_price_amount:
            product.cost_price = Money(
                amount=Decimal(str(request.cost_price_amount)),
                currency=request.base_price_currency
            )
        
        # Configurar inventario
        product.inventory_quantity = request.inventory_quantity
        product.low_stock_threshold = request.low_stock_threshold
        product.inventory_tracking = request.inventory_tracking
        product.allow_backorder = request.allow_backorder
        
        # Configurar dimensiones
        if all(x is not None for x in [request.length, request.width, request.height, request.weight]):
            product.dimensions = Dimensions(
                length=request.length,
                width=request.width,
                height=request.height,
                weight=request.weight
            )
        
        product.requires_shipping = request.requires_shipping
        product.download_url = request.download_url
        product.download_limit = request.download_limit
        
        # Configurar SEO
        product.seo_data = SEOData(
            title=request.seo_title,
            description=request.seo_description,
            keywords=request.seo_keywords,
            slug=request.slug
        )
        
        # Configurar tags y atributos
        product.tags = set(request.tags)
        product.attributes = request.attributes
        product.custom_fields = request.custom_fields
        product.images = request.images
        product.videos = request.videos
        
        # Validar entidad
        errors = product.validate()
        if errors:
            raise ValueError(f"Errores de validación: {', '.join(errors)}")
        
        # Guardar
        saved_product = await self.repository.create(product)
        
        # Generar descripción con IA si se solicita
        if request.auto_generate_description:
            # TODO: Integrar con servicio de IA
            pass
        
        return saved_product


class UpdateProductUseCase:
    """Caso de uso para actualizar productos"""
    
    def __init__(self, repository: IProductRepository):
        
    """__init__ function."""
self.repository = repository
    
    async def execute(self, product_id: str, request: ProductUpdateRequest) -> ProductEntity:
        """Ejecuta la actualización del producto"""
        
        product = await self.repository.get_by_id(product_id)
        if not product:
            raise ValueError(f"Producto no encontrado: {product_id}")
        
        # Actualizar campos
        if request.name is not None:
            product.name = request.name
        if request.description is not None:
            product.description = request.description
        if request.short_description is not None:
            product.short_description = request.short_description
        if request.status is not None:
            product.status = request.status
        
        # Actualizar precios
        if request.base_price_amount is not None:
            product.set_price(Money(
                amount=Decimal(str(request.base_price_amount)),
                currency=product.base_price.currency if product.base_price else "USD"
            ))
        
        if request.sale_price_amount is not None:
            product.set_sale_price(Money(
                amount=Decimal(str(request.sale_price_amount)),
                currency=product.base_price.currency if product.base_price else "USD"
            ))
        
        # Actualizar inventario
        if request.inventory_quantity is not None:
            product.update_inventory(request.inventory_quantity, "set")
        if request.low_stock_threshold is not None:
            product.low_stock_threshold = request.low_stock_threshold
        
        # Actualizar otros campos
        if request.featured is not None:
            product.featured = request.featured
        if request.tags is not None:
            product.tags = set(request.tags)
        if request.attributes is not None:
            product.attributes.update(request.attributes)
        
        # Validar
        errors = product.validate()
        if errors:
            raise ValueError(f"Errores de validación: {', '.join(errors)}")
        
        return await self.repository.update(product)


class SearchProductsUseCase:
    """Caso de uso para búsqueda de productos"""
    
    def __init__(self, repository: IProductRepository):
        
    """__init__ function."""
self.repository = repository
    
    async def execute(self, request: ProductSearchRequest) -> Tuple[List[ProductEntity], int]:
        """Ejecuta la búsqueda de productos"""
        return await self.repository.search(request)


class GetProductAnalyticsUseCase:
    """Caso de uso para analytics de productos"""
    
    def __init__(self, repository: IProductRepository):
        
    """__init__ function."""
self.repository = repository
    
    async def execute(self) -> Dict[str, Any]:
        """Obtiene analytics de productos"""
        
        # TODO: Implementar analytics
        # - Total de productos
        # - Productos por categoría
        # - Valor total del inventario
        # - Productos con bajo stock
        # - Productos más vendidos
        # - Tendencias de precios
        
        return {
            "total_products": 0,
            "by_status": {},
            "by_category": {},
            "inventory_value": 0,
            "low_stock_count": 0,
            "out_of_stock_count": 0,
            "avg_price": 0,
            "timestamp": datetime.utcnow().isoformat()
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

"""
# Crear producto
product_request = ProductCreateRequest(
    name="iPhone 15 Pro",
    description="El último iPhone con tecnología avanzada",
    sku="IPH15PRO128",
    product_type=ProductType.PHYSICAL,
    base_price_amount=999.99,
    inventory_quantity=50,
    length=14.7,
    width=7.1,
    height=0.8,
    weight=0.187,
    tags=["smartphone", "apple", "premium"],
    seo_title="iPhone 15 Pro - Lo último en tecnología móvil",
    auto_generate_description=True
)

# Usar caso de uso
create_use_case = CreateProductUseCase(product_repository)
product = await create_use_case.execute(product_request)

# Actualizar inventario
product.update_inventory(5, "subtract")

# Establecer precio de oferta
product.set_sale_price(Money(amount=Decimal("899.99"), currency="USD"))

# Validar producto
if product.is_valid():
    print("Producto válido")
    print(f"Precio efectivo: {product.get_effective_price().amount}")
    print(f"Descuento: {product.calculate_discount_percentage():.1f}%")
    print(f"En stock: {product.is_in_stock()}")
    print(f"Stock bajo: {product.is_low_stock()}")
""" 