from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import re
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from urllib.parse import urlparse
from pydantic import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Enhanced Quality Schemas Module
==============================

Enterprise-grade Pydantic models with comprehensive validation,
business logic integration, and advanced type safety.
"""


    BaseModel, Field, validator, root_validator, EmailStr,
    HttpUrl, constr, conint, confloat, conlist
)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class ProductStatus(str, Enum):
    """Product status with business meaning."""
    DRAFT = "draft"
    ACTIVE = "active"
    INACTIVE = "inactive"
    DISCONTINUED = "discontinued"
    OUT_OF_STOCK = "out_of_stock"


class ProductType(str, Enum):
    """Product type classification."""
    PHYSICAL = "physical"
    DIGITAL = "digital"
    SERVICE = "service"
    SUBSCRIPTION = "subscription"
    BUNDLE = "bundle"


class Currency(str, Enum):
    """Supported currencies."""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    CAD = "CAD"
    AUD = "AUD"
    JPY = "JPY"


class PriceType(str, Enum):
    """Price type classification."""
    FIXED = "fixed"
    VARIABLE = "variable"
    TIERED = "tiered"
    DYNAMIC = "dynamic"


class InventoryTrackingMethod(str, Enum):
    """Inventory tracking methods."""
    NONE = "none"
    SIMPLE = "simple"
    VARIANTS = "variants"
    SERIAL = "serial"
    BATCH = "batch"


# =============================================================================
# BASE SCHEMAS AND MIXINS
# =============================================================================

class EnhancedBaseModel(BaseModel):
    """Enhanced base model with common configuration and utilities."""
    
    class Config:
        # Performance optimizations
        validate_assignment = True
        str_strip_whitespace = True
        anystr_strip_whitespace = True
        
        # JSON handling
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: float(v),
            uuid.UUID: lambda v: str(v)
        }
        
        # Validation
        validate_all = True
        extra = "forbid"  # Prevent extra fields
        allow_population_by_field_name = True
        
        # Schema generation
        schema_extra = {
            "examples": {}
        }
    
    def dict_clean(self, **kwargs) -> Dict[str, Any]:
        """Get dictionary representation excluding None values."""
        return {k: v for k, v in self.dict(**kwargs).items() if v is not None}
    
    def json_clean(self, **kwargs) -> str:
        """Get JSON representation excluding None values."""
        return self.copy(exclude_none=True).json(**kwargs)


class TimestampMixin(BaseModel):
    """Mixin for timestamp fields with timezone support."""
    
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp in UTC"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp in UTC"
    )
    
    @validator('created_at', 'updated_at', pre=True)
    def ensure_utc_timezone(cls, v) -> Any:
        """Ensure timestamps are in UTC."""
        if isinstance(v, datetime) and v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v


class AuditMixin(BaseModel):
    """Mixin for audit fields."""
    
    created_by: Optional[str] = Field(None, description="User who created the record")
    updated_by: Optional[str] = Field(None, description="User who last updated the record")
    version: int = Field(default=1, ge=1, description="Record version for optimistic locking")


class PaginationMixin(BaseModel):
    """Enhanced pagination mixin with validation."""
    
    page: conint(ge=1, le=10000) = Field(
        default=1,
        description="Page number (1-based)"
    )
    per_page: conint(ge=1, le=500) = Field(
        default=20,
        description="Items per page"
    )
    
    @property
    def offset(self) -> int:
        """Calculate offset for database queries."""
        return (self.page - 1) * self.per_page
    
    @property
    def limit(self) -> int:
        """Get limit for database queries."""
        return self.per_page


# =============================================================================
# VALUE OBJECTS - Domain-driven design
# =============================================================================

class Money(EnhancedBaseModel):
    """Money value object with currency support."""
    
    amount: Decimal = Field(
        ..., ge=0, decimal_places=2,
        description="Monetary amount"
    )
    currency: Currency = Field(
        default=Currency.USD,
        description="Currency code"
    )
    
    def __str__(self) -> str:
        return f"{self.amount:.2f} {self.currency.value}"
    
    def __add__(self, other: 'Money') -> 'Money':
        if self.currency != other.currency:
            raise ValueError("Cannot add different currencies")
        return Money(amount=self.amount + other.amount, currency=self.currency)
    
    def __mul__(self, factor: Union[int, float, Decimal]) -> 'Money':
        return Money(
            amount=self.amount * Decimal(str(factor)),
            currency=self.currency
        )
    
    def round_to_currency(self) -> 'Money':
        """Round amount to currency-appropriate decimal places."""
        if self.currency == Currency.JPY:
            # Japanese Yen doesn't use decimal places
            rounded_amount = self.amount.quantize(Decimal('1'), rounding=ROUND_HALF_UP)
        else:
            rounded_amount = self.amount.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        
        return Money(amount=rounded_amount, currency=self.currency)


class Dimensions(EnhancedBaseModel):
    """Physical dimensions value object."""
    
    length: confloat(ge=0) = Field(..., description="Length in cm")
    width: confloat(ge=0) = Field(..., description="Width in cm")
    height: confloat(ge=0) = Field(..., description="Height in cm")
    weight: confloat(ge=0) = Field(..., description="Weight in kg")
    
    @property
    def volume(self) -> float:
        """Calculate volume in cubic centimeters."""
        return self.length * self.width * self.height
    
    @property
    def is_oversized(self) -> bool:
        """Check if dimensions exceed standard shipping limits."""
        max_dimension = max(self.length, self.width, self.height)
        return max_dimension > 100 or self.weight > 30  # 100cm or 30kg


class SKU(EnhancedBaseModel):
    """SKU value object with validation."""
    
    value: constr(min_length=1, max_length=50, regex=r'^[A-Z0-9\-_]+$') = Field(
        ..., description="Stock Keeping Unit"
    )
    
    def __str__(self) -> str:
        return self.value
    
    @validator('value', pre=True)
    def normalize_sku(cls, v) -> Any:
        """Normalize SKU to uppercase."""
        return str(v).strip().upper()
    
    @classmethod
    def generate(cls, prefix: str = "PROD", category: str = None) -> 'SKU':
        """Generate a new SKU."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        parts = [prefix]
        if category:
            parts.append(category.upper()[:4])
        parts.append(timestamp[-8:])  # Last 8 digits
        
        return cls(value="-".join(parts))


# =============================================================================
# PRODUCT DOMAIN MODELS
# =============================================================================

class ProductIdentity(EnhancedBaseModel):
    """Product identity and basic information."""
    
    name: constr(min_length=1, max_length=200) = Field(
        ..., description="Product name"
    )
    sku: SKU = Field(..., description="Stock Keeping Unit")
    slug: Optional[constr(regex=r'^[a-z0-9\-]+$')] = Field(
        None, description="URL-friendly slug"
    )
    description: constr(max_length=5000) = Field(
        default="", description="Product description"
    )
    short_description: Optional[constr(max_length=300)] = Field(
        None, description="Brief product description"
    )
    
    @validator('name')
    def validate_name(cls, v) -> bool:
        """Validate product name."""
        if not v.strip():
            raise ValueError("Product name cannot be empty")
        
        # Check for special characters that might cause issues
        if re.search(r'[<>"\']', v):
            raise ValueError("Product name contains invalid characters")
        
        return v.strip()
    
    @validator('slug', pre=True, always=True)
    def generate_slug(cls, v, values) -> Any:
        """Auto-generate slug from name if not provided."""
        if v:
            return v
        
        name = values.get('name', '')
        if name:
            # Convert to lowercase, replace spaces and special chars with hyphens
            slug = re.sub(r'[^\w\s-]', '', name.lower())
            slug = re.sub(r'[\s_-]+', '-', slug)
            return slug.strip('-')
        
        return None


class ProductPricing(EnhancedBaseModel):
    """Enhanced product pricing with business logic."""
    
    base_price: Money = Field(..., description="Base selling price")
    cost_price: Optional[Money] = Field(None, description="Cost/purchase price")
    sale_price: Optional[Money] = Field(None, description="Sale/discount price")
    price_type: PriceType = Field(default=PriceType.FIXED, description="Pricing model")
    
    # Tax settings
    tax_inclusive: bool = Field(default=False, description="Whether prices include tax")
    tax_rate: confloat(ge=0, le=1) = Field(default=0.0, description="Tax rate (0-1)")
    
    # Business rules
    minimum_price: Optional[Money] = Field(None, description="Minimum selling price")
    maximum_discount_percent: confloat(ge=0, le=100) = Field(
        default=50.0, description="Maximum discount percentage"
    )
    
    @root_validator
    def validate_pricing_logic(cls, values) -> bool:
        """Validate pricing business rules."""
        base_price = values.get('base_price')
        cost_price = values.get('cost_price')
        sale_price = values.get('sale_price')
        minimum_price = values.get('minimum_price')
        
        # Currency consistency
        prices = [p for p in [base_price, cost_price, sale_price, minimum_price] if p]
        if len(prices) > 1:
            currencies = {p.currency for p in prices}
            if len(currencies) > 1:
                raise ValueError("All prices must use the same currency")
        
        # Sale price validation
        if sale_price and base_price:
            if sale_price.amount >= base_price.amount:
                raise ValueError("Sale price must be less than base price")
            
            # Check maximum discount
            discount_percent = ((base_price.amount - sale_price.amount) / base_price.amount) * 100
            max_discount = values.get('maximum_discount_percent', 50)
            if discount_percent > max_discount:
                raise ValueError(f"Discount exceeds maximum allowed ({max_discount}%)")
        
        # Minimum price validation
        if minimum_price and sale_price:
            if sale_price.amount < minimum_price.amount:
                raise ValueError("Sale price cannot be below minimum price")
        
        # Cost price profitability check
        if cost_price and base_price:
            if cost_price.amount >= base_price.amount:
                # Warning, not error - might be a loss leader
                pass
        
        return values
    
    @property
    def effective_price(self) -> Money:
        """Get the current effective selling price."""
        return self.sale_price or self.base_price
    
    @property
    def is_on_sale(self) -> bool:
        """Check if product is currently on sale."""
        return self.sale_price is not None
    
    @property
    def discount_percentage(self) -> float:
        """Calculate discount percentage."""
        if not self.is_on_sale:
            return 0.0
        
        discount = self.base_price.amount - self.sale_price.amount
        return float((discount / self.base_price.amount) * 100)
    
    @property
    def profit_margin(self) -> Optional[float]:
        """Calculate profit margin percentage."""
        if not self.cost_price:
            return None
        
        effective = self.effective_price
        profit = effective.amount - self.cost_price.amount
        return float((profit / effective.amount) * 100)
    
    def calculate_tax_amount(self) -> Money:
        """Calculate tax amount based on effective price."""
        if self.tax_inclusive:
            # Tax is included in price
            tax_amount = self.effective_price.amount * (self.tax_rate / (1 + self.tax_rate))
        else:
            # Tax is additional
            tax_amount = self.effective_price.amount * self.tax_rate
        
        return Money(amount=tax_amount, currency=self.effective_price.currency)


class ProductInventory(EnhancedBaseModel):
    """Enhanced inventory management."""
    
    quantity: conint(ge=0) = Field(default=0, description="Current stock quantity")
    reserved_quantity: conint(ge=0) = Field(default=0, description="Reserved/allocated quantity")
    
    # Thresholds and alerts
    low_stock_threshold: conint(ge=0) = Field(default=10, description="Low stock alert threshold")
    reorder_point: conint(ge=0) = Field(default=20, description="Automatic reorder point")
    reorder_quantity: conint(ge=0) = Field(default=50, description="Suggested reorder quantity")
    
    # Tracking settings
    track_inventory: bool = Field(default=True, description="Enable inventory tracking")
    tracking_method: InventoryTrackingMethod = Field(
        default=InventoryTrackingMethod.SIMPLE,
        description="Inventory tracking method"
    )
    
    # Physical properties
    dimensions: Optional[Dimensions] = Field(None, description="Product dimensions")
    
    @root_validator
    def validate_inventory_logic(cls, values) -> bool:
        """Validate inventory business rules."""
        quantity = values.get('quantity', 0)
        reserved = values.get('reserved_quantity', 0)
        low_threshold = values.get('low_stock_threshold', 10)
        reorder_point = values.get('reorder_point', 20)
        
        # Reserved cannot exceed available
        if reserved > quantity:
            raise ValueError("Reserved quantity cannot exceed available quantity")
        
        # Reorder point should be higher than low stock threshold
        if reorder_point < low_threshold:
            raise ValueError("Reorder point should be higher than low stock threshold")
        
        return values
    
    @property
    def available_quantity(self) -> int:
        """Get available quantity (total - reserved)."""
        return self.quantity - self.reserved_quantity
    
    @property
    def is_in_stock(self) -> bool:
        """Check if product is in stock."""
        return self.available_quantity > 0
    
    @property
    def is_low_stock(self) -> bool:
        """Check if product is low in stock."""
        return self.track_inventory and self.available_quantity <= self.low_stock_threshold
    
    @property
    def needs_reorder(self) -> bool:
        """Check if product needs reordering."""
        return self.track_inventory and self.available_quantity <= self.reorder_point
    
    def reserve_quantity(self, amount: int) -> bool:
        """Reserve inventory quantity."""
        if amount <= 0:
            return False
        
        if self.available_quantity >= amount:
            self.reserved_quantity += amount
            return True
        return False
    
    def release_reservation(self, amount: int) -> bool:
        """Release reserved inventory."""
        if amount <= 0:
            return False
        
        if self.reserved_quantity >= amount:
            self.reserved_quantity -= amount
            return True
        return False


class ProductSEO(EnhancedBaseModel):
    """Enhanced SEO optimization."""
    
    # Meta tags
    meta_title: Optional[constr(max_length=60)] = Field(
        None, description="SEO meta title"
    )
    meta_description: Optional[constr(max_length=160)] = Field(
        None, description="SEO meta description"
    )
    
    # Keywords and tags
    keywords: conlist(str, max_items=10) = Field(
        default_factory=list, description="SEO keywords"
    )
    tags: conlist(str, max_items=20) = Field(
        default_factory=list, description="Product tags"
    )
    
    # URLs and slugs
    canonical_url: Optional[HttpUrl] = Field(None, description="Canonical URL")
    alt_urls: List[HttpUrl] = Field(default_factory=list, description="Alternative URLs")
    
    # Social media
    og_title: Optional[constr(max_length=60)] = Field(None, description="Open Graph title")
    og_description: Optional[constr(max_length=160)] = Field(None, description="Open Graph description")
    og_image: Optional[HttpUrl] = Field(None, description="Open Graph image URL")
    
    # Schema.org structured data
    schema_type: str = Field(default="Product", description="Schema.org type")
    
    @validator('keywords', 'tags')
    def normalize_keywords_and_tags(cls, v) -> Any:
        """Normalize keywords and tags."""
        return [item.strip().lower() for item in v if item.strip()]
    
    @validator('meta_title', 'og_title')
    def validate_title_length(cls, v) -> bool:
        """Validate title length for SEO."""
        if v and len(v) < 10:
            raise ValueError("Title too short for SEO (minimum 10 characters)")
        return v
    
    @validator('meta_description', 'og_description')
    def validate_description_length(cls, v) -> bool:
        """Validate description length for SEO."""
        if v and len(v) < 50:
            raise ValueError("Description too short for SEO (minimum 50 characters)")
        return v


# =============================================================================
# REQUEST SCHEMAS - Input validation
# =============================================================================

class EnhancedProductCreateRequest(ProductIdentity, EnhancedBaseModel):
    """Enhanced product creation request with comprehensive validation."""
    
    # Core product data
    product_type: ProductType = Field(default=ProductType.PHYSICAL)
    status: ProductStatus = Field(default=ProductStatus.DRAFT)
    
    # Pricing
    pricing: ProductPricing = Field(..., description="Product pricing information")
    
    # Inventory
    inventory: ProductInventory = Field(
        default_factory=ProductInventory,
        description="Inventory management"
    )
    
    # SEO and marketing
    seo: ProductSEO = Field(
        default_factory=ProductSEO,
        description="SEO optimization"
    )
    
    # Categorization
    category_ids: List[str] = Field(default_factory=list, max_items=5)
    brand_id: Optional[str] = Field(None)
    manufacturer_id: Optional[str] = Field(None)
    
    # Digital product specific
    is_digital: bool = Field(default=False)
    download_url: Optional[HttpUrl] = Field(None)
    download_limit: Optional[conint(ge=1)] = Field(None)
    
    # Custom attributes
    attributes: Dict[str, Any] = Field(default_factory=dict, max_items=50)
    
    # Media
    image_urls: List[HttpUrl] = Field(default_factory=list, max_items=20)
    video_urls: List[HttpUrl] = Field(default_factory=list, max_items=5)
    
    @root_validator
    def validate_product_coherence(cls, values) -> bool:
        """Validate product data coherence."""
        product_type = values.get('product_type')
        is_digital = values.get('is_digital', False)
        download_url = values.get('download_url')
        inventory = values.get('inventory')
        
        # Digital product validations
        if is_digital or product_type == ProductType.DIGITAL:
            if not download_url and not values.get('description'):
                raise ValueError("Digital products require download URL or detailed description")
            
            # Digital products don't need physical inventory tracking
            if inventory and inventory.track_inventory and inventory.tracking_method != InventoryTrackingMethod.NONE:
                raise ValueError("Digital products should not track physical inventory")
        
        # Physical product validations
        if product_type == ProductType.PHYSICAL:
            if is_digital:
                raise ValueError("Physical products cannot be marked as digital")
        
        # Service validations
        if product_type == ProductType.SERVICE:
            if inventory and inventory.dimensions:
                raise ValueError("Services cannot have physical dimensions")
        
        return values
    
    class Config(EnhancedBaseModel.Config):
        schema_extra = {
            "example": {
                "name": "Premium Wireless Headphones",
                "sku": {"value": "AUDIO-WH-001"},
                "description": "High-quality wireless headphones with noise cancellation",
                "product_type": "physical",
                "pricing": {
                    "base_price": {"amount": 299.99, "currency": "USD"},
                    "cost_price": {"amount": 150.00, "currency": "USD"}
                },
                "inventory": {
                    "quantity": 100,
                    "low_stock_threshold": 10,
                    "dimensions": {
                        "length": 20, "width": 15, "height": 8, "weight": 0.5
                    }
                },
                "category_ids": ["electronics", "audio"],
                "seo": {
                    "meta_title": "Premium Wireless Headphones - Noise Cancelling",
                    "keywords": ["wireless", "headphones", "audio", "bluetooth"]
                }
            }
        }


# =============================================================================
# RESPONSE SCHEMAS - Output models
# =============================================================================

class EnhancedProductResponse(ProductIdentity, ProductPricing, ProductInventory, ProductSEO, TimestampMixin, AuditMixin, EnhancedBaseModel):
    """Enhanced product response with calculated fields."""
    
    id: str = Field(..., description="Product unique identifier")
    status: ProductStatus = Field(..., description="Product status")
    product_type: ProductType = Field(..., description="Product type")
    
    # Calculated pricing fields
    effective_price_display: str = Field(..., description="Formatted effective price")
    discount_display: Optional[str] = Field(None, description="Formatted discount info")
    
    # Calculated inventory fields
    stock_status: str = Field(..., description="Human-readable stock status")
    availability_message: str = Field(..., description="Availability message")
    
    # Performance metadata
    cache_hit: bool = Field(default=False, description="Response from cache")
    response_time_ms: Optional[float] = Field(None, description="Response time")
    
    # Business intelligence
    popularity_score: float = Field(default=0.0, ge=0, le=100, description="Product popularity")
    conversion_rate: float = Field(default=0.0, ge=0, le=1, description="Conversion rate")
    
    @validator('effective_price_display', pre=True, always=True)
    def format_effective_price(cls, v, values) -> Any:
        """Format effective price for display."""
        pricing = values.get('pricing')
        if pricing:
            return str(pricing.effective_price)
        return "N/A"
    
    @validator('stock_status', pre=True, always=True)
    def calculate_stock_status(cls, v, values) -> Any:
        """Calculate human-readable stock status."""
        inventory = values.get('inventory')
        if not inventory:
            return "Unknown"
        
        if not inventory.track_inventory:
            return "Not Tracked"
        elif not inventory.is_in_stock:
            return "Out of Stock"
        elif inventory.is_low_stock:
            return "Low Stock"
        else:
            return "In Stock"
    
    class Config(EnhancedBaseModel.Config):
        schema_extra = {
            "example": {
                "id": "prod_123456",
                "name": "Premium Wireless Headphones",
                "sku": {"value": "AUDIO-WH-001"},
                "status": "active",
                "effective_price_display": "299.99 USD",
                "stock_status": "In Stock",
                "popularity_score": 85.5,
                "cache_hit": True
            }
        } 