from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from typing import Optional, List, Dict, Any, Union
from decimal import Decimal
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator
from pydantic.config import ConfigDict
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Pydantic Schemas for Enhanced Product API
=========================================

Type-safe schemas following FastAPI best practices:
- Comprehensive validation
- Clear separation of concerns
- Performance optimized
- Developer-friendly error messages
"""



class ProductStatus(str, Enum):
    """Product status enumeration."""
    DRAFT = "draft"
    ACTIVE = "active"
    INACTIVE = "inactive"
    DISCONTINUED = "discontinued"
    OUT_OF_STOCK = "out_of_stock"


class ProductType(str, Enum):
    """Product type enumeration."""
    PHYSICAL = "physical"
    DIGITAL = "digital"
    SERVICE = "service"
    SUBSCRIPTION = "subscription"
    BUNDLE = "bundle"


class PriceType(str, Enum):
    """Price type enumeration."""
    FIXED = "fixed"
    VARIABLE = "variable"
    TIERED = "tiered"
    SUBSCRIPTION = "subscription"


class InventoryTracking(str, Enum):
    """Inventory tracking modes."""
    TRACK = "track"
    NO_TRACK = "no_track"
    BACKORDER = "backorder"


# Value Objects
class MoneySchema(BaseModel):
    """Money representation with currency support."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    amount: Decimal = Field(..., ge=0, description="Amount in specified currency")
    currency: str = Field(..., min_length=3, max_length=3, description="ISO currency code")
    
    @validator('currency')
    def validate_currency(cls, v: str) -> str:
        return v.upper()
    
    @validator('amount')
    def validate_amount(cls, v: Decimal) -> Decimal:
        if v < 0:
            raise ValueError("Amount cannot be negative")
        return round(v, 2)


class DimensionsSchema(BaseModel):
    """Physical dimensions schema."""
    length: float = Field(..., gt=0, description="Length in cm")
    width: float = Field(..., gt=0, description="Width in cm") 
    height: float = Field(..., gt=0, description="Height in cm")
    weight: float = Field(..., gt=0, description="Weight in kg")
    unit: str = Field(default="cm", description="Dimension unit")
    weight_unit: str = Field(default="kg", description="Weight unit")
    
    @property
    def volume(self) -> float:
        """Calculate volume in cubic units."""
        return self.length * self.width * self.height


class SEODataSchema(BaseModel):
    """SEO optimization data."""
    title: Optional[str] = Field(None, max_length=100, description="SEO title")
    description: Optional[str] = Field(None, max_length=300, description="SEO description")
    keywords: List[str] = Field(default_factory=list, description="SEO keywords")
    meta_title: Optional[str] = Field(None, max_length=100, description="Meta title")
    meta_description: Optional[str] = Field(None, max_length=300, description="Meta description")
    slug: Optional[str] = Field(None, max_length=100, description="URL slug")
    
    @validator('keywords')
    def validate_keywords(cls, v: List[str]) -> List[str]:
        return [kw.strip().lower() for kw in v if kw.strip()]


class ProductVariantSchema(BaseModel):
    """Product variant schema."""
    id: str = Field(..., description="Variant ID")
    name: str = Field(..., min_length=1, max_length=200, description="Variant name")
    sku: str = Field(..., min_length=1, max_length=100, description="Variant SKU")
    price: Optional[MoneySchema] = Field(None, description="Variant price")
    inventory_quantity: int = Field(default=0, ge=0, description="Variant inventory")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Variant attributes")
    is_active: bool = Field(default=True, description="Variant active status")


# Request Schemas
class ProductCreateRequest(BaseModel):
    """Schema for creating a new product."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    # Basic Information
    name: str = Field(..., min_length=2, max_length=200, description="Product name")
    description: str = Field(default="", max_length=5000, description="Product description")
    short_description: str = Field(default="", max_length=500, description="Short description")
    sku: str = Field(..., min_length=1, max_length=100, description="Unique SKU")
    product_type: ProductType = Field(default=ProductType.PHYSICAL, description="Product type")
    brand_id: Optional[str] = Field(None, description="Brand identifier")
    category_id: Optional[str] = Field(None, description="Category identifier")
    
    # Pricing
    base_price: Optional[MoneySchema] = Field(None, description="Base price")
    sale_price: Optional[MoneySchema] = Field(None, description="Sale price")
    cost_price: Optional[MoneySchema] = Field(None, description="Cost price")
    price_type: PriceType = Field(default=PriceType.FIXED, description="Price type")
    
    # Inventory
    inventory_quantity: int = Field(default=0, ge=0, description="Initial inventory")
    low_stock_threshold: int = Field(default=10, ge=0, description="Low stock threshold")
    inventory_tracking: InventoryTracking = Field(default=InventoryTracking.TRACK, description="Inventory tracking mode")
    allow_backorder: bool = Field(default=False, description="Allow backorders")
    
    # Physical Properties
    dimensions: Optional[DimensionsSchema] = Field(None, description="Product dimensions")
    requires_shipping: bool = Field(default=True, description="Requires shipping")
    
    # Digital Properties
    download_url: Optional[str] = Field(None, description="Download URL for digital products")
    download_limit: Optional[int] = Field(None, gt=0, description="Download limit")
    
    # SEO and Marketing
    seo_data: Optional[SEODataSchema] = Field(None, description="SEO data")
    tags: List[str] = Field(default_factory=list, description="Product tags")
    featured: bool = Field(default=False, description="Featured product flag")
    
    # Customization
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Custom attributes")
    custom_fields: Dict[str, Any] = Field(default_factory=dict, description="Custom fields")
    
    # Media
    images: List[str] = Field(default_factory=list, description="Image URLs")
    videos: List[str] = Field(default_factory=list, description="Video URLs")
    documents: List[str] = Field(default_factory=list, description="Document URLs")
    
    # AI Integration
    auto_generate_description: bool = Field(default=False, description="Auto-generate AI description")
    
    @validator('sku')
    def validate_sku(cls, v: str) -> str:
        """Validate and normalize SKU."""
        if not v or not v.strip():
            raise ValueError("SKU cannot be empty")
        return v.strip().upper()
    
    @validator('tags')
    def validate_tags(cls, v: List[str]) -> List[str]:
        """Validate and normalize tags."""
        return [tag.strip().lower() for tag in v if tag.strip()]
    
    @root_validator
    def validate_pricing(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate pricing logic."""
        base_price = values.get('base_price')
        sale_price = values.get('sale_price')
        cost_price = values.get('cost_price')
        
        if sale_price and base_price:
            if sale_price.amount >= base_price.amount:
                raise ValueError("Sale price must be less than base price")
        
        if cost_price and base_price:
            if cost_price.amount >= base_price.amount:
                raise ValueError("Cost price should be less than base price")
        
        return values
    
    @root_validator
    def validate_physical_product(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate physical product requirements."""
        product_type = values.get('product_type')
        requires_shipping = values.get('requires_shipping', True)
        dimensions = values.get('dimensions')
        
        if product_type == ProductType.PHYSICAL and requires_shipping and not dimensions:
            raise ValueError("Physical products requiring shipping must have dimensions")
        
        return values
    
    @root_validator
    def validate_digital_product(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate digital product requirements."""
        product_type = values.get('product_type')
        download_url = values.get('download_url')
        description = values.get('description')
        
        if product_type == ProductType.DIGITAL and not download_url and not description:
            raise ValueError("Digital products require download URL or description")
        
        return values


class ProductUpdateRequest(BaseModel):
    """Schema for updating an existing product."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    # Basic Information (all optional for updates)
    name: Optional[str] = Field(None, min_length=2, max_length=200)
    description: Optional[str] = Field(None, max_length=5000)
    short_description: Optional[str] = Field(None, max_length=500)
    status: Optional[ProductStatus] = None
    
    # Pricing Updates
    base_price: Optional[MoneySchema] = None
    sale_price: Optional[MoneySchema] = None
    cost_price: Optional[MoneySchema] = None
    
    # Inventory Updates
    inventory_quantity: Optional[int] = Field(None, ge=0)
    low_stock_threshold: Optional[int] = Field(None, ge=0)
    allow_backorder: Optional[bool] = None
    
    # Marketing Updates
    featured: Optional[bool] = None
    tags: Optional[List[str]] = None
    
    # SEO Updates
    seo_data: Optional[SEODataSchema] = None
    
    # Custom Fields Updates
    attributes: Optional[Dict[str, Any]] = None
    custom_fields: Optional[Dict[str, Any]] = None
    
    # Media Updates
    images: Optional[List[str]] = None
    videos: Optional[List[str]] = None
    documents: Optional[List[str]] = None
    
    @validator('tags')
    def validate_tags(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate and normalize tags."""
        if v is not None:
            return [tag.strip().lower() for tag in v if tag.strip()]
        return v


class ProductSearchRequest(BaseModel):
    """Schema for product search requests."""
    
    # Search Filters
    query: Optional[str] = Field(None, description="Text search query")
    sku: Optional[str] = Field(None, description="SKU filter")
    category_id: Optional[str] = Field(None, description="Category filter")
    brand_id: Optional[str] = Field(None, description="Brand filter")
    status: Optional[ProductStatus] = Field(None, description="Status filter")
    product_type: Optional[ProductType] = Field(None, description="Product type filter")
    
    # Price Filters
    min_price: Optional[Decimal] = Field(None, ge=0, description="Minimum price")
    max_price: Optional[Decimal] = Field(None, ge=0, description="Maximum price")
    on_sale: Optional[bool] = Field(None, description="On sale filter")
    
    # Inventory Filters
    in_stock: Optional[bool] = Field(None, description="In stock filter")
    low_stock: Optional[bool] = Field(None, description="Low stock filter")
    min_quantity: Optional[int] = Field(None, ge=0, description="Minimum quantity")
    max_quantity: Optional[int] = Field(None, ge=0, description="Maximum quantity")
    
    # Marketing Filters
    featured: Optional[bool] = Field(None, description="Featured products filter")
    tags: Optional[List[str]] = Field(None, description="Tags filter")
    
    # Date Filters
    created_after: Optional[datetime] = Field(None, description="Created after date")
    created_before: Optional[datetime] = Field(None, description="Created before date")
    updated_after: Optional[datetime] = Field(None, description="Updated after date")
    updated_before: Optional[datetime] = Field(None, description="Updated before date")
    
    # Pagination
    page: int = Field(default=1, ge=1, description="Page number")
    per_page: int = Field(default=20, ge=1, le=100, description="Items per page")
    
    # Sorting
    sort_by: str = Field(default="updated_at", description="Sort field")
    sort_order: str = Field(default="desc", regex="^(asc|desc)$", description="Sort order")
    
    @validator('sort_by')
    def validate_sort_by(cls, v: str) -> str:
        """Validate sort field."""
        allowed_fields = [
            "name", "sku", "created_at", "updated_at", "published_at",
            "base_price", "inventory_quantity", "status"
        ]
        if v not in allowed_fields:
            raise ValueError(f"sort_by must be one of: {', '.join(allowed_fields)}")
        return v
    
    @root_validator
    def validate_price_range(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate price range."""
        min_price = values.get('min_price')
        max_price = values.get('max_price')
        
        if min_price and max_price and min_price > max_price:
            raise ValueError("min_price cannot be greater than max_price")
        
        return values
    
    @root_validator  
    def validate_quantity_range(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate quantity range."""
        min_quantity = values.get('min_quantity')
        max_quantity = values.get('max_quantity')
        
        if min_quantity and max_quantity and min_quantity > max_quantity:
            raise ValueError("min_quantity cannot be greater than max_quantity")
        
        return values


# Response Schemas
class ProductResponse(BaseModel):
    """Schema for product response."""
    
    # Basic Information
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
    base_price: Optional[MoneySchema]
    sale_price: Optional[MoneySchema]
    cost_price: Optional[MoneySchema]
    effective_price: Optional[MoneySchema]
    price_type: PriceType
    is_on_sale: bool
    discount_percentage: float
    profit_margin: Optional[float]
    
    # Inventory
    inventory_quantity: int
    low_stock_threshold: int
    inventory_tracking: InventoryTracking
    allow_backorder: bool
    is_low_stock: bool
    is_in_stock: bool
    total_inventory_value: Optional[MoneySchema]
    
    # Physical Properties
    dimensions: Optional[DimensionsSchema]
    requires_shipping: bool
    
    # Digital Properties
    download_url: Optional[str]
    download_limit: Optional[int]
    
    # SEO and Marketing
    seo_data: Optional[SEODataSchema]
    tags: List[str]
    featured: bool
    
    # Variants
    has_variants: bool = False
    variants: List[ProductVariantSchema] = Field(default_factory=list)
    
    # Customization
    attributes: Dict[str, Any]
    custom_fields: Dict[str, Any]
    
    # Media
    images: List[str]
    videos: List[str]
    documents: List[str]
    
    # AI Integration
    ai_generated_description: Optional[str]
    ai_confidence_score: Optional[float]
    ai_last_updated: Optional[datetime]
    
    # Timestamps
    created_at: datetime
    updated_at: datetime
    published_at: Optional[datetime]


class ProductListResponse(BaseModel):
    """Schema for paginated product list response."""
    
    products: List[ProductResponse]
    pagination: Dict[str, Any]
    filters_applied: Dict[str, Any]
    total: int
    page: int
    per_page: int
    total_pages: int
    has_next: bool
    has_prev: bool
    
    @validator('total_pages')
    def calculate_total_pages(cls, v: int, values: Dict[str, Any]) -> int:
        """Calculate total pages from total and per_page."""
        total = values.get('total', 0)
        per_page = values.get('per_page', 20)
        return (total + per_page - 1) // per_page if per_page > 0 else 0


class HealthResponse(BaseModel):
    """Schema for health check response."""
    
    status: str = Field(..., description="Overall health status")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    services: Dict[str, str] = Field(..., description="Individual service health")
    metrics: Dict[str, Any] = Field(..., description="Performance metrics")
    timestamp: str = Field(..., description="Response timestamp")


class ErrorResponse(BaseModel):
    """Schema for error responses."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    request_id: Optional[str] = Field(None, description="Request identifier for tracking")


# Bulk Operation Schemas
class BulkCreateRequest(BaseModel):
    """Schema for bulk product creation."""
    
    products: List[ProductCreateRequest] = Field(..., min_items=1, max_items=100)
    validate_skus: bool = Field(default=True, description="Validate SKU uniqueness")
    fail_on_error: bool = Field(default=True, description="Fail entire batch on any error")


class BulkCreateResponse(BaseModel):
    """Schema for bulk creation response."""
    
    created: List[ProductResponse] = Field(..., description="Successfully created products")
    failed: List[Dict[str, Any]] = Field(default_factory=list, description="Failed creations with errors")
    summary: Dict[str, int] = Field(..., description="Operation summary")


# Analytics Schemas
class ProductAnalyticsResponse(BaseModel):
    """Schema for product analytics response."""
    
    total_products: int
    by_status: Dict[str, int]
    by_type: Dict[str, int]
    by_category: Dict[str, int]
    inventory_stats: Dict[str, Any]
    pricing_stats: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    trends: Dict[str, Any]
    timestamp: str


# Configuration Schema
class APIConfigSchema(BaseModel):
    """Schema for API configuration."""
    
    cache_ttl: int = Field(default=3600, ge=60, description="Default cache TTL in seconds")
    rate_limit_requests: int = Field(default=1000, ge=1, description="Rate limit requests per window")
    rate_limit_window: int = Field(default=3600, ge=60, description="Rate limit window in seconds")
    max_search_results: int = Field(default=1000, ge=1, description="Maximum search results")
    bulk_operation_limit: int = Field(default=100, ge=1, description="Bulk operation limit")
    enable_analytics: bool = Field(default=True, description="Enable analytics collection")
    enable_caching: bool = Field(default=True, description="Enable response caching")
    log_level: str = Field(default="INFO", regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$") 