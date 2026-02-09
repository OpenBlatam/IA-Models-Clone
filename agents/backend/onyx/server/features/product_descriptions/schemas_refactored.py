from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import UUID
from pydantic import BaseModel, Field, validator, root_validator
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Refactored Schemas Module
========================

Clean, type-safe Pydantic models with comprehensive validation.
Organized by domain and separated by responsibility.
"""




# =============================================================================
# BASE SCHEMAS
# =============================================================================

class BaseSchema(BaseModel):
    """Base schema with common configuration."""
    
    @dataclass
class Config:
        from_attributes = True
        str_strip_whitespace = True
        validate_assignment = True
        use_enum_values = True


class TimestampMixin(BaseModel):
    """Mixin for timestamp fields."""
    
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class PaginationMixin(BaseModel):
    """Mixin for pagination fields."""
    
    page: int = Field(default=1, ge=1, le=1000, description="Page number")
    per_page: int = Field(default=20, ge=1, le=100, description="Items per page")


# =============================================================================
# PRODUCT SCHEMAS - Domain Models
# =============================================================================

class ProductBase(BaseSchema):
    """Base product schema with core fields."""
    
    name: str = Field(..., min_length=1, max_length=200, description="Product name")
    sku: str = Field(..., min_length=1, max_length=50, description="Stock keeping unit")
    description: str = Field(default="", max_length=5000, description="Product description")
    
    @validator('sku')
    def normalize_sku(cls, v: str) -> str:
        """Normalize SKU to uppercase."""
        return v.strip().upper()
    
    @validator('name')
    def validate_name(cls, v: str) -> str:
        """Validate product name."""
        if not v.strip():
            raise ValueError("Product name cannot be empty")
        return v.strip()


class ProductPricing(BaseSchema):
    """Product pricing information."""
    
    base_price: Optional[Decimal] = Field(None, ge=0, decimal_places=2)
    sale_price: Optional[Decimal] = Field(None, ge=0, decimal_places=2)
    cost_price: Optional[Decimal] = Field(None, ge=0, decimal_places=2)
    currency: str = Field(default="USD", min_length=3, max_length=3)
    
    @validator('currency')
    def normalize_currency(cls, v: str) -> str:
        """Normalize currency to uppercase."""
        return v.upper()
    
    @root_validator
    def validate_pricing(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate pricing logic."""
        base_price = values.get('base_price')
        sale_price = values.get('sale_price')
        cost_price = values.get('cost_price')
        
        # Sale price must be less than base price
        if sale_price and base_price and sale_price >= base_price:
            raise ValueError("Sale price must be less than base price")
        
        # Cost price should be less than base price (for profit)
        if cost_price and base_price and cost_price >= base_price:
            raise ValueError("Cost price should be less than base price for profitability")
        
        return values


class ProductInventory(BaseSchema):
    """Product inventory management."""
    
    quantity: int = Field(default=0, ge=0, description="Stock quantity")
    low_stock_threshold: int = Field(default=10, ge=0, description="Low stock alert threshold")
    track_inventory: bool = Field(default=True, description="Enable inventory tracking")
    
    @property
    def is_low_stock(self) -> bool:
        """Check if product is low in stock."""
        return self.quantity <= self.low_stock_threshold
    
    @property
    def is_in_stock(self) -> bool:
        """Check if product is in stock."""
        return self.quantity > 0


class ProductSEO(BaseSchema):
    """Product SEO optimization."""
    
    seo_title: Optional[str] = Field(None, max_length=100, description="SEO title")
    seo_description: Optional[str] = Field(None, max_length=300, description="SEO description")
    keywords: List[str] = Field(default_factory=list, max_items=10, description="SEO keywords")
    slug: Optional[str] = Field(None, max_length=200, description="URL slug")
    
    @validator('keywords')
    def normalize_keywords(cls, v: List[str]) -> List[str]:
        """Normalize keywords to lowercase."""
        return [kw.strip().lower() for kw in v if kw.strip()]


class ProductMetadata(BaseSchema):
    """Product metadata and categorization."""
    
    category_id: Optional[str] = Field(None, description="Category identifier")
    brand_id: Optional[str] = Field(None, description="Brand identifier")
    tags: List[str] = Field(default_factory=list, max_items=20, description="Product tags")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Custom attributes")
    
    @validator('tags')
    def normalize_tags(cls, v: List[str]) -> List[str]:
        """Normalize tags to lowercase."""
        return [tag.strip().lower() for tag in v if tag.strip()]


# =============================================================================
# REQUEST SCHEMAS - Input Models
# =============================================================================

class ProductCreateRequest(ProductBase, ProductPricing, ProductInventory, ProductSEO, ProductMetadata):
    """Complete product creation request."""
    
    # Additional fields for creation
    is_digital: bool = Field(default=False, description="Is digital product")
    download_url: Optional[str] = Field(None, description="Download URL for digital products")
    
    @root_validator
    def validate_digital_product(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate digital product requirements."""
        is_digital = values.get('is_digital', False)
        download_url = values.get('download_url')
        description = values.get('description')
        
        if is_digital and not download_url and not description:
            raise ValueError("Digital products require download URL or description")
        
        return values


class ProductUpdateRequest(BaseSchema):
    """Product update request with optional fields."""
    
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=5000)
    base_price: Optional[Decimal] = Field(None, ge=0, decimal_places=2)
    sale_price: Optional[Decimal] = Field(None, ge=0, decimal_places=2)
    quantity: Optional[int] = Field(None, ge=0)
    low_stock_threshold: Optional[int] = Field(None, ge=0)
    tags: Optional[List[str]] = Field(None, max_items=20)
    attributes: Optional[Dict[str, Any]] = Field(None)


class ProductSearchRequest(BaseSchema, PaginationMixin):
    """Advanced product search request."""
    
    # Search parameters
    query: Optional[str] = Field(None, max_length=200, description="Search query")
    category_id: Optional[str] = Field(None, description="Filter by category")
    brand_id: Optional[str] = Field(None, description="Filter by brand")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    
    # Price filters
    min_price: Optional[Decimal] = Field(None, ge=0, description="Minimum price")
    max_price: Optional[Decimal] = Field(None, ge=0, description="Maximum price")
    on_sale: Optional[bool] = Field(None, description="Filter sale items")
    
    # Inventory filters
    in_stock: Optional[bool] = Field(None, description="Filter in-stock items")
    low_stock: Optional[bool] = Field(None, description="Filter low-stock items")
    
    # Sorting
    sort_by: str = Field(default="updated_at", description="Sort field")
    sort_order: str = Field(default="desc", regex="^(asc|desc)$", description="Sort direction")
    
    @validator('sort_by')
    def validate_sort_field(cls, v: str) -> str:
        """Validate sort field."""
        allowed = ["name", "price", "created_at", "updated_at", "quantity", "sku"]
        if v not in allowed:
            raise ValueError(f"Invalid sort field. Allowed: {', '.join(allowed)}")
        return v


# =============================================================================
# RESPONSE SCHEMAS - Output Models
# =============================================================================

class ProductResponse(ProductBase, ProductPricing, ProductInventory, ProductSEO, ProductMetadata, TimestampMixin):
    """Complete product response with calculated fields."""
    
    id: str = Field(..., description="Product identifier")
    
    # Calculated fields
    effective_price: Optional[Decimal] = Field(None, description="Effective selling price")
    is_on_sale: bool = Field(default=False, description="Is product on sale")
    discount_percentage: float = Field(default=0.0, description="Discount percentage")
    profit_margin: Optional[float] = Field(None, description="Profit margin percentage")
    
    # Performance metadata
    cache_hit: bool = Field(default=False, description="Was result from cache")
    response_time_ms: Optional[float] = Field(None, description="Response time in milliseconds")


class ProductListResponse(BaseSchema):
    """Paginated product list response."""
    
    products: List[ProductResponse] = Field(..., description="List of products")
    total: int = Field(..., description="Total number of products")
    page: int = Field(..., description="Current page")
    per_page: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total pages")
    has_next: bool = Field(..., description="Has next page")
    has_prev: bool = Field(..., description="Has previous page")
    
    # Search metadata
    search_metadata: Dict[str, Any] = Field(default_factory=dict, description="Search execution info")


class BulkOperationResponse(BaseSchema):
    """Bulk operation response."""
    
    successful: int = Field(..., description="Number of successful operations")
    failed: int = Field(..., description="Number of failed operations")
    total: int = Field(..., description="Total operations attempted")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Error details")
    processing_time_ms: float = Field(..., description="Total processing time")


# =============================================================================
# HEALTH & MONITORING SCHEMAS
# =============================================================================

class HealthResponse(BaseSchema):
    """System health response."""
    
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    uptime_seconds: float = Field(..., description="Application uptime")
    version: str = Field(..., description="Application version")
    
    # Service health
    services: Dict[str, str] = Field(..., description="Individual service health")
    
    # Performance metrics
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")


class MetricsResponse(BaseSchema):
    """Application metrics response."""
    
    # Request metrics
    total_requests: int = Field(..., description="Total requests handled")
    requests_per_second: float = Field(..., description="Current RPS")
    average_response_time: float = Field(..., description="Average response time")
    
    # Cache metrics
    cache_hit_ratio: float = Field(..., description="Cache hit percentage")
    cache_operations: int = Field(..., description="Total cache operations")
    
    # Database metrics
    active_connections: int = Field(..., description="Active DB connections")
    query_count: int = Field(..., description="Total queries executed")
    
    # System metrics
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    cpu_usage_percent: float = Field(..., description="CPU usage percentage")


# =============================================================================
# ERROR SCHEMAS
# =============================================================================

class ErrorResponse(BaseSchema):
    """Standardized error response."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request correlation ID")
    path: Optional[str] = Field(None, description="Request path")


class ValidationErrorResponse(BaseSchema):
    """Validation error response."""
    
    error: str = Field(default="validation_error", description="Error type")
    message: str = Field(..., description="Validation error message")
    field_errors: List[Dict[str, Any]] = Field(..., description="Field-specific errors")
    timestamp: datetime = Field(..., description="Error timestamp")


# =============================================================================
# AI & ANALYTICS SCHEMAS
# =============================================================================

class AIDescriptionRequest(BaseSchema):
    """AI description generation request."""
    
    product_name: str = Field(..., min_length=1, max_length=200)
    features: List[str] = Field(default_factory=list, max_items=10)
    target_audience: Optional[str] = Field(None, max_length=100)
    tone: str = Field(default="professional", description="Description tone")
    
    @validator('tone')
    def validate_tone(cls, v: str) -> str:
        """Validate tone options."""
        allowed = ["professional", "casual", "technical", "marketing", "friendly"]
        if v not in allowed:
            raise ValueError(f"Invalid tone. Allowed: {', '.join(allowed)}")
        return v


class AIDescriptionResponse(BaseSchema):
    """AI description generation response."""
    
    description: str = Field(..., description="Generated description")
    confidence_score: float = Field(..., ge=0, le=1, description="Generation confidence")
    processing_time_ms: float = Field(..., description="Processing time")
    model_used: str = Field(..., description="AI model used")


class AnalyticsRequest(BaseSchema):
    """Analytics request parameters."""
    
    metric: str = Field(..., description="Metric to analyze")
    date_from: Optional[datetime] = Field(None, description="Start date")
    date_to: Optional[datetime] = Field(None, description="End date")
    group_by: Optional[str] = Field(None, description="Grouping dimension")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Additional filters")


class AnalyticsResponse(BaseSchema):
    """Analytics response data."""
    
    metric: str = Field(..., description="Analyzed metric")
    data: List[Dict[str, Any]] = Field(..., description="Analytics data points")
    summary: Dict[str, Any] = Field(..., description="Summary statistics")
    period: Dict[str, datetime] = Field(..., description="Analysis period")
    total_records: int = Field(..., description="Total records analyzed") 