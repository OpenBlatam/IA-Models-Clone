from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import json
import time
from abc import ABC, abstractmethod
from datetime import datetime, date, time as time_type
from decimal import Decimal
from enum import Enum
from functools import lru_cache, wraps
from typing import Any, Dict, List, Optional, Set, Tuple, Union, TypeVar, Generic
from typing_extensions import Annotated, TypeAlias
from uuid import UUID
import orjson
from pydantic import (
from pydantic.json import pydantic_encoder
from typing import Any, List, Dict, Optional
import logging
"""
Advanced Pydantic Serialization System

This module provides optimized serialization and deserialization capabilities
using Pydantic with performance enhancements, custom validators, and caching.
"""


    BaseModel, Field, ValidationError, validator, root_validator,
    ConfigDict, computed_field, model_validator, field_validator,
    BeforeValidator, PlainValidator, AfterValidator
)


# Type aliases for better type hints
JSONDict = Dict[str, Any]
JSONList = List[Any]
JSONValue = Union[str, int, float, bool, None, JSONDict, JSONList]
SerializedData = Union[str, bytes, JSONDict, JSONList]


class SerializationStrategy(Enum):
    """Serialization strategy types"""
    STANDARD = "standard"      # Standard Pydantic serialization
    ORJSON = "orjson"          # Fast orjson serialization
    COMPACT = "compact"        # Compact JSON without whitespace
    CACHED = "cached"          # Cached serialization
    LAZY = "lazy"             # Lazy serialization
    STREAMING = "streaming"    # Streaming serialization


class ValidationLevel(Enum):
    """Validation levels"""
    NONE = "none"             # No validation
    BASIC = "basic"           # Basic field validation
    STRICT = "strict"         # Strict validation with custom validators
    COMPLETE = "complete"     # Complete validation with all checks


# Custom field types with validation
class EmailField(str):
    """Email field with validation"""
    
    @classmethod
    def __get_validators__(cls) -> Optional[Dict[str, Any]]:
        yield cls.validate
    
    @classmethod
    def validate(cls, v) -> bool:
        if not isinstance(v, str):
            raise ValueError('Email must be a string')
        if '@' not in v or '.' not in v:
            raise ValueError('Invalid email format')
        return v.lower()


class PhoneField(str):
    """Phone field with validation"""
    
    @classmethod
    def __get_validators__(cls) -> Optional[Dict[str, Any]]:
        yield cls.validate
    
    @classmethod
    def validate(cls, v) -> bool:
        if not isinstance(v, str):
            raise ValueError('Phone must be a string')
        # Remove all non-digit characters
        digits = ''.join(filter(str.isdigit, v))
        if len(digits) < 10:
            raise ValueError('Phone number too short')
        return digits


class CurrencyField(Decimal):
    """Currency field with validation"""
    
    @classmethod
    def __get_validators__(cls) -> Optional[Dict[str, Any]]:
        yield cls.validate
    
    @classmethod
    def validate(cls, v) -> bool:
        if isinstance(v, str):
            v = Decimal(v)
        elif isinstance(v, (int, float)):
            v = Decimal(str(v))
        elif not isinstance(v, Decimal):
            raise ValueError('Invalid currency value')
        
        # Round to 2 decimal places
        return v.quantize(Decimal('0.01'))


# Custom validators
def validate_positive_number(v: Union[int, float]) -> Union[int, float]:
    """Validate positive number"""
    if v <= 0:
        raise ValueError('Value must be positive')
    return v


def validate_percentage(v: Union[int, float]) -> Union[int, float]:
    """Validate percentage (0-100)"""
    if not 0 <= v <= 100:
        raise ValueError('Percentage must be between 0 and 100')
    return v


def validate_strong_password(v: str) -> str:
    """Validate strong password"""
    if len(v) < 8:
        raise ValueError('Password must be at least 8 characters')
    if not any(c.isupper() for c in v):
        raise ValueError('Password must contain uppercase letter')
    if not any(c.islower() for c in v):
        raise ValueError('Password must contain lowercase letter')
    if not any(c.isdigit() for c in v):
        raise ValueError('Password must contain digit')
    return v


# Base model with optimized configuration
class OptimizedBaseModel(BaseModel):
    """Base model with optimized configuration for serialization"""
    
    model_config = ConfigDict(
        # Performance optimizations
        validate_assignment=True,
        arbitrary_types_allowed=True,
        use_enum_values=True,
        populate_by_name=True,
        
        # JSON serialization options
        json_encoders={
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat(),
            time_type: lambda v: v.isoformat(),
            Decimal: lambda v: str(v),
            UUID: lambda v: str(v),
        },
        
        # Validation options
        extra='forbid',  # Reject extra fields
        str_strip_whitespace=True,
        str_min_length=0,
    )
    
    @computed_field
    @property
    def model_name(self) -> str:
        """Get model name"""
        return self.__class__.__name__
    
    @computed_field
    @property
    def serialization_hash(self) -> str:
        """Get hash for caching"""
        return f"{self.model_name}:{hash(str(self.model_dump()))}"


# Product-related models with optimized serialization
class ProductCategory(OptimizedBaseModel):
    """Product category model"""
    id: str = Field(..., description="Category ID")
    name: str = Field(..., min_length=1, max_length=100, description="Category name")
    description: Optional[str] = Field(None, max_length=500, description="Category description")
    parent_id: Optional[str] = Field(None, description="Parent category ID")
    is_active: bool = Field(True, description="Category active status")
    sort_order: int = Field(0, ge=0, description="Sort order")
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate category name"""
        if not v.strip():
            raise ValueError('Category name cannot be empty')
        return v.strip().title()
    
    @field_validator('sort_order')
    @classmethod
    def validate_sort_order(cls, v: int) -> int:
        """Validate sort order"""
        return validate_positive_number(v)


class ProductTag(OptimizedBaseModel):
    """Product tag model"""
    id: str = Field(..., description="Tag ID")
    name: str = Field(..., min_length=1, max_length=50, description="Tag name")
    color: Optional[str] = Field(None, pattern=r'^#[0-9A-Fa-f]{6}$', description="Tag color")
    description: Optional[str] = Field(None, max_length=200, description="Tag description")
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate tag name"""
        return v.strip().lower()


class ProductImage(OptimizedBaseModel):
    """Product image model"""
    id: str = Field(..., description="Image ID")
    url: str = Field(..., description="Image URL")
    alt_text: Optional[str] = Field(None, max_length=200, description="Alt text")
    width: Optional[int] = Field(None, gt=0, description="Image width")
    height: Optional[int] = Field(None, gt=0, description="Image height")
    is_primary: bool = Field(False, description="Primary image flag")
    sort_order: int = Field(0, ge=0, description="Sort order")
    
    @field_validator('url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate image URL"""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('Invalid image URL')
        return v


class ProductVariant(OptimizedBaseModel):
    """Product variant model"""
    id: str = Field(..., description="Variant ID")
    sku: str = Field(..., min_length=1, max_length=50, description="SKU")
    name: str = Field(..., min_length=1, max_length=100, description="Variant name")
    price: CurrencyField = Field(..., description="Variant price")
    compare_price: Optional[CurrencyField] = Field(None, description="Compare price")
    weight: Optional[float] = Field(None, gt=0, description="Weight in kg")
    dimensions: Optional[Dict[str, float]] = Field(None, description="Dimensions")
    stock_quantity: int = Field(0, ge=0, description="Stock quantity")
    is_active: bool = Field(True, description="Variant active status")
    
    @field_validator('sku')
    @classmethod
    def validate_sku(cls, v: str) -> str:
        """Validate SKU"""
        return v.strip().upper()
    
    @field_validator('compare_price')
    @classmethod
    def validate_compare_price(cls, v: Optional[CurrencyField], info) -> Optional[CurrencyField]:
        """Validate compare price"""
        if v is not None:
            price = info.data.get('price')
            if price and v <= price:
                raise ValueError('Compare price must be greater than regular price')
        return v


class ProductDescription(OptimizedBaseModel):
    """Product description model with optimized serialization"""
    id: str = Field(..., description="Product ID")
    title: str = Field(..., min_length=1, max_length=200, description="Product title")
    short_description: Optional[str] = Field(None, max_length=500, description="Short description")
    long_description: Optional[str] = Field(None, max_length=5000, description="Long description")
    category: ProductCategory = Field(..., description="Product category")
    tags: List[ProductTag] = Field(default_factory=list, description="Product tags")
    images: List[ProductImage] = Field(default_factory=list, description="Product images")
    variants: List[ProductVariant] = Field(default_factory=list, description="Product variants")
    price_range: Dict[str, CurrencyField] = Field(default_factory=dict, description="Price range")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Update timestamp")
    is_active: bool = Field(True, description="Product active status")
    
    @field_validator('title')
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Validate product title"""
        if not v.strip():
            raise ValueError('Product title cannot be empty')
        return v.strip()
    
    @field_validator('price_range')
    @classmethod
    def validate_price_range(cls, v: Dict[str, CurrencyField]) -> Dict[str, CurrencyField]:
        """Validate price range"""
        if 'min' in v and 'max' in v:
            if v['min'] > v['max']:
                raise ValueError('Min price cannot be greater than max price')
        return v
    
    @model_validator(mode='after')
    def validate_product(self) -> 'ProductDescription':
        """Validate product after all fields are set"""
        # Ensure at least one variant
        if not self.variants:
            raise ValueError('Product must have at least one variant')
        
        # Calculate price range from variants
        if not self.price_range:
            prices = [v.price for v in self.variants if v.is_active]
            if prices:
                self.price_range = {
                    'min': min(prices),
                    'max': max(prices)
                }
        
        return self
    
    @computed_field
    @property
    def primary_image(self) -> Optional[ProductImage]:
        """Get primary image"""
        for image in self.images:
            if image.is_primary:
                return image
        return self.images[0] if self.images else None
    
    @computed_field
    @property
    def active_variants(self) -> List[ProductVariant]:
        """Get active variants"""
        return [v for v in self.variants if v.is_active]
    
    @computed_field
    @property
    def total_stock(self) -> int:
        """Get total stock across all variants"""
        return sum(v.stock_quantity for v in self.active_variants)


# Serialization utilities
class SerializationCache:
    """Cache for serialized data"""
    
    def __init__(self, max_size: int = 1000):
        
    """__init__ function."""
self.max_size = max_size
        self._cache: Dict[str, Tuple[SerializedData, float]] = {}
        self._access_times: Dict[str, float] = {}
    
    def get(self, key: str) -> Optional[SerializedData]:
        """Get cached serialized data"""
        if key in self._cache:
            data, expiry = self._cache[key]
            if time.time() < expiry:
                self._access_times[key] = time.time()
                return data
            else:
                self._remove(key)
        return None
    
    def set(self, key: str, data: SerializedData, ttl: float = 300):
        """Set cached serialized data"""
        if len(self._cache) >= self.max_size:
            self._evict_oldest()
        
        expiry = time.time() + ttl
        self._cache[key] = (data, expiry)
        self._access_times[key] = time.time()
    
    def _remove(self, key: str):
        """Remove item from cache"""
        self._cache.pop(key, None)
        self._access_times.pop(key, None)
    
    def _evict_oldest(self) -> Any:
        """Evict oldest accessed item"""
        if not self._access_times:
            return
        
        oldest_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        self._remove(oldest_key)
    
    def clear(self) -> Any:
        """Clear all cached data"""
        self._cache.clear()
        self._access_times.clear()


class PydanticSerializer:
    """Advanced Pydantic serializer with multiple strategies"""
    
    def __init__(self, strategy: SerializationStrategy = SerializationStrategy.ORJSON):
        
    """__init__ function."""
self.strategy = strategy
        self.cache = SerializationCache() if strategy == SerializationStrategy.CACHED else None
    
    def serialize(
        self,
        model: BaseModel,
        strategy: Optional[SerializationStrategy] = None,
        include_none: bool = False,
        exclude_defaults: bool = False,
        exclude_unset: bool = False,
        exclude_empty: bool = False,
        by_alias: bool = False,
        round_trip: bool = False,
        warnings: bool = True,
        max_depth: Optional[int] = None,
        exclude: Optional[Set[str]] = None,
        include: Optional[Set[str]] = None,
    ) -> SerializedData:
        """Serialize Pydantic model with specified strategy"""
        
        strategy = strategy or self.strategy
        
        # Check cache first
        if self.cache and strategy == SerializationStrategy.CACHED:
            cache_key = f"{model.serialization_hash}:{strategy.value}"
            cached_data = self.cache.get(cache_key)
            if cached_data:
                return cached_data
        
        # Serialize based on strategy
        if strategy == SerializationStrategy.ORJSON:
            result = self._serialize_orjson(
                model, include_none, exclude_defaults, exclude_unset,
                exclude_empty, by_alias, round_trip, warnings,
                max_depth, exclude, include
            )
        elif strategy == SerializationStrategy.COMPACT:
            result = self._serialize_compact(
                model, include_none, exclude_defaults, exclude_unset,
                exclude_empty, by_alias, round_trip, warnings,
                max_depth, exclude, include
            )
        elif strategy == SerializationStrategy.STANDARD:
            result = self._serialize_standard(
                model, include_none, exclude_defaults, exclude_unset,
                exclude_empty, by_alias, round_trip, warnings,
                max_depth, exclude, include
            )
        else:
            result = self._serialize_standard(
                model, include_none, exclude_defaults, exclude_unset,
                exclude_empty, by_alias, round_trip, warnings,
                max_depth, exclude, include
            )
        
        # Cache result
        if self.cache and strategy == SerializationStrategy.CACHED:
            cache_key = f"{model.serialization_hash}:{strategy.value}"
            self.cache.set(cache_key, result)
        
        return result
    
    def _serialize_orjson(
        self,
        model: BaseModel,
        include_none: bool,
        exclude_defaults: bool,
        exclude_unset: bool,
        exclude_empty: bool,
        by_alias: bool,
        round_trip: bool,
        warnings: bool,
        max_depth: Optional[int],
        exclude: Optional[Set[str]],
        include: Optional[Set[str]],
    ) -> bytes:
        """Serialize using orjson for maximum performance"""
        model_dict = model.model_dump(
            include_none=include_none,
            exclude_defaults=exclude_defaults,
            exclude_unset=exclude_unset,
            exclude_empty=exclude_empty,
            by_alias=by_alias,
            round_trip=round_trip,
            warnings=warnings,
            max_depth=max_depth,
            exclude=exclude,
            include=include,
        )
        
        return orjson.dumps(
            model_dict,
            option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NAIVE_UTC
        )
    
    def _serialize_compact(
        self,
        model: BaseModel,
        include_none: bool,
        exclude_defaults: bool,
        exclude_unset: bool,
        exclude_empty: bool,
        by_alias: bool,
        round_trip: bool,
        warnings: bool,
        max_depth: Optional[int],
        exclude: Optional[Set[str]],
        include: Optional[Set[str]],
    ) -> str:
        """Serialize to compact JSON string"""
        model_dict = model.model_dump(
            include_none=include_none,
            exclude_defaults=exclude_defaults,
            exclude_unset=exclude_unset,
            exclude_empty=exclude_empty,
            by_alias=by_alias,
            round_trip=round_trip,
            warnings=warnings,
            max_depth=max_depth,
            exclude=exclude,
            include=include,
        )
        
        return json.dumps(model_dict, separators=(',', ':'), ensure_ascii=False)
    
    def _serialize_standard(
        self,
        model: BaseModel,
        include_none: bool,
        exclude_defaults: bool,
        exclude_unset: bool,
        exclude_empty: bool,
        by_alias: bool,
        round_trip: bool,
        warnings: bool,
        max_depth: Optional[int],
        exclude: Optional[Set[str]],
        include: Optional[Set[str]],
    ) -> JSONDict:
        """Serialize using standard Pydantic method"""
        return model.model_dump(
            include_none=include_none,
            exclude_defaults=exclude_defaults,
            exclude_unset=exclude_unset,
            exclude_empty=exclude_empty,
            by_alias=by_alias,
            round_trip=round_trip,
            warnings=warnings,
            max_depth=max_depth,
            exclude=exclude,
            include=include,
        )
    
    def deserialize(
        self,
        data: SerializedData,
        model_class: type[BaseModel],
        validation_level: ValidationLevel = ValidationLevel.STRICT
    ) -> BaseModel:
        """Deserialize data to Pydantic model"""
        
        # Parse input data
        if isinstance(data, bytes):
            parsed_data = orjson.loads(data)
        elif isinstance(data, str):
            parsed_data = json.loads(data)
        else:
            parsed_data = data
        
        # Create model with validation level
        if validation_level == ValidationLevel.NONE:
            # Skip validation for performance
            return model_class.model_construct(**parsed_data)
        else:
            # Full validation
            return model_class.model_validate(parsed_data)


class StreamingSerializer:
    """Streaming serializer for large datasets"""
    
    def __init__(self, chunk_size: int = 1000):
        
    """__init__ function."""
self.chunk_size = chunk_size
    
    async def serialize_stream(
        self,
        models: List[BaseModel],
        strategy: SerializationStrategy = SerializationStrategy.ORJSON
    ) -> AsyncGenerator[bytes, None]:
        """Serialize models in streaming fashion"""
        
        serializer = PydanticSerializer(strategy)
        
        for i in range(0, len(models), self.chunk_size):
            chunk = models[i:i + self.chunk_size]
            
            # Serialize chunk
            chunk_data = []
            for model in chunk:
                serialized = serializer.serialize(model, strategy)
                if isinstance(serialized, bytes):
                    chunk_data.append(serialized.decode())
                else:
                    chunk_data.append(serialized)
            
            # Yield chunk
            yield orjson.dumps(chunk_data)
            
            # Small delay to prevent blocking
            await asyncio.sleep(0.001)
    
    async def deserialize_stream(
        self,
        data_stream: AsyncGenerator[bytes, None],
        model_class: type[BaseModel]
    ) -> AsyncGenerator[BaseModel, None]:
        """Deserialize data stream to models"""
        
        serializer = PydanticSerializer()
        
        async for chunk_data in data_stream:
            chunk = orjson.loads(chunk_data)
            
            for item_data in chunk:
                model = serializer.deserialize(item_data, model_class)
                yield model


# Performance decorators
def cached_serialization(ttl: float = 300):
    """Decorator for caching serialized data"""
    def decorator(func) -> Any:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Generate cache key
            cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Check cache
            cache = SerializationCache()
            cached_result = cache.get(cache_key)
            if cached_result:
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            cache.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator


def timing_decorator(func) -> Any:
    """Decorator for timing serialization operations"""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        print(f"{func.__name__} took {end_time - start_time:.6f} seconds")
        return result
    return wrapper


# Utility functions
@lru_cache(maxsize=128)
def get_serializer(strategy: SerializationStrategy) -> PydanticSerializer:
    """Get cached serializer instance"""
    return PydanticSerializer(strategy)


def batch_serialize(
    models: List[BaseModel],
    strategy: SerializationStrategy = SerializationStrategy.ORJSON,
    batch_size: int = 100
) -> List[SerializedData]:
    """Serialize multiple models in batches"""
    serializer = get_serializer(strategy)
    results = []
    
    for i in range(0, len(models), batch_size):
        batch = models[i:i + batch_size]
        batch_results = [serializer.serialize(model, strategy) for model in batch]
        results.extend(batch_results)
    
    return results


def batch_deserialize(
    data_list: List[SerializedData],
    model_class: type[BaseModel],
    validation_level: ValidationLevel = ValidationLevel.STRICT
) -> List[BaseModel]:
    """Deserialize multiple data items in batches"""
    serializer = PydanticSerializer()
    results = []
    
    for data in data_list:
        model = serializer.deserialize(data, model_class, validation_level)
        results.append(model)
    
    return results


# Global serializer instances
_serializers: Dict[SerializationStrategy, PydanticSerializer] = {}


def get_global_serializer(strategy: SerializationStrategy = SerializationStrategy.ORJSON) -> PydanticSerializer:
    """Get global serializer instance"""
    if strategy not in _serializers:
        _serializers[strategy] = PydanticSerializer(strategy)
    return _serializers[strategy]


def clear_serializer_cache():
    """Clear all serializer caches"""
    for serializer in _serializers.values():
        if serializer.cache:
            serializer.cache.clear()


# Example usage and testing
class SerializationExample:
    """Example usage of the serialization system"""
    
    @staticmethod
    def create_sample_product() -> ProductDescription:
        """Create a sample product for testing"""
        category = ProductCategory(
            id="cat_001",
            name="Electronics",
            description="Electronic devices and gadgets",
            sort_order=1
        )
        
        tag1 = ProductTag(id="tag_001", name="premium", color="#FFD700")
        tag2 = ProductTag(id="tag_002", name="wireless", color="#4CAF50")
        
        image1 = ProductImage(
            id="img_001",
            url="https://example.com/product1.jpg",
            alt_text="Product main image",
            width=800,
            height=600,
            is_primary=True
        )
        
        variant1 = ProductVariant(
            id="var_001",
            sku="PROD-001-BLK",
            name="Black Variant",
            price=Decimal("99.99"),
            stock_quantity=50
        )
        
        variant2 = ProductVariant(
            id="var_002",
            sku="PROD-001-WHT",
            name="White Variant",
            price=Decimal("109.99"),
            stock_quantity=30
        )
        
        return ProductDescription(
            id="prod_001",
            title="Premium Wireless Headphones",
            short_description="High-quality wireless headphones with noise cancellation",
            long_description="Experience crystal clear sound with our premium wireless headphones...",
            category=category,
            tags=[tag1, tag2],
            images=[image1],
            variants=[variant1, variant2]
        )
    
    @staticmethod
    def benchmark_serialization():
        """Benchmark different serialization strategies"""
        product = SerializationExample.create_sample_product()
        
        strategies = [
            SerializationStrategy.STANDARD,
            SerializationStrategy.ORJSON,
            SerializationStrategy.COMPACT,
            SerializationStrategy.CACHED
        ]
        
        print("Serialization Performance Benchmark")
        print("=" * 50)
        
        for strategy in strategies:
            serializer = PydanticSerializer(strategy)
            
            # Warm up
            for _ in range(10):
                serializer.serialize(product, strategy)
            
            # Benchmark
            start_time = time.time()
            for _ in range(1000):
                result = serializer.serialize(product, strategy)
            end_time = time.time()
            
            total_time = end_time - start_time
            avg_time = total_time / 1000
            
            print(f"{strategy.value:12} | {total_time:8.3f}s | {avg_time:10.6f}s per op")
        
        print("=" * 50)


if __name__ == "__main__":
    # Run benchmark
    SerializationExample.benchmark_serialization() 