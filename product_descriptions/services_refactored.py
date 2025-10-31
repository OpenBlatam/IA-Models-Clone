from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import hashlib
import json
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import structlog
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession
from .config_refactored import config
from .schemas_refactored import (
from typing import Any, List, Dict, Optional
import logging
"""
Refactored Services Module
=========================

Clean Architecture services with proper separation of concerns.
Business logic encapsulated in domain services.
"""



    ProductCreateRequest,
    ProductResponse,
    ProductSearchRequest,
    ProductListResponse,
    HealthResponse,
    MetricsResponse,
    AIDescriptionRequest,
    AIDescriptionResponse
)

logger = structlog.get_logger(__name__)


# =============================================================================
# INTERFACES - Abstract Base Classes
# =============================================================================

class ICacheService(ABC):
    """Cache service interface."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check cache health."""
        pass


class IProductRepository(ABC):
    """Product repository interface."""
    
    @abstractmethod
    async def create(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new product."""
        pass
    
    @abstractmethod
    async def get_by_id(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Get product by ID."""
        pass
    
    @abstractmethod
    async def get_by_sku(self, sku: str) -> Optional[Dict[str, Any]]:
        """Get product by SKU."""
        pass
    
    @abstractmethod
    async def search(self, filters: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int]:
        """Search products with filters."""
        pass
    
    @abstractmethod
    async def update(self, product_id: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update product."""
        pass
    
    @abstractmethod
    async def delete(self, product_id: str) -> bool:
        """Delete product."""
        pass


class INotificationService(ABC):
    """Notification service interface."""
    
    @abstractmethod
    async def send_low_stock_alert(self, product: Dict[str, Any]) -> bool:
        """Send low stock notification."""
        pass
    
    @abstractmethod
    async def send_product_created(self, product: Dict[str, Any]) -> bool:
        """Send product creation notification."""
        pass


# =============================================================================
# CACHE SERVICE - Redis Implementation
# =============================================================================

class RedisCacheService(ICacheService):
    """Redis-based cache service with connection pooling."""
    
    def __init__(self, redis_url: str, max_connections: int = 20):
        
    """__init__ function."""
self.redis_url = redis_url
        self.max_connections = max_connections
        self.client: Optional[Redis] = None
        self._connection_pool = None
        
        # Metrics
        self.hits = 0
        self.misses = 0
        self.operations = 0
    
    async def initialize(self) -> None:
        """Initialize Redis connection."""
        try:
            self.client = Redis.from_url(
                self.redis_url,
                max_connections=self.max_connections,
                retry_on_timeout=True,
                socket_timeout=5
            )
            # Test connection
            await self.client.ping()
            logger.info("✅ Redis cache service initialized", url=self.redis_url)
        except Exception as e:
            logger.error("❌ Redis initialization failed", error=str(e))
            self.client = None
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.client:
            return None
        
        try:
            self.operations += 1
            value = await self.client.get(key)
            
            if value:
                self.hits += 1
                logger.debug("Cache hit", key=key)
                return json.loads(value)
            else:
                self.misses += 1
                logger.debug("Cache miss", key=key)
                return None
                
        except Exception as e:
            logger.error("Cache get error", key=key, error=str(e))
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        if not self.client:
            return False
        
        try:
            self.operations += 1
            ttl = ttl or config.cache_ttl
            
            serialized_value = json.dumps(value, default=str)
            await self.client.setex(key, ttl, serialized_value)
            
            logger.debug("Cache set", key=key, ttl=ttl)
            return True
            
        except Exception as e:
            logger.error("Cache set error", key=key, error=str(e))
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not self.client:
            return False
        
        try:
            self.operations += 1
            result = await self.client.delete(key)
            logger.debug("Cache delete", key=key, deleted=bool(result))
            return bool(result)
            
        except Exception as e:
            logger.error("Cache delete error", key=key, error=str(e))
            return False
    
    async def health_check(self) -> bool:
        """Check cache health."""
        if not self.client:
            return False
        
        try:
            await self.client.ping()
            return True
        except Exception:
            return False
    
    async def close(self) -> None:
        """Close cache connections."""
        if self.client:
            await self.client.close()
            logger.info("🔌 Redis cache closed")
    
    @property
    def hit_ratio(self) -> float:
        """Calculate cache hit ratio."""
        total = self.hits + self.misses
        return (self.hits / total) if total > 0 else 0.0


# =============================================================================
# PRODUCT REPOSITORY - In-Memory Implementation (Demo)
# =============================================================================

class InMemoryProductRepository(IProductRepository):
    """In-memory product repository for demonstration."""
    
    def __init__(self) -> Any:
        self.products: Dict[str, Dict[str, Any]] = {}
        self.sku_index: Dict[str, str] = {}
        self._next_id = 1
    
    async def create(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new product."""
        # Generate ID
        product_id = f"prod_{self._next_id:06d}"
        self._next_id += 1
        
        # Add metadata
        now = datetime.utcnow()
        product_data.update({
            "id": product_id,
            "created_at": now,
            "updated_at": now
        })
        
        # Store product
        self.products[product_id] = product_data
        self.sku_index[product_data["sku"]] = product_id
        
        logger.info("Product created", product_id=product_id, sku=product_data["sku"])
        return product_data
    
    async def get_by_id(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Get product by ID."""
        return self.products.get(product_id)
    
    async def get_by_sku(self, sku: str) -> Optional[Dict[str, Any]]:
        """Get product by SKU."""
        product_id = self.sku_index.get(sku.upper())
        return self.products.get(product_id) if product_id else None
    
    async def search(self, filters: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int]:
        """Search products with filters."""
        products = list(self.products.values())
        
        # Apply filters
        if filters.get("query"):
            query = filters["query"].lower()
            products = [
                p for p in products
                if (query in p.get("name", "").lower() or 
                    query in p.get("description", "").lower() or
                    query in p.get("sku", "").lower())
            ]
        
        if filters.get("category_id"):
            products = [p for p in products if p.get("category_id") == filters["category_id"]]
        
        if filters.get("in_stock"):
            products = [p for p in products if p.get("quantity", 0) > 0]
        
        if filters.get("on_sale"):
            products = [p for p in products if p.get("sale_price") is not None]
        
        # Price range
        if filters.get("min_price"):
            products = [p for p in products if (p.get("base_price") or 0) >= filters["min_price"]]
        
        if filters.get("max_price"):
            products = [p for p in products if (p.get("base_price") or 0) <= filters["max_price"]]
        
        # Sorting
        sort_by = filters.get("sort_by", "updated_at")
        sort_order = filters.get("sort_order", "desc")
        
        products.sort(
            key=lambda x: x.get(sort_by, ""),
            reverse=(sort_order == "desc")
        )
        
        total = len(products)
        
        # Pagination
        page = filters.get("page", 1)
        per_page = filters.get("per_page", 20)
        start = (page - 1) * per_page
        end = start + per_page
        
        return products[start:end], total
    
    async def update(self, product_id: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update product."""
        if product_id not in self.products:
            return None
        
        product = self.products[product_id].copy()
        product.update(data)
        product["updated_at"] = datetime.utcnow()
        
        self.products[product_id] = product
        
        logger.info("Product updated", product_id=product_id)
        return product
    
    async def delete(self, product_id: str) -> bool:
        """Delete product."""
        if product_id not in self.products:
            return False
        
        product = self.products[product_id]
        sku = product.get("sku")
        
        del self.products[product_id]
        if sku and sku in self.sku_index:
            del self.sku_index[sku]
        
        logger.info("Product deleted", product_id=product_id)
        return True


# =============================================================================
# BUSINESS SERVICES - Domain Logic
# =============================================================================

class ProductService:
    """Core product business logic service."""
    
    def __init__(self, repository: IProductRepository, cache: ICacheService):
        
    """__init__ function."""
self.repository = repository
        self.cache = cache
    
    async def create_product(self, request: ProductCreateRequest) -> ProductResponse:
        """Create new product with business logic."""
        # Check SKU uniqueness
        existing = await self.repository.get_by_sku(request.sku)
        if existing:
            raise ValueError(f"SKU '{request.sku}' already exists")
        
        # Calculate derived fields
        product_data = request.dict()
        product_data.update(self._calculate_pricing_fields(product_data))
        product_data.update(self._calculate_inventory_fields(product_data))
        
        # Create product
        created_product = await self.repository.create(product_data)
        
        # Cache the product
        cache_key = f"product:{created_product['id']}"
        await self.cache.set(cache_key, created_product, ttl=config.cache_ttl)
        
        # Send notifications if needed
        if created_product.get("quantity", 0) <= created_product.get("low_stock_threshold", 10):
            logger.warning("Product created with low stock", 
                         product_id=created_product['id'],
                         quantity=created_product.get("quantity"))
        
        return ProductResponse(**created_product)
    
    async def get_product(self, product_id: str) -> Optional[ProductResponse]:
        """Get product by ID with caching."""
        # Try cache first
        cache_key = f"product:{product_id}"
        cached_data = await self.cache.get(cache_key)
        
        if cached_data:
            response = ProductResponse(**cached_data)
            response.cache_hit = True
            return response
        
        # Get from repository
        product_data = await self.repository.get_by_id(product_id)
        if not product_data:
            return None
        
        # Update calculated fields
        product_data.update(self._calculate_pricing_fields(product_data))
        product_data.update(self._calculate_inventory_fields(product_data))
        
        # Cache for future requests
        await self.cache.set(cache_key, product_data)
        
        response = ProductResponse(**product_data)
        response.cache_hit = False
        return response
    
    async def search_products(self, request: ProductSearchRequest) -> ProductListResponse:
        """Search products with caching."""
        # Generate cache key from search parameters
        cache_key = self._generate_search_cache_key(request)
        
        # Try cache first
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            return ProductListResponse(**cached_result)
        
        # Search in repository
        filters = request.dict(exclude_unset=True)
        products, total = await self.repository.search(filters)
        
        # Process products
        processed_products = []
        for product in products:
            product.update(self._calculate_pricing_fields(product))
            product.update(self._calculate_inventory_fields(product))
            processed_products.append(ProductResponse(**product))
        
        # Build response
        total_pages = (total + request.per_page - 1) // request.per_page
        response_data = {
            "products": processed_products,
            "total": total,
            "page": request.page,
            "per_page": request.per_page,
            "total_pages": total_pages,
            "has_next": request.page < total_pages,
            "has_prev": request.page > 1,
            "search_metadata": {
                "filters_applied": len([f for f in filters.values() if f is not None]),
                "cache_hit": False
            }
        }
        
        # Cache result for shorter time
        await self.cache.set(cache_key, response_data, ttl=300)  # 5 minutes
        
        return ProductListResponse(**response_data)
    
    def _calculate_pricing_fields(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate pricing-related fields."""
        base_price = product.get("base_price")
        sale_price = product.get("sale_price")
        cost_price = product.get("cost_price")
        
        # Effective price
        effective_price = sale_price or base_price
        
        # Sale status
        is_on_sale = bool(sale_price)
        
        # Discount percentage
        discount_percentage = 0.0
        if is_on_sale and base_price and sale_price:
            discount = base_price - sale_price
            discount_percentage = float((discount / base_price) * 100)
        
        # Profit margin
        profit_margin = None
        if cost_price and effective_price:
            profit = effective_price - cost_price
            profit_margin = float((profit / effective_price) * 100)
        
        return {
            "effective_price": effective_price,
            "is_on_sale": is_on_sale,
            "discount_percentage": round(discount_percentage, 2),
            "profit_margin": round(profit_margin, 2) if profit_margin else None
        }
    
    def _calculate_inventory_fields(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate inventory-related fields."""
        quantity = product.get("quantity", 0)
        low_stock_threshold = product.get("low_stock_threshold", 10)
        
        return {
            "is_low_stock": quantity <= low_stock_threshold,
            "is_in_stock": quantity > 0
        }
    
    def _generate_search_cache_key(self, request: ProductSearchRequest) -> str:
        """Generate cache key for search request."""
        search_data = request.dict(exclude_unset=True)
        search_str = json.dumps(search_data, sort_keys=True, default=str)
        hash_key = hashlib.md5(search_str.encode()).hexdigest()
        return f"search:{hash_key}"


# =============================================================================
# AI SERVICE - LLM Integration
# =============================================================================

class AIService:
    """AI/ML service for intelligent features."""
    
    def __init__(self, cache: ICacheService):
        
    """__init__ function."""
self.cache = cache
        self.enabled = bool(config.openai_api_key)
    
    async def generate_description(self, request: AIDescriptionRequest) -> AIDescriptionResponse:
        """Generate product description using AI."""
        if not self.enabled:
            # Mock response when AI is disabled
            description = f"Professional {request.product_name} with excellent quality and features."
            return AIDescriptionResponse(
                description=description,
                confidence_score=0.8,
                processing_time_ms=50.0,
                model_used="mock"
            )
        
        # Generate cache key
        cache_key = f"ai_desc:{hash(request.product_name + str(request.features))}"
        
        # Try cache first
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            return AIDescriptionResponse(**cached_result)
        
        start_time = time.time()
        
        # TODO: Integrate with actual OpenAI API
        # For demo, generate mock description
        features_text = ", ".join(request.features) if request.features else "quality construction"
        description = f"Discover the {request.product_name} - expertly crafted with {features_text}. Perfect for {request.target_audience or 'discerning customers'} seeking reliability and performance."
        
        processing_time = (time.time() - start_time) * 1000
        
        response_data = {
            "description": description,
            "confidence_score": 0.85,
            "processing_time_ms": processing_time,
            "model_used": config.openai_api_key[:10] + "..." if config.openai_api_key else "mock"
        }
        
        # Cache for 24 hours
        await self.cache.set(cache_key, response_data, ttl=86400)
        
        logger.info("AI description generated", 
                   product=request.product_name, 
                   processing_time_ms=processing_time)
        
        return AIDescriptionResponse(**response_data)


# =============================================================================
# HEALTH & MONITORING SERVICE
# =============================================================================

class HealthService:
    """Health check and monitoring service."""
    
    def __init__(self, cache: ICacheService, repository: IProductRepository):
        
    """__init__ function."""
self.cache = cache
        self.repository = repository
        self.start_time = time.time()
    
    async def get_health_status(self) -> HealthResponse:
        """Get comprehensive health status."""
        # Check individual services
        cache_healthy = await self.cache.health_check()
        
        # Database health (simplified for demo)
        db_healthy = True  # In real implementation, check DB connection
        
        # Overall status
        overall_status = "healthy" if (cache_healthy and db_healthy) else "degraded"
        
        # Service statuses
        services = {
            "cache": "healthy" if cache_healthy else "unhealthy",
            "database": "healthy" if db_healthy else "unhealthy",
            "api": "healthy"
        }
        
        # Performance metrics
        uptime = time.time() - self.start_time
        metrics = {
            "uptime_seconds": uptime,
            "cache_hit_ratio": getattr(self.cache, 'hit_ratio', 0.0),
            "total_products": len(getattr(self.repository, 'products', {}))
        }
        
        return HealthResponse(
            status=overall_status,
            timestamp=datetime.utcnow(),
            uptime_seconds=uptime,
            version=config.version,
            services=services,
            metrics=metrics
        )
    
    async def get_metrics(self) -> MetricsResponse:
        """Get detailed application metrics."""
        # In a real implementation, these would come from actual metrics collection
        return MetricsResponse(
            total_requests=1000,
            requests_per_second=10.5,
            average_response_time=0.150,
            cache_hit_ratio=getattr(self.cache, 'hit_ratio', 0.0),
            cache_operations=getattr(self.cache, 'operations', 0),
            active_connections=5,
            query_count=500,
            memory_usage_mb=128.5,
            cpu_usage_percent=25.3
        )


# =============================================================================
# SERVICE FACTORY - Dependency Injection
# =============================================================================

class ServiceContainer:
    """Service container for dependency injection."""
    
    def __init__(self) -> Any:
        self._cache_service: Optional[ICacheService] = None
        self._product_repository: Optional[IProductRepository] = None
        self._product_service: Optional[ProductService] = None
        self._ai_service: Optional[AIService] = None
        self._health_service: Optional[HealthService] = None
    
    async def get_cache_service(self) -> ICacheService:
        """Get cache service instance."""
        if not self._cache_service:
            self._cache_service = RedisCacheService(
                redis_url=config.redis_url,
                max_connections=config.redis_max_connections
            )
            await self._cache_service.initialize()
        return self._cache_service
    
    async def get_product_repository(self) -> IProductRepository:
        """Get product repository instance."""
        if not self._product_repository:
            self._product_repository = InMemoryProductRepository()
        return self._product_repository
    
    async def get_product_service(self) -> ProductService:
        """Get product service instance."""
        if not self._product_service:
            cache = await self.get_cache_service()
            repository = await self.get_product_repository()
            self._product_service = ProductService(repository, cache)
        return self._product_service
    
    async def get_ai_service(self) -> AIService:
        """Get AI service instance."""
        if not self._ai_service:
            cache = await self.get_cache_service()
            self._ai_service = AIService(cache)
        return self._ai_service
    
    async def get_health_service(self) -> HealthService:
        """Get health service instance."""
        if not self._health_service:
            cache = await self.get_cache_service()
            repository = await self.get_product_repository()
            self._health_service = HealthService(cache, repository)
        return self._health_service
    
    async def cleanup(self) -> None:
        """Cleanup all services."""
        if self._cache_service:
            await self._cache_service.close()


# Global service container
services = ServiceContainer() 