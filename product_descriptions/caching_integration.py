from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from caching_manager import (
from typing import Any, List, Dict, Optional
import logging
"""
Caching Integration Example

This module demonstrates how to integrate the advanced caching system
with the existing Product Descriptions API.
"""



    CacheManager, CacheConfig, CacheStrategy,
    StaticDataCache, CacheWarmingService, CacheMonitor,
    cached, cache_invalidate, get_cache_manager
)


# Pydantic models
class ProductDescription(BaseModel):
    """Product description model"""
    id: str
    title: str
    description: str
    category: str
    tags: List[str] = []
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class CacheIntegrationService:
    """Service demonstrating caching integration"""
    
    def __init__(self, cache_manager: CacheManager):
        
    """__init__ function."""
self.cache_manager = cache_manager
        self.static_cache = StaticDataCache(cache_manager)
        self.warming_service = CacheWarmingService(cache_manager)
        self.monitor = CacheMonitor(cache_manager)
    
    @cached(ttl=300, key_prefix="product_desc")
    async def get_product_description(self, product_id: str) -> Optional[ProductDescription]:
        """Get product description with caching"""
        # Simulate database query
        await asyncio.sleep(0.1)
        
        # Mock data - in real app, this would be a database query
        descriptions = {
            "prod_001": ProductDescription(
                id="prod_001",
                title="Premium Widget",
                description="High-quality widget for professional use",
                category="Electronics",
                tags=["premium", "professional"]
            ),
            "prod_002": ProductDescription(
                id="prod_002",
                title="Basic Gadget",
                description="Simple gadget for everyday use",
                category="Home",
                tags=["basic", "home"]
            )
        }
        
        return descriptions.get(product_id)
    
    @cached(ttl=600, key_prefix="product_list")
    async def get_product_list(self, category: Optional[str] = None) -> List[ProductDescription]:
        """Get product list with caching"""
        await asyncio.sleep(0.2)
        
        all_products = [
            ProductDescription(
                id="prod_001",
                title="Premium Widget",
                description="High-quality widget for professional use",
                category="Electronics",
                tags=["premium", "professional"]
            ),
            ProductDescription(
                id="prod_002",
                title="Basic Gadget",
                description="Simple gadget for everyday use",
                category="Home",
                tags=["basic", "home"]
            ),
            ProductDescription(
                id="prod_003",
                title="Luxury Item",
                description="Exclusive luxury item for discerning customers",
                category="Luxury",
                tags=["luxury", "exclusive"]
            )
        ]
        
        if category:
            return [p for p in all_products if p.category.lower() == category.lower()]
        
        return all_products
    
    @cached(ttl=1800, key_prefix="categories")
    async def get_categories(self) -> List[str]:
        """Get categories with caching"""
        await asyncio.sleep(0.05)
        return ["Electronics", "Home", "Luxury", "Sports", "Books"]
    
    @cache_invalidate(keys=["product_desc:*", "product_list:*"])
    async def update_product_description(self, product_id: str, description: ProductDescription) -> bool:
        """Update product description and invalidate cache"""
        await asyncio.sleep(0.3)  # Simulate database update
        
        # Invalidate specific product cache
        await self.cache_manager.delete(f"product_desc:{product_id}")
        
        return True
    
    async def warm_product_cache(self) -> Any:
        """Warm cache with product data"""
        async def get_all_products():
            
    """get_all_products function."""
return await self.get_product_list()
        
        await self.warming_service.warm_cache(
            get_all_products,
            "product_list",
            batch_size=50
        )
    
    async def cache_static_config(self) -> Any:
        """Cache static configuration data"""
        config_data = {
            "api_version": "1.0.0",
            "features": {
                "caching": True,
                "monitoring": True,
                "warming": True
            },
            "limits": {
                "max_products": 1000,
                "cache_ttl": 3600
            }
        }
        
        await self.static_cache.cache_static_data("app_config", config_data, 86400)
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = self.cache_manager.get_stats()
        return {
            "cache_stats": stats,
            "strategy": self.cache_manager.config.strategy.value
        }
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report"""
        return await self.monitor.get_performance_report()


# FastAPI application with caching integration
app = FastAPI(
    title="Product Descriptions API with Caching",
    description="Demonstrates caching integration",
    version="1.0.0"
)

# Global service instance
service: Optional[CacheIntegrationService] = None


async def get_service() -> CacheIntegrationService:
    """Dependency to get service instance"""
    if service is None:
        raise HTTPException(status_code=500, detail="Service not initialized")
    return service


@app.on_event("startup")
async def startup_event():
    """Initialize cache and services on startup"""
    global service
    
    # Initialize cache manager
    cache_config = CacheConfig(
        strategy=CacheStrategy.HYBRID,
        redis_url="redis://localhost:6379",
        memory_max_size=500,
        memory_ttl=300,
        redis_ttl=3600,
        enable_stats=True,
        enable_warming=True
    )
    
    cache_manager = await get_cache_manager(cache_config)
    service = CacheIntegrationService(cache_manager)
    
    # Warm cache and cache static data
    await service.warm_product_cache()
    await service.cache_static_config()
    
    print("🚀 Cache integrated and warmed!")


# API Routes with caching
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Product Descriptions API with Caching Integration",
        "version": "1.0.0",
        "features": [
            "Redis Caching",
            "In-Memory Caching",
            "Cache Warming",
            "Cache Monitoring",
            "Static Data Caching"
        ]
    }


@app.get("/products/{product_id}")
async def get_product(product_id: str, svc: CacheIntegrationService = Depends(get_service)):
    """Get product description with caching"""
    product = await svc.get_product_description(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    
    return product


@app.get("/products")
async def get_products(
    category: Optional[str] = None,
    svc: CacheIntegrationService = Depends(get_service)
):
    """Get product list with caching"""
    products = await svc.get_product_list(category)
    return {"products": products}


@app.get("/categories")
async def get_categories(svc: CacheIntegrationService = Depends(get_service)):
    """Get categories with caching"""
    categories = await svc.get_categories()
    return {"categories": categories}


@app.put("/products/{product_id}")
async def update_product(
    product_id: str,
    description: ProductDescription,
    svc: CacheIntegrationService = Depends(get_service)
):
    """Update product description and invalidate cache"""
    success = await svc.update_product_description(product_id, description)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update product")
    
    return {"message": "Product updated successfully", "product_id": product_id}


@app.post("/cache/warm")
async def warm_cache(
    background_tasks: BackgroundTasks,
    svc: CacheIntegrationService = Depends(get_service)
):
    """Warm cache with product data"""
    background_tasks.add_task(svc.warm_product_cache)
    return {"message": "Cache warming started"}


@app.get("/cache/stats")
async def get_cache_stats(svc: CacheIntegrationService = Depends(get_service)):
    """Get cache statistics"""
    return await svc.get_cache_stats()


@app.get("/cache/performance")
async def get_performance_report(svc: CacheIntegrationService = Depends(get_service)):
    """Get cache performance report"""
    return await svc.get_performance_report()


@app.delete("/cache/clear")
async def clear_cache(svc: CacheIntegrationService = Depends(get_service)):
    """Clear all cache"""
    success = await svc.cache_manager.clear()
    if not success:
        raise HTTPException(status_code=500, detail="Failed to clear cache")
    
    return {"message": "Cache cleared successfully"}


# Example of middleware integration
@app.middleware("http")
async def cache_middleware(request, call_next) -> Any:
    """Example middleware for request caching"""
    svc = await get_service()
    
    # Only cache GET requests
    if request.method == "GET":
        cache_key = f"request:{request.url.path}:{request.query_params}"
        cached_response = await svc.cache_manager.get(cache_key)
        
        if cached_response:
            return cached_response
    
    # Process request
    response = await call_next(request)
    
    # Cache successful GET responses
    if request.method == "GET" and response.status_code == 200:
        cache_key = f"request:{request.url.path}:{request.query_params}"
        await svc.cache_manager.set(cache_key, response.body, ttl=300)
    
    return response


# Example usage functions
async def demonstrate_caching_integration():
    """Demonstrate caching integration features"""
    print("\n=== Caching Integration Demo ===")
    
    # Initialize cache manager
    cache_config = CacheConfig(strategy=CacheStrategy.HYBRID)
    cache_manager = await get_cache_manager(cache_config)
    service = CacheIntegrationService(cache_manager)
    
    try:
        # 1. Test product description caching
        print("1. Testing product description caching...")
        product = await service.get_product_description("prod_001")
        print(f"   Retrieved product: {product.title}")
        
        # Second call should be cached
        product_cached = await service.get_product_description("prod_001")
        print(f"   Retrieved cached product: {product_cached.title}")
        
        # 2. Test product list caching
        print("\n2. Testing product list caching...")
        products = await service.get_product_list()
        print(f"   Retrieved {len(products)} products")
        
        # 3. Test cache warming
        print("\n3. Testing cache warming...")
        await service.warm_product_cache()
        print("   Cache warming completed")
        
        # 4. Test static data caching
        print("\n4. Testing static data caching...")
        await service.cache_static_config()
        config = await service.static_cache.get_static_data("app_config")
        print(f"   Retrieved config: {config['api_version']}")
        
        # 5. Test cache invalidation
        print("\n5. Testing cache invalidation...")
        new_description = ProductDescription(
            id="prod_001",
            title="Updated Premium Widget",
            description="Updated high-quality widget",
            category="Electronics",
            tags=["premium", "updated"]
        )
        await service.update_product_description("prod_001", new_description)
        print("   Cache invalidation completed")
        
        # 6. Get cache statistics
        print("\n6. Getting cache statistics...")
        stats = await service.get_cache_stats()
        print(f"   Hit rate: {stats['cache_stats']['hit_rate']:.2%}")
        print(f"   Total requests: {stats['cache_stats']['total_requests']}")
        
        # 7. Get performance report
        print("\n7. Getting performance report...")
        report = await service.get_performance_report()
        print(f"   Alerts: {len(report['alerts'])}")
        print(f"   Recommendations: {len(report['recommendations'])}")
        
        print("\n✅ All caching integration features demonstrated!")
        
    finally:
        await cache_manager.close()


if __name__ == "__main__":
    # Run the integration demo
    asyncio.run(demonstrate_caching_integration()) 