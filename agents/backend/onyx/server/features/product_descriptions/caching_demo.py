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
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from caching_manager import (
from typing import Any, List, Dict, Optional
import logging
"""
Comprehensive Caching Demo for Product Descriptions API

This demo showcases:
- Redis caching for distributed environments
- In-memory caching for fast access
- Hybrid caching strategies
- Cache warming and preloading
- Cache invalidation patterns
- Cache monitoring and analytics
- Static data caching
- Performance optimization with caching
"""



    CacheManager, CacheConfig, CacheStrategy, EvictionPolicy,
    StaticDataCache, CacheWarmingService, CacheMonitor,
    cached, cache_invalidate, get_cache_manager, close_cache_manager
)


# Pydantic models for demo
class ProductData(BaseModel):
    """Product data model"""
    id: str
    name: str
    description: str
    price: float
    category: str
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class CacheStatsResponse(BaseModel):
    """Cache statistics response"""
    hits: int
    misses: int
    sets: int
    deletes: int
    errors: int
    hit_rate: float
    total_requests: int
    strategy: str


class CacheWarmingRequest(BaseModel):
    """Cache warming request"""
    data_type: str
    key_pattern: str
    batch_size: int = 100


class StaticDataRequest(BaseModel):
    """Static data request"""
    key: str
    data: Dict[str, Any]
    ttl: int = 86400


# Mock data sources for demo
class MockDataSources:
    """Mock data sources for demonstration"""
    
    @staticmethod
    async def get_product_catalog() -> Dict[str, ProductData]:
        """Simulate product catalog data"""
        await asyncio.sleep(0.1)  # Simulate database query
        
        return {
            "prod_001": ProductData(
                id="prod_001",
                name="Premium Widget",
                description="High-quality widget for professional use",
                price=99.99,
                category="Electronics",
                tags=["premium", "professional", "electronics"]
            ),
            "prod_002": ProductData(
                id="prod_002",
                name="Basic Gadget",
                description="Simple gadget for everyday use",
                price=29.99,
                category="Home",
                tags=["basic", "home", "everyday"]
            ),
            "prod_003": ProductData(
                id="prod_003",
                name="Luxury Item",
                description="Exclusive luxury item for discerning customers",
                price=299.99,
                category="Luxury",
                tags=["luxury", "exclusive", "premium"]
            )
        }
    
    @staticmethod
    async def get_category_list() -> List[str]:
        """Simulate category list data"""
        await asyncio.sleep(0.05)  # Simulate database query
        return ["Electronics", "Home", "Luxury", "Sports", "Books", "Clothing"]
    
    @staticmethod
    async def get_user_preferences(user_id: str) -> Dict[str, Any]:
        """Simulate user preferences data"""
        await asyncio.sleep(0.02)  # Simulate database query
        return {
            "theme": "dark",
            "language": "en",
            "currency": "USD",
            "notifications": True,
            "last_login": datetime.now().isoformat()
        }
    
    @staticmethod
    async def get_configuration_data() -> Dict[str, Any]:
        """Simulate configuration data"""
        await asyncio.sleep(0.01)  # Simulate database query
        return {
            "api_version": "1.0.0",
            "features": {
                "caching": True,
                "analytics": True,
                "notifications": True
            },
            "limits": {
                "max_products": 1000,
                "max_categories": 50,
                "cache_ttl": 3600
            }
        }


# Cached service layer
class CachedProductService:
    """Service layer with caching for product operations"""
    
    def __init__(self, cache_manager: CacheManager):
        
    """__init__ function."""
self.cache_manager = cache_manager
        self.static_cache = StaticDataCache(cache_manager)
        self.warming_service = CacheWarmingService(cache_manager)
        self.monitor = CacheMonitor(cache_manager)
    
    @cached(ttl=300, key_prefix="product")
    async def get_product(self, product_id: str) -> Optional[ProductData]:
        """Get product with caching"""
        # Simulate database query
        await asyncio.sleep(0.1)
        
        catalog = await MockDataSources.get_product_catalog()
        return catalog.get(product_id)
    
    @cached(ttl=600, key_prefix="catalog")
    async def get_product_catalog(self) -> Dict[str, ProductData]:
        """Get full product catalog with caching"""
        return await MockDataSources.get_product_catalog()
    
    @cached(ttl=1800, key_prefix="categories")
    async def get_categories(self) -> List[str]:
        """Get categories with caching"""
        return await MockDataSources.get_category_list()
    
    @cached(ttl=120, key_prefix="user_prefs")
    async def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences with caching"""
        return await MockDataSources.get_user_preferences(user_id)
    
    @cached(ttl=3600, key_prefix="config")
    async def get_configuration(self) -> Dict[str, Any]:
        """Get configuration with caching"""
        return await MockDataSources.get_configuration_data()
    
    @cache_invalidate(keys=["catalog:*", "product:*"])
    async def update_product(self, product_id: str, product_data: ProductData) -> bool:
        """Update product and invalidate related cache"""
        # Simulate database update
        await asyncio.sleep(0.2)
        
        # Invalidate specific product cache
        await self.cache_manager.delete(f"product:{product_id}")
        
        return True
    
    async def warm_product_cache(self) -> Any:
        """Warm cache with product data"""
        await self.warming_service.warm_cache(
            MockDataSources.get_product_catalog,
            "catalog",
            batch_size=50
        )
    
    async def warm_category_cache(self) -> Any:
        """Warm cache with category data"""
        await self.warming_service.warm_cache(
            MockDataSources.get_category_list,
            "categories",
            batch_size=10
        )
    
    async def cache_static_data(self, key: str, data: Dict[str, Any], ttl: int = 86400):
        """Cache static data"""
        return await self.static_cache.cache_static_data(key, data, ttl)
    
    async def get_static_data(self, key: str) -> Optional[Dict[str, Any]]:
        """Get static data from cache"""
        return await self.static_cache.get_static_data(key)
    
    async def get_cache_stats(self) -> CacheStatsResponse:
        """Get cache statistics"""
        stats = self.cache_manager.get_stats()
        return CacheStatsResponse(
            **stats,
            strategy=self.cache_manager.config.strategy.value
        )
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report"""
        return await self.monitor.get_performance_report()


# FastAPI application with caching
app = FastAPI(
    title="Product Descriptions API with Caching",
    description="Demonstrates advanced caching strategies",
    version="1.0.0"
)

# Global service instance
product_service: Optional[CachedProductService] = None


@app.on_event("startup")
async def startup_event():
    """Initialize cache and services on startup"""
    global product_service
    
    # Initialize cache manager with hybrid strategy
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
    product_service = CachedProductService(cache_manager)
    
    # Warm cache with initial data
    await product_service.warm_product_cache()
    await product_service.warm_category_cache()
    
    print("🚀 Cache initialized and warmed!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await close_cache_manager()
    print("🔒 Cache manager closed!")


# API Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Product Descriptions API with Advanced Caching",
        "version": "1.0.0",
        "features": [
            "Redis Caching",
            "In-Memory Caching", 
            "Hybrid Caching",
            "Cache Warming",
            "Cache Monitoring",
            "Static Data Caching"
        ]
    }


@app.get("/products/{product_id}")
async def get_product(product_id: str):
    """Get product by ID with caching"""
    if not product_service:
        raise HTTPException(status_code=500, detail="Service not initialized")
    
    product = await product_service.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    
    return product


@app.get("/products")
async def get_products():
    """Get all products with caching"""
    if not product_service:
        raise HTTPException(status_code=500, detail="Service not initialized")
    
    catalog = await product_service.get_product_catalog()
    return {"products": list(catalog.values())}


@app.get("/categories")
async def get_categories():
    """Get categories with caching"""
    if not product_service:
        raise HTTPException(status_code=500, detail="Service not initialized")
    
    categories = await product_service.get_categories()
    return {"categories": categories}


@app.get("/user/{user_id}/preferences")
async def get_user_preferences(user_id: str):
    """Get user preferences with caching"""
    if not product_service:
        raise HTTPException(status_code=500, detail="Service not initialized")
    
    preferences = await product_service.get_user_preferences(user_id)
    return {"user_id": user_id, "preferences": preferences}


@app.get("/configuration")
async def get_configuration():
    """Get configuration with caching"""
    if not product_service:
        raise HTTPException(status_code=500, detail="Service not initialized")
    
    config = await product_service.get_configuration()
    return config


@app.put("/products/{product_id}")
async def update_product(product_id: str, product_data: ProductData):
    """Update product and invalidate cache"""
    if not product_service:
        raise HTTPException(status_code=500, detail="Service not initialized")
    
    success = await product_service.update_product(product_id, product_data)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update product")
    
    return {"message": "Product updated successfully", "product_id": product_id}


@app.post("/cache/warm")
async def warm_cache(request: CacheWarmingRequest, background_tasks: BackgroundTasks):
    """Warm cache with data"""
    if not product_service:
        raise HTTPException(status_code=500, detail="Service not initialized")
    
    if request.data_type == "products":
        background_tasks.add_task(product_service.warm_product_cache)
    elif request.data_type == "categories":
        background_tasks.add_task(product_service.warm_category_cache)
    else:
        raise HTTPException(status_code=400, detail="Invalid data type")
    
    return {"message": f"Cache warming started for {request.data_type}"}


@app.post("/cache/static")
async def cache_static_data(request: StaticDataRequest):
    """Cache static data"""
    if not product_service:
        raise HTTPException(status_code=500, detail="Service not initialized")
    
    success = await product_service.cache_static_data(
        request.key, 
        request.data, 
        request.ttl
    )
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to cache static data")
    
    return {"message": "Static data cached successfully", "key": request.key}


@app.get("/cache/static/{key}")
async def get_static_data(key: str):
    """Get static data from cache"""
    if not product_service:
        raise HTTPException(status_code=500, detail="Service not initialized")
    
    data = await product_service.get_static_data(key)
    if not data:
        raise HTTPException(status_code=404, detail="Static data not found")
    
    return {"key": key, "data": data}


@app.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics"""
    if not product_service:
        raise HTTPException(status_code=500, detail="Service not initialized")
    
    return await product_service.get_cache_stats()


@app.get("/cache/performance")
async def get_performance_report():
    """Get cache performance report"""
    if not product_service:
        raise HTTPException(status_code=500, detail="Service not initialized")
    
    return await product_service.get_performance_report()


@app.delete("/cache/clear")
async def clear_cache():
    """Clear all cache"""
    if not product_service:
        raise HTTPException(status_code=500, detail="Service not initialized")
    
    success = await product_service.cache_manager.clear()
    if not success:
        raise HTTPException(status_code=500, detail="Failed to clear cache")
    
    return {"message": "Cache cleared successfully"}


@app.delete("/cache/static/{key}")
async def delete_static_data(key: str):
    """Delete static data from cache"""
    if not product_service:
        raise HTTPException(status_code=500, detail="Service not initialized")
    
    success = await product_service.static_cache.invalidate_static_data(key)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete static data")
    
    return {"message": "Static data deleted successfully", "key": key}


# Performance testing endpoints
@app.get("/test/performance")
async def test_performance():
    """Test cache performance"""
    if not product_service:
        raise HTTPException(status_code=500, detail="Service not initialized")
    
    results = {
        "cached_requests": [],
        "uncached_requests": [],
        "cache_stats": None
    }
    
    # Test cached requests
    start_time = time.time()
    for i in range(5):
        product = await product_service.get_product("prod_001")
        results["cached_requests"].append({
            "iteration": i + 1,
            "product_id": "prod_001",
            "found": product is not None
        })
    cached_time = time.time() - start_time
    
    # Test uncached requests (simulate cache miss)
    await product_service.cache_manager.delete("product:prod_001")
    start_time = time.time()
    for i in range(5):
        product = await product_service.get_product("prod_001")
        results["uncached_requests"].append({
            "iteration": i + 1,
            "product_id": "prod_001",
            "found": product is not None
        })
    uncached_time = time.time() - start_time
    
    # Get cache stats
    results["cache_stats"] = await product_service.get_cache_stats()
    results["performance_comparison"] = {
        "cached_time": cached_time,
        "uncached_time": uncached_time,
        "speedup": uncached_time / cached_time if cached_time > 0 else 0
    }
    
    return results


@app.get("/test/cache-strategies")
async def test_cache_strategies():
    """Test different cache strategies"""
    strategies = [
        CacheStrategy.MEMORY,
        CacheStrategy.REDIS,
        CacheStrategy.HYBRID
    ]
    
    results = {}
    
    for strategy in strategies:
        # Create temporary cache manager for testing
        config = CacheConfig(strategy=strategy)
        temp_cache_manager = CacheManager(config)
        await temp_cache_manager.initialize()
        
        # Test performance
        start_time = time.time()
        for i in range(10):
            await temp_cache_manager.set(f"test_key_{i}", f"test_value_{i}")
            await temp_cache_manager.get(f"test_key_{i}")
        end_time = time.time()
        
        results[strategy.value] = {
            "total_time": end_time - start_time,
            "avg_time_per_operation": (end_time - start_time) / 20
        }
        
        await temp_cache_manager.close()
    
    return results


# Demo functions for manual testing
async def demo_basic_caching():
    """Demonstrate basic caching functionality"""
    print("\n=== Basic Caching Demo ===")
    
    # Initialize cache manager
    cache_manager = await get_cache_manager()
    
    # Test basic operations
    print("1. Setting cache value...")
    await cache_manager.set("demo_key", "demo_value", 60)
    
    print("2. Getting cache value...")
    value = await cache_manager.get("demo_key")
    print(f"   Retrieved: {value}")
    
    print("3. Checking if key exists...")
    exists = await cache_manager.exists("demo_key")
    print(f"   Exists: {exists}")
    
    print("4. Getting cache stats...")
    stats = cache_manager.get_stats()
    print(f"   Stats: {stats}")
    
    print("5. Deleting cache value...")
    await cache_manager.delete("demo_key")
    
    print("6. Getting deleted value...")
    value = await cache_manager.get("demo_key")
    print(f"   Retrieved: {value}")


async def demo_static_data_caching():
    """Demonstrate static data caching"""
    print("\n=== Static Data Caching Demo ===")
    
    cache_manager = await get_cache_manager()
    static_cache = StaticDataCache(cache_manager)
    
    # Cache static configuration
    config_data = {
        "api_version": "1.0.0",
        "features": ["caching", "monitoring", "analytics"],
        "limits": {"max_products": 1000, "cache_ttl": 3600}
    }
    
    print("1. Caching static configuration...")
    await static_cache.cache_static_data("app_config", config_data, 86400)
    
    print("2. Retrieving static configuration...")
    retrieved_config = await static_cache.get_static_data("app_config")
    print(f"   Config: {retrieved_config}")
    
    print("3. Invalidating static data...")
    await static_cache.invalidate_static_data("app_config")


async def demo_cache_warming():
    """Demonstrate cache warming"""
    print("\n=== Cache Warming Demo ===")
    
    cache_manager = await get_cache_manager()
    warming_service = CacheWarmingService(cache_manager)
    
    async def mock_data_source():
        
    """mock_data_source function."""
return {"key1": "value1", "key2": "value2", "key3": "value3"}
    
    print("1. Warming cache with mock data...")
    await warming_service.warm_cache(mock_data_source, "demo_data", 10)
    
    print("2. Checking warmed data...")
    for i in range(1, 4):
        value = await cache_manager.get(f"demo_data:key{i}")
        print(f"   key{i}: {value}")


async def demo_cache_monitoring():
    """Demonstrate cache monitoring"""
    print("\n=== Cache Monitoring Demo ===")
    
    cache_manager = await get_cache_manager()
    monitor = CacheMonitor(cache_manager)
    
    # Generate some activity
    for i in range(10):
        await cache_manager.set(f"monitor_key_{i}", f"monitor_value_{i}")
        await cache_manager.get(f"monitor_key_{i}")
    
    print("1. Getting performance report...")
    report = await monitor.get_performance_report()
    print(f"   Report: {json.dumps(report, indent=2, default=str)}")


async def demo_hybrid_caching():
    """Demonstrate hybrid caching strategy"""
    print("\n=== Hybrid Caching Demo ===")
    
    # Create hybrid cache manager
    config = CacheConfig(strategy=CacheStrategy.HYBRID)
    cache_manager = await get_cache_manager(config)
    
    print("1. Setting value in hybrid cache...")
    await cache_manager.set("hybrid_key", "hybrid_value", 300)
    
    print("2. Getting value from hybrid cache...")
    value = await cache_manager.get("hybrid_key")
    print(f"   Retrieved: {value}")
    
    print("3. Getting hybrid cache stats...")
    stats = cache_manager.get_stats()
    print(f"   Stats: {stats}")


async def run_comprehensive_demo():
    """Run comprehensive caching demo"""
    print("🚀 Starting Comprehensive Caching Demo")
    print("=" * 50)
    
    try:
        await demo_basic_caching()
        await demo_static_data_caching()
        await demo_cache_warming()
        await demo_cache_monitoring()
        await demo_hybrid_caching()
        
        print("\n✅ All demos completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
    
    finally:
        await close_cache_manager()
        print("🔒 Cache manager closed")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(run_comprehensive_demo()) 