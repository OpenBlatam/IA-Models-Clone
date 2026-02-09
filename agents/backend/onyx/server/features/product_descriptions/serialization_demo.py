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
from decimal import Decimal
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from pydantic_serialization import (
from typing import Any, List, Dict, Optional
import logging
"""
Comprehensive Pydantic Serialization Demo

This demo showcases:
- Multiple serialization strategies
- Performance optimizations
- Custom validators and field types
- Caching and streaming serialization
- Real-world usage scenarios
- Benchmarking and comparison
"""



    ProductDescription, ProductCategory, ProductTag, ProductImage, ProductVariant,
    SerializationStrategy, ValidationLevel, PydanticSerializer, StreamingSerializer,
    batch_serialize, batch_deserialize, get_global_serializer, clear_serializer_cache,
    SerializationExample, cached_serialization, timing_decorator
)


# Additional models for demo
class UserProfile(BaseModel):
    """User profile model for demo"""
    id: str = Field(..., description="User ID")
    email: str = Field(..., description="User email")
    name: str = Field(..., min_length=1, max_length=100, description="User name")
    age: int = Field(..., ge=0, le=150, description="User age")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")
    is_active: bool = Field(True, description="Active status")


class OrderItem(BaseModel):
    """Order item model for demo"""
    product_id: str = Field(..., description="Product ID")
    variant_id: str = Field(..., description="Variant ID")
    quantity: int = Field(..., gt=0, description="Quantity")
    unit_price: Decimal = Field(..., description="Unit price")
    total_price: Decimal = Field(..., description="Total price")
    
    @classmethod
    def calculate_total(cls, quantity: int, unit_price: Decimal) -> Decimal:
        """Calculate total price"""
        return quantity * unit_price


class Order(BaseModel):
    """Order model for demo"""
    id: str = Field(..., description="Order ID")
    user_id: str = Field(..., description="User ID")
    items: List[OrderItem] = Field(..., description="Order items")
    status: str = Field(..., description="Order status")
    total_amount: Decimal = Field(..., description="Total amount")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")
    updated_at: datetime = Field(default_factory=datetime.now, description="Update time")
    
    @classmethod
    def calculate_total_amount(cls, items: List[OrderItem]) -> Decimal:
        """Calculate total order amount"""
        return sum(item.total_price for item in items)


# Demo service with serialization features
class SerializationDemoService:
    """Service demonstrating serialization features"""
    
    def __init__(self) -> Any:
        self.serializers = {
            strategy: PydanticSerializer(strategy)
            for strategy in SerializationStrategy
        }
        self.streaming_serializer = StreamingSerializer()
    
    def create_sample_data(self) -> Dict[str, Any]:
        """Create sample data for testing"""
        # Create sample products
        products = []
        for i in range(10):
            category = ProductCategory(
                id=f"cat_{i:03d}",
                name=f"Category {i}",
                description=f"Description for category {i}",
                sort_order=i
            )
            
            tags = [
                ProductTag(id=f"tag_{i}_1", name=f"tag{i}_1", color="#FF0000"),
                ProductTag(id=f"tag_{i}_2", name=f"tag{i}_2", color="#00FF00")
            ]
            
            images = [
                ProductImage(
                    id=f"img_{i}_1",
                    url=f"https://example.com/product{i}_1.jpg",
                    alt_text=f"Product {i} image 1",
                    width=800,
                    height=600,
                    is_primary=True
                )
            ]
            
            variants = [
                ProductVariant(
                    id=f"var_{i}_1",
                    sku=f"SKU-{i:03d}-001",
                    name=f"Variant {i} 1",
                    price=Decimal(f"{99.99 + i}"),
                    stock_quantity=100 - i
                ),
                ProductVariant(
                    id=f"var_{i}_2",
                    sku=f"SKU-{i:03d}-002",
                    name=f"Variant {i} 2",
                    price=Decimal(f"{149.99 + i}"),
                    stock_quantity=50 - i
                )
            ]
            
            product = ProductDescription(
                id=f"prod_{i:03d}",
                title=f"Product {i}",
                short_description=f"Short description for product {i}",
                long_description=f"Long detailed description for product {i}...",
                category=category,
                tags=tags,
                images=images,
                variants=variants
            )
            products.append(product)
        
        # Create sample users
        users = []
        for i in range(5):
            user = UserProfile(
                id=f"user_{i:03d}",
                email=f"user{i}@example.com",
                name=f"User {i}",
                age=25 + i,
                preferences={
                    "theme": "dark" if i % 2 == 0 else "light",
                    "language": "en",
                    "notifications": True
                }
            )
            users.append(user)
        
        # Create sample orders
        orders = []
        for i in range(3):
            items = [
                OrderItem(
                    product_id=f"prod_{i:03d}",
                    variant_id=f"var_{i}_1",
                    quantity=2,
                    unit_price=Decimal(f"{99.99 + i}"),
                    total_price=Decimal(f"{199.98 + 2*i}")
                )
            ]
            
            order = Order(
                id=f"order_{i:03d}",
                user_id=f"user_{i:03d}",
                items=items,
                status="pending",
                total_amount=Order.calculate_total_amount(items)
            )
            orders.append(order)
        
        return {
            "products": products,
            "users": users,
            "orders": orders
        }
    
    @timing_decorator
    def benchmark_serialization_strategies(self, data: Any) -> Dict[str, float]:
        """Benchmark different serialization strategies"""
        results = {}
        
        for strategy in SerializationStrategy:
            serializer = self.serializers[strategy]
            
            # Warm up
            for _ in range(10):
                serializer.serialize(data, strategy)
            
            # Benchmark
            start_time = time.time()
            for _ in range(100):
                result = serializer.serialize(data, strategy)
            end_time = time.time()
            
            total_time = end_time - start_time
            avg_time = total_time / 100
            
            results[strategy.value] = {
                "total_time": total_time,
                "avg_time": avg_time,
                "ops_per_second": 100 / total_time
            }
        
        return results
    
    @cached_serialization(ttl=300)
    async def get_cached_serialized_data(self, data: Any, strategy: SerializationStrategy) -> Optional[Dict[str, Any]]:
        """Get cached serialized data"""
        serializer = self.serializers[strategy]
        return serializer.serialize(data, strategy)
    
    async def stream_serialize_data(self, data_list: List[Any]) -> List[bytes]:
        """Stream serialize a list of data"""
        chunks = []
        async for chunk in self.streaming_serializer.serialize_stream(data_list):
            chunks.append(chunk)
        return chunks
    
    async def stream_deserialize_data(self, data_stream: List[bytes], model_class: type) -> List[Any]:
        """Stream deserialize data"""
        models = []
        async for model in self.streaming_serializer.deserialize_stream(
            self._mock_data_stream(data_stream), model_class
        ):
            models.append(model)
        return models
    
    def _mock_data_stream(self, data_list: List[bytes]):
        """Mock async generator for data stream"""
        async def generator():
            
    """generator function."""
for data in data_list:
                yield data
        return generator()
    
    def validate_data_integrity(self, original: Any, deserialized: Any) -> bool:
        """Validate that deserialized data matches original"""
        # Convert both to dict for comparison
        if hasattr(original, 'model_dump'):
            original_dict = original.model_dump()
        else:
            original_dict = original
        
        if hasattr(deserialized, 'model_dump'):
            deserialized_dict = deserialized.model_dump()
        else:
            deserialized_dict = deserialized
        
        return original_dict == deserialized_dict


# FastAPI application with serialization features
app = FastAPI(
    title="Pydantic Serialization Demo API",
    description="Demonstrates optimized serialization with Pydantic",
    version="1.0.0"
)

# Global service instance
demo_service: Optional[SerializationDemoService] = None


@app.on_event("startup")
async def startup_event():
    """Initialize demo service on startup"""
    global demo_service
    demo_service = SerializationDemoService()
    print("🚀 Serialization demo service initialized!")


# API Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Pydantic Serialization Demo API",
        "version": "1.0.0",
        "features": [
            "Multiple Serialization Strategies",
            "Performance Optimization",
            "Custom Validators",
            "Caching",
            "Streaming Serialization",
            "Batch Operations"
        ]
    }


@app.get("/sample-data")
async def get_sample_data():
    """Get sample data for testing"""
    if not demo_service:
        raise HTTPException(status_code=500, detail="Service not initialized")
    
    return demo_service.create_sample_data()


@app.post("/serialize")
async def serialize_data(
    data: Dict[str, Any],
    strategy: SerializationStrategy = SerializationStrategy.ORJSON,
    include_none: bool = False,
    exclude_defaults: bool = False
):
    """Serialize data using specified strategy"""
    if not demo_service:
        raise HTTPException(status_code=500, detail="Service not initialized")
    
    # Create a sample product from the data
    try:
        product = ProductDescription.model_validate(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid data: {e}")
    
    serializer = demo_service.serializers[strategy]
    result = serializer.serialize(
        product,
        strategy,
        include_none=include_none,
        exclude_defaults=exclude_defaults
    )
    
    return {
        "strategy": strategy.value,
        "result": result,
        "size_bytes": len(str(result).encode()) if isinstance(result, str) else len(result)
    }


@app.post("/deserialize")
async def deserialize_data(
    data: Dict[str, Any],
    validation_level: ValidationLevel = ValidationLevel.STRICT
):
    """Deserialize data to Pydantic model"""
    if not demo_service:
        raise HTTPException(status_code=500, detail="Service not initialized")
    
    serializer = PydanticSerializer()
    
    try:
        result = serializer.deserialize(data, ProductDescription, validation_level)
        return {
            "success": True,
            "model": result.model_dump(),
            "validation_level": validation_level.value
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "validation_level": validation_level.value
        }


@app.get("/benchmark")
async def run_benchmark():
    """Run serialization benchmark"""
    if not demo_service:
        raise HTTPException(status_code=500, detail="Service not initialized")
    
    # Create sample data
    sample_data = demo_service.create_sample_data()
    product = sample_data["products"][0]  # Use first product for benchmark
    
    # Run benchmark
    results = demo_service.benchmark_serialization_strategies(product)
    
    return {
        "benchmark_results": results,
        "test_data": "ProductDescription model",
        "iterations": 100
    }


@app.post("/batch-serialize")
async def batch_serialize_data(
    data_list: List[Dict[str, Any]],
    strategy: SerializationStrategy = SerializationStrategy.ORJSON
):
    """Batch serialize multiple data items"""
    if not demo_service:
        raise HTTPException(status_code=500, detail="Service not initialized")
    
    # Convert dicts to models
    models = []
    for data in data_list:
        try:
            model = ProductDescription.model_validate(data)
            models.append(model)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid data: {e}")
    
    # Batch serialize
    start_time = time.time()
    results = batch_serialize(models, strategy)
    end_time = time.time()
    
    return {
        "strategy": strategy.value,
        "count": len(results),
        "total_time": end_time - start_time,
        "avg_time_per_item": (end_time - start_time) / len(results),
        "results": results[:5]  # Return first 5 for preview
    }


@app.post("/batch-deserialize")
async def batch_deserialize_data(
    data_list: List[Dict[str, Any]],
    validation_level: ValidationLevel = ValidationLevel.STRICT
):
    """Batch deserialize multiple data items"""
    if not demo_service:
        raise HTTPException(status_code=500, detail="Service not initialized")
    
    serializer = PydanticSerializer()
    results = []
    errors = []
    
    start_time = time.time()
    for i, data in enumerate(data_list):
        try:
            model = serializer.deserialize(data, ProductDescription, validation_level)
            results.append(model.model_dump())
        except Exception as e:
            errors.append({"index": i, "error": str(e)})
    end_time = time.time()
    
    return {
        "validation_level": validation_level.value,
        "total_count": len(data_list),
        "successful_count": len(results),
        "error_count": len(errors),
        "total_time": end_time - start_time,
        "avg_time_per_item": (end_time - start_time) / len(data_list),
        "errors": errors[:5]  # Return first 5 errors for preview
    }


@app.post("/stream-serialize")
async def stream_serialize_data(
    data_list: List[Dict[str, Any]],
    strategy: SerializationStrategy = SerializationStrategy.ORJSON
):
    """Stream serialize data"""
    if not demo_service:
        raise HTTPException(status_code=500, detail="Service not initialized")
    
    # Convert dicts to models
    models = []
    for data in data_list:
        try:
            model = ProductDescription.model_validate(data)
            models.append(model)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid data: {e}")
    
    # Stream serialize
    start_time = time.time()
    chunks = await demo_service.stream_serialize_data(models)
    end_time = time.time()
    
    return {
        "strategy": strategy.value,
        "chunk_count": len(chunks),
        "total_time": end_time - start_time,
        "chunks": [chunk.decode()[:100] + "..." for chunk in chunks[:3]]  # Preview first 3 chunks
    }


@app.get("/cached-serialization")
async def test_cached_serialization(
    strategy: SerializationStrategy = SerializationStrategy.CACHED
):
    """Test cached serialization"""
    if not demo_service:
        raise HTTPException(status_code=500, detail="Service not initialized")
    
    # Create sample data
    sample_data = demo_service.create_sample_data()
    product = sample_data["products"][0]
    
    # First call (should cache)
    start_time = time.time()
    result1 = await demo_service.get_cached_serialized_data(product, strategy)
    time1 = time.time() - start_time
    
    # Second call (should use cache)
    start_time = time.time()
    result2 = await demo_service.get_cached_serialized_data(product, strategy)
    time2 = time.time() - start_time
    
    return {
        "strategy": strategy.value,
        "first_call_time": time1,
        "second_call_time": time2,
        "speedup": time1 / time2 if time2 > 0 else float('inf'),
        "results_match": result1 == result2
    }


@app.get("/validation-levels")
async def test_validation_levels():
    """Test different validation levels"""
    if not demo_service:
        raise HTTPException(status_code=500, detail="Service not initialized")
    
    # Create sample data with some invalid fields
    invalid_data = {
        "id": "prod_001",
        "title": "",  # Invalid: empty title
        "category": {
            "id": "cat_001",
            "name": "Electronics",
            "description": "Electronic devices",
            "sort_order": -1  # Invalid: negative sort order
        },
        "variants": [
            {
                "id": "var_001",
                "sku": "SKU-001",
                "name": "Variant 1",
                "price": "invalid_price",  # Invalid: not a number
                "stock_quantity": 50
            }
        ]
    }
    
    results = {}
    serializer = PydanticSerializer()
    
    for level in ValidationLevel:
        start_time = time.time()
        try:
            result = serializer.deserialize(invalid_data, ProductDescription, level)
            success = True
            error = None
        except Exception as e:
            success = False
            error = str(e)
        end_time = time.time()
        
        results[level.value] = {
            "success": success,
            "error": error,
            "time": end_time - start_time
        }
    
    return results


@app.delete("/clear-cache")
async def clear_cache():
    """Clear serialization cache"""
    clear_serializer_cache()
    return {"message": "Serialization cache cleared successfully"}


# Demo functions
async def demonstrate_serialization_features():
    """Demonstrate all serialization features"""
    print("\n=== Pydantic Serialization Demo ===")
    
    demo_service = SerializationDemoService()
    
    try:
        # 1. Create sample data
        print("1. Creating sample data...")
        sample_data = demo_service.create_sample_data()
        product = sample_data["products"][0]
        print(f"   ✅ Created {len(sample_data['products'])} products")
        
        # 2. Test different serialization strategies
        print("\n2. Testing serialization strategies...")
        for strategy in SerializationStrategy:
            serializer = demo_service.serializers[strategy]
            result = serializer.serialize(product, strategy)
            size = len(str(result).encode()) if isinstance(result, str) else len(result)
            print(f"   ✅ {strategy.value:12} - Size: {size:6} bytes")
        
        # 3. Benchmark performance
        print("\n3. Running performance benchmark...")
        benchmark_results = demo_service.benchmark_serialization_strategies(product)
        for strategy, results in benchmark_results.items():
            print(f"   ✅ {strategy:12} - {results['avg_time']*1000:6.2f}ms per op")
        
        # 4. Test batch operations
        print("\n4. Testing batch operations...")
        products = sample_data["products"][:5]
        batch_results = batch_serialize(products, SerializationStrategy.ORJSON)
        print(f"   ✅ Batch serialized {len(batch_results)} products")
        
        # 5. Test validation levels
        print("\n5. Testing validation levels...")
        for level in ValidationLevel:
            serializer = PydanticSerializer()
            try:
                result = serializer.deserialize(
                    product.model_dump(), ProductDescription, level
                )
                print(f"   ✅ {level.value:10} validation - Success")
            except Exception as e:
                print(f"   ❌ {level.value:10} validation - Failed: {e}")
        
        # 6. Test data integrity
        print("\n6. Testing data integrity...")
        serializer = PydanticSerializer()
        serialized = serializer.serialize(product, SerializationStrategy.ORJSON)
        deserialized = serializer.deserialize(serialized, ProductDescription)
        integrity_ok = demo_service.validate_data_integrity(product, deserialized)
        print(f"   ✅ Data integrity check: {'PASSED' if integrity_ok else 'FAILED'}")
        
        print("\n✅ All serialization features demonstrated!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")


async def demonstrate_streaming_serialization():
    """Demonstrate streaming serialization"""
    print("\n=== Streaming Serialization Demo ===")
    
    demo_service = SerializationDemoService()
    
    try:
        # Create sample data
        sample_data = demo_service.create_sample_data()
        products = sample_data["products"][:10]
        
        print(f"1. Streaming serialization of {len(products)} products...")
        start_time = time.time()
        chunks = await demo_service.stream_serialize_data(products)
        end_time = time.time()
        
        print(f"   ✅ Generated {len(chunks)} chunks in {end_time - start_time:.3f}s")
        
        print("2. Streaming deserialization...")
        start_time = time.time()
        deserialized_products = await demo_service.stream_deserialize_data(
            chunks, ProductDescription
        )
        end_time = time.time()
        
        print(f"   ✅ Deserialized {len(deserialized_products)} products in {end_time - start_time:.3f}s")
        
        # Verify integrity
        integrity_ok = all(
            demo_service.validate_data_integrity(original, deserialized)
            for original, deserialized in zip(products, deserialized_products)
        )
        print(f"   ✅ Data integrity: {'PASSED' if integrity_ok else 'FAILED'}")
        
        print("✅ Streaming serialization demo completed!")
        
    except Exception as e:
        print(f"❌ Streaming demo failed: {e}")


async def run_comprehensive_demo():
    """Run comprehensive serialization demo"""
    print("🚀 Starting Comprehensive Pydantic Serialization Demo")
    print("=" * 60)
    
    try:
        await demonstrate_serialization_features()
        await demonstrate_streaming_serialization()
        
        print("\n" + "=" * 60)
        print("✅ All demos completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo suite failed: {e}")
        raise


if __name__ == "__main__":
    # Run the comprehensive demo
    asyncio.run(run_comprehensive_demo()) 