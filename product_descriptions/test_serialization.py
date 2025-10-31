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
from typing import Dict, List, Any
from decimal import Decimal
from pydantic_serialization import (
from typing import Any, List, Dict, Optional
import logging
"""
Test script for the optimized Pydantic serialization system

This script tests:
- Different serialization strategies
- Custom field types and validators
- Validation levels
- Performance optimizations
- Caching functionality
- Streaming serialization
- Batch operations
"""


    ProductDescription, ProductCategory, ProductTag, ProductImage, ProductVariant,
    SerializationStrategy, ValidationLevel, PydanticSerializer, StreamingSerializer,
    batch_serialize, batch_deserialize, get_global_serializer, clear_serializer_cache,
    SerializationExample, cached_serialization, timing_decorator,
    EmailField, PhoneField, CurrencyField
)


async def test_serialization_strategies():
    """Test different serialization strategies"""
    print("\n=== Testing Serialization Strategies ===")
    
    # Create sample product
    product = SerializationExample.create_sample_product()
    
    strategies = [
        SerializationStrategy.STANDARD,
        SerializationStrategy.ORJSON,
        SerializationStrategy.COMPACT,
        SerializationStrategy.CACHED
    ]
    
    for strategy in strategies:
        print(f"\nTesting {strategy.value}...")
        
        try:
            serializer = PydanticSerializer(strategy)
            
            # Serialize
            start_time = time.time()
            serialized = serializer.serialize(product, strategy)
            serialize_time = time.time() - start_time
            
            # Deserialize
            start_time = time.time()
            deserialized = serializer.deserialize(serialized, ProductDescription)
            deserialize_time = time.time() - start_time
            
            # Check data integrity
            integrity_ok = product.model_dump() == deserialized.model_dump()
            
            # Get size
            if isinstance(serialized, bytes):
                size = len(serialized)
            else:
                size = len(str(serialized).encode())
            
            print(f"   ✅ Serialize: {serialize_time*1000:.2f}ms")
            print(f"   ✅ Deserialize: {deserialize_time*1000:.2f}ms")
            print(f"   ✅ Size: {size} bytes")
            print(f"   ✅ Integrity: {'PASSED' if integrity_ok else 'FAILED'}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")


async def test_custom_field_types():
    """Test custom field types and validators"""
    print("\n=== Testing Custom Field Types ===")
    
    # Test EmailField
    print("1. Testing EmailField...")
    try:
        email = EmailField("user@example.com")
        print(f"   ✅ Valid email: {email}")
        
        # Test invalid email
        try:
            invalid_email = EmailField("invalid-email")
            print(f"   ❌ Should have failed: {invalid_email}")
        except ValueError:
            print("   ✅ Invalid email correctly rejected")
            
    except Exception as e:
        print(f"   ❌ EmailField test failed: {e}")
    
    # Test PhoneField
    print("\n2. Testing PhoneField...")
    try:
        phone = PhoneField("+1 (555) 123-4567")
        print(f"   ✅ Valid phone: {phone}")
        
        # Test invalid phone
        try:
            invalid_phone = PhoneField("123")
            print(f"   ❌ Should have failed: {invalid_phone}")
        except ValueError:
            print("   ✅ Invalid phone correctly rejected")
            
    except Exception as e:
        print(f"   ❌ PhoneField test failed: {e}")
    
    # Test CurrencyField
    print("\n3. Testing CurrencyField...")
    try:
        currency = CurrencyField("99.99")
        print(f"   ✅ Valid currency: {currency}")
        
        # Test precision
        currency2 = CurrencyField(100.123)
        print(f"   ✅ Currency with precision: {currency2}")
        
        # Test invalid currency
        try:
            invalid_currency = CurrencyField("invalid")
            print(f"   ❌ Should have failed: {invalid_currency}")
        except ValueError:
            print("   ✅ Invalid currency correctly rejected")
            
    except Exception as e:
        print(f"   ❌ CurrencyField test failed: {e}")


async def test_validation_levels():
    """Test different validation levels"""
    print("\n=== Testing Validation Levels ===")
    
    # Create valid data
    valid_data = {
        "id": "prod_001",
        "title": "Test Product",
        "category": {
            "id": "cat_001",
            "name": "Electronics",
            "description": "Electronic devices",
            "sort_order": 1
        },
        "variants": [
            {
                "id": "var_001",
                "sku": "SKU-001",
                "name": "Variant 1",
                "price": "99.99",
                "stock_quantity": 50
            }
        ]
    }
    
    # Create invalid data
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
    
    levels = [
        ValidationLevel.NONE,
        ValidationLevel.BASIC,
        ValidationLevel.STRICT,
        ValidationLevel.COMPLETE
    ]
    
    for level in levels:
        print(f"\nTesting {level.value} validation...")
        
        serializer = PydanticSerializer()
        
        # Test valid data
        try:
            start_time = time.time()
            result = serializer.deserialize(valid_data, ProductDescription, level)
            valid_time = time.time() - start_time
            print(f"   ✅ Valid data: {valid_time*1000:.2f}ms")
        except Exception as e:
            print(f"   ❌ Valid data failed: {e}")
        
        # Test invalid data
        try:
            start_time = time.time()
            result = serializer.deserialize(invalid_data, ProductDescription, level)
            invalid_time = time.time() - start_time
            print(f"   ⚠️  Invalid data accepted: {invalid_time*1000:.2f}ms")
        except Exception as e:
            print(f"   ✅ Invalid data correctly rejected: {str(e)[:50]}...")


async def test_performance_optimizations():
    """Test performance optimizations"""
    print("\n=== Testing Performance Optimizations ===")
    
    # Create multiple products for testing
    products = []
    for i in range(100):
        category = ProductCategory(
            id=f"cat_{i:03d}",
            name=f"Category {i}",
            description=f"Description for category {i}",
            sort_order=i
        )
        
        variant = ProductVariant(
            id=f"var_{i:03d}",
            sku=f"SKU-{i:03d}",
            name=f"Variant {i}",
            price=Decimal(f"{99.99 + i}"),
            stock_quantity=100 - i
        )
        
        product = ProductDescription(
            id=f"prod_{i:03d}",
            title=f"Product {i}",
            short_description=f"Short description for product {i}",
            category=category,
            variants=[variant]
        )
        products.append(product)
    
    # Test batch serialization
    print("1. Testing batch serialization...")
    start_time = time.time()
    batch_results = batch_serialize(products, SerializationStrategy.ORJSON)
    batch_time = time.time() - start_time
    
    print(f"   ✅ Batch serialized {len(batch_results)} products in {batch_time:.3f}s")
    print(f"   ✅ Average time per product: {batch_time/len(products)*1000:.2f}ms")
    
    # Test individual serialization for comparison
    print("\n2. Testing individual serialization...")
    start_time = time.time()
    individual_results = []
    for product in products:
        serializer = PydanticSerializer(SerializationStrategy.ORJSON)
        result = serializer.serialize(product, SerializationStrategy.ORJSON)
        individual_results.append(result)
    individual_time = time.time() - start_time
    
    print(f"   ✅ Individual serialized {len(individual_results)} products in {individual_time:.3f}s")
    print(f"   ✅ Average time per product: {individual_time/len(products)*1000:.2f}ms")
    print(f"   ✅ Speedup: {individual_time/batch_time:.2f}x")
    
    # Test caching
    print("\n3. Testing caching...")
    cached_serializer = PydanticSerializer(SerializationStrategy.CACHED)
    
    # First call (should cache)
    start_time = time.time()
    cached_result1 = cached_serializer.serialize(products[0], SerializationStrategy.CACHED)
    cache_time1 = time.time() - start_time
    
    # Second call (should use cache)
    start_time = time.time()
    cached_result2 = cached_serializer.serialize(products[0], SerializationStrategy.CACHED)
    cache_time2 = time.time() - start_time
    
    print(f"   ✅ First call: {cache_time1*1000:.2f}ms")
    print(f"   ✅ Second call: {cache_time2*1000:.2f}ms")
    print(f"   ✅ Speedup: {cache_time1/cache_time2:.2f}x")
    print(f"   ✅ Results match: {cached_result1 == cached_result2}")


async def test_streaming_serialization():
    """Test streaming serialization"""
    print("\n=== Testing Streaming Serialization ===")
    
    # Create products for streaming
    products = []
    for i in range(50):
        category = ProductCategory(
            id=f"cat_{i:03d}",
            name=f"Category {i}",
            sort_order=i
        )
        
        variant = ProductVariant(
            id=f"var_{i:03d}",
            sku=f"SKU-{i:03d}",
            name=f"Variant {i}",
            price=Decimal(f"{99.99 + i}"),
            stock_quantity=50
        )
        
        product = ProductDescription(
            id=f"prod_{i:03d}",
            title=f"Product {i}",
            category=category,
            variants=[variant]
        )
        products.append(product)
    
    streaming_serializer = StreamingSerializer(chunk_size=10)
    
    # Test streaming serialization
    print("1. Testing streaming serialization...")
    start_time = time.time()
    chunks = []
    async for chunk in streaming_serializer.serialize_stream(products):
        chunks.append(chunk)
    stream_serialize_time = time.time() - start_time
    
    print(f"   ✅ Generated {len(chunks)} chunks in {stream_serialize_time:.3f}s")
    
    # Test streaming deserialization
    print("\n2. Testing streaming deserialization...")
    start_time = time.time()
    deserialized_products = []
    async for product in streaming_serializer.deserialize_stream(
        _mock_data_stream(chunks), ProductDescription
    ):
        deserialized_products.append(product)
    stream_deserialize_time = time.time() - start_time
    
    print(f"   ✅ Deserialized {len(deserialized_products)} products in {stream_deserialize_time:.3f}s")
    
    # Verify data integrity
    integrity_ok = len(products) == len(deserialized_products)
    if integrity_ok:
        for i, (original, deserialized) in enumerate(zip(products, deserialized_products)):
            if original.model_dump() != deserialized.model_dump():
                integrity_ok = False
                break
    
    print(f"   ✅ Data integrity: {'PASSED' if integrity_ok else 'FAILED'}")


def _mock_data_stream(data_list) -> Any:
    """Mock async generator for data stream"""
    async def generator():
        
    """generator function."""
for data in data_list:
            yield data
    return generator()


async def test_error_handling():
    """Test error handling scenarios"""
    print("\n=== Testing Error Handling ===")
    
    serializer = PydanticSerializer()
    
    # Test invalid JSON
    print("1. Testing invalid JSON...")
    try:
        result = serializer.deserialize("invalid json", ProductDescription)
        print("   ❌ Should have failed")
    except Exception as e:
        print(f"   ✅ Correctly handled invalid JSON: {str(e)[:50]}...")
    
    # Test missing required fields
    print("\n2. Testing missing required fields...")
    invalid_data = {
        "id": "prod_001",
        # Missing title, category, variants
    }
    
    try:
        result = serializer.deserialize(invalid_data, ProductDescription)
        print("   ❌ Should have failed")
    except Exception as e:
        print(f"   ✅ Correctly handled missing fields: {str(e)[:50]}...")
    
    # Test invalid field types
    print("\n3. Testing invalid field types...")
    invalid_types_data = {
        "id": 123,  # Should be string
        "title": "Test Product",
        "category": {
            "id": "cat_001",
            "name": "Electronics",
            "description": "Electronic devices",
            "sort_order": "invalid"  # Should be int
        },
        "variants": [
            {
                "id": "var_001",
                "sku": "SKU-001",
                "name": "Variant 1",
                "price": "invalid_price",  # Should be number
                "stock_quantity": 50
            }
        ]
    }
    
    try:
        result = serializer.deserialize(invalid_types_data, ProductDescription)
        print("   ❌ Should have failed")
    except Exception as e:
        print(f"   ✅ Correctly handled invalid types: {str(e)[:50]}...")


async def test_integration_scenarios():
    """Test real-world integration scenarios"""
    print("\n=== Testing Integration Scenarios ===")
    
    # Scenario 1: API Response Serialization
    print("1. Testing API response serialization...")
    product = SerializationExample.create_sample_product()
    
    # Simulate API response with different strategies
    strategies = [SerializationStrategy.ORJSON, SerializationStrategy.COMPACT]
    
    for strategy in strategies:
        serializer = PydanticSerializer(strategy)
        start_time = time.time()
        serialized = serializer.serialize(product, strategy)
        end_time = time.time()
        
        size = len(serialized) if isinstance(serialized, bytes) else len(str(serialized).encode())
        print(f"   ✅ {strategy.value:12} - {end_time*1000:6.2f}ms, {size:6} bytes")
    
    # Scenario 2: Database Storage
    print("\n2. Testing database storage serialization...")
    products = []
    for i in range(10):
        category = ProductCategory(
            id=f"cat_{i:03d}",
            name=f"Category {i}",
            sort_order=i
        )
        
        variant = ProductVariant(
            id=f"var_{i:03d}",
            sku=f"SKU-{i:03d}",
            name=f"Variant {i}",
            price=Decimal(f"{99.99 + i}"),
            stock_quantity=100
        )
        
        product = ProductDescription(
            id=f"prod_{i:03d}",
            title=f"Product {i}",
            category=category,
            variants=[variant]
        )
        products.append(product)
    
    # Serialize for storage
    serializer = PydanticSerializer(SerializationStrategy.COMPACT)
    storage_data = [serializer.serialize(p, SerializationStrategy.COMPACT) for p in products]
    
    print(f"   ✅ Serialized {len(storage_data)} products for storage")
    total_size = sum(len(str(d).encode()) for d in storage_data)
    print(f"   ✅ Total storage size: {total_size} bytes")
    
    # Scenario 3: Cache Integration
    print("\n3. Testing cache integration...")
    cached_serializer = PydanticSerializer(SerializationStrategy.CACHED)
    
    # Simulate cache hits and misses
    cache_times = []
    for i in range(5):
        start_time = time.time()
        result = cached_serializer.serialize(products[0], SerializationStrategy.CACHED)
        end_time = time.time()
        cache_times.append(end_time - start_time)
    
    print(f"   ✅ Cache times: {[f'{t*1000:.2f}ms' for t in cache_times]}")
    print(f"   ✅ Average cache time: {sum(cache_times)/len(cache_times)*1000:.2f}ms")


async def run_all_tests():
    """Run all serialization tests"""
    print("🚀 Starting Pydantic Serialization System Tests")
    print("=" * 60)
    
    try:
        await test_serialization_strategies()
        await test_custom_field_types()
        await test_validation_levels()
        await test_performance_optimizations()
        await test_streaming_serialization()
        await test_error_handling()
        await test_integration_scenarios()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        raise
    
    finally:
        # Cleanup
        clear_serializer_cache()
        print("🔒 Serializer cache cleared")


if __name__ == "__main__":
    # Run the test suite
    asyncio.run(run_all_tests()) 