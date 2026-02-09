from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import time
import json
import random
from typing import List, Dict, Any, AsyncIterator
from pathlib import Path
from core.advanced_lazy_loader import (
from types.optimized_schemas import (
from core.optimized_engine import generate_caption_optimized
import logging
from typing import Any, List, Dict, Optional
"""
Lazy Loading Demo for Instagram Captions API v14.0

This demo showcases advanced lazy loading techniques for:
- Large dataset streaming
- Substantial API response handling
- Memory-efficient pagination
- Chunked data loading
- Background prefetching
- Streaming response generation
"""


# Import lazy loading components
    AdvancedLazyLoader, LargeDataConfig, DataSize, LoadStrategy,
    DataChunk, PageInfo, create_lazy_loader, lazy_load_large_dataset,
    stream_large_response, large_dataset_context
)

# Import schemas
    CaptionGenerationRequest, CaptionStyle, AudienceType, ContentType
)

# Import engine


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LazyLoadingDemo:
    """Demo class for showcasing lazy loading techniques"""
    
    def __init__(self) -> Any:
        # Create different lazy loaders for different data sizes
        self.small_loader = create_lazy_loader(DataSize.SMALL)
        self.medium_loader = create_lazy_loader(DataSize.MEDIUM)
        self.large_loader = create_lazy_loader(DataSize.LARGE, enable_disk_cache=True)
        self.huge_loader = create_lazy_loader(DataSize.HUGE, max_memory_mb=4096)
        
        # Demo data
        self.demo_content_descriptions = [
            "Beautiful sunset at the beach with golden colors",
            "Delicious homemade pizza with fresh ingredients",
            "Adventure hiking in the mountains with stunning views",
            "Cozy coffee shop with warm lighting and books",
            "Urban street art with vibrant colors and creativity",
            "Peaceful garden with blooming flowers and butterflies",
            "Modern city skyline at night with lights",
            "Rustic farmhouse with rolling hills in background",
            "Tropical beach with crystal clear water",
            "Snowy mountain peak with dramatic clouds"
        ]
    
    async def demo_streaming_captions(self) -> Any:
        """Demo streaming caption generation"""
        logger.info("🚀 Demo: Streaming Caption Generation")
        
        # Create a large batch of caption requests
        requests = []
        for i in range(50):  # Large dataset
            request = CaptionGenerationRequest(
                content_description=random.choice(self.demo_content_descriptions),
                style=random.choice(list(CaptionStyle)),
                audience=random.choice(list(AudienceType)),
                content_type=random.choice(list(ContentType)),
                hashtag_count=random.randint(10, 30),
                include_hashtags=True,
                include_emojis=True,
                max_length=2200,
                request_id=f"demo_request_{i}"
            )
            requests.append(request)
        
        logger.info(f"Created {len(requests)} caption requests for streaming demo")
        
        # Simulate streaming response
        async def stream_generator():
            
    """stream_generator function."""
yield "["
            first = True
            
            for i, request in enumerate(requests):
                if not first:
                    yield ","
                
                try:
                    # Simulate caption generation
                    await asyncio.sleep(0.1)  # Simulate processing time
                    
                    # Create mock response
                    stream_item = {
                        "request_id": request.request_id,
                        "variation_index": 0,
                        "caption": f"Amazing {request.content_description.lower()}! 🌟 #amazing #beautiful #instagram",
                        "hashtags": [f"hashtag_{j}" for j in range(request.hashtag_count)],
                        "quality_score": 85.0 + random.random() * 15,
                        "processing_time": 0.1 + random.random() * 0.2,
                        "timestamp": time.time(),
                        "style": request.style.value,
                        "audience": request.audience.value
                    }
                    
                    yield json.dumps(stream_item)
                    first = False
                    
                    # Progress indicator
                    if (i + 1) % 10 == 0:
                        logger.info(f"Streamed {i + 1}/{len(requests)} captions")
                
                except Exception as e:
                    logger.error(f"Error processing request {i}: {e}")
                    if not first:
                        yield ","
                    error_item = {
                        "error": "caption_generation_failed",
                        "request_id": request.request_id,
                        "message": str(e),
                        "timestamp": time.time()
                    }
                    yield json.dumps(error_item)
            
            yield "]"
        
        # Simulate streaming response
        logger.info("Starting streaming response...")
        start_time = time.time()
        
        stream_data = []
        async for chunk in stream_generator():
            stream_data.append(chunk)
        
        processing_time = time.time() - start_time
        logger.info(f"✅ Streaming completed in {processing_time:.2f} seconds")
        logger.info(f"Generated {len(stream_data)} chunks")
        
        return stream_data
    
    async def demo_paginated_data(self) -> Any:
        """Demo paginated data loading"""
        logger.info("📄 Demo: Paginated Data Loading")
        
        # Create mock data loader
        async def load_mock_data(offset: int = 0, limit: int = None, **kwargs) -> List[Dict[str, Any]]:
            """Mock data loader for pagination demo"""
            await asyncio.sleep(0.1)  # Simulate database query
            
            data = []
            for i in range(limit or 50):
                item_id = offset + i + 1
                data.append({
                    "id": f"caption_{item_id}",
                    "user_id": f"user_{item_id % 100}",
                    "caption": f"Generated caption {item_id} with amazing content",
                    "style": random.choice(list(CaptionStyle)).value,
                    "created_at": f"2024-01-{(item_id % 30) + 1:02d}T10:00:00Z",
                    "quality_score": 75.0 + random.random() * 25,
                    "engagement_score": 0.5 + random.random() * 0.5,
                    "hashtags": [f"tag_{j}" for j in range(random.randint(5, 20))],
                    "likes": random.randint(10, 1000),
                    "comments": random.randint(0, 100)
                })
            
            return data
        
        # Test pagination with different page sizes
        page_sizes = [10, 25, 50, 100]
        
        for page_size in page_sizes:
            logger.info(f"Testing pagination with page size: {page_size}")
            
            start_time = time.time()
            
            # Load first page
            data, page_info = await self.large_loader.load_paginated_data(
                loader_func=load_mock_data,
                page=1,
                page_size=page_size
            )
            
            processing_time = time.time() - start_time
            
            logger.info(f"  ✅ Page 1 loaded in {processing_time:.3f}s")
            logger.info(f"  📊 Items: {len(data)}, Total pages: {page_info.total_pages}")
            logger.info(f"  🔄 Has next: {page_info.has_next}, Has previous: {page_info.has_previous}")
            
            # Test cache hit for same page
            start_time = time.time()
            cached_data, cached_page_info = await self.large_loader.load_paginated_data(
                loader_func=load_mock_data,
                page=1,
                page_size=page_size
            )
            cache_time = time.time() - start_time
            
            logger.info(f"  ⚡ Cache hit in {cache_time:.3f}s ({(processing_time/cache_time):.1f}x faster)")
            
            # Load second page
            if page_info.has_next:
                start_time = time.time()
                data2, page_info2 = await self.large_loader.load_paginated_data(
                    loader_func=load_mock_data,
                    page=2,
                    page_size=page_size
                )
                processing_time2 = time.time() - start_time
                
                logger.info(f"  ✅ Page 2 loaded in {processing_time2:.3f}s")
                logger.info(f"  📊 Items: {len(data2)}")
        
        return True
    
    async def demo_chunked_loading(self) -> Any:
        """Demo chunked data loading"""
        logger.info("🧩 Demo: Chunked Data Loading")
        
        # Create mock chunk loader
        async def load_chunk(chunk_id: str, **kwargs) -> DataChunk:
            """Mock chunk loader"""
            await asyncio.sleep(0.2)  # Simulate chunk loading
            
            # Create mock data
            chunk_data = {
                "chunk_id": chunk_id,
                "items": [f"item_{i}" for i in range(1000)],  # 1000 items per chunk
                "metadata": {
                    "created_at": time.time(),
                    "size": random.randint(50000, 200000),  # 50KB - 200KB
                    "compression_ratio": random.uniform(0.3, 0.8)
                }
            }
            
            return DataChunk(
                id=chunk_id,
                data=chunk_data,
                size=len(str(chunk_data)),
                compressed=False,
                created_at=time.time(),
                accessed_at=time.time(),
                access_count=0
            )
        
        # Test different chunk loading strategies
        chunk_ids = [f"chunk_{i}" for i in range(10)]
        
        strategies = [
            LoadStrategy.STREAMING,
            LoadStrategy.BACKGROUND,
            LoadStrategy.IMMEDIATE
        ]
        
        for strategy in strategies:
            logger.info(f"Testing {strategy.value} strategy...")
            
            start_time = time.time()
            chunk_count = 0
            
            async for chunk in self.large_loader.load_chunked_data(
                loader_func=load_chunk,
                chunk_ids=chunk_ids,
                strategy=strategy
            ):
                chunk_count += 1
                logger.info(f"  📦 Loaded chunk {chunk.id} (size: {chunk.size} bytes)")
            
            processing_time = time.time() - start_time
            logger.info(f"  ✅ {strategy.value} strategy completed in {processing_time:.2f}s")
            logger.info(f"  📊 Loaded {chunk_count} chunks")
        
        return True
    
    async def demo_memory_management(self) -> Any:
        """Demo memory management and cleanup"""
        logger.info("🧠 Demo: Memory Management")
        
        # Get initial memory stats
        initial_stats = await self.large_loader.get_stats()
        logger.info(f"Initial memory usage: {initial_stats['memory_usage_mb']:.2f} MB")
        
        # Load many chunks to trigger memory management
        async def load_many_chunks():
            """Load many chunks to test memory management"""
            for i in range(20):
                chunk_id = f"memory_test_chunk_{i}"
                
                # Create large chunk
                large_data = {
                    "chunk_id": chunk_id,
                    "data": [f"large_item_{j}" for j in range(5000)],  # 5000 items
                    "metadata": {
                        "size": "large",
                        "created_at": time.time()
                    }
                }
                
                chunk = DataChunk(
                    id=chunk_id,
                    data=large_data,
                    size=len(str(large_data)),
                    compressed=False,
                    created_at=time.time(),
                    accessed_at=time.time(),
                    access_count=random.randint(1, 10)
                )
                
                # Add to loader
                self.large_loader.chunks[chunk_id] = chunk
                
                await asyncio.sleep(0.1)  # Small delay
        
        # Load chunks
        await load_many_chunks()
        
        # Check memory usage
        mid_stats = await self.large_loader.get_stats()
        logger.info(f"After loading chunks: {mid_stats['memory_usage_mb']:.2f} MB")
        logger.info(f"Active chunks: {mid_stats['active_chunks']}")
        
        # Trigger memory cleanup
        logger.info("Triggering memory cleanup...")
        await self.large_loader._cleanup_memory()
        
        # Check final stats
        final_stats = await self.large_loader.get_stats()
        logger.info(f"After cleanup: {final_stats['memory_usage_mb']:.2f} MB")
        logger.info(f"Active chunks: {final_stats['active_chunks']}")
        
        # Clear all chunks
        await self.large_loader.clear_cache()
        cleared_stats = await self.large_loader.get_stats()
        logger.info(f"After clearing cache: {cleared_stats['memory_usage_mb']:.2f} MB")
        logger.info(f"Active chunks: {cleared_stats['active_chunks']}")
        
        return True
    
    async def demo_batch_processing(self) -> Any:
        """Demo batch processing with lazy loading"""
        logger.info("⚡ Demo: Batch Processing with Lazy Loading")
        
        # Create large batch of requests
        batch_requests = []
        for i in range(100):
            request = CaptionGenerationRequest(
                content_description=f"Batch content {i} with amazing details",
                style=CaptionStyle.CASUAL,
                audience=AudienceType.GENERAL,
                content_type=ContentType.POST,
                hashtag_count=15,
                include_hashtags=True,
                include_emojis=True,
                max_length=2200,
                request_id=f"batch_request_{i}"
            )
            batch_requests.append(request)
        
        logger.info(f"Created batch of {len(batch_requests)} requests")
        
        # Process in chunks
        chunk_size = 20
        results = []
        
        start_time = time.time()
        
        for i in range(0, len(batch_requests), chunk_size):
            chunk = batch_requests[i:i + chunk_size]
            logger.info(f"Processing chunk {i//chunk_size + 1}/{(len(batch_requests) + chunk_size - 1)//chunk_size}")
            
            # Process chunk
            chunk_results = []
            for request in chunk:
                try:
                    # Simulate caption generation
                    await asyncio.sleep(0.05)  # Simulate processing
                    
                    result = {
                        "request_id": request.request_id,
                        "caption": f"Amazing {request.content_description.lower()}! ✨",
                        "hashtags": [f"batch_tag_{j}" for j in range(request.hashtag_count)],
                        "quality_score": 80.0 + random.random() * 20,
                        "success": True
                    }
                    chunk_results.append(result)
                
                except Exception as e:
                    result = {
                        "request_id": request.request_id,
                        "error": str(e),
                        "success": False
                    }
                    chunk_results.append(result)
            
            results.extend(chunk_results)
            
            # Progress indicator
            progress = (i + chunk_size) / len(batch_requests) * 100
            logger.info(f"  📊 Progress: {min(progress, 100):.1f}%")
        
        processing_time = time.time() - start_time
        
        # Calculate statistics
        successful = sum(1 for r in results if r.get("success", False))
        failed = len(results) - successful
        avg_quality = sum(r.get("quality_score", 0) for r in results if r.get("success", False)) / max(successful, 1)
        
        logger.info(f"✅ Batch processing completed in {processing_time:.2f} seconds")
        logger.info(f"📊 Results: {successful} successful, {failed} failed")
        logger.info(f"📈 Average quality score: {avg_quality:.1f}")
        logger.info(f"⚡ Processing rate: {len(results)/processing_time:.1f} requests/second")
        
        return results
    
    async def demo_performance_comparison(self) -> Any:
        """Demo performance comparison between different approaches"""
        logger.info("📊 Demo: Performance Comparison")
        
        # Test data sizes
        data_sizes = [100, 1000, 5000, 10000]
        
        results = {}
        
        for size in data_sizes:
            logger.info(f"Testing with {size} items...")
            
            # Test immediate loading
            start_time = time.time()
            immediate_data = [f"item_{i}" for i in range(size)]
            immediate_time = time.time() - start_time
            
            # Test lazy loading
            async def lazy_loader(offset: int = 0, limit: int = None, **kwargs):
                
    """lazy_loader function."""
await asyncio.sleep(0.1)  # Simulate loading delay
                return [f"item_{i}" for i in range(offset, offset + (limit or size))]
            
            start_time = time.time()
            lazy_data, page_info = await self.large_loader.load_paginated_data(
                loader_func=lazy_loader,
                page=1,
                page_size=size
            )
            lazy_time = time.time() - start_time
            
            # Test streaming
            async def stream_loader(chunk_id: str, **kwargs):
                
    """stream_loader function."""
await asyncio.sleep(0.1)
                return DataChunk(
                    id=chunk_id,
                    data=[f"item_{i}" for i in range(1000)],
                    size=1000,
                    compressed=False,
                    created_at=time.time(),
                    accessed_at=time.time(),
                    access_count=0
                )
            
            start_time = time.time()
            chunk_count = 0
            async for chunk in self.large_loader.load_chunked_data(
                loader_func=stream_loader,
                chunk_ids=[f"chunk_{i}" for i in range((size + 999) // 1000)],
                strategy=LoadStrategy.STREAMING
            ):
                chunk_count += 1
            streaming_time = time.time() - start_time
            
            results[size] = {
                "immediate": immediate_time,
                "lazy": lazy_time,
                "streaming": streaming_time,
                "memory_usage": len(immediate_data) * 100  # Rough estimate
            }
            
            logger.info(f"  ⚡ Immediate: {immediate_time:.3f}s")
            logger.info(f"  🐌 Lazy: {lazy_time:.3f}s")
            logger.info(f"  🌊 Streaming: {streaming_time:.3f}s")
            logger.info(f"  📦 Chunks: {chunk_count}")
        
        # Print comparison table
        logger.info("\n📊 Performance Comparison Summary:")
        logger.info("Size\tImmediate\tLazy\t\tStreaming\tMemory (KB)")
        logger.info("-" * 60)
        
        for size in data_sizes:
            result = results[size]
            logger.info(f"{size}\t{result['immediate']:.3f}s\t\t{result['lazy']:.3f}s\t\t{result['streaming']:.3f}s\t\t{result['memory_usage']}")
        
        return results
    
    async def run_all_demos(self) -> Any:
        """Run all lazy loading demos"""
        logger.info("🎬 Starting Lazy Loading Demo Suite")
        logger.info("=" * 60)
        
        demos = [
            ("Streaming Captions", self.demo_streaming_captions),
            ("Paginated Data", self.demo_paginated_data),
            ("Chunked Loading", self.demo_chunked_loading),
            ("Memory Management", self.demo_memory_management),
            ("Batch Processing", self.demo_batch_processing),
            ("Performance Comparison", self.demo_performance_comparison)
        ]
        
        results = {}
        
        for demo_name, demo_func in demos:
            try:
                logger.info(f"\n🎯 Running: {demo_name}")
                logger.info("-" * 40)
                
                start_time = time.time()
                result = await demo_func()
                demo_time = time.time() - start_time
                
                results[demo_name] = {
                    "success": True,
                    "time": demo_time,
                    "result": result
                }
                
                logger.info(f"✅ {demo_name} completed in {demo_time:.2f} seconds")
                
            except Exception as e:
                logger.error(f"❌ {demo_name} failed: {e}")
                results[demo_name] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("📋 Demo Summary")
        logger.info("=" * 60)
        
        successful_demos = sum(1 for r in results.values() if r["success"])
        total_demos = len(demos)
        
        logger.info(f"✅ Successful: {successful_demos}/{total_demos}")
        
        for demo_name, result in results.items():
            status = "✅" if result["success"] else "❌"
            time_info = f" ({result['time']:.2f}s)" if result["success"] else ""
            logger.info(f"{status} {demo_name}{time_info}")
        
        # Get final statistics
        final_stats = await self.large_loader.get_stats()
        logger.info(f"\n📊 Final Lazy Loader Statistics:")
        logger.info(f"  Memory Usage: {final_stats['memory_usage_mb']:.2f} MB")
        logger.info(f"  Cache Hits: {final_stats['cache_hits']}")
        logger.info(f"  Total Loads: {final_stats['total_chunks_loaded']}")
        logger.info(f"  Compression Savings: {final_stats['compression_savings'] / (1024*1024):.2f} MB")
        
        return results


async def main():
    """Main demo function"""
    demo = LazyLoadingDemo()
    
    try:
        results = await demo.run_all_demos()
        
        # Save results to file
        output_file = Path("lazy_loading_demo_results.json")
        with open(output_file, "w") as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\n💾 Demo results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


match __name__:
    case "__main__":
    asyncio.run(main()) 