from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import aiohttp
import json
import time
from typing import Dict, Any, List
import logging
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
NotebookLM AI - Production API Demo
🚀 Comprehensive demo of the production API with all features
⚡ Shows text, image, audio, vector processing and more
🎯 Production-ready examples with proper error handling
"""


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionAPIClient:
    """Client for the NotebookLM AI Production API."""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_token: str = "your-token-here"):
        
    """__init__ function."""
self.base_url = base_url.rstrip('/')
        self.api_token = api_token
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check API health."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/health", headers=self.headers) as response:
                return await response.json()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/stats", headers=self.headers) as response:
                return await response.json()
    
    async def process_text(self, text: str, operations: List[str] = None, model: str = "gpt-4") -> Dict[str, Any]:
        """Process text using advanced AI capabilities."""
        if operations is None:
            operations = ["all"]
        
        payload = {
            "text": text,
            "operations": operations,
            "model": model
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/process/text", json=payload, headers=self.headers) as response:
                return await response.json()
    
    async def process_image(self, image_path: str, operations: List[str] = None) -> Dict[str, Any]:
        """Process image using advanced computer vision."""
        if operations is None:
            operations = ["all"]
        
        payload = {
            "image_path": image_path,
            "operations": operations
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/process/image", json=payload, headers=self.headers) as response:
                return await response.json()
    
    async def process_audio(self, audio_path: str, operations: List[str] = None) -> Dict[str, Any]:
        """Process audio using advanced audio processing."""
        if operations is None:
            operations = ["all"]
        
        payload = {
            "audio_path": audio_path,
            "operations": operations
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/process/audio", json=payload, headers=self.headers) as response:
                return await response.json()
    
    async def vector_search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Perform vector search."""
        payload = {
            "query": query,
            "top_k": top_k
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/search/vector", json=payload, headers=self.headers) as response:
                return await response.json()
    
    async def process_batch(self, requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process multiple requests in batch."""
        payload = {
            "requests": requests
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/process/batch", json=payload, headers=self.headers) as response:
                return await response.json()
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/cache/stats", headers=self.headers) as response:
                return await response.json()
    
    async def clear_cache(self) -> Dict[str, Any]:
        """Clear all cache."""
        async with aiohttp.ClientSession() as session:
            async with session.delete(f"{self.base_url}/cache/clear", headers=self.headers) as response:
                return await response.json()

async def demo_health_and_stats(client: ProductionAPIClient):
    """Demo health check and statistics."""
    print("\n" + "="*60)
    print("🏥 HEALTH CHECK & STATISTICS")
    print("="*60)
    
    # Health check
    print("\n📊 Health Check:")
    try:
        health = await client.health_check()
        print(f"Status: {health.get('status', 'unknown')}")
        print(f"Version: {health.get('version', 'unknown')}")
        print(f"Environment: {health.get('environment', 'unknown')}")
        print(f"Components: {health.get('components', {})}")
    except Exception as e:
        print(f"Health check failed: {e}")
    
    # Performance stats
    print("\n📈 Performance Statistics:")
    try:
        stats = await client.get_stats()
        engine_stats = stats.get('engine_stats', {})
        print(f"Total Requests: {engine_stats.get('total_requests', 0)}")
        print(f"Success Rate: {engine_stats.get('success_requests', 0)}/{engine_stats.get('total_requests', 1)}")
        print(f"Average Processing Time: {engine_stats.get('avg_processing_time', 0):.3f}s")
    except Exception as e:
        print(f"Stats retrieval failed: {e}")

async def demo_text_processing(client: ProductionAPIClient):
    """Demo text processing capabilities."""
    print("\n" + "="*60)
    print("📝 TEXT PROCESSING")
    print("="*60)
    
    sample_texts = [
        "I love this product! It's amazing and works perfectly.",
        "This is terrible. Worst experience ever. Don't buy it.",
        "The weather is nice today and I'm feeling happy about it."
    ]
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\n🔤 Processing Text {i}:")
        print(f"Text: {text[:50]}...")
        
        try:
            start_time = time.time()
            result = await client.process_text(
                text=text,
                operations=["sentiment", "keywords", "summary"],
                model="gpt-4"
            )
            processing_time = time.time() - start_time
            
            print(f"✅ Success (took {processing_time:.3f}s)")
            print(f"Status: {result.get('status', 'unknown')}")
            print(f"From Cache: {result.get('from_cache', False)}")
            
            if result.get('status') == 'success':
                ai_result = result.get('result', {})
                print(f"Sentiment: {ai_result.get('sentiment', 'N/A')}")
                print(f"Keywords: {ai_result.get('keywords', [])[:3]}")
                print(f"Summary: {ai_result.get('summary', 'N/A')[:100]}...")
            
        except Exception as e:
            print(f"❌ Error: {e}")

async def demo_image_processing(client: ProductionAPIClient):
    """Demo image processing capabilities."""
    print("\n" + "="*60)
    print("🖼️ IMAGE PROCESSING")
    print("="*60)
    
    # Note: In a real scenario, you'd have actual image files
    sample_images = [
        "/path/to/image1.jpg",
        "/path/to/image2.png"
    ]
    
    for i, image_path in enumerate(sample_images, 1):
        print(f"\n🖼️ Processing Image {i}:")
        print(f"Path: {image_path}")
        
        try:
            start_time = time.time()
            result = await client.process_image(
                image_path=image_path,
                operations=["face_detection", "object_detection", "caption"]
            )
            processing_time = time.time() - start_time
            
            print(f"✅ Success (took {processing_time:.3f}s)")
            print(f"Status: {result.get('status', 'unknown')}")
            
            if result.get('status') == 'success':
                ai_result = result.get('result', {})
                print(f"Faces Detected: {ai_result.get('faces_detected', 0)}")
                print(f"Objects: {ai_result.get('objects', [])[:3]}")
                print(f"Caption: {ai_result.get('caption', 'N/A')}")
            
        except Exception as e:
            print(f"❌ Error: {e}")

async def demo_audio_processing(client: ProductionAPIClient):
    """Demo audio processing capabilities."""
    print("\n" + "="*60)
    print("🎵 AUDIO PROCESSING")
    print("="*60)
    
    # Note: In a real scenario, you'd have actual audio files
    sample_audio = [
        "/path/to/audio1.wav",
        "/path/to/audio2.mp3"
    ]
    
    for i, audio_path in enumerate(sample_audio, 1):
        print(f"\n🎵 Processing Audio {i}:")
        print(f"Path: {audio_path}")
        
        try:
            start_time = time.time()
            result = await client.process_audio(
                audio_path=audio_path,
                operations=["transcription", "sentiment", "speaker_detection"]
            )
            processing_time = time.time() - start_time
            
            print(f"✅ Success (took {processing_time:.3f}s)")
            print(f"Status: {result.get('status', 'unknown')}")
            
            if result.get('status') == 'success':
                ai_result = result.get('result', {})
                print(f"Transcription: {ai_result.get('transcription', 'N/A')[:100]}...")
                print(f"Sentiment: {ai_result.get('sentiment', 'N/A')}")
                print(f"Speakers: {ai_result.get('speakers', 0)}")
            
        except Exception as e:
            print(f"❌ Error: {e}")

async def demo_vector_search(client: ProductionAPIClient):
    """Demo vector search capabilities."""
    print("\n" + "="*60)
    print("🔍 VECTOR SEARCH")
    print("="*60)
    
    sample_queries = [
        "machine learning algorithms",
        "natural language processing",
        "computer vision applications"
    ]
    
    for i, query in enumerate(sample_queries, 1):
        print(f"\n🔍 Vector Search {i}:")
        print(f"Query: {query}")
        
        try:
            start_time = time.time()
            result = await client.vector_search(query=query, top_k=3)
            processing_time = time.time() - start_time
            
            print(f"✅ Success (took {processing_time:.3f}s)")
            print(f"Status: {result.get('status', 'unknown')}")
            
            if result.get('status') == 'success':
                search_results = result.get('result', [])
                print(f"Found {len(search_results)} results:")
                for j, doc in enumerate(search_results[:3], 1):
                    print(f"  {j}. {doc.get('title', 'No title')} (score: {doc.get('score', 0):.3f})")
            
        except Exception as e:
            print(f"❌ Error: {e}")

async def demo_batch_processing(client: ProductionAPIClient):
    """Demo batch processing capabilities."""
    print("\n" + "="*60)
    print("📦 BATCH PROCESSING")
    print("="*60)
    
    batch_requests = [
        {
            "type": "text",
            "text": "First text for processing",
            "operations": ["sentiment"],
            "model": "gpt-4"
        },
        {
            "type": "text",
            "text": "Second text for processing",
            "operations": ["keywords"],
            "model": "gpt-4"
        },
        {
            "type": "vector",
            "query": "artificial intelligence",
            "top_k": 2
        }
    ]
    
    print(f"\n📦 Processing {len(batch_requests)} requests in batch:")
    
    try:
        start_time = time.time()
        result = await client.process_batch(batch_requests)
        processing_time = time.time() - start_time
        
        print(f"✅ Success (took {processing_time:.3f}s)")
        print(f"Status: {result.get('status', 'unknown')}")
        print(f"Total Requests: {result.get('total_requests', 0)}")
        
        if result.get('status') == 'success':
            results = result.get('results', [])
            print(f"Processed {len(results)} results:")
            for i, res in enumerate(results, 1):
                status = res.get('status', 'unknown')
                print(f"  {i}. {status}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

async def demo_cache_management(client: ProductionAPIClient):
    """Demo cache management capabilities."""
    print("\n" + "="*60)
    print("💾 CACHE MANAGEMENT")
    print("="*60)
    
    # Get cache stats
    print("\n📊 Cache Statistics:")
    try:
        cache_stats = await client.get_cache_stats()
        local_cache_size = cache_stats.get('local_cache_size', 0)
        cache_stats_data = cache_stats.get('cache_stats', {})
        
        print(f"Local Cache Size: {local_cache_size}")
        print(f"Cache Hits: {cache_stats_data.get('local_hits', 0)}")
        print(f"Cache Misses: {cache_stats_data.get('misses', 0)}")
        print(f"Cache Sets: {cache_stats_data.get('sets', 0)}")
        
        # Calculate hit rate
        hits = cache_stats_data.get('local_hits', 0)
        misses = cache_stats_data.get('misses', 0)
        total = hits + misses
        hit_rate = (hits / total * 100) if total > 0 else 0
        print(f"Cache Hit Rate: {hit_rate:.1f}%")
        
    except Exception as e:
        print(f"❌ Error getting cache stats: {e}")
    
    # Clear cache
    print("\n🗑️ Clearing Cache:")
    try:
        clear_result = await client.clear_cache()
        print(f"Status: {clear_result.get('status', 'unknown')}")
        print(f"Message: {clear_result.get('message', 'No message')}")
    except Exception as e:
        print(f"❌ Error clearing cache: {e}")

async def demo_performance_benchmark(client: ProductionAPIClient):
    """Demo performance benchmarking."""
    print("\n" + "="*60)
    print("⚡ PERFORMANCE BENCHMARK")
    print("="*60)
    
    # Benchmark text processing
    print("\n📝 Text Processing Benchmark:")
    sample_text = "This is a sample text for performance benchmarking. " * 10
    
    times = []
    for i in range(5):
        try:
            start_time = time.time()
            result = await client.process_text(
                text=sample_text,
                operations=["sentiment", "keywords"],
                model="gpt-4"
            )
            processing_time = time.time() - start_time
            times.append(processing_time)
            
            if result.get('status') == 'success':
                print(f"  Run {i+1}: {processing_time:.3f}s ✅")
            else:
                print(f"  Run {i+1}: {processing_time:.3f}s ❌")
                
        except Exception as e:
            print(f"  Run {i+1}: Error - {e}")
    
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        print(f"\n📊 Results:")
        print(f"  Average: {avg_time:.3f}s")
        print(f"  Minimum: {min_time:.3f}s")
        print(f"  Maximum: {max_time:.3f}s")
        print(f"  Throughput: {1/avg_time:.1f} requests/second")

async def main():
    """Main demo function."""
    print("🚀 NotebookLM AI Production API Demo")
    print("="*60)
    
    # Initialize client
    client = ProductionAPIClient(
        base_url="http://localhost:8000",
        api_token="your-token-here"  # Replace with actual token
    )
    
    try:
        # Run all demos
        await demo_health_and_stats(client)
        await demo_text_processing(client)
        await demo_image_processing(client)
        await demo_audio_processing(client)
        await demo_vector_search(client)
        await demo_batch_processing(client)
        await demo_cache_management(client)
        await demo_performance_benchmark(client)
        
        print("\n" + "="*60)
        print("✅ Demo completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logger.error(f"Demo error: {e}", exc_info=True)

match __name__:
    case "__main__":
    asyncio.run(main()) 