from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import asyncio
import httpx
import time
from datetime import datetime
import orjson
from typing import List, Dict, Any
import statistics
        import uvloop
from typing import Any, List, Dict, Optional
import logging
"""
LinkedIn Posts API V2 - Ultra-Optimized Demo
===========================================

Demonstrates all the advanced features and optimizations of the improved API.
"""



class LinkedInAPIDemo:
    """Demo client for LinkedIn Posts API V2."""
    
    def __init__(self, base_url: str = "http://localhost:8000/api/v2"):
        
    """__init__ function."""
self.base_url = base_url
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Content-Type": "application/json"},
            timeout=30.0
        )
        self.auth_token = None
    
    async def close(self) -> Any:
        """Close client connection."""
        await self.client.aclose()
    
    async def authenticate(self) -> Any:
        """Authenticate and get token."""
        # Mock authentication
        self.auth_token = "mock_jwt_token"
        self.client.headers["Authorization"] = f"Bearer {self.auth_token}"
        print("✅ Authenticated successfully")
    
    async def demo_health_checks(self) -> Any:
        """Demonstrate health check endpoints."""
        print("\n🏥 Health Checks Demo")
        print("=" * 50)
        
        # Basic health check
        response = await self.client.get("/linkedin-posts/health")
        print(f"Basic Health: {response.json()}")
        
        # Detailed health check
        response = await self.client.get("/linkedin-posts/health?detailed=true")
        health_data = response.json()
        print(f"\nDetailed Health:")
        print(f"  - Status: {health_data['status']}")
        print(f"  - Version: {health_data['version']}")
        print(f"  - Uptime: {health_data.get('uptime', 0):.1f}s")
        
        if "dependencies" in health_data:
            print(f"  - Dependencies: {health_data['dependencies']}")
        
        if "performance" in health_data:
            perf = health_data["performance"]
            print(f"  - Active Requests: {perf.get('active_requests', 0)}")
            print(f"  - Avg Response Time: {perf.get('avg_response_time', 'N/A')}")
            print(f"  - Cache Hit Rate: {perf.get('cache_hit_rate', 0):.2%}")
    
    async def demo_create_post(self) -> Any:
        """Demonstrate post creation with NLP enhancement."""
        print("\n✍️ Post Creation Demo")
        print("=" * 50)
        
        post_data = {
            "content": "Excited to announce our new AI-powered LinkedIn post optimization feature! 🚀",
            "post_type": "announcement",
            "tone": "professional",
            "target_audience": "tech professionals",
            "industry": "technology"
        }
        
        # Create with fast NLP
        start_time = time.time()
        response = await self.client.post(
            "/linkedin-posts/?use_fast_nlp=true&use_async_nlp=true",
            json=post_data
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 201:
            post = response.json()
            print(f"✅ Post created in {elapsed:.3f}s")
            print(f"  - ID: {post['id']}")
            print(f"  - NLP Enhanced: {post['nlp_enhanced']}")
            print(f"  - Processing Time: {post.get('nlp_processing_time', 0):.3f}s")
            print(f"  - Content Preview: {post['content'][:100]}...")
            
            return post['id']
        else:
            print(f"❌ Failed to create post: {response.status_code}")
            return None
    
    async def demo_batch_operations(self) -> Any:
        """Demonstrate batch operations."""
        print("\n📦 Batch Operations Demo")
        print("=" * 50)
        
        # Create multiple posts
        posts_data = [
            {
                "content": f"LinkedIn tip #{i}: {tip}",
                "post_type": "educational",
                "tone": "friendly",
                "target_audience": "professionals",
                "industry": "business"
            }
            for i, tip in enumerate([
                "Optimize your profile headline",
                "Share valuable content regularly",
                "Engage with your network",
                "Use relevant hashtags",
                "Post at optimal times"
            ], 1)
        ]
        
        # Batch create
        start_time = time.time()
        response = await self.client.post(
            "/linkedin-posts/batch?parallel_processing=true",
            json=posts_data
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            posts = response.json()
            print(f"✅ Created {len(posts)} posts in {elapsed:.3f}s")
            print(f"  - Average time per post: {elapsed/len(posts):.3f}s")
            
            # Batch optimize
            post_ids = [post['id'] for post in posts]
            
            start_time = time.time()
            response = await self.client.post(
                "/linkedin-posts/batch/optimize",
                json={
                    "post_ids": post_ids[:3],  # Optimize first 3
                    "use_async_nlp": True
                }
            )
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                optimized = response.json()
                print(f"\n✅ Optimized {len(optimized)} posts in {elapsed:.3f}s")
            
            return post_ids
        else:
            print(f"❌ Batch creation failed: {response.status_code}")
            return []
    
    async def demo_caching(self, post_id: str):
        """Demonstrate caching behavior."""
        print("\n💾 Caching Demo")
        print("=" * 50)
        
        # First request (cache miss)
        start_time = time.time()
        response = await self.client.get(f"/linkedin-posts/{post_id}")
        elapsed1 = time.time() - start_time
        
        cache_header = response.headers.get("X-Cache", "MISS")
        print(f"First request: {elapsed1:.3f}s (Cache: {cache_header})")
        
        # Second request (cache hit)
        start_time = time.time()
        response = await self.client.get(f"/linkedin-posts/{post_id}")
        elapsed2 = time.time() - start_time
        
        cache_header = response.headers.get("X-Cache", "MISS")
        print(f"Second request: {elapsed2:.3f}s (Cache: {cache_header})")
        
        # Performance improvement
        if elapsed2 < elapsed1:
            improvement = ((elapsed1 - elapsed2) / elapsed1) * 100
            print(f"✅ Performance improvement: {improvement:.1f}%")
        
        # Test ETag support
        etag = response.headers.get("ETag")
        if etag:
            response = await self.client.get(
                f"/linkedin-posts/{post_id}",
                headers={"If-None-Match": etag}
            )
            print(f"\nETag test: Status {response.status_code} (304 = Not Modified)")
    
    async def demo_streaming(self, post_id: str):
        """Demonstrate streaming updates."""
        print("\n📡 Streaming Demo")
        print("=" * 50)
        
        print(f"Connecting to SSE stream for post {post_id}...")
        
        # Simulate SSE connection
        async with self.client.stream(
            "GET",
            f"/linkedin-posts/stream/{post_id}"
        ) as response:
            count = 0
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = orjson.loads(line[6:])
                    print(f"Event: {data.get('type')} at {data.get('timestamp')}")
                    
                    count += 1
                    if count >= 3:  # Stop after 3 events
                        break
    
    async def demo_analytics(self, post_id: str):
        """Demonstrate analytics features."""
        print("\n📊 Analytics Demo")
        print("=" * 50)
        
        # Analyze post
        start_time = time.time()
        response = await self.client.get(
            f"/linkedin-posts/{post_id}/analyze?"
            "use_async_nlp=true&include_competitors=true&include_trends=true"
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            analysis = response.json()
            print(f"✅ Analysis completed in {elapsed:.3f}s")
            print(f"  - Sentiment Score: {analysis['sentiment_score']:.2f}")
            print(f"  - Readability Score: {analysis['readability_score']:.1f}")
            print(f"  - Keywords: {', '.join(analysis['keywords'][:5])}")
            print(f"  - Entities: {', '.join(analysis['entities'][:3])}")
            
            if analysis.get('cached'):
                print(f"  - Result was cached")
    
    async def demo_performance_metrics(self) -> Any:
        """Demonstrate performance monitoring."""
        print("\n📈 Performance Metrics Demo")
        print("=" * 50)
        
        response = await self.client.get(
            "/linkedin-posts/performance/metrics?include_detailed=true"
        )
        
        if response.status_code == 200:
            metrics = response.json()
            
            # NLP metrics
            print("NLP Performance:")
            fast_nlp = metrics.get('fast_nlp_metrics', {})
            print(f"  - Fast NLP Avg Time: {fast_nlp.get('avg_processing_time', 0):.3f}s")
            print(f"  - Fast NLP Cache Hit Rate: {fast_nlp.get('cache_hit_rate', 0):.2%}")
            
            async_nlp = metrics.get('async_nlp_metrics', {})
            print(f"  - Async NLP Avg Time: {async_nlp.get('avg_processing_time', 0):.3f}s")
            print(f"  - Async NLP Throughput: {async_nlp.get('throughput', 0):.1f} posts/s")
            
            # System metrics
            if 'system_metrics' in metrics:
                system = metrics['system_metrics']
                print(f"\nSystem Performance:")
                print(f"  - Active Requests: {system.get('active_requests', 0)}")
                print(f"  - Total Requests: {system.get('total_requests', 0)}")
                print(f"  - Cache Hit Rate: {system.get('cache_hit_rate', 0):.2%}")
    
    async def demo_rate_limiting(self) -> Any:
        """Demonstrate rate limiting."""
        print("\n🚦 Rate Limiting Demo")
        print("=" * 50)
        
        print("Sending rapid requests to test rate limiting...")
        
        # Send multiple requests quickly
        results = []
        for i in range(5):
            start_time = time.time()
            response = await self.client.get("/linkedin-posts/health")
            elapsed = time.time() - start_time
            
            results.append({
                "request": i + 1,
                "status": response.status_code,
                "time": elapsed,
                "rate_limit": response.headers.get("X-RateLimit-Remaining", "N/A")
            })
            
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After", "N/A")
                print(f"  ❌ Rate limit hit! Retry after: {retry_after}s")
                break
            else:
                print(f"  ✅ Request {i+1}: {elapsed:.3f}s (Remaining: {results[-1]['rate_limit']})")
    
    async def demo_performance_optimization(self) -> Any:
        """Demonstrate performance optimization."""
        print("\n⚡ Performance Optimization Demo")
        print("=" * 50)
        
        # Clear cache
        print("Clearing cache...")
        await self.client.post("/linkedin-posts/performance/optimize?clear_cache=true")
        
        # Warm cache
        print("Warming cache...")
        await self.client.post(
            "/linkedin-posts/performance/optimize?"
            "warm_cache=true&optimize_nlp=true"
        )
        
        print("✅ Performance optimization completed")
    
    async def run_load_test(self, num_requests: int = 50):
        """Run a simple load test."""
        print(f"\n🏃 Load Test Demo ({num_requests} requests)")
        print("=" * 50)
        
        # Create test post
        post_data = {
            "content": "Load test post",
            "post_type": "update",
            "tone": "casual",
            "target_audience": "general",
            "industry": "technology"
        }
        
        # Run concurrent requests
        async def make_request():
            
    """make_request function."""
start = time.time()
            response = await self.client.post("/linkedin-posts/", json=post_data)
            return time.time() - start, response.status_code
        
        # Execute requests
        start_time = time.time()
        tasks = [make_request() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Analyze results
        times = [r[0] for r in results]
        statuses = [r[1] for r in results]
        
        successful = sum(1 for s in statuses if s == 201)
        
        print(f"Results:")
        print(f"  - Total Time: {total_time:.2f}s")
        print(f"  - Requests/second: {num_requests/total_time:.1f}")
        print(f"  - Success Rate: {successful/num_requests:.1%}")
        print(f"  - Avg Response Time: {statistics.mean(times):.3f}s")
        print(f"  - Min Response Time: {min(times):.3f}s")
        print(f"  - Max Response Time: {max(times):.3f}s")
        print(f"  - P95 Response Time: {statistics.quantiles(times, n=20)[18]:.3f}s")


async def main():
    """Run all demos."""
    print("🚀 LinkedIn Posts API V2 - Ultra-Optimized Demo")
    print("=" * 70)
    
    demo = LinkedInAPIDemo()
    
    try:
        # Authenticate
        await demo.authenticate()
        
        # Run demos
        await demo.demo_health_checks()
        
        # Create a post
        post_id = await demo.demo_create_post()
        
        if post_id:
            # Test caching
            await demo.demo_caching(post_id)
            
            # Test analytics
            await demo.demo_analytics(post_id)
            
            # Test streaming (brief)
            # await demo.demo_streaming(post_id)
        
        # Batch operations
        post_ids = await demo.demo_batch_operations()
        
        # Performance metrics
        await demo.demo_performance_metrics()
        
        # Rate limiting
        await demo.demo_rate_limiting()
        
        # Performance optimization
        await demo.demo_performance_optimization()
        
        # Load test
        await demo.run_load_test(num_requests=20)
        
        print("\n✅ All demos completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    finally:
        await demo.close()


if __name__ == "__main__":
    # Run with uvloop for maximum performance
    try:
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        print("Using uvloop for maximum performance")
    except ImportError:
        print("uvloop not available, using standard asyncio")
    
    asyncio.run(main()) 