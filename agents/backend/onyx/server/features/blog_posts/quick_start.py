#!/usr/bin/env python3
"""
🚀 Quick Start - Optimized Blog System
======================================

This script demonstrates the optimized blog system with:
- System startup and configuration
- Sample data creation
- Performance testing
- Results comparison
"""

import asyncio
import time
import json
from typing import Dict, Any
import uvicorn
from pathlib import Path

# Import the optimized system
from optimized_blog_system_v2 import (
    create_optimized_blog_system,
    Config,
    DatabaseConfig,
    CacheConfig,
    PerformanceConfig
)

# Import benchmark tools
from performance_benchmark import BlogSystemBenchmark, BenchmarkConfig

class QuickStartDemo:
    """Demonstration of the optimized blog system."""
    
    def __init__(self):
        self.config = self._create_config()
        self.app = None
        self.server = None
    
    def _create_config(self) -> Config:
        """Create optimized configuration."""
        return Config(
            database=DatabaseConfig(
                url="sqlite+aiosqlite:///./quick_start_blog.db",
                pool_size=10,
                max_overflow=20
            ),
            cache=CacheConfig(
                memory_cache_size=500,
                memory_cache_ttl=300,
                enable_compression=True
            ),
            performance=PerformanceConfig(
                enable_gzip=True,
                enable_cors=True,
                rate_limit_requests=200,
                background_tasks=True
            ),
            debug=True
        )
    
    async def start_system(self):
        """Start the optimized blog system."""
        print("🚀 Starting Optimized Blog System...")
        
        # Create the application
        self.app = create_optimized_blog_system(self.config)
        
        # Start the server in background
        config = uvicorn.Config(
            self.app.app,
            host="127.0.0.1",
            port=8000,
            log_level="info"
        )
        self.server = uvicorn.Server(config)
        
        # Start server in background
        await self.server.serve()
    
    async def create_sample_data(self):
        """Create sample blog posts for testing."""
        print("📝 Creating sample data...")
        
        import aiohttp
        
        sample_posts = [
            {
                "title": "Getting Started with FastAPI",
                "content": "FastAPI is a modern, fast web framework for building APIs with Python 3.7+ based on standard Python type hints.",
                "tags": ["python", "fastapi", "web-development"],
                "is_published": True
            },
            {
                "title": "Optimizing Python Applications",
                "content": "Learn the best practices for optimizing Python applications for performance and scalability.",
                "tags": ["python", "optimization", "performance"],
                "is_published": True
            },
            {
                "title": "Async Programming in Python",
                "content": "Understanding async/await patterns and how to build high-performance applications.",
                "tags": ["python", "async", "concurrency"],
                "is_published": True
            },
            {
                "title": "Caching Strategies for Web Applications",
                "content": "Explore different caching strategies to improve application performance.",
                "tags": ["caching", "performance", "web-development"],
                "is_published": True
            },
            {
                "title": "Database Optimization Techniques",
                "content": "Learn how to optimize database queries and improve application performance.",
                "tags": ["database", "optimization", "sql"],
                "is_published": True
            }
        ]
        
        async with aiohttp.ClientSession() as session:
            for i, post_data in enumerate(sample_posts):
                async with session.post(
                    "http://127.0.0.1:8000/posts",
                    json=post_data
                ) as response:
                    if response.status == 201:
                        print(f"  ✅ Created post {i+1}: {post_data['title']}")
                    else:
                        print(f"  ❌ Failed to create post {i+1}")
    
    async def run_performance_test(self):
        """Run a quick performance test."""
        print("⚡ Running performance test...")
        
        benchmark_config = BenchmarkConfig(
            base_url="http://127.0.0.1:8000",
            concurrent_users=10,
            requests_per_user=20,
            warmup_requests=5
        )
        
        benchmark = BlogSystemBenchmark(benchmark_config)
        
        try:
            await benchmark.setup()
            
            # Run read-heavy test
            result = await benchmark.read_heavy_test()
            
            print(f"\n📊 Performance Results:")
            print(f"  Total Requests: {result.total_requests}")
            print(f"  Success Rate: {result.successful_requests/result.total_requests*100:.2f}%")
            print(f"  Requests/Second: {result.requests_per_second:.2f}")
            print(f"  Avg Response Time: {result.average_response_time*1000:.2f}ms")
            print(f"  Memory Usage: {result.memory_usage_mb:.2f} MB")
            print(f"  CPU Usage: {result.cpu_usage_percent:.2f}%")
            
        finally:
            await benchmark.cleanup()
    
    async def demonstrate_features(self):
        """Demonstrate key features of the optimized system."""
        print("🎯 Demonstrating key features...")
        
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            # Health check
            async with session.get("http://127.0.0.1:8000/health") as response:
                health_data = await response.json()
                print(f"  ✅ Health Check: {health_data['status']}")
            
            # Get metrics
            async with session.get("http://127.0.0.1:8000/metrics") as response:
                metrics = await response.json()
                print(f"  📊 System Metrics: Memory {metrics.get('memory', {}).get('percent', 0):.1f}%")
            
            # List posts (cached)
            start_time = time.time()
            async with session.get("http://127.0.0.1:8000/posts") as response:
                posts = await response.json()
                first_request_time = time.time() - start_time
            
            # List posts again (should be faster due to caching)
            start_time = time.time()
            async with session.get("http://127.0.0.1:8000/posts") as response:
                posts_cached = await response.json()
                cached_request_time = time.time() - start_time
            
            print(f"  📖 Posts Retrieved: {len(posts)}")
            print(f"  ⚡ First Request: {first_request_time*1000:.2f}ms")
            print(f"  🚀 Cached Request: {cached_request_time*1000:.2f}ms")
            print(f"  📈 Speed Improvement: {first_request_time/cached_request_time:.1f}x")
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.server:
            self.server.should_exit = True

async def main():
    """Main demonstration function."""
    print("🚀 Blog System Optimization Demo")
    print("=" * 50)
    
    demo = QuickStartDemo()
    
    try:
        # Start the system
        await demo.start_system()
        
        # Wait for system to be ready
        await asyncio.sleep(2)
        
        # Create sample data
        await demo.create_sample_data()
        
        # Demonstrate features
        await demo.demonstrate_features()
        
        # Run performance test
        await demo.run_performance_test()
        
        print("\n🎉 Demo completed successfully!")
        print("\nKey Optimizations Demonstrated:")
        print("  ✅ Multi-tier caching (Memory + Redis)")
        print("  ✅ Async database operations")
        print("  ✅ Performance monitoring")
        print("  ✅ Rate limiting and security")
        print("  ✅ Background task processing")
        print("  ✅ Comprehensive error handling")
        
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
    finally:
        await demo.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 
 
 