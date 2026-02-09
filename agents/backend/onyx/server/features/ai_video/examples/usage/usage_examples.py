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
from typing import Dict, List, Any
from datetime import datetime
from typing import Any, List, Dict, Optional
import logging
"""
🎯 FASTAPI BEST PRACTICES - USAGE EXAMPLES
==========================================

Practical examples demonstrating how to use the FastAPI best practices
implementation for the AI Video system.

Features:
- API client examples
- Request/response patterns
- Error handling examples
- Performance testing
- Integration examples
"""


# ============================================================================
# 1. API CLIENT EXAMPLES
# ============================================================================

class AIVideoAPIClient:
    """Client for interacting with the AI Video API."""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = None):
        
    """__init__ function."""
self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = None
    
    async def __aenter__(self) -> Any:
        self.session = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> Any:
        if self.session:
            await self.session.close()
    
    async def process_video(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single video."""
        url = f"{self.base_url}/api/v1/videos/process"
        
        async with self.session.post(url, json=video_data) as response:
            if response.status == 201:
                return await response.json()
            else:
                error_data = await response.json()
                raise Exception(f"API Error: {error_data.get('detail', 'Unknown error')}")
    
    async def process_video_batch(self, videos: List[Dict[str, Any]], batch_name: str = None) -> Dict[str, Any]:
        """Process multiple videos in batch."""
        url = f"{self.base_url}/api/v1/videos/batch-process"
        
        batch_data = {
            "videos": videos,
            "batch_name": batch_name,
            "priority": "normal"
        }
        
        async with self.session.post(url, json=batch_data) as response:
            if response.status == 201:
                return await response.json()
            else:
                error_data = await response.json()
                raise Exception(f"API Error: {error_data.get('detail', 'Unknown error')}")
    
    async def get_video(self, video_id: str) -> Dict[str, Any]:
        """Get video by ID."""
        url = f"{self.base_url}/api/v1/videos/{video_id}"
        
        async with self.session.get(url) as response:
            if response.status == 200:
                return await response.json()
            elif response.status == 404:
                raise Exception(f"Video {video_id} not found")
            else:
                error_data = await response.json()
                raise Exception(f"API Error: {error_data.get('detail', 'Unknown error')}")
    
    async def list_videos(self, skip: int = 0, limit: int = 100, quality: str = None) -> Dict[str, Any]:
        """List videos with pagination and filtering."""
        url = f"{self.base_url}/api/v1/videos/"
        params = {"skip": skip, "limit": limit}
        if quality:
            params["quality"] = quality
        
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_data = await response.json()
                raise Exception(f"API Error: {error_data.get('detail', 'Unknown error')}")
    
    async def update_video(self, video_id: str, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update video information."""
        url = f"{self.base_url}/api/v1/videos/{video_id}"
        
        async with self.session.put(url, json=video_data) as response:
            if response.status == 200:
                return await response.json()
            elif response.status == 404:
                raise Exception(f"Video {video_id} not found")
            else:
                error_data = await response.json()
                raise Exception(f"API Error: {error_data.get('detail', 'Unknown error')}")
    
    async def delete_video(self, video_id: str) -> bool:
        """Delete video."""
        url = f"{self.base_url}/api/v1/videos/{video_id}"
        
        async with self.session.delete(url) as response:
            if response.status == 204:
                return True
            elif response.status == 404:
                raise Exception(f"Video {video_id} not found")
            else:
                error_data = await response.json()
                raise Exception(f"API Error: {error_data.get('detail', 'Unknown error')}")
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        url = f"{self.base_url}/api/v1/analytics/performance"
        
        async with self.session.get(url) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_data = await response.json()
                raise Exception(f"API Error: {error_data.get('detail', 'Unknown error')}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check system health."""
        url = f"{self.base_url}/health"
        
        async with self.session.get(url) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception("Health check failed")

# ============================================================================
# 2. USAGE EXAMPLES
# ============================================================================

async def example_single_video_processing():
    """Example: Process a single video."""
    print("🎬 Processing Single Video")
    print("=" * 50)
    
    video_data = {
        "video_id": "sample_001",
        "title": "Sample Video - AI Enhancement",
        "duration": 180.5,
        "quality": "high",
        "priority": "normal",
        "description": "A sample video for AI enhancement processing",
        "tags": ["sample", "ai", "enhancement", "demo"],
        "metadata": {
            "source": "upload",
            "format": "mp4",
            "resolution": "1080p"
        }
    }
    
    async with AIVideoAPIClient() as client:
        try:
            start_time = time.time()
            result = await client.process_video(video_data)
            processing_time = time.time() - start_time
            
            print(f"✅ Video processed successfully!")
            print(f"📊 Processing time: {processing_time:.2f}s")
            print(f"🎯 Video ID: {result['video_id']}")
            print(f"📈 Status: {result['status']}")
            print(f"💬 Message: {result['message']}")
            print(f"🔗 Video URL: {result.get('video_url', 'N/A')}")
            print(f"🖼️  Thumbnail URL: {result.get('thumbnail_url', 'N/A')}")
            
        except Exception as e:
            print(f"❌ Error processing video: {e}")

async def example_batch_video_processing():
    """Example: Process multiple videos in batch."""
    print("\n🎬 Processing Video Batch")
    print("=" * 50)
    
    videos = [
        {
            "video_id": "batch_001",
            "title": "Batch Video 1",
            "duration": 120.0,
            "quality": "medium",
            "tags": ["batch", "demo"]
        },
        {
            "video_id": "batch_002",
            "title": "Batch Video 2",
            "duration": 240.0,
            "quality": "high",
            "tags": ["batch", "demo"]
        },
        {
            "video_id": "batch_003",
            "title": "Batch Video 3",
            "duration": 90.0,
            "quality": "low",
            "tags": ["batch", "demo"]
        }
    ]
    
    async with AIVideoAPIClient() as client:
        try:
            start_time = time.time()
            result = await client.process_video_batch(videos, "Demo Batch")
            processing_time = time.time() - start_time
            
            print(f"✅ Batch processed successfully!")
            print(f"📊 Processing time: {processing_time:.2f}s")
            print(f"🎯 Batch ID: {result['batch_id']}")
            print(f"📈 Overall Status: {result['status']}")
            print(f"📊 Progress: {result['overall_progress']:.1f}%")
            print(f"✅ Completed: {result['completed_videos']}/{result['total_videos']}")
            print(f"❌ Failed: {result['failed_videos']}")
            print(f"🔄 Processing: {result['processing_videos']}")
            print(f"📈 Success Rate: {result['success_rate']:.1%}")
            
        except Exception as e:
            print(f"❌ Error processing batch: {e}")

async def example_video_management():
    """Example: Video CRUD operations."""
    print("\n🎬 Video Management Operations")
    print("=" * 50)
    
    async with AIVideoAPIClient() as client:
        # Create a video
        video_data = {
            "video_id": "crud_001",
            "title": "CRUD Test Video",
            "duration": 150.0,
            "quality": "medium",
            "tags": ["crud", "test"]
        }
        
        try:
            # Process video
            result = await client.process_video(video_data)
            print(f"✅ Created video: {result['video_id']}")
            
            # Get video
            video = await client.get_video("crud_001")
            print(f"📖 Retrieved video: {video['video_id']} - {video['status']}")
            
            # Update video
            updated_data = {
                "video_id": "crud_001",
                "title": "Updated CRUD Test Video",
                "duration": 160.0,
                "quality": "high",
                "tags": ["crud", "test", "updated"]
            }
            updated = await client.update_video("crud_001", updated_data)
            print(f"✏️  Updated video: {updated['video_id']} - {updated['message']}")
            
            # List videos
            videos = await client.list_videos(limit=10)
            print(f"📋 Listed {len(videos['items'])} videos (total: {videos['total']})")
            
            # Delete video
            deleted = await client.delete_video("crud_001")
            print(f"🗑️  Deleted video: {deleted}")
            
        except Exception as e:
            print(f"❌ Error in video management: {e}")

async def example_error_handling():
    """Example: Error handling patterns."""
    print("\n🎬 Error Handling Examples")
    print("=" * 50)
    
    async with AIVideoAPIClient() as client:
        # Test invalid video data
        invalid_video = {
            "video_id": "",  # Empty ID
            "title": "",     # Empty title
            "duration": -1   # Invalid duration
        }
        
        try:
            await client.process_video(invalid_video)
        except Exception as e:
            print(f"❌ Expected validation error: {e}")
        
        # Test non-existent video
        try:
            await client.get_video("nonexistent_123")
        except Exception as e:
            print(f"❌ Expected not found error: {e}")
        
        # Test invalid batch
        invalid_batch = {
            "videos": []  # Empty batch
        }
        
        try:
            await client.process_video_batch(invalid_batch)
        except Exception as e:
            print(f"❌ Expected batch validation error: {e}")

async def example_performance_monitoring():
    """Example: Performance monitoring."""
    print("\n🎬 Performance Monitoring")
    print("=" * 50)
    
    async with AIVideoAPIClient() as client:
        try:
            # Health check
            health = await client.health_check()
            print(f"🏥 System Health: {health['status']}")
            print(f"📊 Version: {health['version']}")
            print(f"🕐 Timestamp: {health['timestamp']}")
            
            # Performance metrics
            metrics = await client.get_performance_metrics()
            print(f"📈 Total Videos Processed: {metrics['total_videos_processed']}")
            print(f"📊 Success Rate: {metrics['success_rate']:.1%}")
            print(f"⏱️  Average Processing Time: {metrics['average_processing_time']:.2f}s")
            print(f"🔄 System Uptime: {metrics['system_uptime']}s")
            print(f"🚀 Active Requests: {metrics['active_requests']}")
            
        except Exception as e:
            print(f"❌ Error getting metrics: {e}")

async def example_concurrent_processing():
    """Example: Concurrent video processing."""
    print("\n🎬 Concurrent Processing")
    print("=" * 50)
    
    videos = [
        {
            "video_id": f"concurrent_{i:03d}",
            "title": f"Concurrent Video {i}",
            "duration": 60.0 + (i * 10),
            "quality": "medium",
            "tags": ["concurrent", "test"]
        }
        for i in range(1, 6)
    ]
    
    async with AIVideoAPIClient() as client:
        try:
            start_time = time.time()
            
            # Process videos concurrently
            tasks = [client.process_video(video) for video in videos]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            processing_time = time.time() - start_time
            
            # Analyze results
            successful = [r for r in results if not isinstance(r, Exception)]
            failed = [r for r in results if isinstance(r, Exception)]
            
            print(f"✅ Concurrent processing completed!")
            print(f"📊 Total time: {processing_time:.2f}s")
            print(f"📈 Successful: {len(successful)}")
            print(f"❌ Failed: {len(failed)}")
            print(f"🚀 Concurrency improvement: {len(videos) * 0.1 / processing_time:.1f}x")
            
            # Show results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"  ❌ Video {i+1}: {result}")
                else:
                    print(f"  ✅ Video {i+1}: {result['video_id']} - {result['status']}")
            
        except Exception as e:
            print(f"❌ Error in concurrent processing: {e}")

async def example_pagination_and_filtering():
    """Example: Pagination and filtering."""
    print("\n🎬 Pagination and Filtering")
    print("=" * 50)
    
    async with AIVideoAPIClient() as client:
        try:
            # List first page
            page1 = await client.list_videos(skip=0, limit=5)
            print(f"📄 Page 1: {len(page1['items'])} videos")
            print(f"📊 Total: {page1['total']} videos")
            print(f"🔄 Has next: {page1['has_next']}")
            print(f"🔄 Has previous: {page1['has_previous']}")
            
            # List second page
            if page1['has_next']:
                page2 = await client.list_videos(skip=5, limit=5)
                print(f"📄 Page 2: {len(page2['items'])} videos")
                print(f"🔄 Has next: {page2['has_next']}")
            
            # Filter by quality
            high_quality = await client.list_videos(quality="high")
            print(f"🎯 High quality videos: {len(high_quality['items'])}")
            
            medium_quality = await client.list_videos(quality="medium")
            print(f"🎯 Medium quality videos: {len(medium_quality['items'])}")
            
        except Exception as e:
            print(f"❌ Error in pagination: {e}")

# ============================================================================
# 3. PERFORMANCE TESTING
# ============================================================================

async def performance_test():
    """Run performance tests."""
    print("\n🎬 Performance Testing")
    print("=" * 50)
    
    async with AIVideoAPIClient() as client:
        # Test single video processing
        video_data = {
            "video_id": "perf_test_001",
            "title": "Performance Test Video",
            "duration": 120.0,
            "quality": "medium"
        }
        
        start_time = time.time()
        result = await client.process_video(video_data)
        single_time = time.time() - start_time
        
        print(f"⏱️  Single video processing: {single_time:.3f}s")
        
        # Test batch processing
        batch_videos = [
            {
                "video_id": f"perf_batch_{i:03d}",
                "title": f"Performance Batch Video {i}",
                "duration": 60.0,
                "quality": "medium"
            }
            for i in range(1, 11)
        ]
        
        start_time = time.time()
        batch_result = await client.process_video_batch(batch_videos)
        batch_time = time.time() - start_time
        
        print(f"⏱️  Batch processing (10 videos): {batch_time:.3f}s")
        print(f"📊 Average per video: {batch_time / 10:.3f}s")
        print(f"🚀 Speedup: {single_time * 10 / batch_time:.1f}x")
        
        # Test concurrent processing
        start_time = time.time()
        tasks = [client.process_video(video) for video in batch_videos]
        concurrent_results = await asyncio.gather(*tasks)
        concurrent_time = time.time() - start_time
        
        print(f"⏱️  Concurrent processing (10 videos): {concurrent_time:.3f}s")
        print(f"📊 Average per video: {concurrent_time / 10:.3f}s")
        print(f"🚀 Speedup vs sequential: {single_time * 10 / concurrent_time:.1f}x")
        print(f"🚀 Speedup vs batch: {batch_time / concurrent_time:.1f}x")

# ============================================================================
# 4. MAIN EXECUTION
# ============================================================================

async def run_all_examples():
    """Run all usage examples."""
    print("🚀 FASTAPI BEST PRACTICES - USAGE EXAMPLES")
    print("=" * 60)
    
    examples = [
        example_single_video_processing,
        example_batch_video_processing,
        example_video_management,
        example_error_handling,
        example_performance_monitoring,
        example_concurrent_processing,
        example_pagination_and_filtering,
        performance_test
    ]
    
    for example in examples:
        try:
            await example()
            print()  # Add spacing between examples
        except Exception as e:
            print(f"❌ Error running {example.__name__}: {e}")
            print()

if __name__ == "__main__":
    # Run examples
    asyncio.run(run_all_examples()) 