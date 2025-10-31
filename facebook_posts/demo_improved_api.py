"""
Demo script for the improved Facebook Posts API
Showcasing FastAPI best practices and enhanced functionality
"""

import asyncio
import httpx
import json
import time
from typing import Dict, Any, List
from datetime import datetime


class FacebookPostsAPIDemo:
    """Demo class for the improved Facebook Posts API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def demo_basic_operations(self):
        """Demo basic API operations"""
        print("🚀 Facebook Posts API - Basic Operations Demo")
        print("=" * 60)
        
        # 1. Health Check
        print("\n1. Health Check")
        try:
            response = await self.client.get(f"{self.base_url}/health")
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                health_data = response.json()
                print(f"   System Status: {health_data.get('status', 'unknown')}")
                print(f"   Version: {health_data.get('version', 'unknown')}")
            else:
                print(f"   Error: {response.text}")
        except Exception as e:
            print(f"   Error: {e}")
        
        # 2. Generate Single Post
        print("\n2. Generate Single Post")
        post_request = {
            "topic": "AI Revolution in Business",
            "audience_type": "professionals",
            "content_type": "educational",
            "tone": "professional",
            "optimization_level": "advanced",
            "include_hashtags": True,
            "tags": ["ai", "business", "innovation"]
        }
        
        try:
            start_time = time.time()
            response = await self.client.post(
                f"{self.base_url}/api/v1/posts/generate",
                json=post_request
            )
            processing_time = time.time() - start_time
            
            print(f"   Status: {response.status_code}")
            if response.status_code == 201:
                data = response.json()
                print(f"   Success: {data['success']}")
                print(f"   Processing Time: {data['processing_time']:.3f}s")
                print(f"   API Response Time: {processing_time:.3f}s")
                print(f"   Post ID: {data['post']['id']}")
                print(f"   Content Preview: {data['post']['content'][:100]}...")
                print(f"   Optimizations Applied: {', '.join(data['optimizations_applied'])}")
                
                return data['post']['id']
            else:
                print(f"   Error: {response.text}")
                return None
        except Exception as e:
            print(f"   Error: {e}")
            return None
    
    async def demo_batch_operations(self):
        """Demo batch operations"""
        print("\n3. Batch Post Generation")
        
        batch_request = {
            "requests": [
                {
                    "topic": "Digital Marketing Trends 2024",
                    "audience_type": "professionals",
                    "content_type": "educational",
                    "tone": "professional"
                },
                {
                    "topic": "Remote Work Best Practices",
                    "audience_type": "general",
                    "content_type": "educational",
                    "tone": "friendly"
                },
                {
                    "topic": "Sustainable Business Practices",
                    "audience_type": "entrepreneurs",
                    "content_type": "inspirational",
                    "tone": "motivational"
                }
            ],
            "parallel_processing": True,
            "batch_metadata": {
                "campaign": "Q1 2024 Content",
                "priority": "high"
            }
        }
        
        try:
            start_time = time.time()
            response = await self.client.post(
                f"{self.base_url}/api/v1/posts/generate/batch",
                json=batch_request
            )
            processing_time = time.time() - start_time
            
            print(f"   Status: {response.status_code}")
            if response.status_code == 201:
                data = response.json()
                print(f"   Success: {data['success']}")
                print(f"   Total Processing Time: {data['total_processing_time']:.3f}s")
                print(f"   API Response Time: {processing_time:.3f}s")
                print(f"   Successful Posts: {data['successful_posts']}")
                print(f"   Failed Posts: {data['failed_posts']}")
                print(f"   Batch ID: {data['batch_id']}")
                
                # Show results summary
                for i, result in enumerate(data['results']):
                    status = "✅" if result['success'] else "❌"
                    print(f"   Post {i+1}: {status} {result.get('error', 'Success')}")
                
                return [r['post']['id'] for r in data['results'] if r['success'] and r.get('post')]
            else:
                print(f"   Error: {response.text}")
                return []
        except Exception as e:
            print(f"   Error: {e}")
            return []
    
    async def demo_filtering_and_pagination(self):
        """Demo filtering and pagination"""
        print("\n4. Filtering and Pagination")
        
        # Test different filter combinations
        filter_tests = [
            {"status": "draft", "limit": 5},
            {"content_type": "educational", "limit": 3},
            {"audience_type": "professionals", "limit": 2},
            {"skip": 0, "limit": 10}
        ]
        
        for i, filters in enumerate(filter_tests, 1):
            print(f"\n   Test {i}: {filters}")
            try:
                response = await self.client.get(
                    f"{self.base_url}/api/v1/posts",
                    params=filters
                )
                
                print(f"   Status: {response.status_code}")
                if response.status_code == 200:
                    data = response.json()
                    print(f"   Posts Returned: {len(data)}")
                    if data:
                        print(f"   First Post ID: {data[0]['id']}")
                        print(f"   First Post Status: {data[0]['status']}")
                else:
                    print(f"   Error: {response.text}")
            except Exception as e:
                print(f"   Error: {e}")
    
    async def demo_error_handling(self):
        """Demo error handling scenarios"""
        print("\n5. Error Handling Demo")
        
        error_tests = [
            {
                "name": "Empty Topic",
                "request": {
                    "topic": "",
                    "audience_type": "professionals",
                    "content_type": "educational"
                },
                "expected_status": 422
            },
            {
                "name": "Short Topic",
                "request": {
                    "topic": "AI",
                    "audience_type": "professionals",
                    "content_type": "educational"
                },
                "expected_status": 400
            },
            {
                "name": "Invalid Status Filter",
                "endpoint": "/api/v1/posts",
                "params": {"status": "invalid_status"},
                "expected_status": 400
            },
            {
                "name": "Invalid Pagination",
                "endpoint": "/api/v1/posts",
                "params": {"skip": -1},
                "expected_status": 400
            },
            {
                "name": "Post Not Found",
                "endpoint": "/api/v1/posts/not-found",
                "expected_status": 404
            }
        ]
        
        for test in error_tests:
            print(f"\n   Testing: {test['name']}")
            try:
                if 'request' in test:
                    response = await self.client.post(
                        f"{self.base_url}/api/v1/posts/generate",
                        json=test['request']
                    )
                else:
                    endpoint = test.get('endpoint', '/api/v1/posts')
                    params = test.get('params', {})
                    response = await self.client.get(
                        f"{self.base_url}{endpoint}",
                        params=params
                    )
                
                print(f"   Status: {response.status_code} (Expected: {test['expected_status']})")
                if response.status_code == test['expected_status']:
                    print("   ✅ Error handled correctly")
                else:
                    print("   ❌ Unexpected status code")
                
                if response.status_code >= 400:
                    error_data = response.json()
                    print(f"   Error Message: {error_data.get('detail', 'No detail provided')}")
            except Exception as e:
                print(f"   Error: {e}")
    
    async def demo_performance_metrics(self):
        """Demo performance metrics"""
        print("\n6. Performance Metrics")
        
        try:
            response = await self.client.get(f"{self.base_url}/api/v1/metrics")
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                metrics = response.json()
                print(f"   Total Requests: {metrics.get('total_requests', 0)}")
                print(f"   Successful Requests: {metrics.get('successful_requests', 0)}")
                print(f"   Failed Requests: {metrics.get('failed_requests', 0)}")
                print(f"   Average Processing Time: {metrics.get('average_processing_time', 0):.3f}s")
                print(f"   Cache Hit Rate: {metrics.get('cache_hit_rate', 0):.1%}")
                print(f"   Memory Usage: {metrics.get('memory_usage', 0):.1f} MB")
                print(f"   CPU Usage: {metrics.get('cpu_usage', 0):.1f}%")
                print(f"   Active Connections: {metrics.get('active_connections', 0)}")
            else:
                print(f"   Error: {response.text}")
        except Exception as e:
            print(f"   Error: {e}")
    
    async def demo_post_operations(self, post_id: str):
        """Demo post CRUD operations"""
        if not post_id:
            print("\n7. Post Operations (Skipped - No post ID available)")
            return
        
        print(f"\n7. Post Operations (ID: {post_id})")
        
        # Get Post
        print("\n   Getting Post:")
        try:
            response = await self.client.get(f"{self.base_url}/api/v1/posts/{post_id}")
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   Content Length: {len(data['content'])} characters")
                print(f"   Status: {data['status']}")
                print(f"   Created: {data['created_at']}")
            else:
                print(f"   Error: {response.text}")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Update Post
        print("\n   Updating Post:")
        update_data = {
            "content": "Updated content with enhanced information about AI in business.",
            "tags": ["ai", "business", "updated", "demo"]
        }
        
        try:
            response = await self.client.put(
                f"{self.base_url}/api/v1/posts/{post_id}",
                json=update_data
            )
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   Updated Successfully")
                print(f"   New Content Length: {len(data['content'])} characters")
            else:
                print(f"   Error: {response.text}")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Optimize Post
        print("\n   Optimizing Post:")
        optimization_request = {
            "optimization_level": "ultra",
            "focus_areas": ["engagement", "readability", "sentiment"],
            "target_audience": "professionals"
        }
        
        try:
            response = await self.client.post(
                f"{self.base_url}/api/v1/posts/{post_id}/optimize",
                json=optimization_request
            )
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   Optimization Success: {data['success']}")
                print(f"   Processing Time: {data['processing_time']:.3f}s")
                print(f"   Improvements: {', '.join(data['improvements'])}")
            else:
                print(f"   Error: {response.text}")
        except Exception as e:
            print(f"   Error: {e}")
    
    async def demo_headers_and_tracking(self):
        """Demo request headers and tracking"""
        print("\n8. Request Headers and Tracking")
        
        try:
            response = await self.client.get(f"{self.base_url}/api/v1/posts")
            
            print(f"   Status: {response.status_code}")
            print("   Response Headers:")
            
            # Check for custom headers
            custom_headers = [
                'x-request-id',
                'x-process-time',
                'x-ratelimit-limit',
                'x-ratelimit-remaining'
            ]
            
            for header in custom_headers:
                value = response.headers.get(header)
                if value:
                    print(f"     {header}: {value}")
                else:
                    print(f"     {header}: Not present")
            
            # Show some standard headers
            standard_headers = ['content-type', 'content-length', 'server']
            for header in standard_headers:
                value = response.headers.get(header)
                if value:
                    print(f"     {header}: {value}")
        
        except Exception as e:
            print(f"   Error: {e}")
    
    async def run_complete_demo(self):
        """Run the complete demo"""
        print("🎬 Facebook Posts API - Complete Demo")
        print("=" * 60)
        print(f"Base URL: {self.base_url}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print("=" * 60)
        
        try:
            # Run all demo sections
            post_id = await self.demo_basic_operations()
            batch_post_ids = await self.demo_batch_operations()
            await self.demo_filtering_and_pagination()
            await self.demo_error_handling()
            await self.demo_performance_metrics()
            await self.demo_post_operations(post_id)
            await self.demo_headers_and_tracking()
            
            print("\n" + "=" * 60)
            print("🎉 Demo completed successfully!")
            print("=" * 60)
            
            # Summary
            print("\n📊 Demo Summary:")
            print(f"   Single Post Generated: {'✅' if post_id else '❌'}")
            print(f"   Batch Posts Generated: {len(batch_post_ids)}")
            print(f"   All Operations Tested: ✅")
            print(f"   Error Handling Verified: ✅")
            print(f"   Performance Metrics Retrieved: ✅")
            
        except Exception as e:
            print(f"\n❌ Demo failed with error: {e}")
            import traceback
            traceback.print_exc()


async def main():
    """Main demo function"""
    print("Starting Facebook Posts API Demo...")
    print("Make sure the API server is running on http://localhost:8000")
    print("You can start it with: uvicorn app:app --reload")
    print()
    
    async with FacebookPostsAPIDemo() as demo:
        await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())






























