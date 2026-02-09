from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import asyncio
import time
import json
import uuid
from typing import Dict, Any, List
import httpx
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from ASYNC_NON_BLOCKING_FLOWS_IMPLEMENTATION import (
            from ASYNC_NON_BLOCKING_FLOWS_IMPLEMENTATION import AsyncPipeline
            import traceback
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
LinkedIn Posts - Async Flows Demo
================================

Demo script showcasing asynchronous and non-blocking flows
with event-driven architecture, flow orchestration, and performance optimizations.
"""


# Import the main implementation
    AsyncLinkedInPostsAPI,
    LinkedInPostsFlowOrchestrator,
    FlowRequest,
    Event,
    EventType,
    FlowStatus
)

# Demo configuration
DEMO_CONFIG = {
    "host": "127.0.0.1",
    "port": 8000,
    "base_url": "http://127.0.0.1:8000",
    "test_posts": [
        {
            "content": "🚀 Excited to share some amazing insights about AI and machine learning! The future of technology is here, and it's absolutely incredible what we can achieve with the right tools and mindset. #AI #MachineLearning #Innovation #TechTrends #FutureOfWork",
            "post_type": "educational",
            "tone": "enthusiastic",
            "target_audience": "developers"
        },
        {
            "content": "💼 Professional networking is more important than ever in today's digital world. Building meaningful connections and maintaining authentic relationships can open doors to incredible opportunities. #Networking #ProfessionalDevelopment #CareerGrowth #LinkedInTips",
            "post_type": "professional",
            "tone": "professional",
            "target_audience": "executives"
        },
        {
            "content": "🎯 Marketing strategies that actually work in 2024! From content marketing to social media engagement, here are the proven techniques that drive real results. #Marketing #DigitalMarketing #Strategy #Growth #Business",
            "post_type": "promotional",
            "tone": "casual",
            "target_audience": "marketers"
        },
        {
            "content": "🌟 Personal growth and development are key to success in any field. Continuous learning, adaptability, and resilience are the cornerstones of professional excellence. #PersonalGrowth #Development #Success #Learning #Excellence",
            "post_type": "personal",
            "tone": "thoughtful",
            "target_audience": "general"
        },
        {
            "content": "🏭 Industry insights and trends that are shaping the future of business. Understanding these dynamics is crucial for staying ahead of the competition. #Industry #Trends #Business #Innovation #Strategy",
            "post_type": "industry",
            "tone": "professional",
            "target_audience": "executives"
        }
    ]
}

class AsyncFlowsDemo:
    def __init__(self) -> Any:
        self.api = AsyncLinkedInPostsAPI()
        self.orchestrator = LinkedInPostsFlowOrchestrator()
        self.client = None
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
    
    async def setup(self) -> Any:
        """Setup the demo environment"""
        print("🚀 Setting up Async Flows Demo...")
        
        # Initialize the orchestrator
        await self.orchestrator.initialize()
        
        # Create HTTP client
        self.client = httpx.AsyncClient(
            base_url=DEMO_CONFIG["base_url"],
            timeout=30.0
        )
        
        print("✅ Demo setup complete!")
    
    async def cleanup(self) -> Any:
        """Cleanup demo resources"""
        if self.client:
            await self.client.aclose()
        
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        
        await self.orchestrator.shutdown()
        
        print("🧹 Demo cleanup complete!")
    
    async def demo_sequential_flow(self) -> Any:
        """Demo sequential async flow"""
        print("\n📋 Testing Sequential Async Flow...")
        
        try:
            # Create a sequential flow using the pipeline
            
            pipeline = AsyncPipeline("demo_sequential")
            pipeline.add_stage(self._validate_data)
            pipeline.add_stage(self._process_data)
            pipeline.add_stage(self._save_data)
            pipeline.add_stage(self._notify_completion)
            
            start_time = time.time()
            result = await pipeline.execute({"message": "Hello World"})
            duration = time.time() - start_time
            
            print(f"✅ Sequential flow completed in {duration:.3f}s")
            print(f"📊 Result: {result}")
            
        except Exception as e:
            print(f"❌ Sequential flow failed: {e}")
    
    async def demo_parallel_flow(self) -> Any:
        """Demo parallel async flow"""
        print("\n⚡ Testing Parallel Async Flow...")
        
        try:
            # Create parallel tasks
            tasks = [
                self._fetch_user_data("user1"),
                self._fetch_post_analytics("post1"),
                self._fetch_external_data("https://jsonplaceholder.typicode.com/posts/1"),
                self._process_background_task("task1")
            ]
            
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            duration = time.time() - start_time
            
            print(f"✅ Parallel flow completed in {duration:.3f}s")
            print(f"📊 Results: {len(results)} tasks completed")
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"  ❌ Task {i+1}: {result}")
                else:
                    print(f"  ✅ Task {i+1}: {type(result).__name__}")
            
        except Exception as e:
            print(f"❌ Parallel flow failed: {e}")
    
    async def demo_conditional_flow(self) -> Any:
        """Demo conditional async flow"""
        print("\n🔄 Testing Conditional Async Flow...")
        
        try:
            conditions = [True, False, True, False]
            
            for i, condition in enumerate(conditions, 1):
                print(f"\n🔄 Testing condition {i}: {condition}")
                
                start_time = time.time()
                
                if condition:
                    result = await self._high_priority_flow(f"priority-{i}")
                else:
                    result = await self._background_flow(f"background-{i}")
                
                duration = time.time() - start_time
                
                print(f"✅ Conditional flow {i} completed in {duration:.3f}s")
                print(f"📊 Result: {result}")
            
        except Exception as e:
            print(f"❌ Conditional flow failed: {e}")
    
    async def demo_event_driven_flow(self) -> Any:
        """Demo event-driven flow"""
        print("\n📡 Testing Event-Driven Flow...")
        
        try:
            # Create events
            events = [
                Event(
                    event_type=EventType.POST_CREATED,
                    data={"post_id": "post1", "content": "Test post 1"}
                ),
                Event(
                    event_type=EventType.ANALYTICS_REQUESTED,
                    data={"post_id": "post2", "analytics_type": "engagement"}
                ),
                Event(
                    event_type=EventType.NOTIFICATION_SENT,
                    data={"post_id": "post3", "notification_type": "email"}
                )
            ]
            
            # Emit events
            for event in events:
                await self.orchestrator.event_system.emit_event(event)
                print(f"📡 Emitted event: {event.event_type.value}")
            
            # Wait for events to be processed
            await asyncio.sleep(2)
            
            print("✅ Event-driven flow completed")
            
        except Exception as e:
            print(f"❌ Event-driven flow failed: {e}")
    
    async def demo_resource_limited_flow(self) -> Any:
        """Demo resource-limited flow"""
        print("\n🔒 Testing Resource-Limited Flow...")
        
        try:
            # Create multiple flows to test resource limits
            flow_data_list = [
                {"flow_type": "create_post", "data": {"content": f"Test post {i}"}}
                for i in range(15)  # More than the semaphore limit
            ]
            
            # Enqueue flows
            flow_ids = []
            for flow_data in flow_data_list:
                flow_id = await self.orchestrator.resource_limited_flow.enqueue_flow(flow_data)
                flow_ids.append(flow_id)
                print(f"📋 Enqueued flow: {flow_id}")
            
            # Wait for flows to complete
            await asyncio.sleep(5)
            
            print(f"✅ Resource-limited flow completed: {len(flow_ids)} flows processed")
            
        except Exception as e:
            print(f"❌ Resource-limited flow failed: {e}")
    
    async def demo_create_post_flow(self) -> Any:
        """Demo create post flow via API"""
        print("\n📝 Testing Create Post Flow via API...")
        
        created_posts = []
        
        for i, post_data in enumerate(DEMO_CONFIG["test_posts"][:3], 1):
            print(f"\n📄 Creating post {i}/{3}...")
            
            try:
                start_time = time.time()
                
                request = FlowRequest(
                    flow_type="create_post",
                    data=post_data,
                    priority=1,
                    timeout=30.0
                )
                
                response = await self.client.post(
                    "/api/v1/flows/create-post",
                    json=request.dict()
                )
                
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    created_posts.append(result)
                    print(f"✅ Post flow completed in {duration:.3f}s")
                    print(f"🆔 Flow ID: {result['flow_id']}")
                    print(f"📊 Status: {result['status']}")
                    print(f"⏱️  Duration: {result['duration']:.3f}s")
                else:
                    print(f"❌ Failed to create post: {response.status_code}")
                    print(f"📄 Error: {response.text}")
                
            except Exception as e:
                print(f"❌ Error creating post: {e}")
        
        return created_posts
    
    async def demo_update_post_flow(self) -> Any:
        """Demo update post flow via API"""
        print("\n✏️ Testing Update Post Flow via API...")
        
        try:
            # Create a post first
            create_request = FlowRequest(
                flow_type="create_post",
                data=DEMO_CONFIG["test_posts"][0]
            )
            
            create_response = await self.client.post(
                "/api/v1/flows/create-post",
                json=create_request.dict()
            )
            
            if create_response.status_code == 200:
                created_post = create_response.json()
                post_id = created_post['result']['post_id']
                
                # Update the post
                update_request = FlowRequest(
                    flow_type="update_post",
                    data={
                        "post_id": post_id,
                        "updates": {
                            "content": "Updated content with new insights!",
                            "tone": "enthusiastic"
                        }
                    }
                )
                
                start_time = time.time()
                
                update_response = await self.client.post(
                    "/api/v1/flows/update-post",
                    json=update_request.dict()
                )
                
                duration = time.time() - start_time
                
                if update_response.status_code == 200:
                    result = update_response.json()
                    print(f"✅ Update flow completed in {duration:.3f}s")
                    print(f"🆔 Flow ID: {result['flow_id']}")
                    print(f"📊 Status: {result['status']}")
                else:
                    print(f"❌ Failed to update post: {update_response.status_code}")
            
        except Exception as e:
            print(f"❌ Error updating post: {e}")
    
    async def demo_batch_processing_flow(self) -> Any:
        """Demo batch processing flow via API"""
        print("\n📦 Testing Batch Processing Flow via API...")
        
        try:
            batch_request = FlowRequest(
                flow_type="batch_process",
                data={
                    "posts": DEMO_CONFIG["test_posts"]
                }
            )
            
            start_time = time.time()
            
            response = await self.client.post(
                "/api/v1/flows/batch-process",
                json=batch_request.dict()
            )
            
            duration = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Batch processing flow completed in {duration:.3f}s")
                print(f"🆔 Flow ID: {result['flow_id']}")
                print(f"📊 Status: {result['status']}")
                print(f"📈 Batch Results:")
                print(f"  📊 Total posts: {result['result']['total_posts']}")
                print(f"  ✅ Successful: {result['result']['successful_posts']}")
                print(f"  ❌ Failed: {result['result']['failed_posts']}")
                print(f"  📊 Success rate: {result['result']['success_rate']:.1%}")
            else:
                print(f"❌ Failed to process batch: {response.status_code}")
                print(f"📄 Error: {response.text}")
            
        except Exception as e:
            print(f"❌ Error processing batch: {e}")
    
    async def demo_concurrent_flows(self) -> Any:
        """Demo handling concurrent flows"""
        print("\n⚡ Testing Concurrent Flows...")
        
        try:
            # Create multiple concurrent flow requests
            async def make_flow_request(flow_id: int):
                
    """make_flow_request function."""
try:
                    start_time = time.time()
                    
                    request = FlowRequest(
                        flow_type="create_post",
                        data={
                            "content": f"Concurrent test post {flow_id}",
                            "post_type": "educational"
                        }
                    )
                    
                    response = await self.client.post(
                        "/api/v1/flows/create-post",
                        json=request.dict()
                    )
                    
                    duration = time.time() - start_time
                    
                    return {
                        "flow_id": flow_id,
                        "status_code": response.status_code,
                        "duration": duration,
                        "success": response.status_code == 200
                    }
                    
                except Exception as e:
                    return {
                        "flow_id": flow_id,
                        "error": str(e),
                        "success": False
                    }
            
            # Create 10 concurrent requests
            tasks = [make_flow_request(i) for i in range(1, 11)]
            
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            total_duration = time.time() - start_time
            
            successful_flows = sum(1 for r in results if r["success"])
            avg_duration = sum(r.get("duration", 0) for r in results if r["success"]) / successful_flows if successful_flows > 0 else 0
            
            print(f"✅ Concurrent flows completed in {total_duration:.3f}s")
            print(f"📊 Successful flows: {successful_flows}/10")
            print(f"📈 Average flow duration: {avg_duration:.3f}s")
            
            # Show individual results
            for result in results:
                if result["success"]:
                    print(f"  ✅ Flow {result['flow_id']}: {result['duration']:.3f}s")
                else:
                    print(f"  ❌ Flow {result['flow_id']}: {result.get('error', 'Unknown error')}")
            
        except Exception as e:
            print(f"❌ Error testing concurrent flows: {e}")
    
    async def demo_queue_status(self) -> Any:
        """Demo queue status monitoring"""
        print("\n📊 Testing Queue Status Monitoring...")
        
        try:
            response = await self.client.get("/api/v1/flows/queue-status")
            
            if response.status_code == 200:
                status = response.json()
                print("✅ Queue status retrieved:")
                print(f"  📡 Event queue size: {status['event_queue_size']}")
                print(f"  📋 Flow queue size: {status['flow_queue_size']}")
                print(f"  👥 Active event workers: {status['active_workers']}")
                print(f"  🔧 Flow workers: {status['flow_workers']}")
            else:
                print(f"❌ Failed to get queue status: {response.status_code}")
            
        except Exception as e:
            print(f"❌ Error getting queue status: {e}")
    
    async def demo_metrics(self) -> Any:
        """Demo Prometheus metrics"""
        print("\n📈 Testing Prometheus Metrics...")
        
        try:
            response = await self.client.get("/metrics")
            
            if response.status_code == 200:
                metrics = response.text
                print("✅ Metrics endpoint accessible")
                print(f"📊 Metrics content length: {len(metrics)} characters")
                
                # Parse and display key metrics
                lines = metrics.split('\n')
                key_metrics = [
                    'flow_duration_seconds',
                    'flow_success_total',
                    'flow_failures_total',
                    'flow_concurrent_total',
                    'event_processed_total'
                ]
                
                print("\n📈 Key Metrics:")
                for line in lines:
                    for metric in key_metrics:
                        if metric in line:
                            print(f"  {line}")
                            break
            else:
                print(f"❌ Failed to get metrics: {response.status_code}")
            
        except Exception as e:
            print(f"❌ Error getting metrics: {e}")
    
    async def demo_performance_comparison(self) -> Any:
        """Demo performance comparison between different flow patterns"""
        print("\n⚡ Performance Comparison Demo...")
        
        # Test sequential vs parallel processing
        async def sequential_processing():
            """Sequential processing of multiple operations"""
            results = []
            for i in range(5):
                result = await self._simulate_operation(f"seq-{i}")
                results.append(result)
            return results
        
        async def parallel_processing():
            """Parallel processing of multiple operations"""
            tasks = [self._simulate_operation(f"par-{i}") for i in range(5)]
            return await asyncio.gather(*tasks)
        
        # Test sequential processing
        print("🔄 Testing sequential processing...")
        start_time = time.time()
        seq_results = await sequential_processing()
        seq_duration = time.time() - start_time
        print(f"⏱️  Sequential processing completed in {seq_duration:.3f}s")
        
        # Test parallel processing
        print("⚡ Testing parallel processing...")
        start_time = time.time()
        par_results = await parallel_processing()
        par_duration = time.time() - start_time
        print(f"⏱️  Parallel processing completed in {par_duration:.3f}s")
        
        # Calculate improvement
        improvement = ((seq_duration - par_duration) / seq_duration) * 100
        print(f"🚀 Performance improvement: {improvement:.1f}%")
        print(f"📊 Speedup factor: {seq_duration / par_duration:.2f}x")
    
    # Helper methods for demo flows
    async def _validate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate data validation"""
        await asyncio.sleep(0.1)
        data['validated'] = True
        return data
    
    async def _process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate data processing"""
        await asyncio.sleep(0.2)
        data['processed'] = True
        return data
    
    async def _save_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate data saving"""
        await asyncio.sleep(0.1)
        data['saved'] = True
        return data
    
    async def _notify_completion(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate notification sending"""
        await asyncio.sleep(0.1)
        data['notified'] = True
        return data
    
    async async def _fetch_user_data(self, user_id: str) -> Dict[str, Any]:
        """Simulate fetching user data"""
        await asyncio.sleep(0.3)
        return {"user_id": user_id, "name": f"User {user_id}"}
    
    async async def _fetch_post_analytics(self, post_id: str) -> Dict[str, Any]:
        """Simulate fetching post analytics"""
        await asyncio.sleep(0.2)
        return {"post_id": post_id, "engagement": 0.75}
    
    async async def _fetch_external_data(self, url: str) -> Dict[str, Any]:
        """Simulate fetching external data"""
        await asyncio.sleep(0.4)
        return {"url": url, "data": "External data"}
    
    async def _process_background_task(self, task_id: str) -> Dict[str, Any]:
        """Simulate background task processing"""
        await asyncio.sleep(0.1)
        return {"task_id": task_id, "status": "completed"}
    
    async def _high_priority_flow(self, flow_id: str) -> Dict[str, Any]:
        """Simulate high-priority flow"""
        await asyncio.sleep(0.2)
        return {"flow_id": flow_id, "priority": "high", "completed": True}
    
    async def _background_flow(self, flow_id: str) -> Dict[str, Any]:
        """Simulate background flow"""
        await asyncio.sleep(0.5)
        return {"flow_id": flow_id, "priority": "background", "completed": True}
    
    async def _simulate_operation(self, operation_id: str) -> Dict[str, Any]:
        """Simulate a generic operation"""
        await asyncio.sleep(0.2)  # Simulate work
        return {"operation_id": operation_id, "result": "success"}
    
    async def run_full_demo(self) -> Any:
        """Run the complete demo"""
        print("🎬 Starting Async Flows Demo")
        print("=" * 50)
        
        try:
            # Setup
            await self.setup()
            
            # Run all demo scenarios
            await self.demo_sequential_flow()
            await self.demo_parallel_flow()
            await self.demo_conditional_flow()
            await self.demo_event_driven_flow()
            await self.demo_resource_limited_flow()
            
            # API demos
            await self.demo_create_post_flow()
            await self.demo_update_post_flow()
            await self.demo_batch_processing_flow()
            await self.demo_concurrent_flows()
            
            # Monitoring demos
            await self.demo_queue_status()
            await self.demo_metrics()
            
            # Performance comparison
            await self.demo_performance_comparison()
            
            print("\n" + "=" * 50)
            print("🎉 Async Flows Demo completed successfully!")
            print("\n📋 Key Takeaways:")
            print("✅ All flows are non-blocking and async")
            print("✅ Event-driven architecture for decoupled processing")
            print("✅ Resource limits prevent system overload")
            print("✅ Parallel processing for improved performance")
            print("✅ Comprehensive monitoring and metrics")
            print("✅ Error handling and resilience patterns")
            print("✅ Flow orchestration and pipeline patterns")
            print("✅ Background task processing")
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            traceback.print_exc()
        
        finally:
            # Cleanup
            await self.cleanup()

# CLI interface
async def main():
    """Main function"""
    demo = AsyncFlowsDemo()
    await demo.run_full_demo()

match __name__:
    case "__main__":
    asyncio.run(main()) 