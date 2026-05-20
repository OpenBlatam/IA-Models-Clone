from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import json
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import random
import string
from DEDICATED_ASYNC_OPERATIONS_IMPLEMENTATION import (
        import traceback
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
LinkedIn Posts - Dedicated Async Operations Demo
===============================================

Comprehensive demo showcasing dedicated async functions for database
and external API operations with performance testing and real-world usage.
"""


# Import the main implementation
    DatabaseConnectionPool,
    LinkedInPostsDatabase,
    ExternalAPISession,
    LinkedInAPI,
    AIAnalysisAPI,
    NotificationAPI,
    LinkedInPostsService,
    PostData,
    PostUpdate,
    LinkedInPostRequest
)

# Mock external services for demo
class MockLinkedInAPI:
    """Mock LinkedIn API for demo purposes"""
    
    async def create_linkedin_post(self, post_data: PostData, access_token: str) -> Dict[str, Any]:
        """Mock LinkedIn post creation"""
        await asyncio.sleep(random.uniform(0.1, 0.5))  # Simulate API delay
        
        # Simulate occasional failures
        if random.random() < 0.05:  # 5% failure rate
            raise Exception("LinkedIn API temporarily unavailable")
        
        return {
            "id": f"urn:li:activity:{uuid.uuid4()}",
            "status": "published",
            "created_at": datetime.now(timezone.utc).isoformat()
        }
    
    async def get_linkedin_profile(self, user_id: str, access_token: str) -> Dict[str, Any]:
        """Mock LinkedIn profile retrieval"""
        await asyncio.sleep(random.uniform(0.05, 0.2))
        return {
            "id": user_id,
            "firstName": "John",
            "lastName": "Doe",
            "profilePicture": "https://example.com/profile.jpg"
        }
    
    async def get_linkedin_analytics(self, post_id: str, access_token: str) -> Dict[str, Any]:
        """Mock LinkedIn analytics retrieval"""
        await asyncio.sleep(random.uniform(0.1, 0.3))
        return {
            "views": random.randint(100, 10000),
            "likes": random.randint(10, 500),
            "comments": random.randint(0, 100),
            "shares": random.randint(0, 50)
        }

class MockAIAnalysisAPI:
    """Mock AI Analysis API for demo purposes"""
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Mock sentiment analysis"""
        await asyncio.sleep(random.uniform(0.2, 0.8))
        
        # Simulate sentiment analysis
        sentiment_score = random.uniform(-1.0, 1.0)
        readability_score = random.uniform(0.0, 1.0)
        engagement_prediction = random.uniform(0.0, 1.0)
        
        return {
            "sentiment_score": sentiment_score,
            "readability_score": readability_score,
            "engagement_prediction": engagement_prediction,
            "confidence": random.uniform(0.7, 0.95)
        }
    
    async def generate_hashtags(self, content: str) -> List[str]:
        """Mock hashtag generation"""
        await asyncio.sleep(random.uniform(0.1, 0.4))
        
        # Generate mock hashtags based on content
        hashtags = []
        content_lower = content.lower()
        
        if "ai" in content_lower or "artificial intelligence" in content_lower:
            hashtags.extend(["#AI", "#ArtificialIntelligence", "#MachineLearning"])
        if "business" in content_lower:
            hashtags.extend(["#Business", "#Entrepreneurship", "#Leadership"])
        if "technology" in content_lower:
            hashtags.extend(["#Technology", "#Innovation", "#DigitalTransformation"])
        if "marketing" in content_lower:
            hashtags.extend(["#Marketing", "#DigitalMarketing", "#Growth"])
        
        # Add some generic hashtags
        hashtags.extend(["#LinkedIn", "#Professional", "#Networking"])
        
        return hashtags[:5]  # Return max 5 hashtags
    
    async def optimize_content(self, content: str, optimization_type: str) -> str:
        """Mock content optimization"""
        await asyncio.sleep(random.uniform(0.3, 1.0))
        
        # Simple content optimization simulation
        if optimization_type == "engagement":
            optimized = f"🚀 {content} 💡"
        elif optimization_type == "professional":
            optimized = f"📈 {content} 📊"
        elif optimization_type == "casual":
            optimized = f"Hey there! {content} 😊"
        else:
            optimized = content
        
        return optimized

class MockNotificationAPI:
    """Mock Notification API for demo purposes"""
    
    async def send_email_notification(self, user_email: str, subject: str, content: str) -> bool:
        """Mock email notification"""
        await asyncio.sleep(random.uniform(0.05, 0.15))
        return True
    
    async def send_push_notification(self, user_id: str, title: str, message: str) -> bool:
        """Mock push notification"""
        await asyncio.sleep(random.uniform(0.02, 0.08))
        return True

class MockDatabase:
    """Mock database for demo purposes"""
    
    def __init__(self) -> Any:
        self.posts = {}
        self.analytics = {}
    
    async def create_post(self, post_data: PostData) -> str:
        """Mock post creation"""
        await asyncio.sleep(random.uniform(0.01, 0.05))
        
        post_id = str(uuid.uuid4())
        self.posts[post_id] = {
            "id": post_id,
            "content": post_data.content,
            "post_type": post_data.post_type,
            "tone": post_data.tone,
            "target_audience": post_data.target_audience,
            "user_id": post_data.user_id,
            "hashtags": post_data.hashtags,
            "call_to_action": post_data.call_to_action,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "status": "draft"
        }
        
        return post_id
    
    async def get_post_by_id(self, post_id: str) -> Optional[Dict[str, Any]]:
        """Mock post retrieval"""
        await asyncio.sleep(random.uniform(0.005, 0.02))
        return self.posts.get(post_id)
    
    async def update_post(self, post_id: str, updates: PostUpdate) -> bool:
        """Mock post update"""
        await asyncio.sleep(random.uniform(0.01, 0.03))
        
        if post_id not in self.posts:
            return False
        
        update_dict = updates.dict(exclude_unset=True)
        for key, value in update_dict.items():
            self.posts[post_id][key] = value
        
        self.posts[post_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
        return True
    
    async def update_post_analytics(self, post_id: str, analytics: Dict[str, Any]) -> bool:
        """Mock analytics update"""
        await asyncio.sleep(random.uniform(0.005, 0.015))
        
        if post_id not in self.posts:
            return False
        
        self.analytics[post_id] = analytics
        return True
    
    async def get_posts_by_user(self, user_id: str, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        """Mock user posts retrieval"""
        await asyncio.sleep(random.uniform(0.01, 0.05))
        
        user_posts = [
            post for post in self.posts.values() 
            if post["user_id"] == user_id
        ]
        
        # Sort by created_at descending
        user_posts.sort(key=lambda x: x["created_at"], reverse=True)
        
        return user_posts[offset:offset + limit]
    
    async def search_posts(self, search_term: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Mock post search"""
        await asyncio.sleep(random.uniform(0.02, 0.08))
        
        matching_posts = [
            post for post in self.posts.values()
            if search_term.lower() in post["content"].lower()
        ]
        
        matching_posts.sort(key=lambda x: x["created_at"], reverse=True)
        return matching_posts[:limit]

class DedicatedAsyncOperationsDemo:
    """Demo class for showcasing dedicated async operations"""
    
    def __init__(self) -> Any:
        self.mock_db = MockDatabase()
        self.mock_linkedin_api = MockLinkedInAPI()
        self.mock_ai_api = MockAIAnalysisAPI()
        self.mock_notification_api = MockNotificationAPI()
        
        # Performance tracking
        self.performance_metrics = {
            "database_operations": [],
            "api_operations": [],
            "combined_operations": []
        }
    
    def generate_sample_post_data(self, post_type: str = "educational") -> PostData:
        """Generate sample post data for testing"""
        sample_contents = {
            "educational": [
                "🚀 The future of AI in business is here! Companies are leveraging machine learning to transform their operations and drive innovation. Here's what you need to know about implementing AI solutions in your organization. #AI #Business #Innovation",
                "📊 Data-driven decision making is no longer optional. Organizations that embrace analytics and insights are outperforming their competitors by 23%. Learn how to build a data-first culture in your team. #DataAnalytics #Leadership #Growth",
                "💡 Remote work is here to stay, but it requires a new approach to team collaboration. Discover the tools and strategies that successful remote teams use to maintain productivity and engagement. #RemoteWork #Collaboration #Productivity"
            ],
            "promotional": [
                "🎉 Excited to announce our new AI-powered LinkedIn post generator! Create engaging, professional content in seconds. Try it free today and see the difference it makes in your social media strategy. #ProductLaunch #AI #Marketing",
                "🔥 Limited time offer: 50% off our premium LinkedIn automation tools! Boost your social media presence and save hours every week. Don't miss this opportunity to transform your digital marketing. #SpecialOffer #LinkedIn #Automation",
                "🌟 Join thousands of professionals who have transformed their LinkedIn presence with our platform. Start your free trial today and see why we're the #1 choice for social media automation. #FreeTrial #LinkedIn #Success"
            ],
            "personal": [
                "🙏 Grateful for the amazing team that made this project possible. Sometimes the best innovations come from collaboration and shared vision. Thank you to everyone who believed in this idea from day one. #Gratitude #Teamwork #Innovation",
                "🎯 After months of hard work, we finally launched our biggest feature yet! The journey has been incredible, and I've learned so much about building products that users actually love. Here's to the next chapter! #ProductLaunch #Journey #Learning",
                "💪 Today marks 5 years since I started this entrepreneurial journey. The ups and downs have taught me more than I ever imagined. To anyone thinking about starting their own business: just start. You'll figure it out along the way. #Entrepreneurship #Journey #Growth"
            ]
        }
        
        content = random.choice(sample_contents[post_type])
        
        return PostData(
            content=content,
            post_type=post_type,
            tone=random.choice(["professional", "casual", "enthusiastic", "thoughtful"]),
            target_audience=random.choice(["general", "executives", "developers", "marketers"]),
            user_id=f"user_{random.randint(1000, 9999)}",
            hashtags=[],
            call_to_action=random.choice([
                "What's your experience with this?",
                "Share your thoughts below!",
                "Let me know what you think!",
                "Would love to hear your perspective!"
            ])
        )
    
    async def demo_database_operations(self, num_operations: int = 10):
        """Demo dedicated database operations"""
        print(f"\n🔍 Demo: Database Operations ({num_operations} operations)")
        print("=" * 60)
        
        start_time = time.time()
        created_posts = []
        
        # Create posts
        for i in range(num_operations):
            post_data = self.generate_sample_post_data()
            operation_start = time.time()
            
            post_id = await self.mock_db.create_post(post_data)
            created_posts.append(post_id)
            
            operation_duration = time.time() - operation_start
            self.performance_metrics["database_operations"].append({
                "operation": "create_post",
                "duration": operation_duration,
                "post_id": post_id
            })
            
            print(f"✅ Created post {i+1}/{num_operations}: {post_id[:8]}... ({operation_duration:.3f}s)")
        
        # Retrieve posts
        for i, post_id in enumerate(created_posts[:5]):  # Test first 5
            operation_start = time.time()
            
            post = await self.mock_db.get_post_by_id(post_id)
            
            operation_duration = time.time() - operation_start
            self.performance_metrics["database_operations"].append({
                "operation": "get_post",
                "duration": operation_duration,
                "post_id": post_id
            })
            
            print(f"📖 Retrieved post {i+1}/5: {post_id[:8]}... ({operation_duration:.3f}s)")
        
        # Update posts
        for i, post_id in enumerate(created_posts[:3]):  # Test first 3
            operation_start = time.time()
            
            updates = PostUpdate(
                content=f"Updated content for post {i+1}",
                hashtags=["#Updated", "#Demo", "#LinkedIn"]
            )
            
            success = await self.mock_db.update_post(post_id, updates)
            
            operation_duration = time.time() - operation_start
            self.performance_metrics["database_operations"].append({
                "operation": "update_post",
                "duration": operation_duration,
                "post_id": post_id,
                "success": success
            })
            
            print(f"✏️  Updated post {i+1}/3: {post_id[:8]}... ({operation_duration:.3f}s)")
        
        total_duration = time.time() - start_time
        avg_duration = total_duration / len(self.performance_metrics["database_operations"])
        
        print(f"\n📊 Database Operations Summary:")
        print(f"   Total operations: {len(self.performance_metrics['database_operations'])}")
        print(f"   Total duration: {total_duration:.3f}s")
        print(f"   Average duration: {avg_duration:.3f}s")
        print(f"   Operations per second: {len(self.performance_metrics['database_operations']) / total_duration:.1f}")
    
    async def demo_api_operations(self, num_operations: int = 10):
        """Demo dedicated API operations"""
        print(f"\n🌐 Demo: API Operations ({num_operations} operations)")
        print("=" * 60)
        
        start_time = time.time()
        
        # Test LinkedIn API operations
        for i in range(num_operations // 3):
            post_data = self.generate_sample_post_data()
            
            # Create LinkedIn post
            operation_start = time.time()
            try:
                linkedin_result = await self.mock_linkedin_api.create_linkedin_post(
                    post_data, "mock_access_token"
                )
                operation_duration = time.time() - operation_start
                
                self.performance_metrics["api_operations"].append({
                    "operation": "create_linkedin_post",
                    "duration": operation_duration,
                    "success": True
                })
                
                print(f"✅ LinkedIn post {i+1}: {linkedin_result['id'][:20]}... ({operation_duration:.3f}s)")
                
            except Exception as e:
                operation_duration = time.time() - operation_start
                self.performance_metrics["api_operations"].append({
                    "operation": "create_linkedin_post",
                    "duration": operation_duration,
                    "success": False,
                    "error": str(e)
                })
                print(f"❌ LinkedIn post {i+1}: Failed ({operation_duration:.3f}s) - {e}")
        
        # Test AI Analysis operations
        sample_texts = [
            "AI is transforming the way we work and live.",
            "Data-driven decisions lead to better business outcomes.",
            "Remote work requires new collaboration strategies."
        ]
        
        for i, text in enumerate(sample_texts):
            # Sentiment analysis
            operation_start = time.time()
            sentiment_result = await self.mock_ai_api.analyze_sentiment(text)
            operation_duration = time.time() - operation_start
            
            self.performance_metrics["api_operations"].append({
                "operation": "analyze_sentiment",
                "duration": operation_duration,
                "success": True
            })
            
            print(f"🧠 Sentiment analysis {i+1}: {sentiment_result['sentiment_score']:.2f} ({operation_duration:.3f}s)")
            
            # Hashtag generation
            operation_start = time.time()
            hashtags = await self.mock_ai_api.generate_hashtags(text)
            operation_duration = time.time() - operation_start
            
            self.performance_metrics["api_operations"].append({
                "operation": "generate_hashtags",
                "duration": operation_duration,
                "success": True
            })
            
            print(f"🏷️  Hashtags {i+1}: {', '.join(hashtags)} ({operation_duration:.3f}s)")
        
        # Test notification operations
        for i in range(3):
            operation_start = time.time()
            email_success = await self.mock_notification_api.send_email_notification(
                f"user{i}@example.com",
                "Test Notification",
                "This is a test notification from the demo."
            )
            operation_duration = time.time() - operation_start
            
            self.performance_metrics["api_operations"].append({
                "operation": "send_email_notification",
                "duration": operation_duration,
                "success": email_success
            })
            
            print(f"📧 Email notification {i+1}: {'✅' if email_success else '❌'} ({operation_duration:.3f}s)")
        
        total_duration = time.time() - start_time
        successful_operations = [op for op in self.performance_metrics["api_operations"] if op["success"]]
        
        print(f"\n📊 API Operations Summary:")
        print(f"   Total operations: {len(self.performance_metrics['api_operations'])}")
        print(f"   Successful operations: {len(successful_operations)}")
        print(f"   Total duration: {total_duration:.3f}s")
        print(f"   Average duration: {total_duration / len(self.performance_metrics['api_operations']):.3f}s")
        print(f"   Operations per second: {len(self.performance_metrics['api_operations']) / total_duration:.1f}")
    
    async def demo_combined_operations(self, num_posts: int = 5):
        """Demo combined database and API operations"""
        print(f"\n🔄 Demo: Combined Operations ({num_posts} posts)")
        print("=" * 60)
        
        start_time = time.time()
        results = []
        
        for i in range(num_posts):
            print(f"\n📝 Processing post {i+1}/{num_posts}...")
            post_start_time = time.time()
            
            # Generate post data
            post_data = self.generate_sample_post_data()
            
            # Step 1: Create post in database
            db_start = time.time()
            post_id = await self.mock_db.create_post(post_data)
            db_duration = time.time() - db_start
            
            print(f"   💾 Database: Created post {post_id[:8]}... ({db_duration:.3f}s)")
            
            # Step 2: Perform AI analysis in parallel
            ai_start = time.time()
            ai_tasks = [
                self.mock_ai_api.analyze_sentiment(post_data.content),
                self.mock_ai_api.generate_hashtags(post_data.content)
            ]
            
            sentiment_result, hashtags = await asyncio.gather(*ai_tasks)
            ai_duration = time.time() - ai_start
            
            print(f"   🤖 AI Analysis: Sentiment {sentiment_result['sentiment_score']:.2f}, {len(hashtags)} hashtags ({ai_duration:.3f}s)")
            
            # Step 3: Update post with analytics
            analytics_start = time.time()
            analytics = {
                'sentiment_score': sentiment_result['sentiment_score'],
                'readability_score': sentiment_result['readability_score'],
                'engagement_prediction': sentiment_result['engagement_prediction']
            }
            
            await self.mock_db.update_post_analytics(post_id, analytics)
            analytics_duration = time.time() - analytics_start
            
            print(f"   📊 Analytics: Updated post analytics ({analytics_duration:.3f}s)")
            
            # Step 4: Publish to LinkedIn
            linkedin_start = time.time()
            try:
                linkedin_result = await self.mock_linkedin_api.create_linkedin_post(
                    post_data, "mock_access_token"
                )
                linkedin_duration = time.time() - linkedin_start
                linkedin_success = True
                print(f"   🔗 LinkedIn: Published successfully ({linkedin_duration:.3f}s)")
                
            except Exception as e:
                linkedin_duration = time.time() - linkedin_start
                linkedin_success = False
                print(f"   ❌ LinkedIn: Failed to publish ({linkedin_duration:.3f}s) - {e}")
            
            # Step 5: Send notifications (fire and forget)
            notification_start = time.time()
            notification_tasks = [
                self.mock_notification_api.send_email_notification(
                    f"user-{post_data.user_id}@example.com",
                    "Post Created Successfully",
                    f"Your post '{post_data.content[:50]}...' has been created."
                ),
                self.mock_notification_api.send_push_notification(
                    post_data.user_id,
                    "Post Created",
                    "Your LinkedIn post has been created successfully!"
                )
            ]
            
            # Don't wait for notifications to complete
            asyncio.create_task(asyncio.gather(*notification_tasks, return_exceptions=True))
            notification_duration = time.time() - notification_start
            
            print(f"   📢 Notifications: Sent ({notification_duration:.3f}s)")
            
            # Record results
            post_duration = time.time() - post_start_time
            results.append({
                "post_id": post_id,
                "total_duration": post_duration,
                "db_duration": db_duration,
                "ai_duration": ai_duration,
                "analytics_duration": analytics_duration,
                "linkedin_duration": linkedin_duration,
                "notification_duration": notification_duration,
                "linkedin_success": linkedin_success
            })
            
            self.performance_metrics["combined_operations"].append({
                "operation": "create_post_with_analysis",
                "duration": post_duration,
                "post_id": post_id,
                "success": True
            })
            
            print(f"   ⏱️  Total: {post_duration:.3f}s")
        
        total_duration = time.time() - start_time
        
        print(f"\n📊 Combined Operations Summary:")
        print(f"   Total posts processed: {len(results)}")
        print(f"   Total duration: {total_duration:.3f}s")
        print(f"   Average per post: {total_duration / len(results):.3f}s")
        print(f"   Posts per second: {len(results) / total_duration:.2f}")
        
        # Calculate parallel efficiency
        total_sequential_time = sum(
            r["db_duration"] + r["ai_duration"] + r["analytics_duration"] + 
            r["linkedin_duration"] + r["notification_duration"] 
            for r in results
        )
        parallel_efficiency = total_sequential_time / total_duration
        
        print(f"   Parallel efficiency: {parallel_efficiency:.2f}x speedup")
        
        # Success rates
        linkedin_success_rate = sum(1 for r in results if r["linkedin_success"]) / len(results) * 100
        print(f"   LinkedIn success rate: {linkedin_success_rate:.1f}%")
    
    async def demo_concurrent_operations(self, num_concurrent: int = 10):
        """Demo concurrent operations for performance testing"""
        print(f"\n⚡ Demo: Concurrent Operations ({num_concurrent} concurrent)")
        print("=" * 60)
        
        start_time = time.time()
        
        # Create concurrent tasks
        tasks = []
        for i in range(num_concurrent):
            post_data = self.generate_sample_post_data()
            task = self.mock_db.create_post(post_data)
            tasks.append(task)
        
        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_duration = time.time() - start_time
        successful_operations = [r for r in results if not isinstance(r, Exception)]
        
        print(f"📊 Concurrent Operations Results:")
        print(f"   Concurrent operations: {num_concurrent}")
        print(f"   Successful operations: {len(successful_operations)}")
        print(f"   Failed operations: {len(results) - len(successful_operations)}")
        print(f"   Total duration: {total_duration:.3f}s")
        print(f"   Average per operation: {total_duration / len(results):.3f}s")
        print(f"   Operations per second: {len(results) / total_duration:.1f}")
        
        # Compare with sequential execution
        sequential_start = time.time()
        for i in range(min(5, num_concurrent)):  # Test with first 5
            post_data = self.generate_sample_post_data()
            await self.mock_db.create_post(post_data)
        sequential_duration = time.time() - sequential_start
        
        if num_concurrent >= 5:
            sequential_estimate = sequential_duration * (num_concurrent / 5)
            speedup = sequential_estimate / total_duration
            print(f"   Estimated sequential time: {sequential_estimate:.3f}s")
            print(f"   Concurrent speedup: {speedup:.2f}x")
    
    def print_performance_summary(self) -> Any:
        """Print comprehensive performance summary"""
        print(f"\n📈 Performance Summary")
        print("=" * 60)
        
        # Database operations
        db_ops = self.performance_metrics["database_operations"]
        if db_ops:
            db_avg = sum(op["duration"] for op in db_ops) / len(db_ops)
            db_total = sum(op["duration"] for op in db_ops)
            print(f"💾 Database Operations:")
            print(f"   Count: {len(db_ops)}")
            print(f"   Total time: {db_total:.3f}s")
            print(f"   Average time: {db_avg:.3f}s")
            print(f"   Throughput: {len(db_ops) / db_total:.1f} ops/sec")
        
        # API operations
        api_ops = self.performance_metrics["api_operations"]
        if api_ops:
            api_avg = sum(op["duration"] for op in api_ops) / len(api_ops)
            api_total = sum(op["duration"] for op in api_ops)
            api_success = sum(1 for op in api_ops if op["success"])
            print(f"\n🌐 API Operations:")
            print(f"   Count: {len(api_ops)}")
            print(f"   Successful: {api_success}")
            print(f"   Success rate: {api_success / len(api_ops) * 100:.1f}%")
            print(f"   Total time: {api_total:.3f}s")
            print(f"   Average time: {api_avg:.3f}s")
            print(f"   Throughput: {len(api_ops) / api_total:.1f} ops/sec")
        
        # Combined operations
        combined_ops = self.performance_metrics["combined_operations"]
        if combined_ops:
            combined_avg = sum(op["duration"] for op in combined_ops) / len(combined_ops)
            combined_total = sum(op["duration"] for op in combined_ops)
            print(f"\n🔄 Combined Operations:")
            print(f"   Count: {len(combined_ops)}")
            print(f"   Total time: {combined_total:.3f}s")
            print(f"   Average time: {combined_avg:.3f}s")
            print(f"   Throughput: {len(combined_ops) / combined_total:.2f} ops/sec")
        
        # Overall statistics
        all_ops = db_ops + api_ops + combined_ops
        if all_ops:
            total_time = sum(op["duration"] for op in all_ops)
            print(f"\n🎯 Overall Statistics:")
            print(f"   Total operations: {len(all_ops)}")
            print(f"   Total time: {total_time:.3f}s")
            print(f"   Average time: {total_time / len(all_ops):.3f}s")
            print(f"   Overall throughput: {len(all_ops) / total_time:.1f} ops/sec")

async def main():
    """Main demo function"""
    print("🚀 LinkedIn Posts - Dedicated Async Operations Demo")
    print("=" * 60)
    print("This demo showcases dedicated async functions for database")
    print("and external API operations with performance optimization.")
    print()
    
    demo = DedicatedAsyncOperationsDemo()
    
    try:
        # Run database operations demo
        await demo.demo_database_operations(15)
        
        # Run API operations demo
        await demo.demo_api_operations(12)
        
        # Run combined operations demo
        await demo.demo_combined_operations(3)
        
        # Run concurrent operations demo
        await demo.demo_concurrent_operations(20)
        
        # Print performance summary
        demo.print_performance_summary()
        
        print(f"\n✅ Demo completed successfully!")
        print("Key takeaways:")
        print("  • Dedicated async functions provide clear separation of concerns")
        print("  • Connection pooling improves database performance")
        print("  • Parallel execution reduces total processing time")
        print("  • Error handling and retry logic improve reliability")
        print("  • Monitoring and metrics enable performance optimization")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        traceback.print_exc()

match __name__:
    case "__main__":
    asyncio.run(main()) 