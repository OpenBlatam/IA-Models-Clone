"""
🚀 ENHANCED BLOG SYSTEM DEMO
============================

Comprehensive demonstration of the enhanced blog system features:
- Full-text search with Elasticsearch
- Real-time analytics and metrics
- AI-powered content analysis
- WebSocket notifications
- Advanced pagination and filtering
- Content recommendation engine
- SEO optimization
"""

import asyncio
import time
import json
import aiohttp
import websockets
from typing import Dict, Any, List
from dataclasses import dataclass

# Import the enhanced system
from enhanced_blog_system_v3 import (
    EnhancedBlogSystem, EnhancedConfig, BlogPostCreate, 
    AnalyticsEvent, ContentAnalysis
)

@dataclass
class DemoConfig:
    """Demo configuration."""
    server_url: str = "http://localhost:8000"
    websocket_url: str = "ws://localhost:8000/ws/notifications"
    demo_posts: int = 10
    search_queries: List[str] = None
    analytics_events: List[str] = None
    
    def __post_init__(self):
        if self.search_queries is None:
            self.search_queries = [
                "technology", "business", "python", "fastapi", "machine learning",
                "web development", "data science", "artificial intelligence"
            ]
        if self.analytics_events is None:
            self.analytics_events = ["view", "like", "share", "comment"]

class EnhancedBlogDemo:
    """Enhanced blog system demonstration."""
    
    def __init__(self, config: DemoConfig):
        self.config = config
        self.session = None
        self.websocket = None
        self.notifications_received = []
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
        if self.websocket:
            await self.websocket.close()
    
    async def start_server(self):
        """Start the enhanced blog server."""
        print("🚀 Starting Enhanced Blog System...")
        
        # Create enhanced configuration
        enhanced_config = EnhancedConfig(
            debug=True,
            search=SearchConfig(enable_full_text_search=True),
            analytics=AnalyticsConfig(enable_real_time_analytics=True),
            ai=AIConfig(enable_content_analysis=True),
            notifications=NotificationConfig(enable_websocket_notifications=True)
        )
        
        # Create and start system
        self.system = create_enhanced_blog_system(enhanced_config)
        
        # Start server in background
        import uvicorn
        config = uvicorn.Config(
            self.system.app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
        self.server = uvicorn.Server(config)
        
        # Start server in background task
        self.server_task = asyncio.create_task(self.server.serve())
        
        # Wait for server to start
        await asyncio.sleep(3)
        print("✅ Enhanced Blog System started successfully!")
    
    async def create_sample_posts(self):
        """Create sample blog posts with rich content."""
        print(f"📝 Creating {self.config.demo_posts} sample posts...")
        
        sample_posts = [
            {
                "title": "The Future of Web Development with FastAPI",
                "content": """
                FastAPI has revolutionized web development with its modern approach to building APIs. 
                This framework combines the best of both worlds: the simplicity of Flask and the 
                performance of modern async frameworks. With automatic OpenAPI documentation, 
                type hints, and async/await support, FastAPI is becoming the go-to choice for 
                developers building high-performance web applications.
                
                Key benefits include:
                - Automatic API documentation
                - Type safety with Pydantic
                - High performance with async/await
                - Easy testing and validation
                - Built-in security features
                
                The future of web development is here, and FastAPI is leading the charge.
                """,
                "tags": ["fastapi", "python", "web development", "api"],
                "category": "technology",
                "author": "Tech Blogger",
                "is_published": True
            },
            {
                "title": "Machine Learning in Production: Best Practices",
                "content": """
                Deploying machine learning models in production requires careful consideration 
                of many factors. From data preprocessing to model serving, every step must be 
                optimized for reliability and performance. This comprehensive guide covers the 
                essential best practices for ML production systems.
                
                Topics covered:
                - Model versioning and management
                - Data pipeline optimization
                - Monitoring and observability
                - A/B testing strategies
                - Performance optimization
                
                Learn how to build robust ML systems that scale with your business needs.
                """,
                "tags": ["machine learning", "ai", "production", "best practices"],
                "category": "technology",
                "author": "ML Expert",
                "is_published": True
            },
            {
                "title": "Building Scalable Microservices with Python",
                "content": """
                Microservices architecture has become the standard for building scalable 
                applications. Python, with its rich ecosystem and excellent async support, 
                is an ideal choice for microservices development. This guide explores how 
                to design and implement microservices using modern Python tools and practices.
                
                Key concepts:
                - Service discovery and communication
                - Data consistency patterns
                - Monitoring and tracing
                - Deployment strategies
                - Testing microservices
                
                Discover how to build resilient, scalable systems with Python microservices.
                """,
                "tags": ["microservices", "python", "architecture", "scalability"],
                "category": "technology",
                "author": "Architecture Expert",
                "is_published": True
            },
            {
                "title": "Data Science Workflow Optimization",
                "content": """
                Efficient data science workflows are crucial for productivity and innovation. 
                From data exploration to model deployment, every step can be optimized for 
                better results and faster iteration cycles. This article presents proven 
                strategies for streamlining your data science workflow.
                
                Optimization areas:
                - Data pipeline automation
                - Experiment tracking and management
                - Model evaluation frameworks
                - Deployment automation
                - Team collaboration tools
                
                Transform your data science process with these optimization techniques.
                """,
                "tags": ["data science", "workflow", "optimization", "automation"],
                "category": "technology",
                "author": "Data Scientist",
                "is_published": True
            },
            {
                "title": "The Rise of Edge Computing in IoT",
                "content": """
                Edge computing is transforming the Internet of Things landscape by bringing 
                computation closer to data sources. This paradigm shift enables real-time 
                processing, reduced latency, and improved privacy for IoT applications. 
                Explore the latest trends and technologies in edge computing for IoT.
                
                Edge computing benefits:
                - Reduced latency and bandwidth usage
                - Enhanced privacy and security
                - Real-time processing capabilities
                - Cost-effective data processing
                - Improved reliability
                
                Discover how edge computing is shaping the future of IoT applications.
                """,
                "tags": ["edge computing", "iot", "real-time", "privacy"],
                "category": "technology",
                "author": "IoT Specialist",
                "is_published": True
            }
        ]
        
        created_posts = []
        for i, post_data in enumerate(sample_posts[:self.config.demo_posts]):
            try:
                post = BlogPostCreate(**post_data)
                async with self.session.post(
                    f"{self.config.server_url}/posts",
                    json=post.model_dump()
                ) as response:
                    if response.status == 201:
                        created_post = await response.json()
                        created_posts.append(created_post)
                        print(f"✅ Created post: {created_post['title']}")
                    else:
                        print(f"❌ Failed to create post {i+1}")
            except Exception as e:
                print(f"❌ Error creating post {i+1}: {e}")
        
        print(f"📊 Created {len(created_posts)} posts successfully!")
        return created_posts
    
    async def demonstrate_search(self):
        """Demonstrate full-text search capabilities."""
        print("🔍 Demonstrating search capabilities...")
        
        for query in self.config.search_queries:
            try:
                async with self.session.get(
                    f"{self.config.server_url}/posts/search",
                    params={"query": query, "limit": 5}
                ) as response:
                    if response.status == 200:
                        results = await response.json()
                        print(f"🔍 Search for '{query}': {len(results['results'])} results")
                        
                        for result in results['results'][:2]:  # Show first 2 results
                            print(f"  - {result['title']} (score: {result['score']:.2f})")
                    else:
                        print(f"❌ Search failed for query: {query}")
            except Exception as e:
                print(f"❌ Error searching for '{query}': {e}")
    
    async def demonstrate_analytics(self):
        """Demonstrate analytics tracking."""
        print("📊 Demonstrating analytics tracking...")
        
        # Get posts first
        async with self.session.get(f"{self.config.server_url}/posts") as response:
            if response.status == 200:
                posts_data = await response.json()
                posts = posts_data.get("posts", [])
                
                if posts:
                    # Track events for the first post
                    post_id = posts[0]["id"]
                    
                    for event_type in self.config.analytics_events:
                        event = AnalyticsEvent(
                            post_id=post_id,
                            event_type=event_type,
                            user_id="demo_user",
                            ip_address="127.0.0.1",
                            user_agent="Demo Browser",
                            session_id="demo_session"
                        )
                        
                        try:
                            async with self.session.post(
                                f"{self.config.server_url}/posts/{post_id}/track",
                                json=event.model_dump()
                            ) as track_response:
                                if track_response.status == 200:
                                    print(f"📊 Tracked {event_type} event for post {post_id}")
                                else:
                                    print(f"❌ Failed to track {event_type} event")
                        except Exception as e:
                            print(f"❌ Error tracking {event_type}: {e}")
                    
                    # Get analytics
                    try:
                        async with self.session.get(
                            f"{self.config.server_url}/posts/{post_id}/analytics"
                        ) as analytics_response:
                            if analytics_response.status == 200:
                                analytics = await analytics_response.json()
                                print(f"📈 Analytics for post {post_id}:")
                                print(f"  - Event counts: {analytics.get('event_counts', {})}")
                                print(f"  - Daily trends: {len(analytics.get('daily_trends', []))} days")
                            else:
                                print("❌ Failed to get analytics")
                    except Exception as e:
                        print(f"❌ Error getting analytics: {e}")
    
    async def demonstrate_ai_analysis(self):
        """Demonstrate AI content analysis."""
        print("🤖 Demonstrating AI content analysis...")
        
        sample_content = """
        FastAPI is an amazing framework for building high-performance web APIs. 
        It provides excellent developer experience with automatic documentation, 
        type safety, and modern async/await support. The performance is incredible, 
        making it perfect for production applications.
        """
        
        try:
            async with self.session.post(
                f"{self.config.server_url}/content/analyze",
                json={"content": sample_content}
            ) as response:
                if response.status == 200:
                    analysis = await response.json()
                    print("🤖 Content Analysis Results:")
                    print(f"  - Sentiment Score: {analysis['sentiment_score']:.2f}")
                    print(f"  - Readability Score: {analysis['readability_score']:.2f}")
                    print(f"  - Content Quality: {analysis['content_quality_score']:.2f}")
                    print(f"  - SEO Score: {analysis['seo_score']:.2f}")
                    print(f"  - Engagement Prediction: {analysis['engagement_prediction']:.2f}")
                    print(f"  - Reading Time: {analysis['reading_time_minutes']} minutes")
                    print(f"  - Top Keywords: {list(analysis['keyword_density'].keys())[:5]}")
                else:
                    print("❌ Failed to analyze content")
        except Exception as e:
            print(f"❌ Error analyzing content: {e}")
    
    async def demonstrate_recommendations(self):
        """Demonstrate content recommendations."""
        print("🎯 Demonstrating content recommendations...")
        
        # Get posts first
        async with self.session.get(f"{self.config.server_url}/posts") as response:
            if response.status == 200:
                posts_data = await response.json()
                posts = posts_data.get("posts", [])
                
                if len(posts) > 1:
                    post_id = posts[0]["id"]
                    
                    try:
                        async with self.session.get(
                            f"{self.config.server_url}/posts/{post_id}/recommendations",
                            params={"limit": 3}
                        ) as rec_response:
                            if rec_response.status == 200:
                                recommendations = await rec_response.json()
                                print(f"🎯 Recommendations for post '{posts[0]['title']}':")
                                for i, rec in enumerate(recommendations, 1):
                                    print(f"  {i}. {rec['title']}")
                            else:
                                print("❌ Failed to get recommendations")
                    except Exception as e:
                        print(f"❌ Error getting recommendations: {e}")
    
    async def demonstrate_websocket_notifications(self):
        """Demonstrate WebSocket notifications."""
        print("🔌 Demonstrating WebSocket notifications...")
        
        try:
            # Connect to WebSocket
            self.websocket = await websockets.connect(self.config.websocket_url)
            print("✅ Connected to WebSocket")
            
            # Listen for notifications
            async def listen_notifications():
                try:
                    while True:
                        message = await self.websocket.recv()
                        notification = json.loads(message)
                        self.notifications_received.append(notification)
                        print(f"🔔 Notification: {notification['type']} - {notification.get('title', 'N/A')}")
                except websockets.exceptions.ConnectionClosed:
                    print("🔌 WebSocket connection closed")
            
            # Start listening in background
            listen_task = asyncio.create_task(listen_notifications())
            
            # Create a new post to trigger notification
            new_post = BlogPostCreate(
                title="WebSocket Test Post",
                content="This post will trigger a WebSocket notification.",
                tags=["websocket", "test"],
                category="technology",
                author="Demo User",
                is_published=True
            )
            
            async with self.session.post(
                f"{self.config.server_url}/posts",
                json=new_post.model_dump()
            ) as response:
                if response.status == 201:
                    print("✅ Created test post to trigger notification")
                else:
                    print("❌ Failed to create test post")
            
            # Wait for notification
            await asyncio.sleep(2)
            
            # Cancel listening task
            listen_task.cancel()
            
        except Exception as e:
            print(f"❌ WebSocket error: {e}")
    
    async def demonstrate_advanced_filtering(self):
        """Demonstrate advanced filtering and pagination."""
        print("🔍 Demonstrating advanced filtering...")
        
        # Test different filter combinations
        filters = [
            {"category": "technology", "sort_by": "created_at", "sort_order": "desc"},
            {"author": "Tech Blogger", "limit": 5},
            {"status": "published", "sort_by": "views", "sort_order": "desc"},
        ]
        
        for i, filter_params in enumerate(filters, 1):
            try:
                async with self.session.get(
                    f"{self.config.server_url}/posts",
                    params=filter_params
                ) as response:
                    if response.status == 200:
                        results = await response.json()
                        print(f"🔍 Filter {i}: {len(results['posts'])} posts found")
                        print(f"  - Total: {results['total']}")
                        print(f"  - Has more: {results['has_more']}")
                        
                        if results['posts']:
                            print(f"  - First post: {results['posts'][0]['title']}")
                    else:
                        print(f"❌ Filter {i} failed")
            except Exception as e:
                print(f"❌ Error with filter {i}: {e}")
    
    async def run_comprehensive_demo(self):
        """Run the complete enhanced blog system demo."""
        print("🚀 ENHANCED BLOG SYSTEM DEMO")
        print("=" * 50)
        
        try:
            # Start server
            await self.start_server()
            
            # Create sample posts
            posts = await self.create_sample_posts()
            
            # Wait a moment for indexing
            await asyncio.sleep(2)
            
            # Demonstrate features
            await self.demonstrate_search()
            await asyncio.sleep(1)
            
            await self.demonstrate_analytics()
            await asyncio.sleep(1)
            
            await self.demonstrate_ai_analysis()
            await asyncio.sleep(1)
            
            await self.demonstrate_recommendations()
            await asyncio.sleep(1)
            
            await self.demonstrate_websocket_notifications()
            await asyncio.sleep(1)
            
            await self.demonstrate_advanced_filtering()
            
            # Show system metrics
            print("\n📊 System Metrics:")
            async with self.session.get(f"{self.config.server_url}/metrics") as response:
                if response.status == 200:
                    metrics = await response.json()
                    print(f"  - CPU: {metrics['system'].get('cpu_percent', 'N/A')}%")
                    print(f"  - Memory: {metrics['system'].get('memory_percent', 'N/A')}%")
                    print(f"  - Disk: {metrics['system'].get('disk_percent', 'N/A')}%")
            
            print("\n✅ Enhanced Blog System Demo Completed Successfully!")
            print(f"📝 Created {len(posts)} posts")
            print(f"🔔 Received {len(self.notifications_received)} notifications")
            
        except Exception as e:
            print(f"❌ Demo error: {e}")
        finally:
            # Stop server
            if hasattr(self, 'server_task'):
                self.server_task.cancel()
                try:
                    await self.server_task
                except asyncio.CancelledError:
                    pass

async def main():
    """Main demo function."""
    config = DemoConfig()
    
    async with EnhancedBlogDemo(config) as demo:
        await demo.run_comprehensive_demo()

if __name__ == "__main__":
    asyncio.run(main()) 
 
 