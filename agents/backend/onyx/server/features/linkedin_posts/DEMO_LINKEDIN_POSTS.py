from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List
from uuid import uuid4
from core.entities.linkedin_post import LinkedInPost, PostType, ContentTone, PostStatus
from infrastructure.langchain_integration import LinkedInPostGenerator, ContentOptimizer, EngagementAnalyzer
from application.use_cases.linkedin_post_use_cases import (
from shared.config.settings import LinkedInPostSettings
from shared.logging import setup_logging, get_logger
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
LinkedIn Posts System Demo
==========================

Comprehensive demo showcasing the LinkedIn Posts system with LangChain integration.
This demo demonstrates all major features including post generation, optimization,
analysis, and A/B testing.
"""


    GenerateLinkedInPostUseCase,
    OptimizeLinkedInPostUseCase,
    AnalyzeEngagementUseCase,
    CreateABTestUseCase,
)

# Setup logging
setup_logging(level="INFO")
logger = get_logger(__name__)


class LinkedInPostsDemo:
    """Demo class for LinkedIn Posts system."""
    
    def __init__(self) -> Any:
        """Initialize the demo."""
        self.settings = LinkedInPostSettings(
            openai_api_key="demo-api-key",  # Replace with actual key
            langchain_model="gpt-4",
            enable_auto_optimization=True,
            enable_engagement_prediction=True,
        )
        
        # Initialize components
        self.post_generator = LinkedInPostGenerator(
            api_key=self.settings.openai_api_key,
            model_name=self.settings.langchain_model
        )
        
        self.content_optimizer = ContentOptimizer(
            llm=None  # Would be actual LLM instance
        )
        
        self.engagement_analyzer = EngagementAnalyzer(
            llm=None  # Would be actual LLM instance
        )
        
        # Mock repository for demo
        self.posts = []
    
    async def run_complete_demo(self) -> Any:
        """Run the complete LinkedIn Posts demo."""
        logger.info("🚀 Starting LinkedIn Posts System Demo")
        logger.info("=" * 60)
        
        try:
            # 1. Generate LinkedIn Post
            await self.demo_post_generation()
            
            # 2. Optimize Content
            await self.demo_content_optimization()
            
            # 3. Analyze Engagement
            await self.demo_engagement_analysis()
            
            # 4. Create A/B Test
            await self.demo_ab_testing()
            
            # 5. Industry-specific Generation
            await self.demo_industry_specific_posts()
            
            # 6. Tone Variations
            await self.demo_tone_variations()
            
            # 7. Bulk Operations
            await self.demo_bulk_operations()
            
            # 8. Performance Metrics
            await self.demo_performance_metrics()
            
            logger.info("✅ LinkedIn Posts System Demo Completed Successfully!")
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            raise
    
    async def demo_post_generation(self) -> Any:
        """Demo LinkedIn post generation."""
        logger.info("\n📝 Demo: LinkedIn Post Generation")
        logger.info("-" * 40)
        
        # Generate different types of posts
        generation_scenarios = [
            {
                "name": "Thought Leadership Post",
                "topic": "The Future of AI in Business",
                "key_points": [
                    "AI is transforming business operations",
                    "Companies need to adapt to stay competitive",
                    "The role of human creativity remains crucial"
                ],
                "target_audience": "Business leaders and executives",
                "industry": "technology",
                "tone": ContentTone.AUTHORITATIVE,
                "post_type": PostType.TEXT,
            },
            {
                "name": "Storytelling Post",
                "topic": "How I Failed and Learned",
                "key_points": [
                    "Personal experience with a failed project",
                    "Key lessons learned from the failure",
                    "How it shaped my approach to business"
                ],
                "target_audience": "Professionals seeking growth",
                "industry": "professional development",
                "tone": ContentTone.INSPIRATIONAL,
                "post_type": PostType.TEXT,
            },
            {
                "name": "Educational Post",
                "topic": "LinkedIn Algorithm Secrets",
                "key_points": [
                    "How the LinkedIn algorithm works",
                    "Best practices for content optimization",
                    "Timing and frequency strategies"
                ],
                "target_audience": "Social media professionals",
                "industry": "marketing",
                "tone": ContentTone.EDUCATIONAL,
                "post_type": PostType.TEXT,
            }
        ]
        
        for scenario in generation_scenarios:
            logger.info(f"\n🎯 Generating: {scenario['name']}")
            
            try:
                # Generate post
                generated_data = await self.post_generator.generate_post(
                    topic=scenario["topic"],
                    key_points=scenario["key_points"],
                    target_audience=scenario["target_audience"],
                    industry=scenario["industry"],
                    tone=scenario["tone"],
                    post_type=scenario["post_type"],
                    keywords=["linkedin", "social media", "professional"],
                    additional_context="Focus on providing value and driving engagement"
                )
                
                # Create post entity
                post = LinkedInPost(
                    title=generated_data["title"],
                    content=generated_data["content"],
                    summary=generated_data["summary"],
                    user_id=uuid4(),
                    creator_id=uuid4(),
                    post_type=scenario["post_type"],
                    tone=scenario["tone"],
                    keywords=generated_data["keywords"],
                    hashtags=generated_data["hashtags"],
                    industry=scenario["industry"],
                    langchain_prompt=generated_data["langchain_data"]["prompt"],
                    langchain_model=generated_data["langchain_data"]["model"],
                    generation_parameters=generated_data["langchain_data"]["parameters"],
                )
                
                self.posts.append(post)
                
                # Display results
                logger.info(f"✅ Generated: {post.title}")
                logger.info(f"📊 Estimated Engagement: {generated_data['estimated_engagement']:.1f}%")
                logger.info(f"🏷️  Hashtags: {', '.join(post.hashtags[:5])}")
                logger.info(f"📝 Content Length: {len(post.content)} characters")
                
            except Exception as e:
                logger.error(f"❌ Failed to generate {scenario['name']}: {e}")
    
    async def demo_content_optimization(self) -> Any:
        """Demo content optimization."""
        logger.info("\n🔧 Demo: Content Optimization")
        logger.info("-" * 40)
        
        if not self.posts:
            logger.warning("No posts available for optimization")
            return
        
        post = self.posts[0]  # Use first generated post
        
        logger.info(f"🎯 Optimizing: {post.title}")
        
        try:
            # Perform comprehensive optimization
            optimization_result = await self.content_optimizer.optimize_comprehensive(
                content=post.content,
                target_audience="professional audience",
                keywords=post.keywords
            )
            
            # Display optimization results
            logger.info("📈 Optimization Results:")
            logger.info(f"   Original Length: {len(optimization_result['original'])} chars")
            logger.info(f"   Optimized Length: {len(optimization_result['final_optimized'])} chars")
            logger.info(f"   Readability Optimized: {len(optimization_result['readability_optimized'])} chars")
            logger.info(f"   Engagement Optimized: {len(optimization_result['engagement_optimized'])} chars")
            logger.info(f"   SEO Optimized: {len(optimization_result['seo_optimized'])} chars")
            
            # Update post with optimized content
            original_content = post.content
            post.content = optimization_result["final_optimized"]
            
            logger.info("✅ Content optimization completed")
            
            # Show content analysis
            analysis = self.content_optimizer.analyze_content_structure(post.content)
            logger.info("📊 Content Analysis:")
            logger.info(f"   Word Count: {analysis.get('word_count', 0)}")
            logger.info(f"   Readability Score: {analysis.get('readability_score', 0):.1f}")
            logger.info(f"   Engagement Potential: {analysis.get('engagement_potential', 0):.1f}")
            logger.info(f"   Has Questions: {analysis.get('has_questions', False)}")
            logger.info(f"   Has Call to Action: {analysis.get('has_call_to_action', False)}")
            
        except Exception as e:
            logger.error(f"❌ Content optimization failed: {e}")
    
    async def demo_engagement_analysis(self) -> Any:
        """Demo engagement analysis."""
        logger.info("\n📊 Demo: Engagement Analysis")
        logger.info("-" * 40)
        
        if not self.posts:
            logger.warning("No posts available for analysis")
            return
        
        post = self.posts[0]  # Use first generated post
        
        logger.info(f"🎯 Analyzing: {post.title}")
        
        try:
            # Perform comprehensive analysis
            analysis = await self.engagement_analyzer.comprehensive_analysis(
                content=post.content,
                target_audience="professional audience",
                industry=post.industry or "general"
            )
            
            # Display analysis results
            logger.info("📈 Engagement Analysis Results:")
            logger.info(f"   Overall Score: {analysis.get('composite_score', 0):.1f}/100")
            
            prediction = analysis.get("prediction", {})
            logger.info(f"   Predicted Likes: {prediction.get('predictions', {}).get('likes', 'N/A')}")
            logger.info(f"   Predicted Comments: {prediction.get('predictions', {}).get('comments', 'N/A')}")
            logger.info(f"   Predicted Shares: {prediction.get('predictions', {}).get('shares', 'N/A')}")
            
            sentiment = analysis.get("sentiment", {})
            logger.info(f"   Sentiment: {sentiment.get('sentiment', 'N/A')}")
            logger.info(f"   Sentiment Score: {sentiment.get('score', 0):.2f}")
            logger.info(f"   Tone: {sentiment.get('tone', 'N/A')}")
            
            audience = analysis.get("audience", {})
            logger.info(f"   Audience Resonance: {audience.get('resonance_score', 0):.1f}/100")
            
            # Display recommendations
            recommendations = analysis.get("recommendations", [])
            if recommendations:
                logger.info("💡 Recommendations:")
                for i, rec in enumerate(recommendations[:3], 1):
                    logger.info(f"   {i}. {rec}")
            
            logger.info("✅ Engagement analysis completed")
            
        except Exception as e:
            logger.error(f"❌ Engagement analysis failed: {e}")
    
    async def demo_ab_testing(self) -> Any:
        """Demo A/B testing."""
        logger.info("\n🧪 Demo: A/B Testing")
        logger.info("-" * 40)
        
        if not self.posts:
            logger.warning("No posts available for A/B testing")
            return
        
        post = self.posts[0]  # Use first generated post
        
        logger.info(f"🎯 Creating A/B test for: {post.title}")
        
        try:
            # Generate A/B test variants
            variants = await self.post_generator.generate_multiple_variants(
                topic=post.title,
                key_points=[post.content[:200]],  # Simplified key points
                target_audience="professional audience",
                industry=post.industry or "general",
                num_variants=3,
                tone=post.tone,
                post_type=post.post_type,
                keywords=post.keywords,
            )
            
            logger.info(f"✅ Generated {len(variants)} A/B test variants")
            
            # Display variant details
            for i, variant_data in enumerate(variants, 1):
                logger.info(f"\n📝 Variant {i}:")
                logger.info(f"   Title: {variant_data['title']}")
                logger.info(f"   Content Length: {len(variant_data['content'])} chars")
                logger.info(f"   Hashtags: {', '.join(variant_data['hashtags'][:3])}")
                logger.info(f"   Estimated Engagement: {variant_data['estimated_engagement']:.1f}%")
                
                # Create variant post
                variant_post = LinkedInPost(
                    title=variant_data["title"],
                    content=variant_data["content"],
                    summary=variant_data["summary"],
                    user_id=post.user_id,
                    creator_id=post.creator_id,
                    post_type=post.post_type,
                    tone=post.tone,
                    keywords=variant_data["keywords"],
                    hashtags=variant_data["hashtags"],
                    is_ab_test=True,
                    ab_test_id=post.id,
                    variant_id=f"variant_{i}",
                )
                
                self.posts.append(variant_post)
            
            logger.info("✅ A/B test creation completed")
            
        except Exception as e:
            logger.error(f"❌ A/B testing failed: {e}")
    
    async def demo_industry_specific_posts(self) -> Any:
        """Demo industry-specific post generation."""
        logger.info("\n🏭 Demo: Industry-Specific Posts")
        logger.info("-" * 40)
        
        industries = [
            ("technology", "AI and Machine Learning Trends"),
            ("finance", "Investment Strategies for 2024"),
            ("healthcare", "Digital Health Innovations"),
            ("education", "Future of Online Learning"),
            ("marketing", "Digital Marketing Evolution"),
        ]
        
        for industry, topic in industries:
            logger.info(f"\n🎯 Generating {industry} post: {topic}")
            
            try:
                post_data = await self.post_generator.generate_industry_specific_post(
                    topic=topic,
                    industry=industry,
                    company_size="enterprise",
                    target_role="professionals"
                )
                
                logger.info(f"✅ Generated {industry} post")
                logger.info(f"   Title: {post_data['title']}")
                logger.info(f"   Content Length: {len(post_data['content'])} chars")
                logger.info(f"   Hashtags: {', '.join(post_data['hashtags'][:3])}")
                
            except Exception as e:
                logger.error(f"❌ Failed to generate {industry} post: {e}")
    
    async def demo_tone_variations(self) -> Any:
        """Demo tone variations."""
        logger.info("\n🎭 Demo: Tone Variations")
        logger.info("-" * 40)
        
        topic = "Remote Work Best Practices"
        key_points = [
            "Setting up a productive home office",
            "Maintaining work-life balance",
            "Effective communication with remote teams"
        ]
        
        tones = [
            ContentTone.PROFESSIONAL,
            ContentTone.CASUAL,
            ContentTone.INSPIRATIONAL,
            ContentTone.EDUCATIONAL,
        ]
        
        for tone in tones:
            logger.info(f"\n🎯 Generating {tone.value} tone post")
            
            try:
                post_data = await self.post_generator.generate_post(
                    topic=topic,
                    key_points=key_points,
                    target_audience="remote workers and managers",
                    industry="professional development",
                    tone=tone,
                    post_type=PostType.TEXT,
                )
                
                logger.info(f"✅ Generated {tone.value} post")
                logger.info(f"   Title: {post_data['title']}")
                logger.info(f"   Content Preview: {post_data['content'][:100]}...")
                logger.info(f"   Estimated Engagement: {post_data['estimated_engagement']:.1f}%")
                
            except Exception as e:
                logger.error(f"❌ Failed to generate {tone.value} post: {e}")
    
    async def demo_bulk_operations(self) -> Any:
        """Demo bulk operations."""
        logger.info("\n📦 Demo: Bulk Operations")
        logger.info("-" * 40)
        
        # Generate multiple posts in bulk
        bulk_topics = [
            "Digital Transformation Strategies",
            "Customer Experience Innovation",
            "Sustainable Business Practices",
            "Leadership in Crisis",
            "Future of Work",
        ]
        
        logger.info(f"🎯 Generating {len(bulk_topics)} posts in bulk")
        
        start_time = time.time()
        
        # Generate posts concurrently
        tasks = []
        for topic in bulk_topics:
            task = self.post_generator.generate_post(
                topic=topic,
                key_points=[f"Key insights about {topic}"],
                target_audience="business professionals",
                industry="business",
                tone=ContentTone.PROFESSIONAL,
                post_type=PostType.TEXT,
            )
            tasks.append(task)
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            successful_posts = 0
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"❌ Failed to generate post {i+1}: {result}")
                else:
                    successful_posts += 1
                    logger.info(f"✅ Generated post {i+1}: {result['title']}")
            
            total_time = time.time() - start_time
            logger.info(f"📊 Bulk Generation Results:")
            logger.info(f"   Total Posts: {len(bulk_topics)}")
            logger.info(f"   Successful: {successful_posts}")
            logger.info(f"   Failed: {len(bulk_topics) - successful_posts}")
            logger.info(f"   Total Time: {total_time:.2f} seconds")
            logger.info(f"   Average Time per Post: {total_time/len(bulk_topics):.2f} seconds")
            
        except Exception as e:
            logger.error(f"❌ Bulk operations failed: {e}")
    
    async def demo_performance_metrics(self) -> Any:
        """Demo performance metrics."""
        logger.info("\n📈 Demo: Performance Metrics")
        logger.info("-" * 40)
        
        if not self.posts:
            logger.warning("No posts available for metrics")
            return
        
        # Calculate metrics
        total_posts = len(self.posts)
        published_posts = len([p for p in self.posts if p.status == PostStatus.PUBLISHED])
        draft_posts = len([p for p in self.posts if p.status == PostStatus.DRAFT])
        
        # Engagement metrics (mock data)
        total_likes = sum(p.likes_count for p in self.posts)
        total_comments = sum(p.comments_count for p in self.posts)
        total_shares = sum(p.shares_count for p in self.posts)
        total_views = sum(p.views_count for p in self.posts)
        
        # Content analysis
        avg_content_length = sum(len(p.content) for p in self.posts) / total_posts
        posts_with_hashtags = len([p for p in self.posts if p.hashtags])
        posts_with_keywords = len([p for p in self.posts if p.keywords])
        
        # Display metrics
        logger.info("📊 Performance Metrics:")
        logger.info(f"   Total Posts: {total_posts}")
        logger.info(f"   Published: {published_posts}")
        logger.info(f"   Drafts: {draft_posts}")
        logger.info(f"   Total Likes: {total_likes}")
        logger.info(f"   Total Comments: {total_comments}")
        logger.info(f"   Total Shares: {total_shares}")
        logger.info(f"   Total Views: {total_views}")
        logger.info(f"   Average Content Length: {avg_content_length:.0f} chars")
        logger.info(f"   Posts with Hashtags: {posts_with_hashtags}")
        logger.info(f"   Posts with Keywords: {posts_with_keywords}")
        
        # Top performing posts
        top_posts = sorted(
            self.posts,
            key=lambda p: p.likes_count + p.comments_count * 2 + p.shares_count * 3,
            reverse=True
        )[:3]
        
        logger.info("\n🏆 Top Performing Posts:")
        for i, post in enumerate(top_posts, 1):
            engagement_score = post.likes_count + post.comments_count * 2 + post.shares_count * 3
            logger.info(f"   {i}. {post.title}")
            logger.info(f"      Engagement Score: {engagement_score}")
            logger.info(f"      Likes: {post.likes_count}, Comments: {post.comments_count}, Shares: {post.shares_count}")
        
        logger.info("✅ Performance metrics calculated")


async def main():
    """Main demo function."""
    demo = LinkedInPostsDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main()) 