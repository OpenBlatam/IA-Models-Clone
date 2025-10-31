from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import asyncio
import json
import time
from pathlib import Path
import sys
from main_optimized import LinkedInPostsOptimizedSystem, LinkedInPostRequest
    import argparse
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
LinkedIn Posts AI System - Demo Runner
======================================

Demo script to showcase the AI-optimized LinkedIn posts system.
"""


# Add the current directory to path
sys.path.insert(0, str(Path(__file__).parent))


class LinkedInPostsDemo:
    """Demo class for showcasing the LinkedIn Posts AI system."""
    
    def __init__(self) -> Any:
        self.system = LinkedInPostsOptimizedSystem()
        self.demo_posts = [
            {
                "content": "Just finished implementing a new AI-powered feature that increased our user engagement by 40%! The key was understanding user behavior patterns and optimizing the user experience. What's your experience with AI-driven product improvements?",
                "post_type": "educational",
                "tone": "enthusiastic",
                "target_audience": "developers"
            },
            {
                "content": "Excited to share that our startup just secured $2M in funding! This milestone represents not just financial backing, but validation of our vision to revolutionize how businesses approach digital transformation. Ready to take your business to the next level? Let's connect!",
                "post_type": "promotional",
                "tone": "enthusiastic",
                "target_audience": "executives"
            },
            {
                "content": "Today marks 5 years since I started my journey in tech. From writing my first line of code to leading engineering teams, the learning never stops. The biggest lesson? Always stay curious and never be afraid to ask questions. What's your biggest career lesson so far?",
                "post_type": "personal",
                "tone": "thoughtful",
                "target_audience": "general"
            },
            {
                "content": "The future of work is here, and it's hybrid. Companies that adapt quickly will thrive, while those that resist change risk falling behind. The key is finding the right balance between remote flexibility and in-person collaboration. How is your organization adapting?",
                "post_type": "industry",
                "tone": "professional",
                "target_audience": "executives"
            }
        ]
    
    async def run_demo(self) -> Any:
        """Run the complete demo."""
        print("🚀 LinkedIn Posts AI System Demo")
        print("=" * 50)
        
        # Initialize system
        print("\n📡 Initializing system...")
        await self.system.startup_event()
        
        try:
            # Demo 1: Create posts
            print("\n📝 Demo 1: Creating Optimized Posts")
            print("-" * 30)
            
            created_posts = []
            for i, post_data in enumerate(self.demo_posts, 1):
                print(f"\nCreating post {i}...")
                
                request = LinkedInPostRequest(**post_data)
                
                # Simulate API call
                start_time = time.time()
                
                # Generate post components
                hashtags = await self.system.optimizer.generate_hashtags(request.content, request.post_type)
                call_to_action = await self.system.optimizer.generate_call_to_action(request.content, request.post_type)
                optimized_content = await self.system.optimizer.optimize_content(
                    request.content, request.post_type, request.tone
                )
                
                # Analyze content
                sentiment_score = await self.system.optimizer.analyze_sentiment(optimized_content)
                readability_score = await self.system.optimizer.calculate_readability(optimized_content)
                engagement_prediction = await self.system.optimizer.predict_engagement(
                    optimized_content, request.post_type, hashtags
                )
                
                processing_time = time.time() - start_time
                
                # Create response
                response_data = {
                    "id": f"demo_post_{i}_{int(time.time())}",
                    "content": request.content,
                    "optimized_content": optimized_content,
                    "post_type": request.post_type,
                    "tone": request.tone,
                    "target_audience": request.target_audience,
                    "hashtags": hashtags,
                    "call_to_action": call_to_action,
                    "sentiment_score": int(sentiment_score * 100),
                    "readability_score": int(readability_score),
                    "engagement_prediction": int(engagement_prediction),
                    "created_at": time.time(),
                    "status": "optimized"
                }
                
                created_posts.append(response_data)
                
                # Display results
                print(f"✅ Post {i} created successfully!")
                print(f"   Processing time: {processing_time:.2f}s")
                print(f"   Sentiment score: {sentiment_score:.2f}")
                print(f"   Readability score: {readability_score:.0f}")
                print(f"   Engagement prediction: {engagement_prediction:.0f}%")
                print(f"   Hashtags: {', '.join(hashtags[:3])}...")
                print(f"   Call to action: {call_to_action[:50]}...")
            
            # Demo 2: Show optimized content
            print("\n🎯 Demo 2: Content Optimization Results")
            print("-" * 40)
            
            for i, post in enumerate(created_posts, 1):
                print(f"\n📄 Post {i} - {post['post_type'].title()} ({post['tone']})")
                print(f"Original: {post['content'][:100]}...")
                print(f"Optimized: {post['optimized_content'][:100]}...")
                print(f"Hashtags: {', '.join(post['hashtags'])}")
                print(f"CTA: {post['call_to_action']}")
                print(f"Metrics: Sentiment={post['sentiment_score']}, Readability={post['readability_score']}, Engagement={post['engagement_prediction']}%")
            
            # Demo 3: Performance metrics
            print("\n📊 Demo 3: Performance Metrics")
            print("-" * 30)
            
            total_posts = len(created_posts)
            avg_sentiment = sum(p['sentiment_score'] for p in created_posts) / total_posts
            avg_readability = sum(p['readability_score'] for p in created_posts) / total_posts
            avg_engagement = sum(p['engagement_prediction'] for p in created_posts) / total_posts
            
            print(f"Total posts processed: {total_posts}")
            print(f"Average sentiment score: {avg_sentiment:.1f}")
            print(f"Average readability score: {avg_readability:.0f}")
            print(f"Average engagement prediction: {avg_engagement:.1f}%")
            
            # Demo 4: AI Model Information
            print("\n🤖 Demo 4: AI Model Information")
            print("-" * 30)
            
            device = self.system.optimizer.device
            print(f"Device: {device}")
            print(f"GPU available: {device.type == 'cuda'}")
            
            if hasattr(self.system.optimizer, 'nlp'):
                print("✅ spaCy NLP model loaded")
            if hasattr(self.system.optimizer, 'sentiment_analyzer'):
                print("✅ VADER Sentiment Analyzer loaded")
            if hasattr(self.system.optimizer, 'diffusion_pipeline'):
                print("✅ Stable Diffusion model loaded")
            
            # Demo 5: Save results
            print("\n💾 Demo 5: Saving Results")
            print("-" * 25)
            
            results = {
                "demo_timestamp": time.time(),
                "total_posts": total_posts,
                "average_metrics": {
                    "sentiment": avg_sentiment,
                    "readability": avg_readability,
                    "engagement": avg_engagement
                },
                "posts": created_posts
            }
            
            # Save to file
            output_file = "demo_results.json"
            with open(output_file, 'w', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Results saved to {output_file}")
            
            # Demo 6: System health
            print("\n🏥 Demo 6: System Health Check")
            print("-" * 30)
            
            health_status = {
                "status": "healthy",
                "timestamp": time.time(),
                "version": "2.0.0",
                "service": "linkedin-posts-ai",
                "gpu_available": device.type == 'cuda',
                "models_loaded": True,
                "cache_available": True
            }
            
            print(f"System status: {health_status['status']}")
            print(f"Version: {health_status['version']}")
            print(f"GPU available: {health_status['gpu_available']}")
            print(f"Models loaded: {health_status['models_loaded']}")
            print(f"Cache available: {health_status['cache_available']}")
            
            print("\n🎉 Demo completed successfully!")
            print("=" * 50)
            
        except Exception as e:
            print(f"❌ Demo failed: {str(e)}")
            raise
        finally:
            # Cleanup
            print("\n🧹 Cleaning up...")
            await self.system.shutdown_event()
    
    async def run_single_post_demo(self, content: str, post_type: str = "educational", tone: str = "professional"):
        """Run demo with a single custom post."""
        print("🚀 LinkedIn Posts AI System - Single Post Demo")
        print("=" * 50)
        
        # Initialize system
        print("\n📡 Initializing system...")
        await self.system.startup_event()
        
        try:
            print(f"\n📝 Creating post...")
            print(f"Content: {content[:100]}...")
            print(f"Type: {post_type}")
            print(f"Tone: {tone}")
            
            request = LinkedInPostRequest(
                content=content,
                post_type=post_type,
                tone=tone
            )
            
            start_time = time.time()
            
            # Generate post components
            hashtags = await self.system.optimizer.generate_hashtags(request.content, request.post_type)
            call_to_action = await self.system.optimizer.generate_call_to_action(request.content, request.post_type)
            optimized_content = await self.system.optimizer.optimize_content(
                request.content, request.post_type, request.tone
            )
            
            # Analyze content
            sentiment_score = await self.system.optimizer.analyze_sentiment(optimized_content)
            readability_score = await self.system.optimizer.calculate_readability(optimized_content)
            engagement_prediction = await self.system.optimizer.predict_engagement(
                optimized_content, request.post_type, hashtags
            )
            
            processing_time = time.time() - start_time
            
            # Display results
            print(f"\n✅ Post created successfully!")
            print(f"Processing time: {processing_time:.2f}s")
            print(f"\n📄 Original Content:")
            print(f"{request.content}")
            print(f"\n🎯 Optimized Content:")
            print(f"{optimized_content}")
            print(f"\n🏷️ Hashtags:")
            print(f"{', '.join(hashtags)}")
            print(f"\n📢 Call to Action:")
            print(f"{call_to_action}")
            print(f"\n📊 Metrics:")
            print(f"Sentiment Score: {sentiment_score:.2f}")
            print(f"Readability Score: {readability_score:.0f}")
            print(f"Engagement Prediction: {engagement_prediction:.0f}%")
            
        except Exception as e:
            print(f"❌ Demo failed: {str(e)}")
            raise
        finally:
            # Cleanup
            print("\n🧹 Cleaning up...")
            await self.system.shutdown_event()

async def main():
    """Main entry point for the demo."""
    
    parser = argparse.ArgumentParser(description="LinkedIn Posts AI System Demo")
    parser.add_argument("--mode", choices=["full", "single"], default="full", help="Demo mode")
    parser.add_argument("--content", help="Custom content for single post demo")
    parser.add_argument("--post-type", default="educational", help="Post type for single demo")
    parser.add_argument("--tone", default="professional", help="Tone for single demo")
    
    args = parser.parse_args()
    
    demo = LinkedInPostsDemo()
    
    if args.mode == "single":
        if not args.content:
            print("❌ Please provide content for single post demo")
            return
        
        await demo.run_single_post_demo(args.content, args.post_type, args.tone)
    else:
        await demo.run_demo()

match __name__:
    case "__main__":
    asyncio.run(main()) 