"""
LangChain Video Processing Examples

Comprehensive examples demonstrating LangChain integration for intelligent
video content analysis and optimization for short-form videos.
"""

import asyncio
import time
from typing import List, Dict, Optional
import structlog

from ..models.video_models import VideoClipRequest
from ..models.viral_models import (
    ViralVideoVariant,
    ViralVideoBatchResponse,
    LangChainAnalysis,
    ContentOptimization,
    ShortVideoOptimization,
    ContentType,
    EngagementType,
    TransitionType,
    ScreenDivisionType,
    CaptionStyle,
    VideoEffect,
    create_default_caption_config
)
from ..processors.langchain_processor import (
    LangChainVideoProcessor,
    LangChainConfig,
    create_langchain_processor,
    create_optimized_langchain_processor
)
from ..processors.viral_processor import (
    ViralVideoProcessor,
    ViralProcessorConfig,
    create_viral_processor,
    create_optimized_viral_processor
)

logger = structlog.get_logger()

# =============================================================================
# BASIC LANGCHAIN EXAMPLES
# =============================================================================

def example_basic_langchain_processing():
    """Basic example of LangChain video processing."""
    print("=== Basic LangChain Video Processing ===")
    
    # Create processor
    processor = create_langchain_processor(
        api_key="your-openai-api-key",  # Replace with actual key
        model_name="gpt-4",
        enable_agents=True
    )
    
    # Create video request
    request = VideoClipRequest(
        youtube_url="https://www.youtube.com/watch?v=example",
        language="en",
        max_clip_length=30.0,
        target_platform="tiktok"
    )
    
    # Process with LangChain
    response = processor.process_video_with_langchain(
        request=request,
        n_variants=5,
        audience_profile={
            "age": "18-25",
            "interests": ["entertainment", "comedy", "viral"],
            "platform": "tiktok"
        }
    )
    
    # Display results
    print(f"Processing successful: {response.success}")
    print(f"Variants generated: {response.successful_variants}")
    print(f"Average viral score: {response.average_viral_score:.3f}")
    print(f"Best viral score: {response.best_viral_score:.3f}")
    print(f"LangChain analysis time: {response.langchain_analysis_time:.2f}s")
    print(f"Content optimization time: {response.content_optimization_time:.2f}s")
    
    # Display top variant
    if response.variants:
        top_variant = response.variants[0]
        print(f"\nTop Variant:")
        print(f"  Title: {top_variant.title}")
        print(f"  Viral Score: {top_variant.viral_score:.3f}")
        print(f"  Estimated Views: {top_variant.estimated_views:,}")
        print(f"  Captions: {len(top_variant.captions)}")
        print(f"  AI Generated Hooks: {len(top_variant.ai_generated_hooks)}")

def example_langchain_content_analysis():
    """Example of detailed content analysis with LangChain."""
    print("\n=== LangChain Content Analysis ===")
    
    # Create processor with detailed analysis
    config = LangChainConfig(
        openai_api_key="your-openai-api-key",
        model_name="gpt-4",
        enable_content_analysis=True,
        enable_engagement_analysis=True,
        enable_viral_analysis=True,
        enable_audience_analysis=True,
        batch_size=3,
        enable_debug=True
    )
    
    processor = LangChainVideoProcessor(config)
    
    # Create request
    request = VideoClipRequest(
        youtube_url="https://www.youtube.com/watch?v=educational-content",
        language="en",
        max_clip_length=45.0,
        target_platform="youtube_shorts"
    )
    
    # Process with detailed analysis
    response = processor.process_video_with_langchain(
        request=request,
        n_variants=3,
        audience_profile={
            "age": "25-35",
            "interests": ["education", "technology", "learning"],
            "platform": "youtube_shorts"
        }
    )
    
    # Display analysis results
    if response.variants and response.variants[0].langchain_analysis:
        analysis = response.variants[0].langchain_analysis
        print(f"Content Type: {analysis.content_type.value}")
        print(f"Sentiment: {analysis.sentiment}")
        print(f"Engagement Score: {analysis.engagement_score:.3f}")
        print(f"Viral Potential: {analysis.viral_potential:.3f}")
        print(f"Target Audience: {', '.join(analysis.target_audience)}")
        print(f"Trending Keywords: {', '.join(analysis.trending_keywords[:5])}")
        print(f"Optimal Duration: {analysis.optimal_duration:.1f}s")
        print(f"Optimal Format: {analysis.optimal_format}")

def example_langchain_optimization():
    """Example of content optimization with LangChain."""
    print("\n=== LangChain Content Optimization ===")
    
    processor = create_optimized_langchain_processor(
        api_key="your-openai-api-key",
        batch_size=2,
        max_retries=3
    )
    
    # Create request for different content types
    requests = [
        VideoClipRequest(
            youtube_url="https://www.youtube.com/watch?v=comedy-video",
            language="en",
            max_clip_length=30.0,
            target_platform="tiktok"
        ),
        VideoClipRequest(
            youtube_url="https://www.youtube.com/watch?v=tutorial-video",
            language="en",
            max_clip_length=60.0,
            target_platform="instagram_reels"
        )
    ]
    
    audience_profiles = [
        {"age": "18-25", "interests": ["comedy", "entertainment"]},
        {"age": "25-40", "interests": ["education", "how-to"]}
    ]
    
    # Process each request
    for i, (request, profile) in enumerate(zip(requests, audience_profiles)):
        print(f"\nProcessing request {i+1}: {request.target_platform}")
        
        response = processor.process_video_with_langchain(
            request=request,
            n_variants=3,
            audience_profile=profile
        )
        
        if response.variants and response.variants[0].content_optimization:
            optimization = response.variants[0].content_optimization
            print(f"  Optimal Title: {optimization.optimal_title}")
            print(f"  Optimal Tags: {', '.join(optimization.optimal_tags[:5])}")
            print(f"  Optimal Hashtags: {', '.join(optimization.optimal_hashtags[:5])}")
            print(f"  Engagement Hooks: {len(optimization.engagement_hooks)}")
            print(f"  Viral Elements: {len(optimization.viral_elements)}")

# =============================================================================
# ADVANCED LANGCHAIN EXAMPLES
# =============================================================================

def example_short_video_optimization():
    """Example of short-form video optimization with LangChain."""
    print("\n=== Short Video Optimization ===")
    
    processor = create_langchain_processor(
        api_key="your-openai-api-key",
        model_name="gpt-4"
    )
    
    # Create request for short-form video
    request = VideoClipRequest(
        youtube_url="https://www.youtube.com/watch?v=short-video",
        language="en",
        max_clip_length=15.0,  # Very short for maximum engagement
        target_platform="tiktok"
    )
    
    response = processor.process_video_with_langchain(
        request=request,
        n_variants=5,
        audience_profile={
            "age": "16-24",
            "interests": ["trending", "viral", "entertainment"],
            "platform": "tiktok"
        }
    )
    
    # Display short video optimization results
    if response.variants and response.variants[0].short_video_optimization:
        short_opt = response.variants[0].short_video_optimization
        print(f"Optimal Clip Length: {short_opt.optimal_clip_length:.1f}s")
        print(f"Hook Duration: {short_opt.hook_duration:.1f}s")
        print(f"Retention Duration: {short_opt.retention_duration:.1f}s")
        print(f"CTA Duration: {short_opt.call_to_action_duration:.1f}s")
        print(f"Hook Type: {short_opt.hook_type}")
        print(f"Vertical Format: {short_opt.vertical_format}")
        print(f"Engagement Triggers: {len(short_opt.engagement_triggers)}")
        print(f"Viral Hooks: {len(short_opt.viral_hooks)}")
        print(f"Emotional Impact: {short_opt.emotional_impact:.3f}")

def example_langchain_batch_processing():
    """Example of batch processing with LangChain."""
    print("\n=== LangChain Batch Processing ===")
    
    processor = create_optimized_langchain_processor(
        api_key="your-openai-api-key",
        batch_size=3
    )
    
    # Create multiple requests
    requests = [
        VideoClipRequest(
            youtube_url=f"https://www.youtube.com/watch?v=video-{i}",
            language="en",
            max_clip_length=30.0,
            target_platform="tiktok"
        )
        for i in range(5)
    ]
    
    audience_profiles = [
        {"age": "18-25", "interests": ["comedy"]},
        {"age": "25-35", "interests": ["education"]},
        {"age": "16-24", "interests": ["gaming"]},
        {"age": "25-40", "interests": ["technology"]},
        {"age": "18-30", "interests": ["lifestyle"]}
    ]
    
    # Process in batch
    start_time = time.perf_counter()
    
    responses = []
    for request, profile in zip(requests, audience_profiles):
        response = processor.process_video_with_langchain(
            request=request,
            n_variants=3,
            audience_profile=profile
        )
        responses.append(response)
    
    total_time = time.perf_counter() - start_time
    
    # Display batch results
    print(f"Batch processing completed in {total_time:.2f}s")
    print(f"Total requests processed: {len(responses)}")
    
    successful_responses = [r for r in responses if r.success]
    print(f"Successful requests: {len(successful_responses)}")
    
    if successful_responses:
        total_variants = sum(r.successful_variants for r in successful_responses)
        avg_viral_score = sum(r.average_viral_score for r in successful_responses) / len(successful_responses)
        print(f"Total variants generated: {total_variants}")
        print(f"Average viral score: {avg_viral_score:.3f}")

# =============================================================================
# INTEGRATED PROCESSING EXAMPLES
# =============================================================================

def example_integrated_langchain_viral_processing():
    """Example of integrated LangChain and viral processing."""
    print("\n=== Integrated LangChain + Viral Processing ===")
    
    # Create viral processor with LangChain
    viral_processor = create_optimized_viral_processor(
        api_key="your-openai-api-key",
        batch_size=3,
        max_workers=4
    )
    
    # Create request
    request = VideoClipRequest(
        youtube_url="https://www.youtube.com/watch?v=integrated-video",
        language="en",
        max_clip_length=45.0,
        target_platform="tiktok"
    )
    
    # Process with both LangChain and viral optimization
    response = viral_processor.process_viral_variants(
        request=request,
        n_variants=8,
        audience_profile={
            "age": "18-25",
            "interests": ["entertainment", "viral", "trending"],
            "platform": "tiktok"
        },
        use_langchain=True
    )
    
    # Display integrated results
    print(f"Processing successful: {response.success}")
    print(f"Variants generated: {response.successful_variants}")
    print(f"Average viral score: {response.average_viral_score:.3f}")
    print(f"Best viral score: {response.best_viral_score:.3f}")
    print(f"AI enhancement score: {response.ai_enhancement_score:.3f}")
    
    # Display optimization insights
    if response.optimization_insights:
        print("\nOptimization Insights:")
        for key, value in response.optimization_insights.items():
            print(f"  {key}: {value}")
    
    # Display top 3 variants
    print("\nTop 3 Variants:")
    for i, variant in enumerate(response.variants[:3]):
        print(f"  {i+1}. {variant.title}")
        print(f"     Viral Score: {variant.viral_score:.3f}")
        print(f"     Estimated Views: {variant.estimated_views:,}")
        print(f"     AI Generated Hooks: {len(variant.ai_generated_hooks)}")
        print(f"     AI Viral Elements: {len(variant.ai_viral_elements)}")

def example_comparison_processing():
    """Compare processing with and without LangChain."""
    print("\n=== LangChain vs Standard Processing Comparison ===")
    
    # Create processor
    processor = create_viral_processor(
        enable_langchain=True,
        api_key="your-openai-api-key"
    )
    
    # Create request
    request = VideoClipRequest(
        youtube_url="https://www.youtube.com/watch?v=comparison-video",
        language="en",
        max_clip_length=30.0,
        target_platform="tiktok"
    )
    
    audience_profile = {
        "age": "18-25",
        "interests": ["entertainment", "viral"],
        "platform": "tiktok"
    }
    
    # Process with LangChain
    print("Processing with LangChain...")
    start_time = time.perf_counter()
    langchain_response = processor.process_viral_variants(
        request=request,
        n_variants=5,
        audience_profile=audience_profile,
        use_langchain=True
    )
    langchain_time = time.perf_counter() - start_time
    
    # Process without LangChain
    print("Processing without LangChain...")
    start_time = time.perf_counter()
    standard_response = processor.process_viral_variants(
        request=request,
        n_variants=5,
        audience_profile=audience_profile,
        use_langchain=False
    )
    standard_time = time.perf_counter() - start_time
    
    # Compare results
    print(f"\nComparison Results:")
    print(f"LangChain Processing:")
    print(f"  Time: {langchain_time:.2f}s")
    print(f"  Average Viral Score: {langchain_response.average_viral_score:.3f}")
    print(f"  Best Viral Score: {langchain_response.best_viral_score:.3f}")
    print(f"  AI Enhancement Score: {langchain_response.ai_enhancement_score:.3f}")
    
    print(f"\nStandard Processing:")
    print(f"  Time: {standard_time:.2f}s")
    print(f"  Average Viral Score: {standard_response.average_viral_score:.3f}")
    print(f"  Best Viral Score: {standard_response.best_viral_score:.3f}")
    
    # Calculate improvements
    viral_improvement = ((langchain_response.average_viral_score - standard_response.average_viral_score) / 
                        standard_response.average_viral_score * 100)
    time_difference = langchain_time - standard_time
    
    print(f"\nImprovements:")
    print(f"  Viral Score Improvement: {viral_improvement:+.1f}%")
    print(f"  Additional Processing Time: {time_difference:+.2f}s")

# =============================================================================
# ERROR HANDLING EXAMPLES
# =============================================================================

def example_error_handling():
    """Example of error handling in LangChain processing."""
    print("\n=== Error Handling Examples ===")
    
    # Create processor without API key (will trigger fallback)
    processor = create_langchain_processor(
        api_key=None,  # No API key
        model_name="gpt-4"
    )
    
    # Create request
    request = VideoClipRequest(
        youtube_url="https://www.youtube.com/watch?v=error-test",
        language="en",
        max_clip_length=30.0,
        target_platform="tiktok"
    )
    
    # Process (should fall back to standard processing)
    print("Processing with missing API key (should fallback)...")
    response = processor.process_video_with_langchain(
        request=request,
        n_variants=3
    )
    
    print(f"Processing successful: {response.success}")
    print(f"Variants generated: {response.successful_variants}")
    print(f"Errors: {response.errors}")
    
    # Test with invalid request
    print("\nTesting with invalid request...")
    invalid_request = VideoClipRequest(
        youtube_url="",  # Invalid URL
        language="en",
        max_clip_length=0.0,  # Invalid duration
        target_platform="invalid"
    )
    
    response = processor.process_video_with_langchain(
        request=invalid_request,
        n_variants=1
    )
    
    print(f"Processing successful: {response.success}")
    print(f"Errors: {response.errors}")

# =============================================================================
# PERFORMANCE BENCHMARKING
# =============================================================================

def example_performance_benchmarking():
    """Example of performance benchmarking for LangChain processing."""
    print("\n=== Performance Benchmarking ===")
    
    # Create different processor configurations
    configs = [
        ("Basic", create_langchain_processor(api_key="your-openai-api-key")),
        ("Optimized", create_optimized_langchain_processor(api_key="your-openai-api-key")),
        ("Viral + LangChain", create_optimized_viral_processor(api_key="your-openai-api-key"))
    ]
    
    # Create test request
    request = VideoClipRequest(
        youtube_url="https://www.youtube.com/watch?v=benchmark-test",
        language="en",
        max_clip_length=30.0,
        target_platform="tiktok"
    )
    
    audience_profile = {
        "age": "18-25",
        "interests": ["entertainment", "viral"],
        "platform": "tiktok"
    }
    
    # Benchmark each configuration
    results = []
    
    for name, processor in configs:
        print(f"\nBenchmarking {name} configuration...")
        
        # Warm up
        try:
            processor.process_video_with_langchain(
                request=request,
                n_variants=1,
                audience_profile=audience_profile
            )
        except:
            pass
        
        # Actual benchmark
        times = []
        viral_scores = []
        
        for i in range(3):  # Run 3 times for average
            start_time = time.perf_counter()
            try:
                response = processor.process_video_with_langchain(
                    request=request,
                    n_variants=3,
                    audience_profile=audience_profile
                )
                end_time = time.perf_counter()
                
                times.append(end_time - start_time)
                viral_scores.append(response.average_viral_score)
                
            except Exception as e:
                print(f"  Run {i+1} failed: {str(e)}")
        
        if times:
            avg_time = sum(times) / len(times)
            avg_viral_score = sum(viral_scores) / len(viral_scores)
            
            results.append({
                "name": name,
                "avg_time": avg_time,
                "avg_viral_score": avg_viral_score,
                "success_rate": len(times) / 3
            })
            
            print(f"  Average Time: {avg_time:.2f}s")
            print(f"  Average Viral Score: {avg_viral_score:.3f}")
            print(f"  Success Rate: {len(times)/3:.1%}")
    
    # Display comparison
    print(f"\nPerformance Comparison:")
    print(f"{'Configuration':<20} {'Time (s)':<10} {'Viral Score':<12} {'Success Rate':<12}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['name']:<20} {result['avg_time']:<10.2f} {result['avg_viral_score']:<12.3f} {result['success_rate']:<12.1%}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_all_examples():
    """Run all LangChain examples."""
    print("🚀 LangChain Video Processing Examples")
    print("=" * 50)
    
    try:
        # Basic examples
        example_basic_langchain_processing()
        example_langchain_content_analysis()
        example_langchain_optimization()
        
        # Advanced examples
        example_short_video_optimization()
        example_langchain_batch_processing()
        
        # Integrated examples
        example_integrated_langchain_viral_processing()
        example_comparison_processing()
        
        # Error handling
        example_error_handling()
        
        # Performance benchmarking
        example_performance_benchmarking()
        
        print("\n✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {str(e)}")
        logger.error("Example execution failed", error=str(e))

if __name__ == "__main__":
    run_all_examples() 