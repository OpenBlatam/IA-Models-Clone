"""
Async External APIs Examples for Video-OpusClip

Comprehensive examples demonstrating real-world usage of async external API operations
including YouTube API integration, OpenAI text generation, Stability AI image generation,
ElevenLabs text-to-speech, batch operations, and performance monitoring.
"""

import asyncio
import time
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import structlog

# Import async external API modules
from async_external_apis import (
    APIConfig, APIType, HTTPMethod,
    AsyncHTTPClient, RateLimiter, APICache,
    AsyncExternalAPIOperations, AsyncYouTubeAPI,
    AsyncOpenAIAPI, AsyncStabilityAIAPI, AsyncElevenLabsAPI,
    AsyncBatchAPIOperations, APIMetricsCollector,
    create_api_config, create_async_http_client,
    create_async_external_api_operations, create_async_youtube_api,
    create_async_openai_api, create_async_stability_ai_api,
    create_async_elevenlabs_api, create_async_batch_api_operations,
    create_api_metrics_collector, setup_external_api,
    close_external_api, get_api_metrics
)

logger = structlog.get_logger()

# =============================================================================
# EXAMPLE 1: BASIC API OPERATIONS
# =============================================================================

async def example_basic_api_operations():
    """Demonstrate basic API operations with different services."""
    print("\n=== Example 1: Basic API Operations ===")
    
    try:
        # Example 1.1: YouTube API operations
        youtube_config = create_api_config(
            base_url="https://www.googleapis.com/youtube/v3",
            api_key="your_youtube_api_key",
            rate_limit_per_minute=100,
            enable_caching=True,
            cache_ttl=600
        )
        
        youtube_api = await setup_external_api(
            APIType.YOUTUBE,
            base_url=youtube_config.base_url,
            api_key=youtube_config.api_key
        )
        
        # Get video information
        video_info = await youtube_api.get_video_info("dQw4w9WgXcQ")
        print(f"YouTube video info: {video_info.get('title', 'Unknown')}")
        
        # Search for videos
        search_results = await youtube_api.search_videos("sunset timelapse", max_results=5)
        print(f"Found {len(search_results)} sunset timelapse videos")
        
        # Example 1.2: OpenAI API operations
        openai_config = create_api_config(
            base_url="https://api.openai.com/v1",
            api_key="your_openai_api_key",
            rate_limit_per_minute=60,
            enable_caching=True,
            cache_ttl=300
        )
        
        openai_api = await setup_external_api(
            APIType.OPENAI,
            base_url=openai_config.base_url,
            api_key=openai_config.api_key
        )
        
        # Generate text
        text_response = await openai_api.generate_text(
            prompt="Write a short description of a beautiful sunset video",
            model="gpt-3.5-turbo",
            max_tokens=100,
            temperature=0.7
        )
        print(f"OpenAI generated text: {text_response.get('choices', [{}])[0].get('message', {}).get('content', 'No content')}")
        
        # Generate captions
        captions = await openai_api.generate_captions(
            audio_text="Beautiful sunset over the ocean with gentle waves",
            style="casual",
            language="en"
        )
        print(f"Generated captions: {captions}")
        
        # Example 1.3: Stability AI API operations
        stability_config = create_api_config(
            base_url="https://api.stability.ai/v1",
            api_key="your_stability_api_key",
            rate_limit_per_minute=30,
            enable_caching=True,
            cache_ttl=1800
        )
        
        stability_api = await setup_external_api(
            APIType.STABILITY_AI,
            base_url=stability_config.base_url,
            api_key=stability_config.api_key
        )
        
        # Generate image
        image_response = await stability_api.generate_image(
            prompt="Beautiful sunset over ocean, photorealistic, high quality",
            width=1024,
            height=1024,
            steps=30,
            cfg_scale=7.0
        )
        print(f"Generated image: {image_response.get('artifacts', [{}])[0].get('base64', 'No image')[:50]}...")
        
        # Example 1.4: ElevenLabs API operations
        elevenlabs_config = create_api_config(
            base_url="https://api.elevenlabs.io/v1",
            api_key="your_elevenlabs_api_key",
            rate_limit_per_minute=50,
            enable_caching=True,
            cache_ttl=600
        )
        
        elevenlabs_api = await setup_external_api(
            APIType.ELEVENLABS,
            base_url=elevenlabs_config.base_url,
            api_key=elevenlabs_config.api_key
        )
        
        # Get available voices
        voices = await elevenlabs_api.get_available_voices()
        print(f"Available voices: {len(voices)} voices found")
        
        # Generate speech
        speech_response = await elevenlabs_api.text_to_speech(
            text="Welcome to our beautiful sunset video. Enjoy the view!",
            voice_id="21m00Tcm4TlvDq8ikWAM",
            model_id="eleven_monolingual_v1"
        )
        print(f"Generated speech: {speech_response.get('audio', 'No audio')[:50]}...")
        
    except Exception as e:
        logger.error(f"Basic API operation failed: {e}")

# =============================================================================
# EXAMPLE 2: BATCH API OPERATIONS
# =============================================================================

async def example_batch_api_operations():
    """Demonstrate batch API operations for high-throughput scenarios."""
    print("\n=== Example 2: Batch API Operations ===")
    
    try:
        # Setup batch API operations
        api_ops = await setup_external_api(
            APIType.CUSTOM,
            base_url="https://api.example.com"
        )
        
        batch_ops = create_async_batch_api_operations(api_ops)
        
        # Example 2.1: Batch video info retrieval
        video_ids = [
            "dQw4w9WgXcQ",
            "9bZkp7q19f0",
            "kJQP7kiw5Fk",
            "y6120QOlsfU",
            "ZZ5LpwO-An4"
        ]
        
        video_infos = await batch_ops.batch_get_video_info(video_ids)
        print(f"Retrieved info for {len(video_infos)} videos")
        
        for i, video_info in enumerate(video_infos):
            if video_info:
                print(f"  Video {i+1}: {video_info.get('title', 'Unknown')}")
        
        # Example 2.2: Batch caption generation
        audio_texts = [
            "Beautiful sunset over the ocean with gentle waves",
            "Mountain landscape with snow-capped peaks",
            "Urban cityscape at night with lights",
            "Forest scene with birds chirping",
            "Desert landscape with sand dunes"
        ]
        
        captions_batch = await batch_ops.batch_generate_captions(
            audio_texts=audio_texts,
            style="casual"
        )
        
        print(f"Generated captions for {len(captions_batch)} audio texts")
        for i, captions in enumerate(captions_batch):
            print(f"  Audio {i+1}: {len(captions)} captions generated")
        
    except Exception as e:
        logger.error(f"Batch API operation failed: {e}")

# =============================================================================
# EXAMPLE 3: RATE LIMITING AND CACHING
# =============================================================================

async def example_rate_limiting_and_caching():
    """Demonstrate rate limiting and caching mechanisms."""
    print("\n=== Example 3: Rate Limiting and Caching ===")
    
    try:
        # Setup API with aggressive rate limiting for testing
        config = create_api_config(
            base_url="https://api.example.com",
            rate_limit_per_minute=10,  # Low limit for testing
            rate_limit_per_hour=100,
            enable_caching=True,
            cache_ttl=300,
            timeout=30.0
        )
        
        http_client = create_async_http_client(config)
        await http_client.initialize()
        
        # Example 3.1: Test rate limiting
        print("Testing rate limiting with 15 requests (limit: 10/min)...")
        start_time = time.perf_counter()
        
        tasks = []
        for i in range(15):
            task = http_client.get(
                endpoint="/test",
                cache_key=f"test_request_{i}",
                use_cache=True
            )
            tasks.append(task)
        
        # Execute requests
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        successful_requests = len([r for r in results if not isinstance(r, Exception)])
        print(f"Completed {successful_requests}/15 requests in {total_time:.2f} seconds")
        
        # Example 3.2: Test caching
        print("Testing caching with repeated requests...")
        
        # First request (cache miss)
        start_time = time.perf_counter()
        result1 = await http_client.get(
            endpoint="/cached_data",
            cache_key="test_cache_key"
        )
        time1 = time.perf_counter() - start_time
        
        # Second request (cache hit)
        start_time = time.perf_counter()
        result2 = await http_client.get(
            endpoint="/cached_data",
            cache_key="test_cache_key"
        )
        time2 = time.perf_counter() - start_time
        
        print(f"First request (cache miss): {time1:.3f}s")
        print(f"Second request (cache hit): {time2:.3f}s")
        print(f"Cache speedup: {time1/time2:.1f}x faster")
        
        await http_client.close()
        
    except Exception as e:
        logger.error(f"Rate limiting and caching test failed: {e}")

# =============================================================================
# EXAMPLE 4: ERROR HANDLING AND RETRY LOGIC
# =============================================================================

async def example_error_handling_and_retry():
    """Demonstrate error handling and retry logic."""
    print("\n=== Example 4: Error Handling and Retry Logic ===")
    
    try:
        # Setup API with retry configuration
        config = create_api_config(
            base_url="https://httpstat.us",  # Test service
            max_retries=3,
            retry_delay=1.0,
            timeout=10.0
        )
        
        api_ops = await setup_external_api(
            APIType.CUSTOM,
            base_url=config.base_url
        )
        
        # Example 4.1: Test successful request
        print("Testing successful request...")
        try:
            result = await api_ops.make_request(
                method=HTTPMethod.GET,
                endpoint="/200",
                retry_on_error=True
            )
            print(f"Successful request: {result}")
        except Exception as e:
            print(f"Request failed: {e}")
        
        # Example 4.2: Test retry on 500 error
        print("Testing retry on 500 error...")
        try:
            result = await api_ops.make_request(
                method=HTTPMethod.GET,
                endpoint="/500",
                retry_on_error=True
            )
            print(f"Request succeeded after retries: {result}")
        except Exception as e:
            print(f"Request failed after retries: {e}")
        
        # Example 4.3: Test timeout handling
        print("Testing timeout handling...")
        try:
            result = await api_ops.make_request(
                method=HTTPMethod.GET,
                endpoint="/200?sleep=5000",  # 5 second delay
                retry_on_error=True
            )
            print(f"Request succeeded: {result}")
        except Exception as e:
            print(f"Request timed out: {e}")
        
    except Exception as e:
        logger.error(f"Error handling test failed: {e}")

# =============================================================================
# EXAMPLE 5: VIDEO PROCESSING PIPELINE WITH APIS
# =============================================================================

async def example_video_processing_pipeline():
    """Demonstrate a complete video processing pipeline using multiple APIs."""
    print("\n=== Example 5: Video Processing Pipeline ===")
    
    try:
        # Setup multiple APIs
        youtube_api = await setup_external_api(
            APIType.YOUTUBE,
            base_url="https://www.googleapis.com/youtube/v3",
            api_key="your_youtube_api_key"
        )
        
        openai_api = await setup_external_api(
            APIType.OPENAI,
            base_url="https://api.openai.com/v1",
            api_key="your_openai_api_key"
        )
        
        stability_api = await setup_external_api(
            APIType.STABILITY_AI,
            base_url="https://api.stability.ai/v1",
            api_key="your_stability_api_key"
        )
        
        elevenlabs_api = await setup_external_api(
            APIType.ELEVENLABS,
            base_url="https://api.elevenlabs.io/v1",
            api_key="your_elevenlabs_api_key"
        )
        
        # Step 1: Get video information from YouTube
        print("Step 1: Getting video information...")
        video_id = "dQw4w9WgXcQ"
        video_info = await youtube_api.get_video_info(video_id)
        video_title = video_info.get('title', 'Unknown Video')
        video_description = video_info.get('description', '')
        
        print(f"Video: {video_title}")
        
        # Step 2: Generate captions using OpenAI
        print("Step 2: Generating captions...")
        captions = await openai_api.generate_captions(
            audio_text=f"{video_title}. {video_description[:200]}",
            style="casual",
            language="en"
        )
        
        print(f"Generated {len(captions)} captions")
        
        # Step 3: Generate thumbnail using Stability AI
        print("Step 3: Generating thumbnail...")
        thumbnail_prompt = f"Thumbnail for video: {video_title}. Eye-catching, modern design."
        thumbnail = await stability_api.generate_video_thumbnail(
            video_title=video_title,
            video_description=video_description[:100]
        )
        
        print(f"Generated thumbnail: {thumbnail.get('artifacts', [{}])[0].get('base64', 'No image')[:50]}...")
        
        # Step 4: Generate narration using ElevenLabs
        print("Step 4: Generating narration...")
        narration_script = f"Welcome to {video_title}. This is an amazing video that you won't want to miss!"
        narration = await elevenlabs_api.generate_video_narration(
            script=narration_script,
            voice_id="21m00Tcm4TlvDq8ikWAM"
        )
        
        print(f"Generated narration: {narration.get('audio', 'No audio')[:50]}...")
        
        # Step 5: Analyze video content
        print("Step 5: Analyzing video content...")
        content_analysis = await openai_api.analyze_video_content(
            video_description=video_description,
            video_title=video_title
        )
        
        print(f"Content analysis: {content_analysis}")
        
        # Step 6: Compile results
        pipeline_results = {
            "video_id": video_id,
            "video_title": video_title,
            "captions": captions,
            "thumbnail": thumbnail,
            "narration": narration,
            "content_analysis": content_analysis,
            "processed_at": datetime.now().isoformat()
        }
        
        print(f"Pipeline completed successfully!")
        print(f"Results: {json.dumps(pipeline_results, indent=2, default=str)}")
        
    except Exception as e:
        logger.error(f"Video processing pipeline failed: {e}")

# =============================================================================
# EXAMPLE 6: PERFORMANCE MONITORING AND METRICS
# =============================================================================

async def example_performance_monitoring():
    """Demonstrate performance monitoring and metrics collection."""
    print("\n=== Example 6: Performance Monitoring ===")
    
    try:
        # Create metrics collector
        metrics_collector = create_api_metrics_collector()
        
        # Setup API with metrics
        api_ops = await setup_external_api(
            APIType.CUSTOM,
            base_url="https://api.example.com"
        )
        
        # Example 6.1: Monitor API performance
        print("Monitoring API performance...")
        
        # Simulate multiple API calls
        endpoints = ["/users", "/videos", "/analytics", "/reports", "/stats"]
        methods = ["GET", "POST", "GET", "GET", "POST"]
        
        for i, (endpoint, method) in enumerate(zip(endpoints, methods)):
            start_time = time.perf_counter()
            
            try:
                result = await api_ops.make_request(
                    method=HTTPMethod(method),
                    endpoint=endpoint,
                    retry_on_error=True
                )
                
                response_time = time.perf_counter() - start_time
                await metrics_collector.record_request(
                    endpoint=endpoint,
                    method=method,
                    success=True,
                    response_time=response_time
                )
                
                print(f"Request {i+1}: {method} {endpoint} - {response_time:.3f}s")
                
            except Exception as e:
                response_time = time.perf_counter() - start_time
                await metrics_collector.record_request(
                    endpoint=endpoint,
                    method=method,
                    success=False,
                    response_time=response_time,
                    error_type=type(e).__name__
                )
                
                print(f"Request {i+1}: {method} {endpoint} - FAILED ({response_time:.3f}s)")
        
        # Example 6.2: Monitor cache performance
        print("Monitoring cache performance...")
        
        for i in range(10):
            # Simulate cache hits and misses
            is_hit = i < 7  # 70% cache hit rate
            await metrics_collector.record_cache_access(hit=is_hit)
        
        # Example 6.3: Get comprehensive metrics
        metrics = await metrics_collector.get_metrics()
        
        print(f"Performance metrics:")
        print(f"  Total requests: {metrics['total_requests']}")
        print(f"  Successful requests: {metrics['successful_requests']}")
        print(f"  Failed requests: {metrics['failed_requests']}")
        print(f"  Success rate: {metrics['success_rate']:.1f}%")
        print(f"  Average response time: {metrics['average_response_time']:.3f}s")
        print(f"  Cache hit rate: {metrics['cache_hit_rate']:.1f}%")
        print(f"  Requests per minute: {metrics['requests_per_minute']:.1f}")
        
        # Example 6.4: Endpoint-specific metrics
        print(f"Endpoint metrics:")
        for endpoint, endpoint_metrics in metrics['endpoint_metrics'].items():
            print(f"  {endpoint}:")
            print(f"    Requests: {endpoint_metrics['requests']}")
            print(f"    Success rate: {endpoint_metrics['success_rate']:.1f}%")
            print(f"    Average time: {endpoint_metrics['average_time']:.3f}s")
        
    except Exception as e:
        logger.error(f"Performance monitoring failed: {e}")

# =============================================================================
# EXAMPLE 7: INTEGRATED WORKFLOW
# =============================================================================

async def example_integrated_workflow():
    """Demonstrate an integrated workflow using multiple APIs."""
    print("\n=== Example 7: Integrated Workflow ===")
    
    try:
        # Setup all APIs
        apis = {}
        
        # YouTube API
        apis['youtube'] = await setup_external_api(
            APIType.YOUTUBE,
            base_url="https://www.googleapis.com/youtube/v3",
            api_key="your_youtube_api_key"
        )
        
        # OpenAI API
        apis['openai'] = await setup_external_api(
            APIType.OPENAI,
            base_url="https://api.openai.com/v1",
            api_key="your_openai_api_key"
        )
        
        # Stability AI API
        apis['stability'] = await setup_external_api(
            APIType.STABILITY_AI,
            base_url="https://api.stability.ai/v1",
            api_key="your_stability_api_key"
        )
        
        # ElevenLabs API
        apis['elevenlabs'] = await setup_external_api(
            APIType.ELEVENLABS,
            base_url="https://api.elevenlabs.io/v1",
            api_key="your_elevenlabs_api_key"
        )
        
        # Workflow: Process multiple videos
        video_ids = ["dQw4w9WgXcQ", "9bZkp7q19f0", "kJQP7q19f0"]
        
        workflow_results = []
        
        for i, video_id in enumerate(video_ids):
            print(f"Processing video {i+1}/{len(video_ids)}: {video_id}")
            
            # Parallel processing tasks
            tasks = [
                # Get video info
                apis['youtube'].get_video_info(video_id),
                
                # Generate description
                apis['openai'].generate_text(
                    prompt=f"Write a compelling description for video {video_id}",
                    max_tokens=150
                ),
                
                # Generate thumbnail
                apis['stability'].generate_image(
                    prompt="Video thumbnail, modern design, eye-catching",
                    width=1280,
                    height=720
                ),
                
                # Generate voice intro
                apis['elevenlabs'].text_to_speech(
                    text=f"Welcome to video {video_id}. Let's get started!",
                    voice_id="21m00Tcm4TlvDq8ikWAM"
                )
            ]
            
            # Execute tasks in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Compile results
            video_result = {
                "video_id": video_id,
                "video_info": results[0] if not isinstance(results[0], Exception) else None,
                "description": results[1] if not isinstance(results[1], Exception) else None,
                "thumbnail": results[2] if not isinstance(results[2], Exception) else None,
                "voice_intro": results[3] if not isinstance(results[3], Exception) else None,
                "processed_at": datetime.now().isoformat()
            }
            
            workflow_results.append(video_result)
            
            print(f"  Completed video {video_id}")
        
        print(f"Workflow completed! Processed {len(workflow_results)} videos")
        
        # Summary
        successful_videos = len([r for r in workflow_results if r['video_info']])
        print(f"Successfully processed: {successful_videos}/{len(video_ids)} videos")
        
    except Exception as e:
        logger.error(f"Integrated workflow failed: {e}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

async def main():
    """Run all external API examples."""
    print("Async External APIs Examples")
    print("=" * 50)
    
    # Run examples
    await example_basic_api_operations()
    await example_batch_api_operations()
    await example_rate_limiting_and_caching()
    await example_error_handling_and_retry()
    await example_video_processing_pipeline()
    await example_performance_monitoring()
    await example_integrated_workflow()
    
    print("\nAll examples completed!")

if __name__ == "__main__":
    asyncio.run(main()) 