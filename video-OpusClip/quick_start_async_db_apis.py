#!/usr/bin/env python3
"""
Quick Start: Async Database and External API Operations

This script demonstrates the key features of async database and external API
operations for Video-OpusClip with practical examples.
"""

import asyncio
import time
import json
from typing import List, Dict, Any
import structlog

# Import async database components
from async_database import (
    create_database_config,
    create_database_pool,
    create_async_database_operations,
    create_async_video_database,
    create_async_batch_database_operations,
    create_async_transaction_manager,
    create_async_database_setup,
    DatabaseType,
    QueryType
)

# Import async external API components
from async_external_apis import (
    create_api_config,
    create_async_http_client,
    create_async_external_api_operations,
    create_async_youtube_api,
    create_async_openai_api,
    create_async_stability_ai_api,
    create_async_elevenlabs_api,
    create_async_batch_api_operations,
    create_api_metrics_collector,
    APIType,
    HTTPMethod
)

logger = structlog.get_logger()

# =============================================================================
# EXAMPLE 1: BASIC DATABASE OPERATIONS
# =============================================================================

async def example_basic_database_operations():
    """Demonstrate basic async database operations."""
    print("=== Example 1: Basic Database Operations ===")
    
    # Create database configuration
    config = create_database_config(
        host="localhost",
        port=5432,
        database="video_opusclip",
        username="postgres",
        password="password",
        max_connections=10
    )
    
    # Create database pool (using SQLite for demo)
    pool = create_database_pool(config, DatabaseType.SQLITE)
    await pool.initialize()
    
    # Create database operations
    db_ops = create_async_database_operations(pool)
    
    # Create database setup
    setup = create_async_database_setup(db_ops)
    await setup.create_tables()
    
    print("✅ Database initialized and tables created")
    
    # Create video database operations
    video_db = create_async_video_database(db_ops)
    
    # Create video record
    video_data = {
        "url": "https://youtube.com/watch?v=example",
        "title": "Amazing Cooking Tutorial",
        "duration": 180,
        "status": "pending"
    }
    
    video_id = await video_db.create_video_record(video_data)
    print(f"✅ Created video record with ID: {video_id}")
    
    # Get video by ID
    video = await video_db.get_video_by_id(video_id)
    print(f"✅ Retrieved video: {video['title']}")
    
    # Update video status
    success = await video_db.update_video_status(video_id, "processing")
    print(f"✅ Updated video status: {success}")
    
    # Create clip record
    clip_data = {
        "video_id": video_id,
        "start_time": 0,
        "end_time": 30,
        "duration": 30,
        "caption": "Amazing cooking tutorial! 🍳",
        "effects": ["fade_in", "fade_out"],
        "file_path": "/path/to/clip.mp4"
    }
    
    clip_id = await video_db.create_clip_record(clip_data)
    print(f"✅ Created clip record with ID: {clip_id}")
    
    # Get clips for video
    clips = await video_db.get_clips_by_video_id(video_id)
    print(f"✅ Retrieved {len(clips)} clips for video")
    
    # Get database statistics
    stats = await setup.get_database_stats()
    print(f"📊 Database stats: {stats}")
    
    await pool.close()
    print()

# =============================================================================
# EXAMPLE 2: BATCH DATABASE OPERATIONS
# =============================================================================

async def example_batch_database_operations():
    """Demonstrate batch database operations."""
    print("=== Example 2: Batch Database Operations ===")
    
    # Setup database
    config = create_database_config(
        host="localhost",
        port=5432,
        database="video_opusclip",
        username="postgres",
        password="password",
        max_connections=10
    )
    
    pool = create_database_pool(config, DatabaseType.SQLITE)
    await pool.initialize()
    
    db_ops = create_async_database_operations(pool)
    setup = create_async_database_setup(db_ops)
    await setup.create_tables()
    
    # Create batch operations
    batch_db = create_async_batch_database_operations(db_ops)
    
    # Prepare batch video data
    videos = [
        {
            "url": f"https://youtube.com/watch?v=video_{i}",
            "title": f"Video Tutorial {i}",
            "duration": 120 + i * 30,
            "status": "pending"
        }
        for i in range(5)
    ]
    
    print(f"🔄 Batch inserting {len(videos)} videos...")
    start_time = time.perf_counter()
    
    # Batch insert videos
    video_ids = await batch_db.batch_insert_videos(videos)
    duration = time.perf_counter() - start_time
    
    print(f"✅ Batch insert completed in {duration:.3f}s")
    print(f"   Inserted {len(video_ids)} videos")
    
    # Prepare batch clip data
    clips = []
    for video_id in video_ids:
        for j in range(3):  # 3 clips per video
            clips.append({
                "video_id": video_id,
                "start_time": j * 30,
                "end_time": (j + 1) * 30,
                "duration": 30,
                "caption": f"Clip {j + 1} from video {video_id}",
                "effects": ["fade_in", "fade_out"],
                "file_path": f"/path/to/clip_{video_id}_{j}.mp4"
            })
    
    print(f"🔄 Batch inserting {len(clips)} clips...")
    start_time = time.perf_counter()
    
    # Batch insert clips
    clip_ids = await batch_db.batch_insert_clips(clips)
    duration = time.perf_counter() - start_time
    
    print(f"✅ Batch insert completed in {duration:.3f}s")
    print(f"   Inserted {len(clip_ids)} clips")
    
    # Batch update video statuses
    updates = [(video_id, "completed") for video_id in video_ids]
    
    print("🔄 Batch updating video statuses...")
    start_time = time.perf_counter()
    
    updated_count = await batch_db.batch_update_video_status(updates)
    duration = time.perf_counter() - start_time
    
    print(f"✅ Batch update completed in {duration:.3f}s")
    print(f"   Updated {updated_count} videos")
    
    await pool.close()
    print()

# =============================================================================
# EXAMPLE 3: TRANSACTION MANAGEMENT
# =============================================================================

async def example_transaction_management():
    """Demonstrate transaction management."""
    print("=== Example 3: Transaction Management ===")
    
    # Setup database
    config = create_database_config(
        host="localhost",
        port=5432,
        database="video_opusclip",
        username="postgres",
        password="password",
        max_connections=10
    )
    
    pool = create_database_pool(config, DatabaseType.SQLITE)
    await pool.initialize()
    
    db_ops = create_async_database_operations(pool)
    setup = create_async_database_setup(db_ops)
    await setup.create_tables()
    
    # Create transaction manager
    tx_manager = create_async_transaction_manager(db_ops)
    video_db = create_async_video_database(db_ops)
    
    # Define transaction operations
    async def create_video_with_clips(video_data, clips_data):
        async with tx_manager.transaction() as connection:
            # Create video
            video_id = await video_db.create_video_record(video_data)
            print(f"   Created video with ID: {video_id}")
            
            # Create clips
            clip_ids = []
            for clip_data in clips_data:
                clip_data["video_id"] = video_id
                clip_id = await video_db.create_clip_record(clip_data)
                clip_ids.append(clip_id)
                print(f"   Created clip with ID: {clip_id}")
            
            # Update video status
            await video_db.update_video_status(video_id, "completed")
            print(f"   Updated video status to completed")
            
            return video_id, clip_ids
    
    # Execute transaction
    video_data = {
        "url": "https://youtube.com/watch?v=transaction_example",
        "title": "Transaction Example Video",
        "duration": 240,
        "status": "pending"
    }
    
    clips_data = [
        {
            "start_time": 0,
            "end_time": 60,
            "duration": 60,
            "caption": "Introduction",
            "effects": ["fade_in"],
            "file_path": "/path/to/intro.mp4"
        },
        {
            "start_time": 60,
            "end_time": 180,
            "duration": 120,
            "caption": "Main content",
            "effects": ["zoom"],
            "file_path": "/path/to/main.mp4"
        },
        {
            "start_time": 180,
            "end_time": 240,
            "duration": 60,
            "caption": "Conclusion",
            "effects": ["fade_out"],
            "file_path": "/path/to/conclusion.mp4"
        }
    ]
    
    print("🔄 Executing transaction...")
    start_time = time.perf_counter()
    
    try:
        video_id, clip_ids = await create_video_with_clips(video_data, clips_data)
        duration = time.perf_counter() - start_time
        
        print(f"✅ Transaction completed successfully in {duration:.3f}s")
        print(f"   Video ID: {video_id}")
        print(f"   Clip IDs: {clip_ids}")
        
    except Exception as e:
        print(f"❌ Transaction failed: {e}")
        # Transaction automatically rolled back
    
    await pool.close()
    print()

# =============================================================================
# EXAMPLE 4: BASIC EXTERNAL API OPERATIONS
# =============================================================================

async def example_basic_external_api_operations():
    """Demonstrate basic async external API operations."""
    print("=== Example 4: Basic External API Operations ===")
    
    # Create API configuration
    config = create_api_config(
        base_url="https://api.example.com",
        api_key="your_api_key",
        timeout=30.0,
        max_retries=3,
        rate_limit_per_minute=60
    )
    
    # Create HTTP client
    http_client = create_async_http_client(config)
    await http_client.initialize()
    
    # Create API operations
    api_ops = create_async_external_api_operations(http_client)
    
    print("✅ HTTP client initialized")
    
    # Make GET request
    try:
        result = await api_ops.make_request(
            HTTPMethod.GET,
            "videos",
            params={"id": "example_video_id"},
            cache_key="example:video:example_video_id",
            use_cache=True
        )
        print(f"✅ GET request successful: {len(result)} items")
    except Exception as e:
        print(f"⚠️  GET request failed (expected for demo): {e}")
    
    # Make POST request
    try:
        result = await api_ops.make_request(
            HTTPMethod.POST,
            "videos",
            data={"title": "Test Video", "url": "https://example.com/video.mp4"}
        )
        print(f"✅ POST request successful")
    except Exception as e:
        print(f"⚠️  POST request failed (expected for demo): {e}")
    
    # Get API metrics
    metrics = get_api_metrics(api_ops)
    print(f"📊 API metrics: {metrics}")
    
    await http_client.close()
    print()

# =============================================================================
# EXAMPLE 5: YOUTUBE API OPERATIONS
# =============================================================================

async def example_youtube_api_operations():
    """Demonstrate YouTube API operations."""
    print("=== Example 5: YouTube API Operations ===")
    
    # Create YouTube API configuration
    config = create_api_config(
        base_url="https://www.googleapis.com/youtube/v3",
        api_key="your_youtube_api_key",  # Replace with actual key
        timeout=30.0,
        max_retries=3,
        rate_limit_per_minute=60
    )
    
    # Create HTTP client
    http_client = create_async_http_client(config)
    await http_client.initialize()
    
    # Create API operations
    api_ops = create_async_external_api_operations(http_client)
    youtube_api = create_async_youtube_api(api_ops)
    
    print("✅ YouTube API client initialized")
    
    # Example video ID (Rick Astley - Never Gonna Give You Up)
    video_id = "dQw4w9WgXcQ"
    
    try:
        # Get video information
        print(f"🔄 Getting video info for {video_id}...")
        video_info = await youtube_api.get_video_info(video_id)
        
        if video_info and "items" in video_info and video_info["items"]:
            video = video_info["items"][0]
            print(f"✅ Video info retrieved:")
            print(f"   Title: {video['snippet']['title']}")
            print(f"   Channel: {video['snippet']['channelTitle']}")
            print(f"   Duration: {video['contentDetails']['duration']}")
            print(f"   Views: {video['statistics']['viewCount']}")
        else:
            print("⚠️  No video info found")
            
    except Exception as e:
        print(f"⚠️  YouTube API request failed (expected without valid API key): {e}")
    
    try:
        # Search for videos
        print("🔄 Searching for videos...")
        search_results = await youtube_api.search_videos("machine learning", max_results=3)
        
        if search_results:
            print(f"✅ Search results: {len(search_results)} videos found")
            for i, video in enumerate(search_results[:3]):
                print(f"   {i+1}. {video['snippet']['title']}")
        else:
            print("⚠️  No search results found")
            
    except Exception as e:
        print(f"⚠️  YouTube search failed (expected without valid API key): {e}")
    
    await http_client.close()
    print()

# =============================================================================
# EXAMPLE 6: OPENAI API OPERATIONS
# =============================================================================

async def example_openai_api_operations():
    """Demonstrate OpenAI API operations."""
    print("=== Example 6: OpenAI API Operations ===")
    
    # Create OpenAI API configuration
    config = create_api_config(
        base_url="https://api.openai.com/v1",
        api_key="your_openai_api_key",  # Replace with actual key
        timeout=60.0,
        max_retries=3,
        rate_limit_per_minute=60
    )
    
    # Create HTTP client
    http_client = create_async_http_client(config)
    await http_client.initialize()
    
    # Create API operations
    api_ops = create_async_external_api_operations(http_client)
    openai_api = create_async_openai_api(api_ops)
    
    print("✅ OpenAI API client initialized")
    
    try:
        # Generate text
        print("🔄 Generating text...")
        result = await openai_api.generate_text(
            prompt="Write a short, engaging caption for a cooking video about making pasta.",
            model="gpt-3.5-turbo",
            max_tokens=100,
            temperature=0.7
        )
        
        if result and "choices" in result:
            content = result["choices"][0]["message"]["content"]
            print(f"✅ Generated text: {content}")
        else:
            print("⚠️  No text generated")
            
    except Exception as e:
        print(f"⚠️  OpenAI API request failed (expected without valid API key): {e}")
    
    try:
        # Generate captions
        print("🔄 Generating captions...")
        captions = await openai_api.generate_captions(
            audio_text="Today we're making the most amazing pasta dish you've ever tasted. This recipe is so simple yet so delicious!",
            style="casual",
            language="en"
        )
        
        if captions:
            print(f"✅ Generated {len(captions)} captions:")
            for i, caption in enumerate(captions):
                print(f"   {i+1}. {caption}")
        else:
            print("⚠️  No captions generated")
            
    except Exception as e:
        print(f"⚠️  Caption generation failed (expected without valid API key): {e}")
    
    await http_client.close()
    print()

# =============================================================================
# EXAMPLE 7: BATCH API OPERATIONS
# =============================================================================

async def example_batch_api_operations():
    """Demonstrate batch API operations."""
    print("=== Example 7: Batch API Operations ===")
    
    # Create API configuration
    config = create_api_config(
        base_url="https://api.example.com",
        api_key="your_api_key",
        timeout=30.0,
        max_retries=3,
        rate_limit_per_minute=60
    )
    
    # Create HTTP client
    http_client = create_async_http_client(config)
    await http_client.initialize()
    
    # Create API operations
    api_ops = create_async_external_api_operations(http_client)
    batch_api = create_async_batch_api_operations(api_ops)
    
    print("✅ Batch API client initialized")
    
    # Example video IDs
    video_ids = ["video1", "video2", "video3", "video4", "video5"]
    
    try:
        # Batch get video information
        print(f"🔄 Batch getting info for {len(video_ids)} videos...")
        start_time = time.perf_counter()
        
        video_infos = await batch_api.batch_get_video_info(video_ids)
        duration = time.perf_counter() - start_time
        
        print(f"✅ Batch API request completed in {duration:.3f}s")
        print(f"   Retrieved info for {len(video_infos)} videos")
        
    except Exception as e:
        print(f"⚠️  Batch API request failed (expected for demo): {e}")
    
    try:
        # Batch generate captions
        audio_texts = [
            "Amazing cooking tutorial that will change your life!",
            "Learn to code in Python in just 10 minutes.",
            "Travel vlog from the most beautiful places in Japan.",
            "Fitness workout that burns maximum calories.",
            "DIY home decoration ideas on a budget."
        ]
        
        print(f"🔄 Batch generating captions for {len(audio_texts)} texts...")
        start_time = time.perf_counter()
        
        captions = await batch_api.batch_generate_captions(audio_texts, style="engaging")
        duration = time.perf_counter() - start_time
        
        print(f"✅ Batch caption generation completed in {duration:.3f}s")
        print(f"   Generated captions for {len(captions)} texts")
        
        for i, caption_list in enumerate(captions):
            print(f"   Text {i+1}: {len(caption_list)} captions")
            
    except Exception as e:
        print(f"⚠️  Batch caption generation failed (expected for demo): {e}")
    
    await http_client.close()
    print()

# =============================================================================
# EXAMPLE 8: API METRICS AND MONITORING
# =============================================================================

async def example_api_metrics_and_monitoring():
    """Demonstrate API metrics and monitoring."""
    print("=== Example 8: API Metrics and Monitoring ===")
    
    # Create metrics collector
    metrics_collector = create_api_metrics_collector()
    
    # Create API configuration
    config = create_api_config(
        base_url="https://api.example.com",
        api_key="your_api_key",
        timeout=30.0,
        max_retries=3,
        rate_limit_per_minute=60
    )
    
    # Create HTTP client
    http_client = create_async_http_client(config)
    await http_client.initialize()
    
    # Create API operations
    api_ops = create_async_external_api_operations(http_client)
    
    print("✅ API metrics collector initialized")
    
    # Simulate monitored API calls
    async def monitored_api_call(endpoint: str, should_succeed: bool = True):
        start_time = time.perf_counter()
        
        try:
            if should_succeed:
                # Simulate successful API call
                await asyncio.sleep(0.1)
                result = {"status": "success", "data": "example_data"}
            else:
                # Simulate failed API call
                await asyncio.sleep(0.05)
                raise Exception("Simulated API error")
            
            response_time = time.perf_counter() - start_time
            await metrics_collector.record_request(
                endpoint, "GET", True, response_time
            )
            
            return result
            
        except Exception as e:
            response_time = time.perf_counter() - start_time
            await metrics_collector.record_request(
                endpoint, "GET", False, response_time, type(e).__name__
            )
            raise
    
    # Execute monitored API calls
    endpoints = ["videos", "users", "analytics", "settings", "notifications"]
    
    print("🔄 Executing monitored API calls...")
    
    for i, endpoint in enumerate(endpoints):
        try:
            # Simulate some failures
            should_succeed = i != 2  # Third call will fail
            result = await monitored_api_call(endpoint, should_succeed)
            print(f"   ✅ {endpoint}: Success")
        except Exception as e:
            print(f"   ❌ {endpoint}: Failed - {e}")
    
    # Get metrics
    metrics = await metrics_collector.get_metrics()
    
    print(f"\n📊 API Metrics Summary:")
    print(f"   Total requests: {metrics['total_requests']}")
    print(f"   Successful requests: {metrics['successful_requests']}")
    print(f"   Failed requests: {metrics['failed_requests']}")
    print(f"   Success rate: {metrics.get('success_rate', 0):.2%}")
    print(f"   Average response time: {metrics.get('avg_response_time', 0):.3f}s")
    
    print(f"\n📈 Requests by Endpoint:")
    for endpoint, count in metrics['requests_by_endpoint'].items():
        print(f"   {endpoint}: {count}")
    
    print(f"\n❌ Errors by Type:")
    for error_type, count in metrics['errors_by_type'].items():
        print(f"   {error_type}: {count}")
    
    await http_client.close()
    print()

# =============================================================================
# EXAMPLE 9: INTEGRATED DATABASE AND API OPERATIONS
# =============================================================================

async def example_integrated_operations():
    """Demonstrate integrated database and API operations."""
    print("=== Example 9: Integrated Database and API Operations ===")
    
    # Setup database
    db_config = create_database_config(
        host="localhost",
        port=5432,
        database="video_opusclip",
        username="postgres",
        password="password",
        max_connections=10
    )
    
    db_pool = create_database_pool(db_config, DatabaseType.SQLITE)
    await db_pool.initialize()
    
    db_ops = create_async_database_operations(db_pool)
    setup = create_async_database_setup(db_ops)
    await setup.create_tables()
    
    video_db = create_async_video_database(db_ops)
    
    # Setup API
    api_config = create_api_config(
        base_url="https://api.example.com",
        api_key="your_api_key",
        timeout=30.0,
        max_retries=3,
        rate_limit_per_minute=60
    )
    
    http_client = create_async_http_client(api_config)
    await http_client.initialize()
    
    api_ops = create_async_external_api_operations(http_client)
    
    print("✅ Database and API clients initialized")
    
    try:
        # Simulate video processing pipeline
        video_url = "https://youtube.com/watch?v=integrated_example"
        
        # 1. Create video record in database
        print("🔄 Step 1: Creating video record...")
        video_id = await video_db.create_video_record({
            "url": video_url,
            "title": "Integrated Example Video",
            "duration": 300,
            "status": "downloading"
        })
        print(f"   ✅ Created video record: {video_id}")
        
        # 2. Simulate API call to get video info
        print("🔄 Step 2: Getting video info from API...")
        try:
            # Simulate API call
            await asyncio.sleep(0.1)
            video_info = {
                "title": "Updated Video Title",
                "description": "This is an amazing video about cooking",
                "duration": 320
            }
            print(f"   ✅ Retrieved video info from API")
        except Exception as e:
            print(f"   ⚠️  API call failed: {e}")
            video_info = {"title": "Fallback Title", "description": "", "duration": 300}
        
        # 3. Update video record with API data
        print("🔄 Step 3: Updating video record...")
        await video_db.update_video_status(video_id, "processing")
        print(f"   ✅ Updated video status to processing")
        
        # 4. Simulate processing and create clips
        print("🔄 Step 4: Creating video clips...")
        clips_data = [
            {
                "video_id": video_id,
                "start_time": 0,
                "end_time": 60,
                "duration": 60,
                "caption": "Introduction to amazing cooking! 🍳",
                "effects": ["fade_in"],
                "file_path": f"/clips/{video_id}_intro.mp4"
            },
            {
                "video_id": video_id,
                "start_time": 60,
                "end_time": 180,
                "duration": 120,
                "caption": "Main cooking tutorial 🔥",
                "effects": ["zoom", "text_overlay"],
                "file_path": f"/clips/{video_id}_main.mp4"
            },
            {
                "video_id": video_id,
                "start_time": 180,
                "end_time": 300,
                "duration": 120,
                "caption": "Final result and tips! ✨",
                "effects": ["fade_out", "highlight"],
                "file_path": f"/clips/{video_id}_final.mp4"
            }
        ]
        
        for clip_data in clips_data:
            clip_id = await video_db.create_clip_record(clip_data)
            print(f"   ✅ Created clip: {clip_id}")
        
        # 5. Final status update
        print("🔄 Step 5: Finalizing processing...")
        await video_db.update_video_status(video_id, "completed")
        print(f"   ✅ Updated video status to completed")
        
        # 6. Get final results
        final_video = await video_db.get_video_by_id(video_id)
        final_clips = await video_db.get_clips_by_video_id(video_id)
        
        print(f"\n🎉 Integrated processing completed successfully!")
        print(f"   Video: {final_video['title']} (Status: {final_video['status']})")
        print(f"   Clips: {len(final_clips)} clips created")
        
        for clip in final_clips:
            print(f"     - Clip {clip['id']}: {clip['caption']}")
        
    except Exception as e:
        print(f"❌ Integrated processing failed: {e}")
    
    finally:
        # Cleanup
        await db_pool.close()
        await http_client.close()
        print()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

async def main():
    """Run all async database and API examples."""
    print("🚀 Async Database and External API Operations - Quick Start Examples")
    print("=" * 70)
    print()
    
    try:
        # Run examples
        await example_basic_database_operations()
        await example_batch_database_operations()
        await example_transaction_management()
        await example_basic_external_api_operations()
        await example_youtube_api_operations()
        await example_openai_api_operations()
        await example_batch_api_operations()
        await example_api_metrics_and_monitoring()
        await example_integrated_operations()
        
        print("🎉 All examples completed successfully!")
        print()
        print("📚 Key Takeaways:")
        print("   ✅ Async database operations provide non-blocking data access")
        print("   ✅ Connection pooling optimizes resource usage")
        print("   ✅ Batch operations improve performance for large datasets")
        print("   ✅ Transaction management ensures data consistency")
        print("   ✅ External API operations handle rate limiting and caching")
        print("   ✅ Metrics collection enables performance monitoring")
        print("   ✅ Integrated operations combine database and API workflows")
        print()
        print("🔧 Next Steps:")
        print("   1. Configure your actual database and API credentials")
        print("   2. Review the ASYNC_DATABASE_AND_APIS_GUIDE.md for detailed documentation")
        print("   3. Explore the async_database.py and async_external_apis.py modules")
        print("   4. Integrate these operations into your Video-OpusClip application")
        print("   5. Monitor performance with the built-in metrics collectors")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the async database and API examples
    asyncio.run(main()) 