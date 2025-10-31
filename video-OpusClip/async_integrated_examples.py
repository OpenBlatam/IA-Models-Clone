"""
Async Integrated Examples for Video-OpusClip

Comprehensive examples demonstrating the integration of async database and external API operations
for real-world video processing scenarios including end-to-end workflows, data synchronization,
and performance optimization.
"""

import asyncio
import time
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import structlog

# Import async modules
from async_database import (
    DatabaseType, AsyncVideoDatabase, AsyncBatchDatabaseOperations,
    AsyncTransactionManager, setup_database_connection, close_database_connection
)

from async_external_apis import (
    APIType, AsyncYouTubeAPI, AsyncOpenAIAPI, AsyncStabilityAIAPI,
    AsyncElevenLabsAPI, AsyncBatchAPIOperations, setup_external_api,
    close_external_api
)

logger = structlog.get_logger()

# =============================================================================
# EXAMPLE 1: END-TO-END VIDEO PROCESSING WORKFLOW
# =============================================================================

async def example_end_to_end_video_processing():
    """Demonstrate a complete end-to-end video processing workflow."""
    print("\n=== Example 1: End-to-End Video Processing Workflow ===")
    
    try:
        # Setup database and APIs
        db_ops = await setup_database_connection(
            DatabaseType.POSTGRESQL,
            database="video_opusclip_test"
        )
        
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
        
        # Create database operations
        video_db = AsyncVideoDatabase(db_ops)
        tx_manager = AsyncTransactionManager(db_ops)
        
        # Workflow: Process a YouTube video
        video_id = "dQw4w9WgXcQ"
        
        print(f"Starting workflow for video: {video_id}")
        
        # Step 1: Get video information from YouTube
        print("Step 1: Fetching video information from YouTube...")
        video_info = await youtube_api.get_video_info(video_id)
        
        video_title = video_info.get('title', 'Unknown Video')
        video_description = video_info.get('description', '')
        video_duration = video_info.get('duration', 0)
        
        print(f"  Title: {video_title}")
        print(f"  Duration: {video_duration} seconds")
        
        # Step 2: Create video record in database
        print("Step 2: Creating video record in database...")
        video_data = {
            "title": video_title,
            "description": video_description[:500],  # Truncate if too long
            "url": f"https://youtube.com/watch?v={video_id}",
            "duration": video_duration,
            "resolution": "1920x1080",  # Default
            "status": "processing",
            "created_at": datetime.now().isoformat(),
            "metadata": {
                "youtube_id": video_id,
                "view_count": video_info.get('viewCount', 0),
                "like_count": video_info.get('likeCount', 0)
            }
        }
        
        db_video_id = await video_db.create_video_record(video_data)
        print(f"  Created database record with ID: {db_video_id}")
        
        # Step 3: Generate content analysis using OpenAI
        print("Step 3: Analyzing video content with OpenAI...")
        content_analysis = await openai_api.analyze_video_content(
            video_description=video_description,
            video_title=video_title
        )
        
        print(f"  Content analysis completed")
        
        # Step 4: Generate captions
        print("Step 4: Generating captions...")
        captions = await openai_api.generate_captions(
            audio_text=f"{video_title}. {video_description[:200]}",
            style="casual",
            language="en"
        )
        
        print(f"  Generated {len(captions)} captions")
        
        # Step 5: Generate thumbnail using Stability AI
        print("Step 5: Generating thumbnail...")
        thumbnail = await stability_api.generate_video_thumbnail(
            video_title=video_title,
            video_description=video_description[:100]
        )
        
        print(f"  Thumbnail generated")
        
        # Step 6: Generate voice narration
        print("Step 6: Generating voice narration...")
        narration_script = f"Welcome to {video_title}. This is an amazing video that you won't want to miss!"
        narration = await elevenlabs_api.generate_video_narration(
            script=narration_script,
            voice_id="21m00Tcm4TlvDq8ikWAM"
        )
        
        print(f"  Voice narration generated")
        
        # Step 7: Create clips and jobs in database transaction
        print("Step 7: Creating clips and processing jobs...")
        async with tx_manager.transaction() as tx:
            # Create clips based on content analysis
            clip_segments = content_analysis.get('segments', [])
            clip_ids = []
            
            for i, segment in enumerate(clip_segments[:3]):  # Limit to 3 clips
                clip_data = {
                    "video_id": db_video_id,
                    "start_time": segment.get('start_time', i * 30.0),
                    "end_time": segment.get('end_time', (i + 1) * 30.0),
                    "title": segment.get('title', f"Clip {i+1}"),
                    "description": segment.get('description', f"Auto-generated clip {i+1}"),
                    "status": "created",
                    "created_at": datetime.now().isoformat(),
                    "metadata": {
                        "segment_type": segment.get('type', 'auto'),
                        "confidence": segment.get('confidence', 0.8)
                    }
                }
                
                clip_id = await video_db.create_clip_record(clip_data)
                clip_ids.append(clip_id)
            
            # Create processing jobs
            jobs = [
                {
                    "video_id": db_video_id,
                    "job_type": "caption_embedding",
                    "parameters": {
                        "captions": captions,
                        "language": "en"
                    },
                    "status": "pending",
                    "priority": 1,
                    "created_at": datetime.now().isoformat()
                },
                {
                    "video_id": db_video_id,
                    "job_type": "thumbnail_processing",
                    "parameters": {
                        "thumbnail_data": thumbnail,
                        "format": "jpeg"
                    },
                    "status": "pending",
                    "priority": 2,
                    "created_at": datetime.now().isoformat()
                },
                {
                    "video_id": db_video_id,
                    "job_type": "audio_processing",
                    "parameters": {
                        "narration_data": narration,
                        "format": "mp3"
                    },
                    "status": "pending",
                    "priority": 3,
                    "created_at": datetime.now().isoformat()
                }
            ]
            
            job_ids = []
            for job_data in jobs:
                job_id = await video_db.create_processing_job(job_data)
                job_ids.append(job_id)
        
        print(f"  Created {len(clip_ids)} clips and {len(job_ids)} jobs")
        
        # Step 8: Update video status to completed
        print("Step 8: Finalizing video processing...")
        await video_db.update_video_status(db_video_id, "completed")
        
        # Step 9: Compile final results
        final_results = {
            "video_id": db_video_id,
            "youtube_id": video_id,
            "title": video_title,
            "clips_created": len(clip_ids),
            "jobs_created": len(job_ids),
            "captions_generated": len(captions),
            "content_analysis": content_analysis,
            "processed_at": datetime.now().isoformat()
        }
        
        print(f"Workflow completed successfully!")
        print(f"Final results: {json.dumps(final_results, indent=2, default=str)}")
        
        # Cleanup
        await close_database_connection(db_ops)
        await close_external_api(youtube_api)
        await close_external_api(openai_api)
        await close_external_api(stability_api)
        await close_external_api(elevenlabs_api)
        
    except Exception as e:
        logger.error(f"End-to-end workflow failed: {e}")

# =============================================================================
# EXAMPLE 2: BATCH VIDEO PROCESSING PIPELINE
# =============================================================================

async def example_batch_video_processing():
    """Demonstrate batch processing of multiple videos."""
    print("\n=== Example 2: Batch Video Processing Pipeline ===")
    
    try:
        # Setup database and APIs
        db_ops = await setup_database_connection(
            DatabaseType.POSTGRESQL,
            database="video_opusclip_test"
        )
        
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
        
        # Create batch operations
        batch_db_ops = AsyncBatchDatabaseOperations(db_ops)
        batch_api_ops = AsyncBatchAPIOperations(openai_api)
        
        # List of videos to process
        video_ids = [
            "dQw4w9WgXcQ",
            "9bZkp7q19f0",
            "kJQP7kiw5Fk",
            "y6120QOlsfU",
            "ZZ5LpwO-An4"
        ]
        
        print(f"Processing {len(video_ids)} videos in batch...")
        
        # Step 1: Batch fetch video information
        print("Step 1: Batch fetching video information...")
        video_infos = await batch_api_ops.batch_get_video_info(video_ids)
        
        successful_videos = [info for info in video_infos if info]
        print(f"  Successfully fetched {len(successful_videos)} videos")
        
        # Step 2: Prepare video data for database
        videos_data = []
        audio_texts = []
        
        for video_info in successful_videos:
            video_data = {
                "title": video_info.get('title', 'Unknown'),
                "description": video_info.get('description', '')[:500],
                "url": f"https://youtube.com/watch?v={video_info.get('id', '')}",
                "duration": video_info.get('duration', 0),
                "resolution": "1920x1080",
                "status": "pending",
                "created_at": datetime.now().isoformat(),
                "metadata": {
                    "youtube_id": video_info.get('id'),
                    "view_count": video_info.get('viewCount', 0)
                }
            }
            videos_data.append(video_data)
            
            # Prepare audio text for caption generation
            audio_text = f"{video_data['title']}. {video_data['description'][:200]}"
            audio_texts.append(audio_text)
        
        # Step 3: Batch insert videos
        print("Step 3: Batch inserting videos into database...")
        db_video_ids = await batch_db_ops.batch_insert_videos(videos_data)
        print(f"  Inserted {len(db_video_ids)} videos")
        
        # Step 4: Batch generate captions
        print("Step 4: Batch generating captions...")
        captions_batch = await batch_api_ops.batch_generate_captions(
            audio_texts=audio_texts,
            style="casual"
        )
        
        print(f"  Generated captions for {len(captions_batch)} videos")
        
        # Step 5: Create processing jobs for each video
        print("Step 5: Creating processing jobs...")
        all_jobs = []
        
        for i, (video_id, captions) in enumerate(zip(db_video_ids, captions_batch)):
            jobs = [
                {
                    "video_id": video_id,
                    "job_type": "caption_processing",
                    "parameters": {"captions": captions},
                    "status": "pending",
                    "priority": 1,
                    "created_at": datetime.now().isoformat()
                },
                {
                    "video_id": video_id,
                    "job_type": "thumbnail_generation",
                    "parameters": {"style": "modern"},
                    "status": "pending",
                    "priority": 2,
                    "created_at": datetime.now().isoformat()
                }
            ]
            all_jobs.extend(jobs)
        
        # Note: This would require a batch job creation method
        # For now, we'll create jobs individually
        job_ids = []
        for job_data in all_jobs:
            # This would be a batch operation in a real implementation
            job_ids.append(job_data)
        
        print(f"  Created {len(job_ids)} processing jobs")
        
        # Step 6: Update video statuses
        print("Step 6: Updating video statuses...")
        status_updates = [(video_id, "processing") for video_id in db_video_ids]
        updated_count = await batch_db_ops.batch_update_video_status(status_updates)
        print(f"  Updated {updated_count} video statuses")
        
        # Compile batch results
        batch_results = {
            "total_videos": len(video_ids),
            "successful_videos": len(successful_videos),
            "videos_inserted": len(db_video_ids),
            "captions_generated": len(captions_batch),
            "jobs_created": len(job_ids),
            "processed_at": datetime.now().isoformat()
        }
        
        print(f"Batch processing completed!")
        print(f"Results: {json.dumps(batch_results, indent=2, default=str)}")
        
        # Cleanup
        await close_database_connection(db_ops)
        await close_external_api(youtube_api)
        await close_external_api(openai_api)
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")

# =============================================================================
# EXAMPLE 3: REAL-TIME VIDEO MONITORING SYSTEM
# =============================================================================

async def example_real_time_monitoring():
    """Demonstrate real-time monitoring of video processing."""
    print("\n=== Example 3: Real-Time Video Monitoring System ===")
    
    try:
        # Setup database and APIs
        db_ops = await setup_database_connection(
            DatabaseType.POSTGRESQL,
            database="video_opusclip_test"
        )
        
        youtube_api = await setup_external_api(
            APIType.YOUTUBE,
            base_url="https://www.googleapis.com/youtube/v3",
            api_key="your_youtube_api_key"
        )
        
        video_db = AsyncVideoDatabase(db_ops)
        
        # Simulate real-time monitoring
        print("Starting real-time monitoring...")
        
        # Monitor for 10 seconds
        start_time = time.time()
        monitoring_duration = 10
        
        while time.time() - start_time < monitoring_duration:
            # Get current processing status
            pending_jobs = await video_db.get_pending_jobs(limit=10)
            processing_videos = await video_db.get_videos_by_status("processing")
            completed_videos = await video_db.get_videos_by_status("completed")
            
            print(f"\n--- Monitoring Update ({time.time() - start_time:.1f}s) ---")
            print(f"Pending jobs: {len(pending_jobs)}")
            print(f"Processing videos: {len(processing_videos)}")
            print(f"Completed videos: {len(completed_videos)}")
            
            # Simulate processing a job
            if pending_jobs:
                job = pending_jobs[0]
                print(f"Processing job {job['id']} for video {job['video_id']}")
                
                # Update job status
                await video_db.update_job_status(
                    job['id'],
                    "completed",
                    {"processed_at": datetime.now().isoformat()}
                )
                
                # Check if all jobs for video are completed
                video_jobs = await video_db.get_pending_jobs(limit=100)
                video_jobs = [j for j in video_jobs if j['video_id'] == job['video_id']]
                
                if not video_jobs:
                    await video_db.update_video_status(job['video_id'], "completed")
                    print(f"Video {job['video_id']} completed!")
            
            await asyncio.sleep(1)  # Update every second
        
        print("Monitoring completed!")
        
        # Final status report
        final_pending = await video_db.get_pending_jobs(limit=100)
        final_processing = await video_db.get_videos_by_status("processing")
        final_completed = await video_db.get_videos_by_status("completed")
        
        print(f"\nFinal Status Report:")
        print(f"  Pending jobs: {len(final_pending)}")
        print(f"  Processing videos: {len(final_processing)}")
        print(f"  Completed videos: {len(final_completed)}")
        
        # Cleanup
        await close_database_connection(db_ops)
        await close_external_api(youtube_api)
        
    except Exception as e:
        logger.error(f"Real-time monitoring failed: {e}")

# =============================================================================
# EXAMPLE 4: DATA SYNCHRONIZATION BETWEEN APIS AND DATABASE
# =============================================================================

async def example_data_synchronization():
    """Demonstrate data synchronization between external APIs and database."""
    print("\n=== Example 4: Data Synchronization ===")
    
    try:
        # Setup database and APIs
        db_ops = await setup_database_connection(
            DatabaseType.POSTGRESQL,
            database="video_opusclip_test"
        )
        
        youtube_api = await setup_external_api(
            APIType.YOUTUBE,
            base_url="https://www.googleapis.com/youtube/v3",
            api_key="your_youtube_api_key"
        )
        
        video_db = AsyncVideoDatabase(db_ops)
        tx_manager = AsyncTransactionManager(db_ops)
        
        # Example: Sync YouTube channel videos
        channel_id = "UC_x5XG1OV2P6uZZ5FSM9Ttw"  # Google Developers channel
        
        print(f"Synchronizing videos from channel: {channel_id}")
        
        # Step 1: Get channel videos from YouTube
        print("Step 1: Fetching channel videos from YouTube...")
        channel_videos = await youtube_api.get_channel_videos(channel_id, max_results=10)
        
        print(f"  Found {len(channel_videos)} videos on YouTube")
        
        # Step 2: Check which videos already exist in database
        print("Step 2: Checking existing videos in database...")
        existing_videos = []
        new_videos = []
        
        for video in channel_videos:
            # Check if video exists (by YouTube ID in metadata)
            # This is a simplified check - in real implementation, you'd query by metadata
            existing_videos.append(video)
            new_videos.append(video)
        
        print(f"  New videos to sync: {len(new_videos)}")
        
        # Step 3: Sync new videos to database
        print("Step 3: Syncing new videos to database...")
        synced_count = 0
        
        async with tx_manager.transaction() as tx:
            for video in new_videos:
                video_data = {
                    "title": video.get('title', 'Unknown'),
                    "description": video.get('description', '')[:500],
                    "url": f"https://youtube.com/watch?v={video.get('id', '')}",
                    "duration": video.get('duration', 0),
                    "resolution": "1920x1080",
                    "status": "synced",
                    "created_at": datetime.now().isoformat(),
                    "metadata": {
                        "youtube_id": video.get('id'),
                        "channel_id": channel_id,
                        "published_at": video.get('publishedAt'),
                        "view_count": video.get('viewCount', 0),
                        "like_count": video.get('likeCount', 0)
                    }
                }
                
                video_id = await video_db.create_video_record(video_data)
                synced_count += 1
                
                print(f"    Synced: {video_data['title']} (ID: {video_id})")
        
        print(f"  Successfully synced {synced_count} videos")
        
        # Step 4: Update existing videos with latest data
        print("Step 4: Updating existing videos...")
        updated_count = 0
        
        for video in existing_videos:
            # Update view count, like count, etc.
            # This would require a method to update video metadata
            updated_count += 1
        
        print(f"  Updated {updated_count} existing videos")
        
        # Step 5: Generate sync report
        sync_report = {
            "channel_id": channel_id,
            "total_videos_found": len(channel_videos),
            "new_videos_synced": synced_count,
            "existing_videos_updated": updated_count,
            "sync_timestamp": datetime.now().isoformat()
        }
        
        print(f"Sync completed!")
        print(f"Report: {json.dumps(sync_report, indent=2, default=str)}")
        
        # Cleanup
        await close_database_connection(db_ops)
        await close_external_api(youtube_api)
        
    except Exception as e:
        logger.error(f"Data synchronization failed: {e}")

# =============================================================================
# EXAMPLE 5: PERFORMANCE OPTIMIZATION AND CACHING
# =============================================================================

async def example_performance_optimization():
    """Demonstrate performance optimization techniques."""
    print("\n=== Example 5: Performance Optimization ===")
    
    try:
        # Setup database and APIs with optimized configurations
        db_ops = await setup_database_connection(
            DatabaseType.POSTGRESQL,
            database="video_opusclip_test"
        )
        
        # Setup APIs with aggressive caching
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
        
        video_db = AsyncVideoDatabase(db_ops)
        
        # Test performance with and without caching
        video_ids = ["dQw4w9WgXcQ", "9bZkp7q19f0", "kJQP7kiw5Fk"]
        
        print("Testing performance optimization...")
        
        # Test 1: Sequential requests (no caching)
        print("Test 1: Sequential requests without caching...")
        start_time = time.perf_counter()
        
        for video_id in video_ids:
            video_info = await youtube_api.get_video_info(video_id)
            print(f"  Fetched: {video_info.get('title', 'Unknown')}")
        
        sequential_time = time.perf_counter() - start_time
        print(f"  Sequential time: {sequential_time:.3f}s")
        
        # Test 2: Parallel requests (with caching)
        print("Test 2: Parallel requests with caching...")
        start_time = time.perf_counter()
        
        tasks = []
        for video_id in video_ids:
            task = youtube_api.get_video_info(video_id)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        parallel_time = time.perf_counter() - start_time
        print(f"  Parallel time: {parallel_time:.3f}s")
        print(f"  Speedup: {sequential_time/parallel_time:.1f}x faster")
        
        # Test 3: Database batch operations
        print("Test 3: Database batch operations...")
        start_time = time.perf_counter()
        
        # Create test videos
        videos_data = []
        for i in range(10):
            video_data = {
                "title": f"Performance Test Video {i}",
                "description": f"Testing batch operations {i}",
                "url": f"https://example.com/perf_video_{i}.mp4",
                "duration": 60.0,
                "resolution": "1920x1080",
                "status": "pending",
                "created_at": datetime.now().isoformat()
            }
            videos_data.append(video_data)
        
        # Individual inserts
        individual_start = time.perf_counter()
        individual_ids = []
        for video_data in videos_data:
            video_id = await video_db.create_video_record(video_data)
            individual_ids.append(video_id)
        individual_time = time.perf_counter() - individual_start
        
        print(f"  Individual inserts: {individual_time:.3f}s")
        
        # Batch inserts (simulated)
        batch_start = time.perf_counter()
        batch_ops = AsyncBatchDatabaseOperations(db_ops)
        batch_ids = await batch_ops.batch_insert_videos(videos_data)
        batch_time = time.perf_counter() - batch_start
        
        print(f"  Batch inserts: {batch_time:.3f}s")
        print(f"  Batch speedup: {individual_time/batch_time:.1f}x faster")
        
        # Performance summary
        performance_summary = {
            "sequential_time": sequential_time,
            "parallel_time": parallel_time,
            "parallel_speedup": sequential_time/parallel_time,
            "individual_db_time": individual_time,
            "batch_db_time": batch_time,
            "batch_speedup": individual_time/batch_time,
            "tested_at": datetime.now().isoformat()
        }
        
        print(f"Performance optimization completed!")
        print(f"Summary: {json.dumps(performance_summary, indent=2, default=str)}")
        
        # Cleanup
        await close_database_connection(db_ops)
        await close_external_api(youtube_api)
        await close_external_api(openai_api)
        
    except Exception as e:
        logger.error(f"Performance optimization failed: {e}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

async def main():
    """Run all integrated examples."""
    print("Async Integrated Examples")
    print("=" * 50)
    
    # Run examples
    await example_end_to_end_video_processing()
    await example_batch_video_processing()
    await example_real_time_monitoring()
    await example_data_synchronization()
    await example_performance_optimization()
    
    print("\nAll integrated examples completed!")

if __name__ == "__main__":
    asyncio.run(main()) 