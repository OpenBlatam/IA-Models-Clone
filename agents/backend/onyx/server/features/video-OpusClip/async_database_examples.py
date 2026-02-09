"""
Async Database Operations Examples for Video-OpusClip

Comprehensive examples demonstrating real-world usage of async database operations
including video processing workflows, batch operations, transaction management,
and performance monitoring.
"""

import asyncio
import time
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import structlog

# Import async database modules
from async_database import (
    DatabaseConfig, DatabaseType, QueryType,
    PostgreSQLPool, MySQLPool, SQLitePool, RedisPool,
    AsyncDatabaseOperations, AsyncVideoDatabase,
    AsyncBatchDatabaseOperations, AsyncTransactionManager,
    AsyncDatabaseSetup,
    create_database_config, create_database_pool,
    create_async_database_operations, create_async_video_database,
    create_async_batch_database_operations, create_async_transaction_manager,
    create_async_database_setup, setup_database_connection,
    close_database_connection, get_query_metrics
)

logger = structlog.get_logger()

# =============================================================================
# EXAMPLE 1: BASIC DATABASE OPERATIONS
# =============================================================================

async def example_basic_database_operations():
    """Demonstrate basic database operations with PostgreSQL."""
    print("\n=== Example 1: Basic Database Operations ===")
    
    # Setup database connection
    config = create_database_config(
        host="localhost",
        port=5432,
        database="video_opusclip_test",
        username="postgres",
        password="password"
    )
    
    try:
        # Create database operations
        db_ops = await setup_database_connection(
            DatabaseType.POSTGRESQL,
            host=config.host,
            port=config.port,
            database=config.database,
            username=config.username,
            password=config.password
        )
        
        # Create video database
        video_db = create_async_video_database(db_ops)
        
        # Example 1.1: Create video record
        video_data = {
            "title": "Amazing Sunset Timelapse",
            "description": "Beautiful sunset captured over the ocean",
            "url": "https://example.com/video1.mp4",
            "duration": 120.5,
            "resolution": "1920x1080",
            "status": "pending",
            "created_at": datetime.now().isoformat()
        }
        
        video_id = await video_db.create_video_record(video_data)
        print(f"Created video record with ID: {video_id}")
        
        # Example 1.2: Retrieve video by ID
        video = await video_db.get_video_by_id(video_id)
        print(f"Retrieved video: {video['title']}")
        
        # Example 1.3: Update video status
        success = await video_db.update_video_status(video_id, "processing")
        print(f"Updated video status: {success}")
        
        # Example 1.4: Get videos by status
        pending_videos = await video_db.get_videos_by_status("pending")
        print(f"Found {len(pending_videos)} pending videos")
        
        # Example 1.5: Create clip record
        clip_data = {
            "video_id": video_id,
            "start_time": 10.0,
            "end_time": 30.0,
            "title": "Sunset Peak Moment",
            "description": "The most beautiful moment of the sunset",
            "status": "created",
            "created_at": datetime.now().isoformat()
        }
        
        clip_id = await video_db.create_clip_record(clip_data)
        print(f"Created clip record with ID: {clip_id}")
        
        # Example 1.6: Get clips for video
        clips = await video_db.get_clips_by_video_id(video_id)
        print(f"Found {len(clips)} clips for video {video_id}")
        
        # Example 1.7: Create processing job
        job_data = {
            "video_id": video_id,
            "job_type": "caption_generation",
            "parameters": {
                "language": "en",
                "style": "casual"
            },
            "status": "pending",
            "priority": 1,
            "created_at": datetime.now().isoformat()
        }
        
        job_id = await video_db.create_processing_job(job_data)
        print(f"Created processing job with ID: {job_id}")
        
        # Example 1.8: Get pending jobs
        pending_jobs = await video_db.get_pending_jobs(limit=5)
        print(f"Found {len(pending_jobs)} pending jobs")
        
        # Example 1.9: Update job status
        job_success = await video_db.update_job_status(
            job_id, 
            "completed", 
            {"captions": ["Beautiful sunset", "Ocean waves"]}
        )
        print(f"Updated job status: {job_success}")
        
        # Get metrics
        metrics = get_query_metrics(db_ops)
        print(f"Database metrics: {metrics}")
        
    except Exception as e:
        logger.error(f"Database operation failed: {e}")
    finally:
        await close_database_connection(db_ops)

# =============================================================================
# EXAMPLE 2: BATCH DATABASE OPERATIONS
# =============================================================================

async def example_batch_database_operations():
    """Demonstrate batch database operations for high-throughput scenarios."""
    print("\n=== Example 2: Batch Database Operations ===")
    
    try:
        # Setup database connection
        db_ops = await setup_database_connection(
            DatabaseType.POSTGRESQL,
            database="video_opusclip_test"
        )
        
        # Create batch operations
        batch_ops = create_async_batch_database_operations(db_ops)
        
        # Example 2.1: Batch insert videos
        videos = []
        for i in range(10):
            video_data = {
                "title": f"Video Batch {i+1}",
                "description": f"Batch video description {i+1}",
                "url": f"https://example.com/batch_video_{i+1}.mp4",
                "duration": 60.0 + i * 10,
                "resolution": "1920x1080",
                "status": "pending",
                "created_at": datetime.now().isoformat()
            }
            videos.append(video_data)
        
        video_ids = await batch_ops.batch_insert_videos(videos)
        print(f"Batch inserted {len(video_ids)} videos: {video_ids}")
        
        # Example 2.2: Batch insert clips
        clips = []
        for video_id in video_ids[:5]:  # Create clips for first 5 videos
            for j in range(3):  # 3 clips per video
                clip_data = {
                    "video_id": video_id,
                    "start_time": j * 20.0,
                    "end_time": (j + 1) * 20.0,
                    "title": f"Clip {j+1} for Video {video_id}",
                    "description": f"Batch clip description {j+1}",
                    "status": "created",
                    "created_at": datetime.now().isoformat()
                }
                clips.append(clip_data)
        
        clip_ids = await batch_ops.batch_insert_clips(clips)
        print(f"Batch inserted {len(clip_ids)} clips: {clip_ids}")
        
        # Example 2.3: Batch update video status
        updates = [(video_id, "processing") for video_id in video_ids[:5]]
        updated_count = await batch_ops.batch_update_video_status(updates)
        print(f"Batch updated {updated_count} videos")
        
    except Exception as e:
        logger.error(f"Batch operation failed: {e}")

# =============================================================================
# EXAMPLE 3: TRANSACTION MANAGEMENT
# =============================================================================

async def example_transaction_management():
    """Demonstrate transaction management for complex operations."""
    print("\n=== Example 3: Transaction Management ===")
    
    try:
        # Setup database connection
        db_ops = await setup_database_connection(
            DatabaseType.POSTGRESQL,
            database="video_opusclip_test"
        )
        
        # Create transaction manager
        tx_manager = create_async_transaction_manager(db_ops)
        video_db = create_async_video_database(db_ops)
        
        # Example 3.1: Simple transaction
        async with tx_manager.transaction() as tx:
            # Create video
            video_data = {
                "title": "Transaction Test Video",
                "description": "Testing transaction management",
                "url": "https://example.com/tx_video.mp4",
                "duration": 90.0,
                "resolution": "1920x1080",
                "status": "pending",
                "created_at": datetime.now().isoformat()
            }
            
            video_id = await video_db.create_video_record(video_data)
            print(f"Created video in transaction: {video_id}")
            
            # Create clip
            clip_data = {
                "video_id": video_id,
                "start_time": 0.0,
                "end_time": 30.0,
                "title": "Transaction Test Clip",
                "description": "Clip created in transaction",
                "status": "created",
                "created_at": datetime.now().isoformat()
            }
            
            clip_id = await video_db.create_clip_record(clip_data)
            print(f"Created clip in transaction: {clip_id}")
            
            # Update video status
            await video_db.update_video_status(video_id, "processing")
            print("Updated video status in transaction")
        
        print("Transaction completed successfully")
        
        # Example 3.2: Complex transaction with multiple operations
        async def create_video_with_clips_and_jobs(video_data: Dict, clips_data: List[Dict], job_data: Dict):
            """Complex operation creating video, clips, and job in one transaction."""
            video_id = await video_db.create_video_record(video_data)
            
            clip_ids = []
            for clip_data in clips_data:
                clip_data["video_id"] = video_id
                clip_id = await video_db.create_clip_record(clip_data)
                clip_ids.append(clip_id)
            
            job_data["video_id"] = video_id
            job_id = await video_db.create_processing_job(job_data)
            
            return video_id, clip_ids, job_id
        
        # Execute complex transaction
        video_data = {
            "title": "Complex Transaction Video",
            "description": "Video with multiple related records",
            "url": "https://example.com/complex_video.mp4",
            "duration": 120.0,
            "resolution": "1920x1080",
            "status": "pending",
            "created_at": datetime.now().isoformat()
        }
        
        clips_data = [
            {
                "start_time": 0.0,
                "end_time": 30.0,
                "title": "Clip 1",
                "description": "First clip",
                "status": "created",
                "created_at": datetime.now().isoformat()
            },
            {
                "start_time": 30.0,
                "end_time": 60.0,
                "title": "Clip 2",
                "description": "Second clip",
                "status": "created",
                "created_at": datetime.now().isoformat()
            }
        ]
        
        job_data = {
            "job_type": "caption_generation",
            "parameters": {"language": "en", "style": "formal"},
            "status": "pending",
            "priority": 2,
            "created_at": datetime.now().isoformat()
        }
        
        result = await tx_manager.execute_in_transaction([
            lambda tx: create_video_with_clips_and_jobs(video_data, clips_data, job_data)
        ])
        
        video_id, clip_ids, job_id = result[0]
        print(f"Complex transaction completed: Video={video_id}, Clips={clip_ids}, Job={job_id}")
        
    except Exception as e:
        logger.error(f"Transaction failed: {e}")

# =============================================================================
# EXAMPLE 4: PERFORMANCE MONITORING AND OPTIMIZATION
# =============================================================================

async def example_performance_monitoring():
    """Demonstrate performance monitoring and optimization."""
    print("\n=== Example 4: Performance Monitoring ===")
    
    try:
        # Setup database connection
        db_ops = await setup_database_connection(
            DatabaseType.POSTGRESQL,
            database="video_opusclip_test"
        )
        
        video_db = create_async_video_database(db_ops)
        
        # Example 4.1: Monitor query performance
        start_time = time.perf_counter()
        
        # Execute multiple queries
        for i in range(10):
            video_data = {
                "title": f"Performance Test Video {i}",
                "description": f"Testing query performance {i}",
                "url": f"https://example.com/perf_video_{i}.mp4",
                "duration": 60.0,
                "resolution": "1920x1080",
                "status": "pending",
                "created_at": datetime.now().isoformat()
            }
            
            video_id = await video_db.create_video_record(video_data)
            
            # Retrieve with caching
            video = await video_db.get_video_by_id(video_id)
            
            # Update status
            await video_db.update_video_status(video_id, "processing")
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        print(f"Executed 30 queries in {total_time:.2f} seconds")
        print(f"Average time per query: {total_time / 30:.3f} seconds")
        
        # Example 4.2: Get detailed metrics
        metrics = get_query_metrics(db_ops)
        print(f"Database metrics:")
        print(f"  Queries executed: {metrics['queries_executed']}")
        print(f"  Total execution time: {metrics['total_execution_time']:.2f}s")
        print(f"  Cache hits: {metrics['cache_hits']}")
        print(f"  Cache misses: {metrics['cache_misses']}")
        print(f"  Average query time: {metrics['total_execution_time'] / metrics['queries_executed']:.3f}s")
        
        # Example 4.3: Cache performance analysis
        cache_hit_rate = metrics['cache_hits'] / (metrics['cache_hits'] + metrics['cache_misses']) * 100
        print(f"Cache hit rate: {cache_hit_rate:.1f}%")
        
    except Exception as e:
        logger.error(f"Performance monitoring failed: {e}")

# =============================================================================
# EXAMPLE 5: VIDEO PROCESSING WORKFLOW
# =============================================================================

async def example_video_processing_workflow():
    """Demonstrate a complete video processing workflow using async database operations."""
    print("\n=== Example 5: Video Processing Workflow ===")
    
    try:
        # Setup database connection
        db_ops = await setup_database_connection(
            DatabaseType.POSTGRESQL,
            database="video_opusclip_test"
        )
        
        video_db = create_async_video_database(db_ops)
        tx_manager = create_async_transaction_manager(db_ops)
        
        # Step 1: Create video record
        video_data = {
            "title": "Workflow Test Video",
            "description": "Testing complete video processing workflow",
            "url": "https://example.com/workflow_video.mp4",
            "duration": 180.0,
            "resolution": "1920x1080",
            "status": "uploaded",
            "created_at": datetime.now().isoformat()
        }
        
        video_id = await video_db.create_video_record(video_data)
        print(f"Step 1: Created video record (ID: {video_id})")
        
        # Step 2: Create processing jobs
        jobs = [
            {
                "video_id": video_id,
                "job_type": "video_analysis",
                "parameters": {"analyze_audio": True, "detect_scenes": True},
                "status": "pending",
                "priority": 1,
                "created_at": datetime.now().isoformat()
            },
            {
                "video_id": video_id,
                "job_type": "caption_generation",
                "parameters": {"language": "en", "style": "casual"},
                "status": "pending",
                "priority": 2,
                "created_at": datetime.now().isoformat()
            },
            {
                "video_id": video_id,
                "job_type": "thumbnail_generation",
                "parameters": {"style": "modern", "include_text": True},
                "status": "pending",
                "priority": 3,
                "created_at": datetime.now().isoformat()
            }
        ]
        
        job_ids = []
        for job_data in jobs:
            job_id = await video_db.create_processing_job(job_data)
            job_ids.append(job_id)
        
        print(f"Step 2: Created {len(job_ids)} processing jobs")
        
        # Step 3: Simulate job processing
        for i, job_id in enumerate(job_ids):
            # Update job status to processing
            await video_db.update_job_status(job_id, "processing")
            print(f"Step 3.{i+1}: Started processing job {job_id}")
            
            # Simulate processing time
            await asyncio.sleep(0.1)
            
            # Update job status to completed with results
            results = {
                "job_type": jobs[i]["job_type"],
                "processing_time": 0.1,
                "status": "completed"
            }
            await video_db.update_job_status(job_id, "completed", results)
            print(f"Step 3.{i+1}: Completed job {job_id}")
        
        # Step 4: Create clips based on analysis
        async with tx_manager.transaction() as tx:
            clips_data = [
                {
                    "video_id": video_id,
                    "start_time": 0.0,
                    "end_time": 30.0,
                    "title": "Introduction",
                    "description": "Video introduction segment",
                    "status": "created",
                    "created_at": datetime.now().isoformat()
                },
                {
                    "video_id": video_id,
                    "start_time": 30.0,
                    "end_time": 90.0,
                    "title": "Main Content",
                    "description": "Main video content segment",
                    "status": "created",
                    "created_at": datetime.now().isoformat()
                },
                {
                    "video_id": video_id,
                    "start_time": 90.0,
                    "end_time": 120.0,
                    "title": "Conclusion",
                    "description": "Video conclusion segment",
                    "status": "created",
                    "created_at": datetime.now().isoformat()
                }
            ]
            
            clip_ids = []
            for clip_data in clips_data:
                clip_id = await video_db.create_clip_record(clip_data)
                clip_ids.append(clip_id)
            
            print(f"Step 4: Created {len(clip_ids)} clips in transaction")
        
        # Step 5: Update video status to completed
        await video_db.update_video_status(video_id, "completed")
        print(f"Step 5: Updated video status to completed")
        
        # Step 6: Retrieve final results
        final_video = await video_db.get_video_by_id(video_id)
        final_clips = await video_db.get_clips_by_video_id(video_id)
        final_jobs = await video_db.get_pending_jobs(limit=100)  # Get all jobs
        
        print(f"Step 6: Workflow completed successfully")
        print(f"  Video status: {final_video['status']}")
        print(f"  Clips created: {len(final_clips)}")
        print(f"  Jobs completed: {len([j for j in final_jobs if j['video_id'] == video_id and j['status'] == 'completed'])}")
        
    except Exception as e:
        logger.error(f"Video processing workflow failed: {e}")

# =============================================================================
# EXAMPLE 6: DATABASE SETUP AND MAINTENANCE
# =============================================================================

async def example_database_setup_and_maintenance():
    """Demonstrate database setup and maintenance operations."""
    print("\n=== Example 6: Database Setup and Maintenance ===")
    
    try:
        # Setup database connection
        db_ops = await setup_database_connection(
            DatabaseType.POSTGRESQL,
            database="video_opusclip_test"
        )
        
        # Create database setup
        db_setup = create_async_database_setup(db_ops)
        
        # Example 6.1: Create tables
        await db_setup.create_tables()
        print("Created database tables")
        
        # Example 6.2: Get database statistics
        stats = await db_setup.get_database_stats()
        print(f"Database statistics:")
        print(f"  Total videos: {stats['total_videos']}")
        print(f"  Total clips: {stats['total_clips']}")
        print(f"  Total jobs: {stats['total_jobs']}")
        print(f"  Pending jobs: {stats['pending_jobs']}")
        print(f"  Completed jobs: {stats['completed_jobs']}")
        
    except Exception as e:
        logger.error(f"Database setup failed: {e}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

async def main():
    """Run all database examples."""
    print("Async Database Operations Examples")
    print("=" * 50)
    
    # Run examples
    await example_basic_database_operations()
    await example_batch_database_operations()
    await example_transaction_management()
    await example_performance_monitoring()
    await example_video_processing_workflow()
    await example_database_setup_and_maintenance()
    
    print("\nAll examples completed!")

if __name__ == "__main__":
    asyncio.run(main()) 