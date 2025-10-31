from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import time
import logging
import json
import hashlib
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from pathlib import Path
import aiofiles
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import psutil
import gc
import torch
import numpy as np
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from sqlalchemy.orm import selectinload
from typing import Any, List, Dict, Optional
"""
🚀 PERFORMANCE EXAMPLES - REAL-WORLD AI VIDEO OPTIMIZATION
==========================================================

Practical examples of performance optimization for AI Video systems:
- Async video processing pipelines
- Model caching and lazy loading
- Database query optimization
- Memory management for large models
- Background task processing
"""



logger = logging.getLogger(__name__)

# ============================================================================
# EXAMPLE 1: ASYNC VIDEO PROCESSING PIPELINE
# ============================================================================

class AsyncVideoProcessor:
    """Async video processing pipeline with optimization."""
    
    def __init__(self, max_concurrent: int = 3):
        
    """__init__ function."""
self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
    
    async def process_video_batch(self, video_requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple video requests concurrently."""
        
        async def process_single_video(request: Dict[str, Any]) -> Dict[str, Any]:
            async with self.semaphore:
                return await self._process_video_async(request)
        
        # Process videos concurrently
        tasks = [process_single_video(request) for request in video_requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Video {i} processing failed: {result}")
            else:
                successful_results.append(result)
        
        return successful_results
    
    async def _process_video_async(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single video asynchronously."""
        start_time = time.time()
        
        try:
            # 1. Validate request (CPU-bound, run in executor)
            validation_result = await asyncio.to_thread(
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                self._validate_video_request, request
            )
            
            if not validation_result["valid"]:
                return {
                    "video_id": request.get("video_id"),
                    "status": "failed",
                    "error": validation_result["error"],
                    "processing_time": time.time() - start_time
                }
            
            # 2. Load model (I/O-bound, cached)
            model = await self._load_video_model_async(request["model_name"])
            
            # 3. Generate video (CPU-bound, run in executor)
            video_data = await asyncio.to_thread(
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                self._generate_video_sync, model, request
            )
            
            # 4. Save video file (I/O-bound)
            file_path = await self._save_video_file_async(
                video_data, request["video_id"]
            )
            
            # 5. Generate thumbnail (CPU-bound, run in executor)
            thumbnail_path = await asyncio.to_thread(
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                self._generate_thumbnail_sync, video_data
            )
            
            processing_time = time.time() - start_time
            
            return {
                "video_id": request["video_id"],
                "status": "completed",
                "file_path": file_path,
                "thumbnail_path": thumbnail_path,
                "processing_time": processing_time,
                "file_size": len(video_data)
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Video processing failed: {e}")
            
            return {
                "video_id": request.get("video_id"),
                "status": "failed",
                "error": str(e),
                "processing_time": processing_time
            }
    
    async def _validate_video_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Validate video request (CPU-bound)."""
        required_fields = ["video_id", "prompt", "model_name", "width", "height"]
        
        for field in required_fields:
            if field not in request:
                return {"valid": False, "error": f"Missing required field: {field}"}
        
        if not (64 <= request["width"] <= 4096 and 64 <= request["height"] <= 4096):
            return {"valid": False, "error": "Invalid dimensions"}
        
        if len(request["prompt"]) < 1 or len(request["prompt"]) > 1000:
            return {"valid": False, "error": "Invalid prompt length"}
        
        return {"valid": True, "error": None}
    
    async def _load_video_model_async(self, model_name: str) -> Any:
        """Load video model asynchronously with caching."""
        # This would integrate with the ModelCache from performance_optimization.py
        # For now, simulate model loading
        await asyncio.sleep(0.5)  # Simulate loading time
        return {"model": model_name, "loaded": True}
    
    def _generate_video_sync(self, model: Any, request: Dict[str, Any]) -> bytes:
        """Generate video synchronously (CPU-bound)."""
        # Simulate video generation
        time.sleep(2)  # Simulate processing time
        return b"fake_video_data" * 1000  # Simulate video data
    
    async def _save_video_file_async(self, video_data: bytes, video_id: str) -> str:
        """Save video file asynchronously."""
        file_path = f"/videos/{video_id}.mp4"
        
        # Use aiofiles for async file I/O
        async with aiofiles.open(file_path, 'wb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            await f.write(video_data)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        logger.info(f"Saved video file: {file_path}")
        return file_path
    
    def _generate_thumbnail_sync(self, video_data: bytes) -> str:
        """Generate thumbnail synchronously (CPU-bound)."""
        # Simulate thumbnail generation
        time.sleep(0.5)
        thumbnail_path = f"/thumbnails/{hash(video_data)}.jpg"
        return thumbnail_path

# ============================================================================
# EXAMPLE 2: MODEL CACHING AND LAZY LOADING
# ============================================================================

class AIVideoModelManager:
    """AI Video model manager with caching and lazy loading."""
    
    def __init__(self, cache: Any):  # Would use AsyncCache from performance_optimization.py
        self.cache = cache
        self.loaded_models = {}
        self.model_loaders = {}
        self._lock = asyncio.Lock()
    
    def register_model(self, model_name: str, loader: Callable):
        """Register a model loader function."""
        self.model_loaders[model_name] = loader
        logger.info(f"Registered model loader for: {model_name}")
    
    async def get_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get model with lazy loading and caching."""
        # Check if model is already loaded in memory
        if model_name in self.loaded_models:
            logger.info(f"Using cached model: {model_name}")
            return self.loaded_models[model_name]
        
        async with self._lock:
            # Double-check after acquiring lock
            if model_name in self.loaded_models:
                return self.loaded_models[model_name]
            
            # Try to load from persistent cache
            cached_model = await self._load_from_cache(model_name)
            if cached_model:
                self.loaded_models[model_name] = cached_model
                logger.info(f"Loaded model from cache: {model_name}")
                return cached_model
            
            # Load model using registered loader
            if model_name in self.model_loaders:
                logger.info(f"Loading model: {model_name}")
                model = await self.model_loaders[model_name]()
                
                # Store in memory and persistent cache
                self.loaded_models[model_name] = model
                await self._save_to_cache(model_name, model)
                
                logger.info(f"Model loaded successfully: {model_name}")
                return model
            
            raise ValueError(f"Model {model_name} not found and no loader registered")
    
    async def _load_from_cache(self, model_name: str) -> Optional[Any]:
        """Load model from persistent cache."""
        try:
            # This would use the cache system from performance_optimization.py
            # For now, simulate cache loading
            await asyncio.sleep(0.1)
            return None  # Simulate cache miss
        except Exception as e:
            logger.error(f"Cache load failed for {model_name}: {e}")
            return None
    
    async def _save_to_cache(self, model_name: str, model: Any):
        """Save model to persistent cache."""
        try:
            # This would use the cache system from performance_optimization.py
            # For now, simulate cache saving
            await asyncio.sleep(0.1)
            logger.info(f"Model saved to cache: {model_name}")
        except Exception as e:
            logger.error(f"Cache save failed for {model_name}: {e}")
    
    async def preload_models(self, model_names: List[str]):
        """Preload multiple models concurrently."""
        logger.info(f"Preloading models: {model_names}")
        
        tasks = [self.get_model(name) for name in model_names]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to preload {model_names[i]}: {result}")
            else:
                successful += 1
        
        logger.info(f"Preloaded {successful}/{len(model_names)} models")
    
    async def unload_model(self, model_name: str):
        """Unload model from memory."""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            logger.info(f"Unloaded model from memory: {model_name}")
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models."""
        return list(self.loaded_models.keys())

# ============================================================================
# EXAMPLE 3: DATABASE QUERY OPTIMIZATION
# ============================================================================

class VideoDatabaseOptimizer:
    """Database query optimizer for video operations."""
    
    def __init__(self, cache: Any):  # Would use AsyncCache from performance_optimization.py
        self.cache = cache
        self.query_stats = {}
    
    async def get_video_by_id(self, session: AsyncSession, video_id: str) -> Optional[Dict[str, Any]]:
        """Get video by ID with caching."""
        cache_key = f"video:{video_id}"
        
        # Try cache first
        cached_video = await self._get_from_cache(cache_key)
        if cached_video:
            return cached_video
        
        # Query database
        video = await self._query_video_from_db(session, video_id)
        
        if video:
            # Cache result
            await self._save_to_cache(cache_key, video, ttl=3600)
        
        return video
    
    async def get_user_videos(self, session: AsyncSession, user_id: str, 
                            limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        """Get user videos with pagination and caching."""
        cache_key = f"user_videos:{user_id}:{limit}:{offset}"
        
        # Try cache first
        cached_videos = await self._get_from_cache(cache_key)
        if cached_videos:
            return cached_videos
        
        # Query database
        videos = await self._query_user_videos_from_db(session, user_id, limit, offset)
        
        # Cache result
        await self._save_to_cache(cache_key, videos, ttl=1800)
        
        return videos
    
    async def batch_get_videos(self, session: AsyncSession, video_ids: List[str]) -> List[Dict[str, Any]]:
        """Get multiple videos in batch with optimization."""
        # Group by cache hit/miss
        cache_hits = []
        cache_misses = []
        
        # Check cache for all videos
        for video_id in video_ids:
            cache_key = f"video:{video_id}"
            cached_video = await self._get_from_cache(cache_key)
            
            if cached_video:
                cache_hits.append(cached_video)
            else:
                cache_misses.append(video_id)
        
        # Query database for cache misses
        db_videos = []
        if cache_misses:
            db_videos = await self._batch_query_videos_from_db(session, cache_misses)
            
            # Cache new results
            for video in db_videos:
                cache_key = f"video:{video['id']}"
                await self._save_to_cache(cache_key, video, ttl=3600)
        
        # Combine results
        all_videos = cache_hits + db_videos
        
        # Sort by original order
        video_map = {v['id']: v for v in all_videos}
        return [video_map.get(vid) for vid in video_ids if vid in video_map]
    
    async def _query_video_from_db(self, session: AsyncSession, video_id: str) -> Optional[Dict[str, Any]]:
        """Query video from database."""
        # This would use actual SQLAlchemy models
        # For now, simulate database query
        await asyncio.sleep(0.1)  # Simulate query time
        
        # Simulate video data
        return {
            "id": video_id,
            "title": f"Video {video_id}",
            "status": "completed",
            "created_at": "2024-01-01T00:00:00Z"
        }
    
    async def _query_user_videos_from_db(self, session: AsyncSession, user_id: str, 
                                       limit: int, offset: int) -> List[Dict[str, Any]]:
        """Query user videos from database."""
        # This would use actual SQLAlchemy models
        # For now, simulate database query
        await asyncio.sleep(0.2)  # Simulate query time
        
        # Simulate video list
        return [
            {
                "id": f"video_{i}",
                "title": f"User Video {i}",
                "status": "completed",
                "created_at": "2024-01-01T00:00:00Z"
            }
            for i in range(offset, offset + limit)
        ]
    
    async def _batch_query_videos_from_db(self, session: AsyncSession, video_ids: List[str]) -> List[Dict[str, Any]]:
        """Batch query videos from database."""
        # This would use actual SQLAlchemy models with IN clause
        # For now, simulate batch query
        await asyncio.sleep(0.1)  # Simulate query time
        
        return [
            {
                "id": video_id,
                "title": f"Video {video_id}",
                "status": "completed",
                "created_at": "2024-01-01T00:00:00Z"
            }
            for video_id in video_ids
        ]
    
    async def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            # This would use the cache system from performance_optimization.py
            # For now, simulate cache get
            await asyncio.sleep(0.01)
            return None  # Simulate cache miss
        except Exception as e:
            logger.error(f"Cache get failed: {e}")
            return None
    
    async def _save_to_cache(self, key: str, value: Any, ttl: int = 3600):
        """Save value to cache."""
        try:
            # This would use the cache system from performance_optimization.py
            # For now, simulate cache save
            await asyncio.sleep(0.01)
        except Exception as e:
            logger.error(f"Cache save failed: {e}")

# ============================================================================
# EXAMPLE 4: MEMORY MANAGEMENT FOR LARGE MODELS
# ============================================================================

class ModelMemoryManager:
    """Memory manager for large AI models."""
    
    def __init__(self, max_memory_gb: float = 8.0):
        
    """__init__ function."""
self.max_memory_gb = max_memory_gb
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        self.loaded_models = {}
        self.model_sizes = {}
        self._lock = asyncio.Lock()
    
    async def load_model(self, model_name: str, model_data: Any, estimated_size_mb: float) -> bool:
        """Load model with memory management."""
        model_size_bytes = estimated_size_mb * 1024 * 1024
        
        async with self._lock:
            # Check if we have enough memory
            if not await self._can_load_model(model_size_bytes):
                # Try to free memory
                await self._free_memory(model_size_bytes)
                
                # Check again
                if not await self._can_load_model(model_size_bytes):
                    logger.error(f"Insufficient memory to load model: {model_name}")
                    return False
            
            # Load model
            self.loaded_models[model_name] = model_data
            self.model_sizes[model_name] = model_size_bytes
            
            logger.info(f"Loaded model: {model_name} ({estimated_size_mb:.1f}MB)")
            return True
    
    async def unload_model(self, model_name: str):
        """Unload model and free memory."""
        async with self._lock:
            if model_name in self.loaded_models:
                del self.loaded_models[model_name]
                del self.model_sizes[model_name]
                
                # Force garbage collection
                gc.collect()
                
                logger.info(f"Unloaded model: {model_name}")
    
    async def get_model(self, model_name: str) -> Optional[Any]:
        """Get loaded model."""
        return self.loaded_models.get(model_name)
    
    async def _can_load_model(self, model_size_bytes: int) -> bool:
        """Check if we can load a model of given size."""
        current_memory = psutil.Process().memory_info().rss
        total_loaded_size = sum(self.model_sizes.values())
        
        return (current_memory + total_loaded_size + model_size_bytes) <= self.max_memory_bytes
    
    async def _free_memory(self, required_bytes: int):
        """Free memory by unloading least recently used models."""
        # Simple LRU implementation
        if not self.loaded_models:
            return
        
        # Unload models until we have enough space
        freed_bytes = 0
        models_to_unload = []
        
        for model_name, size in self.model_sizes.items():
            if freed_bytes >= required_bytes:
                break
            
            models_to_unload.append(model_name)
            freed_bytes += size
        
        # Unload models
        for model_name in models_to_unload:
            await self.unload_model(model_name)
        
        logger.info(f"Freed {freed_bytes / 1024 / 1024:.1f}MB of memory")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        current_memory = psutil.Process().memory_info().rss
        total_loaded_size = sum(self.model_sizes.values())
        
        return {
            "current_memory_mb": current_memory / 1024 / 1024,
            "loaded_models_size_mb": total_loaded_size / 1024 / 1024,
            "max_memory_gb": self.max_memory_gb,
            "available_memory_mb": (self.max_memory_bytes - current_memory - total_loaded_size) / 1024 / 1024,
            "loaded_models": list(self.loaded_models.keys()),
            "model_count": len(self.loaded_models)
        }

# ============================================================================
# EXAMPLE 5: BACKGROUND TASK PROCESSING
# ============================================================================

class VideoBackgroundProcessor:
    """Background processor for video-related tasks."""
    
    def __init__(self, max_workers: int = 4):
        
    """__init__ function."""
self.max_workers = max_workers
        self.task_queue = asyncio.Queue()
        self.workers = []
        self.running = False
        self.task_stats = {}
    
    async def start(self) -> Any:
        """Start background processor."""
        self.running = True
        self.workers = [
            asyncio.create_task(self._worker(f"video-worker-{i}"))
            for i in range(self.max_workers)
        ]
        logger.info(f"Started {self.max_workers} video background workers")
    
    async def stop(self) -> Any:
        """Stop background processor."""
        self.running = False
        
        # Wait for all workers to complete
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)
        
        logger.info("Video background processor stopped")
    
    async def add_task(self, task_type: str, task_func: Callable, *args, **kwargs):
        """Add task to processing queue."""
        task_id = f"{task_type}_{int(time.time() * 1000)}"
        await self.task_queue.put({
            "id": task_id,
            "type": task_type,
            "func": task_func,
            "args": args,
            "kwargs": kwargs,
            "created_at": time.time()
        })
        
        logger.info(f"Added background task: {task_id} ({task_type})")
    
    async def _worker(self, worker_name: str):
        """Background worker function."""
        logger.info(f"Video worker {worker_name} started")
        
        while self.running:
            try:
                # Get task from queue with timeout
                task_data = await asyncio.wait_for(
                    self.task_queue.get(), timeout=1.0
                )
                
                # Execute task
                start_time = time.time()
                try:
                    result = await task_data["func"](*task_data["args"], **task_data["kwargs"])
                    
                    # Update stats
                    duration = time.time() - start_time
                    task_type = task_data["type"]
                    if task_type not in self.task_stats:
                        self.task_stats[task_type] = {"completed": 0, "failed": 0, "total_time": 0}
                    
                    self.task_stats[task_type]["completed"] += 1
                    self.task_stats[task_type]["total_time"] += duration
                    
                    logger.info(f"Worker {worker_name} completed task {task_data['id']} in {duration:.2f}s")
                    
                except Exception as e:
                    # Update stats
                    task_type = task_data["type"]
                    if task_type not in self.task_stats:
                        self.task_stats[task_type] = {"completed": 0, "failed": 0, "total_time": 0}
                    
                    self.task_stats[task_type]["failed"] += 1
                    logger.error(f"Worker {worker_name} task {task_data['id']} failed: {e}")
                
                # Mark task as done
                self.task_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Video worker {worker_name} error: {e}")
        
        logger.info(f"Video worker {worker_name} stopped")
    
    def get_task_stats(self) -> Dict[str, Any]:
        """Get task processing statistics."""
        stats = {}
        for task_type, data in self.task_stats.items():
            total_tasks = data["completed"] + data["failed"]
            avg_time = data["total_time"] / data["completed"] if data["completed"] > 0 else 0
            
            stats[task_type] = {
                "total_tasks": total_tasks,
                "completed": data["completed"],
                "failed": data["failed"],
                "success_rate": data["completed"] / total_tasks if total_tasks > 0 else 0,
                "avg_time": avg_time,
                "total_time": data["total_time"]
            }
        
        return stats

# ============================================================================
# EXAMPLE 6: INTEGRATED PERFORMANCE SYSTEM
# ============================================================================

class AIVideoPerformanceSystem:
    """Integrated performance system for AI Video operations."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        
    """__init__ function."""
# Initialize components
        self.video_processor = AsyncVideoProcessor()
        self.model_manager = AIVideoModelManager(None)  # Would use cache
        self.db_optimizer = VideoDatabaseOptimizer(None)  # Would use cache
        self.memory_manager = ModelMemoryManager()
        self.background_processor = VideoBackgroundProcessor()
        
        # Register model loaders
        self._register_model_loaders()
        
        # Start background processor
        asyncio.create_task(self.background_processor.start())
    
    def _register_model_loaders(self) -> Any:
        """Register model loaders."""
        async def load_stable_diffusion():
            
    """load_stable_diffusion function."""
await asyncio.sleep(2)  # Simulate loading
            return {"model": "stable-diffusion", "version": "1.5"}
        
        async def load_text_to_video():
            
    """load_text_to_video function."""
await asyncio.sleep(3)  # Simulate loading
            return {"model": "text-to-video", "version": "2.0"}
        
        self.model_manager.register_model("stable-diffusion", load_stable_diffusion)
        self.model_manager.register_model("text-to-video", load_text_to_video)
    
    async def generate_video(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate video with full optimization."""
        start_time = time.time()
        
        try:
            # 1. Load model (lazy loading)
            model = await self.model_manager.get_model(request["model_name"])
            
            # 2. Process video (async processing)
            result = await self.video_processor._process_video_async(request)
            
            # 3. Add to background tasks for cleanup
            await self.background_processor.add_task(
                "cleanup", self._cleanup_video_files, request["video_id"]
            )
            
            processing_time = time.time() - start_time
            
            return {
                **result,
                "total_processing_time": processing_time,
                "optimizations_used": ["async_processing", "lazy_loading", "background_tasks"]
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Video generation failed: {e}")
            
            return {
                "video_id": request.get("video_id"),
                "status": "failed",
                "error": str(e),
                "total_processing_time": processing_time
            }
    
    async def batch_generate_videos(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate multiple videos with batch optimization."""
        # Preload models
        model_names = list(set(req["model_name"] for req in requests))
        await self.model_manager.preload_models(model_names)
        
        # Process videos in batch
        results = await self.video_processor.process_video_batch(requests)
        
        return results
    
    async def get_video_with_optimization(self, session: AsyncSession, video_id: str) -> Optional[Dict[str, Any]]:
        """Get video with database optimization."""
        return await self.db_optimizer.get_video_by_id(session, video_id)
    
    async def _cleanup_video_files(self, video_id: str):
        """Cleanup temporary video files."""
        # Simulate cleanup
        await asyncio.sleep(1)
        logger.info(f"Cleaned up temporary files for video: {video_id}")
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        return {
            "memory": self.memory_manager.get_memory_stats(),
            "background_tasks": self.background_processor.get_task_stats(),
            "loaded_models": self.model_manager.get_loaded_models(),
            "queue_size": self.background_processor.task_queue.qsize()
        }
    
    async def cleanup(self) -> Any:
        """Cleanup system resources."""
        await self.background_processor.stop()

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

async def example_video_processing():
    """Example of optimized video processing."""
    
    system = AIVideoPerformanceSystem()
    
    # Single video generation
    request = {
        "video_id": "video_123",
        "prompt": "A beautiful sunset over mountains",
        "model_name": "stable-diffusion",
        "width": 1920,
        "height": 1080
    }
    
    result = await system.generate_video(request)
    logger.info(f"Video generation result: {result}")
    
    # Batch video generation
    requests = [
        {
            "video_id": f"video_{i}",
            "prompt": f"Video {i} prompt",
            "model_name": "stable-diffusion",
            "width": 1280,
            "height": 720
        }
        for i in range(5)
    ]
    
    batch_results = await system.batch_generate_videos(requests)
    logger.info(f"Batch generation completed: {len(batch_results)} videos")
    
    # Get system stats
    stats = await system.get_system_stats()
    logger.info(f"System stats: {json.dumps(stats, indent=2)}")
    
    # Cleanup
    await system.cleanup()

async def example_memory_management():
    """Example of memory management for large models."""
    
    memory_manager = ModelMemoryManager(max_memory_gb=4.0)
    
    # Load models
    models = [
        ("model_1", {"data": "model1"}, 1024),  # 1GB
        ("model_2", {"data": "model2"}, 1024),  # 1GB
        ("model_3", {"data": "model3"}, 1024),  # 1GB
        ("model_4", {"data": "model4"}, 1024),  # 1GB
    ]
    
    for model_name, model_data, size_mb in models:
        success = await memory_manager.load_model(model_name, model_data, size_mb)
        logger.info(f"Loaded {model_name}: {success}")
    
    # Get memory stats
    stats = memory_manager.get_memory_stats()
    logger.info(f"Memory stats: {stats}")

if __name__ == "__main__":
    # Run examples
    async def main():
        
    """main function."""
await example_video_processing()
        await example_memory_management()
    
    asyncio.run(main()) 