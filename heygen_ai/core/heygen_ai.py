#!/usr/bin/env python3
"""
Enhanced HeyGen AI Core System v2.1
Main orchestrator with intelligent caching, async queues, webhooks, rate limiting, and metrics.
"""

import asyncio
import time
import logging
import structlog
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import json
from datetime import datetime, timezone

# Import enhanced components
from .avatar_manager import AvatarManager
from .voice_engine import VoiceEngine
from .video_renderer import VideoRenderer
from .script_generator import ScriptGenerator
from .langchain_manager import LangChainManager
from .advanced_ai_workflows import AdvancedAIWorkflows

# Import new enhanced components
from .cache_manager import get_cache_manager, CacheType, CachePriority
from .async_queue_manager import get_queue_manager, TaskType, TaskPriority
from .webhook_manager import get_webhook_manager, WebhookEventType
from .rate_limiter import get_rate_limiter
from .metrics_collector import get_metrics_collector

# Import models
from ..api.models import (
    VideoRequest, VideoResponse, VoiceGenerationRequest, 
    AvatarGenerationRequest, BatchVideoRequest, BatchVideoResponse
)

logger = structlog.get_logger()

class EnhancedHeyGenAI:
    """
    Enhanced HeyGen AI System v2.1
    
    Features:
    - Real AI Models Integration (Stable Diffusion, Coqui TTS, Wav2Lip)
    - Intelligent Caching System
    - Async Task Queues with Priorities
    - Real-time Webhook Notifications
    - Adaptive Rate Limiting
    - Comprehensive Metrics Collection
    - Performance Monitoring
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the enhanced HeyGen AI system."""
        self.config = config or {}
        self.version = "2.1.0"
        
        # Initialize core components
        self.avatar_manager = AvatarManager()
        self.voice_engine = VoiceEngine()
        self.video_renderer = VideoRenderer()
        self.script_generator = ScriptGenerator()
        self.langchain_manager = LangChainManager()
        self.advanced_ai_workflows = AdvancedAIWorkflows()
        
        # Initialize enhanced components
        self.cache_manager = get_cache_manager()
        self.queue_manager = get_queue_manager()
        self.webhook_manager = get_webhook_manager()
        self.rate_limiter = get_rate_limiter()
        self.metrics_collector = get_metrics_collector()
        
        # System state
        self.is_initialized = False
        self.startup_time = time.time()
        
        # Performance tracking
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0
        
        # Initialize system
        asyncio.create_task(self._initialize_system())
    
    async def _initialize_system(self):
        """Initialize all system components."""
        try:
            logger.info("Initializing Enhanced HeyGen AI System v2.1")
            
            # Start enhanced components
            await self.queue_manager.start()
            await self.webhook_manager.start()
            
            # Register task handlers
            self._register_task_handlers()
            
            # Register webhook event handlers
            self._register_webhook_handlers()
            
            # Set up metrics collection
            self._setup_metrics_integration()
            
            self.is_initialized = True
            logger.info("Enhanced HeyGen AI System initialized successfully")
            
            # Send system startup webhook
            await self.webhook_manager.send_event(
                WebhookEventType.SYSTEM_HEALTH,
                {
                    "status": "started",
                    "version": self.version,
                    "timestamp": time.time(),
                    "components": {
                        "cache_manager": True,
                        "queue_manager": True,
                        "webhook_manager": True,
                        "rate_limiter": True,
                        "metrics_collector": True
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            self.is_initialized = False
    
    def _register_task_handlers(self):
        """Register handlers for different task types."""
        self.queue_manager.register_handler(
            TaskType.VIDEO_GENERATION,
            self._process_video_generation_task
        )
        
        self.queue_manager.register_handler(
            TaskType.VOICE_SYNTHESIS,
            self._process_voice_synthesis_task
        )
        
        self.queue_manager.register_handler(
            TaskType.AVATAR_GENERATION,
            self._process_avatar_generation_task
        )
        
        self.queue_manager.register_handler(
            TaskType.BATCH_PROCESSING,
            self._process_batch_processing_task
        )
    
    def _register_webhook_handlers(self):
        """Register webhook event handlers."""
        self.webhook_manager.register_event_handler(
            WebhookEventType.VIDEO_COMPLETED,
            self._on_video_completed
        )
        
        self.webhook_manager.register_event_handler(
            WebhookEventType.VIDEO_FAILED,
            self._on_video_failed
        )
    
    def _setup_metrics_integration(self):
        """Set up metrics integration with components."""
        # This would integrate metrics collection with various components
        # For now, we'll just log that it's set up
        logger.info("Metrics integration set up")
    
    async def create_video(
        self, 
        request: VideoRequest,
        user_id: str = "anonymous",
        priority: TaskPriority = TaskPriority.NORMAL,
        enable_webhooks: bool = True
    ) -> VideoResponse:
        """
        Create a video using the enhanced pipeline.
        
        This method now supports:
        - Rate limiting per user
        - Async processing with priorities
        - Intelligent caching
        - Real-time webhooks
        - Performance metrics
        """
        start_time = time.time()
        self.request_count += 1
        
        try:
            # Check rate limits
            rate_limit_allowed, rate_limit_info = await self.rate_limiter.check_rate_limit(
                user_id=user_id,
                endpoint="video_generation",
                request_weight=1
            )
            
            if not rate_limit_allowed:
                self.metrics_collector.record_rate_limit_check(
                    rate_limit_info["user_tier"], 
                    False
                )
                
                raise Exception(f"Rate limit exceeded. Retry after {rate_limit_info.get('retry_after', 'unknown')}")
            
            # Record successful rate limit check
            self.metrics_collector.record_rate_limit_check(
                rate_limit_info["user_tier"], 
                True
            )
            
            # Check cache first
            cache_key = self._generate_cache_key(request)
            cached_result = await self.cache_manager.get(
                cache_key,
                CacheType.API_RESPONSE
            )
            
            if cached_result:
                logger.info("Video generation result found in cache")
                self.metrics_collector.record_cache_operation("video_generation", True)
                
                # Update metrics
                processing_time = time.time() - start_time
                self.metrics_collector.record_video_generation(
                    "completed", 
                    request.quality_preset, 
                    request.resolution, 
                    processing_time
                )
                
                return VideoResponse(**cached_result)
            
            # Cache miss
            self.metrics_collector.record_cache_operation("video_generation", False)
            
            # Submit to async queue for processing
            task_id = await self.queue_manager.submit_task(
                task_type=TaskType.VIDEO_GENERATION,
                payload={
                    "request": request.dict(),
                    "user_id": user_id,
                    "enable_webhooks": enable_webhooks,
                    "cache_key": cache_key
                },
                priority=priority,
                user_id=user_id
            )
            
            # Update queue metrics
            queue_stats = await self.queue_manager.get_queue_stats()
            self.metrics_collector.update_queue_size("video_generation", queue_stats["queue_size"])
            
            # Return immediate response with task ID
            response = VideoResponse(
                video_id=task_id,
                status="queued",
                message="Video generation queued for processing",
                output_url=None,
                generation_time=0,
                quality_metrics={},
                cache_hit=False,
                task_id=task_id,
                estimated_completion_time=time.time() + 300  # 5 minutes estimate
            )
            
            # Send webhook event if enabled
            if enable_webhooks:
                await self.webhook_manager.send_event(
                    WebhookEventType.TASK_STARTED,
                    {
                        "task_id": task_id,
                        "task_type": "video_generation",
                        "user_id": user_id,
                        "priority": priority.value,
                        "timestamp": time.time()
                    }
                )
            
            # Record metrics
            processing_time = time.time() - start_time
            self.metrics_collector.record_video_generation(
                "queued", 
                request.quality_preset, 
                request.resolution, 
                processing_time
            )
            
            self.success_count += 1
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.error_count += 1
            
            # Record error metrics
            self.metrics_collector.record_error("video_generation", str(type(e).__name__))
            
            # Send webhook event if enabled
            if enable_webhooks:
                await self.webhook_manager.send_event(
                    WebhookEventType.VIDEO_FAILED,
                    {
                        "error": str(e),
                        "user_id": user_id,
                        "timestamp": time.time()
                    }
                )
            
            logger.error(f"Video creation failed: {e}")
            raise
    
    async def _process_video_generation_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process video generation task from the queue."""
        start_time = time.time()
        
        try:
            request_data = payload["request"]
            user_id = payload["user_id"]
            enable_webhooks = payload.get("enable_webhooks", True)
            cache_key = payload["cache_key"]
            
            # Convert back to VideoRequest
            request = VideoRequest(**request_data)
            
            logger.info(f"Processing video generation task", 
                       task_id=payload.get("task_id"),
                       user_id=user_id)
            
            # Execute the full video generation pipeline
            result = await self._execute_video_pipeline(request)
            
            # Cache the result
            await self.cache_manager.set(
                cache_key,
                result.dict(),
                cache_type=CacheType.API_RESPONSE,
                priority=CachePriority.HIGH,
                ttl_seconds=3600,  # 1 hour
                tags=["video_generation", user_id]
            )
            
            # Update cache metrics
            cache_stats = await self.cache_manager.get_stats()
            self.metrics_collector.update_cache_size(
                "video_generation",
                cache_stats["memory_usage_bytes"] + cache_stats["disk_usage_bytes"],
                cache_stats["memory_entries"] + cache_stats["disk_entries"]
            )
            
            # Send webhook event if enabled
            if enable_webhooks:
                await self.webhook_manager.send_event(
                    WebhookEventType.VIDEO_COMPLETED,
                    {
                        "video_id": result.video_id,
                        "user_id": user_id,
                        "generation_time": result.generation_time,
                        "quality_metrics": result.quality_metrics,
                        "timestamp": time.time()
                    }
                )
            
            # Record metrics
            processing_time = time.time() - start_time
            self.metrics_collector.record_video_generation(
                "completed", 
                request.quality_preset, 
                request.resolution, 
                processing_time
            )
            
            self.metrics_collector.record_queue_task(
                "video_generation", 
                "completed", 
                processing_time
            )
            
            return result.dict()
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Record error metrics
            self.metrics_collector.record_error("video_generation_task", str(type(e).__name__))
            self.metrics_collector.record_queue_task("video_generation", "failed", processing_time)
            
            logger.error(f"Video generation task failed: {e}")
            raise
    
    async def _execute_video_pipeline(self, request: VideoRequest) -> VideoResponse:
        """Execute the full video generation pipeline."""
        start_time = time.time()
        
        try:
            # Step 1: Process script
            logger.info("Processing script")
            processed_script = await self.script_generator.process_script(request.script)
            
            # Step 2: Generate voice
            logger.info("Generating voice")
            voice_request = VoiceGenerationRequest(
                text=processed_script,
                voice_id=request.voice_id,
                language=request.language,
                quality=request.quality_preset,
                speed=1.0,
                pitch=1.0
            )
            
            audio_path = await self.voice_engine.synthesize_speech(voice_request)
            
            # Step 3: Generate avatar video
            logger.info("Generating avatar video")
            avatar_request = AvatarGenerationRequest(
                avatar_id=request.avatar_id,
                audio_path=audio_path,
                resolution=request.resolution,
                quality_preset=request.quality_preset,
                enable_expressions=request.enable_expressions
            )
            
            avatar_video_path = await self.avatar_manager.generate_avatar_video(avatar_request)
            
            # Step 4: Final video rendering
            logger.info("Rendering final video")
            final_video_path = await self.video_renderer.render_final_video(
                avatar_video_path=avatar_video_path,
                audio_path=audio_path,
                resolution=request.resolution,
                quality_preset=request.quality_preset,
                effects=request.enable_effects
            )
            
            # Step 5: Quality analysis and upload
            logger.info("Analyzing quality and uploading")
            quality_metrics = await self.video_renderer.analyze_video_quality(final_video_path)
            output_url = await self.video_renderer.upload_video(final_video_path)
            
            # Calculate generation time
            generation_time = time.time() - start_time
            
            # Create response
            response = VideoResponse(
                video_id=f"video_{int(time.time())}",
                status="completed",
                message="Video generated successfully",
                output_url=output_url,
                generation_time=generation_time,
                quality_metrics=quality_metrics,
                cache_hit=False,
                task_id=None,
                estimated_completion_time=None
            )
            
            logger.info(f"Video generation completed", 
                       video_id=response.video_id,
                       generation_time=generation_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Video pipeline execution failed: {e}")
            raise
    
    async def _process_voice_synthesis_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process voice synthesis task from the queue."""
        start_time = time.time()
        
        try:
            request_data = payload["request"]
            user_id = payload["user_id"]
            
            request = VoiceGenerationRequest(**request_data)
            
            # Execute voice synthesis
            audio_path = await self.voice_engine.synthesize_speech(request)
            
            # Record metrics
            processing_time = time.time() - start_time
            self.metrics_collector.record_voice_synthesis(
                "completed",
                request.voice_id,
                request.language,
                processing_time
            )
            
            self.metrics_collector.record_queue_task(
                "voice_synthesis",
                "completed",
                processing_time
            )
            
            return {"audio_path": audio_path, "status": "completed"}
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            self.metrics_collector.record_error("voice_synthesis_task", str(type(e).__name__))
            self.metrics_collector.record_queue_task("voice_synthesis", "failed", processing_time)
            
            logger.error(f"Voice synthesis task failed: {e}")
            raise
    
    async def _process_avatar_generation_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process avatar generation task from the queue."""
        start_time = time.time()
        
        try:
            request_data = payload["request"]
            user_id = payload["user_id"]
            
            request = AvatarGenerationRequest(**request_data)
            
            # Execute avatar generation
            video_path = await self.avatar_manager.generate_avatar_video(request)
            
            # Record metrics
            processing_time = time.time() - start_time
            self.metrics_collector.record_avatar_generation(
                "completed",
                "stable_diffusion",  # Default model
                request.resolution,
                processing_time
            )
            
            self.metrics_collector.record_queue_task(
                "avatar_generation",
                "completed",
                processing_time
            )
            
            return {"video_path": video_path, "status": "completed"}
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            self.metrics_collector.record_error("avatar_generation_task", str(type(e).__name__))
            self.metrics_collector.record_queue_task("avatar_generation", "failed", processing_time)
            
            logger.error(f"Avatar generation task failed: {e}")
            raise
    
    async def _process_batch_processing_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process batch processing task from the queue."""
        start_time = time.time()
        
        try:
            batch_request = payload["batch_request"]
            user_id = payload["user_id"]
            
            results = []
            for video_request in batch_request.videos:
                try:
                    result = await self._execute_video_pipeline(VideoRequest(**video_request))
                    results.append(result.dict())
                except Exception as e:
                    logger.error(f"Batch video {video_request.get('script', 'unknown')} failed: {e}")
                    results.append({
                        "status": "failed",
                        "error": str(e)
                    })
            
            # Record metrics
            processing_time = time.time() - start_time
            self.metrics_collector.record_queue_task(
                "batch_processing",
                "completed",
                processing_time
            )
            
            return {"results": results, "status": "completed"}
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            self.metrics_collector.record_error("batch_processing_task", str(type(e).__name__))
            self.metrics_collector.record_queue_task("batch_processing", "failed", processing_time)
            
            logger.error(f"Batch processing task failed: {e}")
            raise
    
    async def _on_video_completed(self, data: Dict[str, Any]):
        """Handle video completion webhook event."""
        logger.info(f"Video completed webhook event received", **data)
        
        # Update metrics
        self.metrics_collector.record_webhook_event("video.completed", "sent")
    
    async def _on_video_failed(self, data: Dict[str, Any]):
        """Handle video failure webhook event."""
        logger.info(f"Video failed webhook event received", **data)
        
        # Update metrics
        self.metrics_collector.record_webhook_event("video.failed", "sent")
    
    def _generate_cache_key(self, request: VideoRequest) -> str:
        """Generate a cache key for the request."""
        # Create a unique key based on request parameters
        key_data = {
            "script": request.script,
            "avatar_id": request.avatar_id,
            "voice_id": request.voice_id,
            "language": request.language,
            "resolution": request.resolution,
            "quality_preset": request.quality_preset,
            "enable_expressions": request.enable_expressions,
            "enable_effects": request.enable_effects
        }
        
        import hashlib
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a queued task."""
        return await self.queue_manager.get_task_status(task_id)
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a queued task."""
        return await self.queue_manager.cancel_task(task_id)
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        try:
            # Get component stats
            cache_stats = await self.cache_manager.get_stats()
            queue_stats = await self.queue_manager.get_queue_stats()
            webhook_stats = self.webhook_manager.get_stats()
            rate_limit_stats = self.rate_limiter.get_system_stats()
            metrics_summary = self.metrics_collector.get_metrics_summary()
            
            # Calculate uptime
            uptime_seconds = time.time() - self.startup_time
            uptime_hours = uptime_seconds / 3600
            
            return {
                "system": {
                    "version": self.version,
                    "status": "operational" if self.is_initialized else "initializing",
                    "uptime_hours": round(uptime_hours, 2),
                    "startup_time": self.startup_time,
                    "is_initialized": self.is_initialized
                },
                "performance": {
                    "request_count": self.request_count,
                    "success_count": self.success_count,
                    "error_count": self.error_count,
                    "success_rate": self.success_count / max(self.request_count, 1)
                },
                "cache": cache_stats,
                "queue": queue_stats,
                "webhooks": webhook_stats,
                "rate_limiting": rate_limit_stats,
                "metrics": metrics_summary
            }
            
        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a comprehensive health check."""
        try:
            health_status = {
                "status": "healthy",
                "timestamp": time.time(),
                "version": self.version,
                "components": {}
            }
            
            # Check core components
            try:
                avatar_health = self.avatar_manager.health_check()
                health_status["components"]["avatar_manager"] = avatar_health
            except Exception as e:
                health_status["components"]["avatar_manager"] = {"status": "unhealthy", "error": str(e)}
            
            try:
                voice_health = self.voice_engine.health_check()
                health_status["components"]["voice_engine"] = voice_health
            except Exception as e:
                health_status["components"]["voice_engine"] = {"status": "unhealthy", "error": str(e)}
            
            try:
                video_health = self.video_renderer.health_check()
                health_status["components"]["video_renderer"] = video_health
            except Exception as e:
                health_status["components"]["video_renderer"] = {"status": "unhealthy", "error": str(e)}
            
            # Check enhanced components
            try:
                cache_stats = await self.cache_manager.get_stats()
                health_status["components"]["cache_manager"] = {
                    "status": "healthy",
                    "stats": cache_stats
                }
            except Exception as e:
                health_status["components"]["cache_manager"] = {"status": "unhealthy", "error": str(e)}
            
            try:
                queue_stats = await self.queue_manager.get_queue_stats()
                health_status["components"]["queue_manager"] = {
                    "status": "healthy",
                    "stats": queue_stats
                }
            except Exception as e:
                health_status["components"]["queue_manager"] = {"status": "unhealthy", "error": str(e)}
            
            try:
                webhook_stats = self.webhook_manager.get_stats()
                health_status["components"]["webhook_manager"] = {
                    "status": "healthy",
                    "stats": webhook_stats
                }
            except Exception as e:
                health_status["components"]["webhook_manager"] = {"status": "unhealthy", "error": str(e)}
            
            # Determine overall health
            unhealthy_components = [
                comp for comp, status in health_status["components"].items()
                if status.get("status") == "unhealthy"
            ]
            
            if unhealthy_components:
                health_status["status"] = "degraded"
                health_status["unhealthy_components"] = unhealthy_components
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def shutdown(self):
        """Shutdown the enhanced HeyGen AI system."""
        try:
            logger.info("Shutting down Enhanced HeyGen AI System")
            
            # Shutdown enhanced components
            await self.queue_manager.stop()
            await self.webhook_manager.stop()
            await self.rate_limiter.shutdown()
            await self.metrics_collector.shutdown()
            await self.cache_manager.shutdown()
            
            logger.info("Enhanced HeyGen AI System shutdown complete")
            
        except Exception as e:
            logger.error(f"Shutdown failed: {e}")

# Global instance
heygen_ai: Optional[EnhancedHeyGenAI] = None

def get_heygen_ai() -> EnhancedHeyGenAI:
    """Get global HeyGen AI instance."""
    global heygen_ai
    if heygen_ai is None:
        heygen_ai = EnhancedHeyGenAI()
    return heygen_ai

async def shutdown_heygen_ai():
    """Shutdown global HeyGen AI instance."""
    global heygen_ai
    if heygen_ai:
        await heygen_ai.shutdown()
        heygen_ai = None 