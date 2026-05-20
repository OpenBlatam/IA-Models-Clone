"""
Batch Processor Service
Processes multiple video generation requests in batch
"""

import asyncio
from typing import List, Dict, Any, Optional
from uuid import UUID
import logging
from datetime import datetime

from ..core.models import VideoGenerationRequest, VideoGenerationResponse, VideoStatus
from .video_orchestrator import VideoOrchestrator

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Processes multiple video generation requests"""
    
    def __init__(self, max_concurrent: int = 3):
        self.max_concurrent = max_concurrent
        self.orchestrator = VideoOrchestrator()
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_batch(
        self,
        requests: List[VideoGenerationRequest],
        webhook_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process multiple video generation requests
        
        Args:
            requests: List of video generation requests
            webhook_url: Optional webhook URL for batch completion
            
        Returns:
            Batch processing result with job IDs and status
        """
        logger.info(f"Starting batch processing for {len(requests)} videos")
        
        batch_id = UUID(int=datetime.utcnow().timestamp() * 1000000)
        results = []
        
        # Start all jobs
        for request in requests:
            try:
                response = await self.orchestrator.start_generation(request)
                results.append({
                    "video_id": str(response.video_id),
                    "status": response.status.value,
                    "request": request.dict(),
                })
                
                # Register webhook if provided
                if webhook_url:
                    self.orchestrator.webhook_service.register_webhook(
                        response.video_id,
                        webhook_url
                    )
                
                # Process in background with concurrency limit
                asyncio.create_task(
                    self._process_with_semaphore(response.video_id, request)
                )
            except Exception as e:
                logger.error(f"Failed to start video generation in batch: {str(e)}")
                results.append({
                    "error": str(e),
                    "status": "failed",
                })
        
        return {
            "batch_id": str(batch_id),
            "total": len(requests),
            "started": len([r for r in results if "video_id" in r]),
            "failed": len([r for r in results if "error" in r]),
            "jobs": results,
        }
    
    async def _process_with_semaphore(
        self,
        video_id: UUID,
        request: VideoGenerationRequest
    ):
        """Process video generation with semaphore for concurrency control"""
        async with self.semaphore:
            try:
                await self.orchestrator.process_generation(video_id, request)
            except Exception as e:
                logger.error(f"Batch processing failed for {video_id}: {str(e)}")
    
    async def get_batch_status(
        self,
        video_ids: List[UUID]
    ) -> Dict[str, Any]:
        """Get status of multiple video generation jobs"""
        statuses = []
        
        for video_id in video_ids:
            job = self.orchestrator.jobs.get(video_id)
            if job:
                statuses.append({
                    "video_id": str(video_id),
                    "status": job.status.value,
                    "progress": job.progress.progress,
                    "current_step": job.progress.current_step,
                })
            else:
                statuses.append({
                    "video_id": str(video_id),
                    "status": "not_found",
                })
        
        completed = len([s for s in statuses if s["status"] == "completed"])
        failed = len([s for s in statuses if s["status"] == "failed"])
        processing = len([s for s in statuses if s["status"] in ["processing", "generating_images", "generating_audio", "adding_subtitles", "compositing"]])
        
        return {
            "total": len(video_ids),
            "completed": completed,
            "failed": failed,
            "processing": processing,
            "pending": len(video_ids) - completed - failed - processing,
            "jobs": statuses,
        }

