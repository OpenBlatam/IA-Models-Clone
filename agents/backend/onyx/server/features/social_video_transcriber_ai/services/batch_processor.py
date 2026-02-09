"""
Batch Processor Service
Handles batch processing of multiple video URLs
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable
from uuid import UUID, uuid4
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from ..config.settings import get_settings
from ..core.models import (
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionStatus,
    SupportedPlatform,
)
from .video_downloader import get_video_downloader
from .transcription_service import get_transcription_service
from .ai_analyzer import get_ai_analyzer
from .cache_service import get_cache_service

logger = logging.getLogger(__name__)


class BatchStatus(str, Enum):
    """Status of batch processing"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    PARTIALLY_COMPLETED = "partially_completed"
    FAILED = "failed"


@dataclass
class BatchJob:
    """Batch job definition"""
    batch_id: UUID = field(default_factory=uuid4)
    urls: List[str] = field(default_factory=list)
    status: BatchStatus = BatchStatus.PENDING
    include_timestamps: bool = True
    include_analysis: bool = True
    language: Optional[str] = None
    webhook_url: Optional[str] = None
    
    # Results
    results: Dict[str, TranscriptionResponse] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
    
    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Progress
    total_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": str(self.batch_id),
            "status": self.status.value,
            "total_jobs": self.total_jobs,
            "completed_jobs": self.completed_jobs,
            "failed_jobs": self.failed_jobs,
            "progress_percent": round(
                (self.completed_jobs + self.failed_jobs) / max(self.total_jobs, 1) * 100, 1
            ),
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "urls": self.urls,
            "errors": self.errors,
        }


class BatchProcessor:
    """Service for batch processing videos"""
    
    def __init__(self, max_concurrent: int = 3):
        self.settings = get_settings()
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._batches: Dict[UUID, BatchJob] = {}
    
    async def create_batch(
        self,
        urls: List[str],
        include_timestamps: bool = True,
        include_analysis: bool = True,
        language: Optional[str] = None,
        webhook_url: Optional[str] = None,
    ) -> BatchJob:
        """
        Create a new batch job
        
        Args:
            urls: List of video URLs to process
            include_timestamps: Include timestamps in transcriptions
            include_analysis: Include AI analysis
            language: Language code (None for auto-detect)
            webhook_url: URL for completion webhook
            
        Returns:
            BatchJob object
        """
        batch = BatchJob(
            urls=urls,
            include_timestamps=include_timestamps,
            include_analysis=include_analysis,
            language=language,
            webhook_url=webhook_url,
            total_jobs=len(urls),
        )
        
        self._batches[batch.batch_id] = batch
        logger.info(f"Created batch {batch.batch_id} with {len(urls)} URLs")
        
        return batch
    
    async def process_batch(
        self,
        batch_id: UUID,
        progress_callback: Optional[Callable[[BatchJob], None]] = None,
    ) -> BatchJob:
        """
        Process a batch job
        
        Args:
            batch_id: Batch ID to process
            progress_callback: Optional callback for progress updates
            
        Returns:
            Completed BatchJob
        """
        batch = self._batches.get(batch_id)
        if not batch:
            raise ValueError(f"Batch {batch_id} not found")
        
        batch.status = BatchStatus.PROCESSING
        batch.started_at = datetime.utcnow()
        
        logger.info(f"Starting batch processing: {batch_id}")
        
        # Process URLs concurrently with semaphore
        tasks = [
            self._process_url(batch, url, i, progress_callback)
            for i, url in enumerate(batch.urls)
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Determine final status
        batch.completed_at = datetime.utcnow()
        
        if batch.failed_jobs == batch.total_jobs:
            batch.status = BatchStatus.FAILED
        elif batch.failed_jobs > 0:
            batch.status = BatchStatus.PARTIALLY_COMPLETED
        else:
            batch.status = BatchStatus.COMPLETED
        
        logger.info(
            f"Batch {batch_id} completed: "
            f"{batch.completed_jobs}/{batch.total_jobs} successful, "
            f"{batch.failed_jobs} failed"
        )
        
        # Send webhook if configured
        if batch.webhook_url:
            await self._send_webhook(batch)
        
        return batch
    
    async def _process_url(
        self,
        batch: BatchJob,
        url: str,
        index: int,
        progress_callback: Optional[Callable[[BatchJob], None]] = None,
    ):
        """Process a single URL within the batch"""
        async with self._semaphore:
            logger.debug(f"Processing URL {index + 1}/{batch.total_jobs}: {url}")
            
            try:
                # Check cache first
                cache = get_cache_service()
                cached = await cache.get(url, {
                    "timestamps": batch.include_timestamps,
                    "analysis": batch.include_analysis,
                })
                
                if cached:
                    logger.debug(f"Using cached result for {url}")
                    batch.results[url] = TranscriptionResponse(**cached)
                    batch.completed_jobs += 1
                    if progress_callback:
                        progress_callback(batch)
                    return
                
                # Process video
                downloader = get_video_downloader()
                transcriber = get_transcription_service()
                analyzer = get_ai_analyzer()
                
                # Download
                job_id = uuid4()
                audio_path, video_info = await downloader.download_video(
                    url=url,
                    job_id=job_id,
                    extract_audio=True,
                )
                
                # Transcribe
                result = await transcriber.transcribe(
                    audio_path=audio_path,
                    language=batch.language,
                    include_timestamps=batch.include_timestamps,
                )
                
                # Create response
                response = TranscriptionResponse(
                    job_id=job_id,
                    status=TranscriptionStatus.COMPLETED,
                    platform_detected=downloader.detect_platform(url),
                    video_title=video_info.get('title'),
                    video_duration=video_info.get('duration'),
                    video_author=video_info.get('author'),
                    full_text=result['full_text'],
                    full_text_with_timestamps=result['full_text_with_timestamps'],
                    segments=result['segments'],
                    completed_at=datetime.utcnow(),
                )
                
                # Analyze if requested
                if batch.include_analysis and response.full_text:
                    analysis = await analyzer.analyze_content(response.full_text)
                    response.analysis = analysis
                
                # Cache result
                await cache.set(url, response.model_dump(), {
                    "timestamps": batch.include_timestamps,
                    "analysis": batch.include_analysis,
                })
                
                batch.results[url] = response
                batch.completed_jobs += 1
                
                # Cleanup
                await downloader.cleanup(job_id)
                
            except Exception as e:
                logger.error(f"Failed to process {url}: {e}")
                batch.errors[url] = str(e)
                batch.failed_jobs += 1
            
            if progress_callback:
                progress_callback(batch)
    
    async def _send_webhook(self, batch: BatchJob):
        """Send webhook notification for batch completion"""
        if not batch.webhook_url:
            return
        
        import httpx
        
        payload = {
            "event": "batch_completed",
            "batch_id": str(batch.batch_id),
            "status": batch.status.value,
            "total_jobs": batch.total_jobs,
            "completed_jobs": batch.completed_jobs,
            "failed_jobs": batch.failed_jobs,
            "completed_at": batch.completed_at.isoformat() if batch.completed_at else None,
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                await client.post(batch.webhook_url, json=payload)
            logger.info(f"Webhook sent for batch {batch.batch_id}")
        except Exception as e:
            logger.error(f"Failed to send webhook: {e}")
    
    def get_batch(self, batch_id: UUID) -> Optional[BatchJob]:
        """Get batch job by ID"""
        return self._batches.get(batch_id)
    
    def list_batches(self, limit: int = 50) -> List[BatchJob]:
        """List all batch jobs"""
        batches = list(self._batches.values())
        batches.sort(key=lambda b: b.created_at, reverse=True)
        return batches[:limit]


_batch_processor: Optional[BatchProcessor] = None


def get_batch_processor() -> BatchProcessor:
    """Get batch processor singleton"""
    global _batch_processor
    if _batch_processor is None:
        _batch_processor = BatchProcessor()
    return _batch_processor












