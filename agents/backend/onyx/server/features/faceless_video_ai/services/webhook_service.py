"""
Webhook Service
Sends notifications when video generation completes
"""

import asyncio
from typing import Optional, Dict, Any, List
from uuid import UUID
import logging
import httpx
from datetime import datetime

logger = logging.getLogger(__name__)


class WebhookService:
    """Manages webhook notifications for video generation events"""
    
    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout
        self.webhook_urls: Dict[UUID, List[str]] = {}
    
    def register_webhook(self, video_id: UUID, webhook_url: str):
        """Register a webhook URL for a video generation job"""
        if video_id not in self.webhook_urls:
            self.webhook_urls[video_id] = []
        
        if webhook_url not in self.webhook_urls[video_id]:
            self.webhook_urls[video_id].append(webhook_url)
            logger.info(f"Registered webhook for video {video_id}: {webhook_url}")
    
    async def send_webhook(
        self,
        video_id: UUID,
        status: str,
        data: Dict[str, Any],
        webhook_url: Optional[str] = None
    ):
        """
        Send webhook notification
        
        Args:
            video_id: Video generation job ID
            status: Status (completed, failed, etc.)
            data: Additional data to send
            webhook_url: Specific webhook URL (optional)
        """
        urls = [webhook_url] if webhook_url else self.webhook_urls.get(video_id, [])
        
        if not urls:
            return
        
        payload = {
            "video_id": str(video_id),
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
            **data
        }
        
        # Send to all registered webhooks
        tasks = []
        for url in urls:
            task = self._send_single_webhook(url, payload)
            tasks.append(task)
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for url, result in zip(urls, results):
                if isinstance(result, Exception):
                    logger.error(f"Webhook failed for {url}: {str(result)}")
                else:
                    logger.debug(f"Webhook sent successfully to {url}")
    
    async def _send_single_webhook(self, url: str, payload: Dict[str, Any]) -> bool:
        """Send webhook to a single URL"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                return True
        except httpx.TimeoutException:
            logger.warning(f"Webhook timeout for {url}")
            return False
        except Exception as e:
            logger.error(f"Webhook error for {url}: {str(e)}")
            return False
    
    async def notify_completion(
        self,
        video_id: UUID,
        video_url: str,
        duration: float,
        file_size: int
    ):
        """Send completion notification"""
        await self.send_webhook(
            video_id=video_id,
            status="completed",
            data={
                "video_url": video_url,
                "duration": duration,
                "file_size": file_size,
            }
        )
    
    async def notify_failure(
        self,
        video_id: UUID,
        error: str
    ):
        """Send failure notification"""
        await self.send_webhook(
            video_id=video_id,
            status="failed",
            data={
                "error": error,
            }
        )
    
    def unregister_webhook(self, video_id: UUID, webhook_url: Optional[str] = None):
        """Unregister webhook(s) for a video"""
        if video_id not in self.webhook_urls:
            return
        
        if webhook_url:
            if webhook_url in self.webhook_urls[video_id]:
                self.webhook_urls[video_id].remove(webhook_url)
        else:
            del self.webhook_urls[video_id]

