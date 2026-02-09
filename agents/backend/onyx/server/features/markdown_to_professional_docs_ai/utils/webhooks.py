"""Webhook support for async conversions"""
import httpx
import json
from typing import Optional, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class WebhookClient:
    """Client for sending webhooks"""
    
    def __init__(self, timeout: int = 30):
        """
        Initialize webhook client
        
        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
    
    async def send_webhook(
        self,
        url: str,
        event: str,
        data: Dict[str, Any],
        secret: Optional[str] = None
    ) -> bool:
        """
        Send webhook notification
        
        Args:
            url: Webhook URL
            event: Event type
            data: Event data
            secret: Optional webhook secret for signing
            
        Returns:
            True if successful, False otherwise
        """
        try:
            payload = {
                "event": event,
                "timestamp": datetime.now().isoformat(),
                "data": data
            }
            
            # Add signature if secret provided
            if secret:
                import hmac
                import hashlib
                signature = hmac.new(
                    secret.encode(),
                    json.dumps(payload, sort_keys=True).encode(),
                    hashlib.sha256
                ).hexdigest()
                payload["signature"] = signature
            
            response = await self.client.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Error sending webhook to {url}: {e}")
            return False
    
    async def send_conversion_started(
        self,
        webhook_url: str,
        conversion_id: str,
        format_name: str,
        secret: Optional[str] = None
    ) -> bool:
        """Send conversion started webhook"""
        return await self.send_webhook(
            webhook_url,
            "conversion.started",
            {
                "conversion_id": conversion_id,
                "format": format_name,
                "status": "started"
            },
            secret
        )
    
    async def send_conversion_completed(
        self,
        webhook_url: str,
        conversion_id: str,
        format_name: str,
        output_path: str,
        file_size: int,
        secret: Optional[str] = None
    ) -> bool:
        """Send conversion completed webhook"""
        return await self.send_webhook(
            webhook_url,
            "conversion.completed",
            {
                "conversion_id": conversion_id,
                "format": format_name,
                "status": "completed",
                "output_path": output_path,
                "file_size": file_size
            },
            secret
        )
    
    async def send_conversion_failed(
        self,
        webhook_url: str,
        conversion_id: str,
        format_name: str,
        error: str,
        secret: Optional[str] = None
    ) -> bool:
        """Send conversion failed webhook"""
        return await self.send_webhook(
            webhook_url,
            "conversion.failed",
            {
                "conversion_id": conversion_id,
                "format": format_name,
                "status": "failed",
                "error": error
            },
            secret
        )
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()


# Global webhook client
_webhook_client: Optional[WebhookClient] = None


def get_webhook_client() -> WebhookClient:
    """Get global webhook client"""
    global _webhook_client
    if _webhook_client is None:
        _webhook_client = WebhookClient()
    return _webhook_client

