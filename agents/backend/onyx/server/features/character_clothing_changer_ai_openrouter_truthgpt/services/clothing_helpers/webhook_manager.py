"""
Webhook Manager
===============
Manages webhook notifications for clothing change operations
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class WebhookManager:
    """
    Manages webhook notifications for clothing change operations.
    """
    
    def __init__(self, webhook_service: Optional[Any] = None):
        """
        Initialize webhook manager.
        
        Args:
            webhook_service: Optional webhook service instance
        """
        self.webhook_service = webhook_service
        self.enabled = webhook_service is not None
    
    async def send_clothing_change_completed(
        self,
        prompt_id: str,
        success: bool,
        openrouter_used: bool = False,
        truthgpt_used: bool = False,
        error: Optional[str] = None
    ) -> bool:
        """
        Send webhook notification for clothing change completion.
        
        Args:
            prompt_id: ComfyUI prompt ID
            success: Whether operation succeeded
            openrouter_used: Whether OpenRouter was used
            truthgpt_used: Whether TruthGPT was used
            error: Optional error message
            
        Returns:
            True if webhook sent successfully, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            from services.webhook_service import WebhookEvent
            
            event = WebhookEvent(
                event_type="clothing_change_completed",
                prompt_id=prompt_id,
                data={
                    "success": success,
                    "openrouter_used": openrouter_used,
                    "truthgpt_used": truthgpt_used
                }
            )
            
            if error:
                event.data["error"] = error
            
            await self.webhook_service.send_event(event)
            return True
            
        except Exception as e:
            logger.warning(f"Failed to send webhook notification: {e}")
            return False
    
    async def send_face_swap_completed(
        self,
        prompt_id: str,
        success: bool,
        error: Optional[str] = None
    ) -> bool:
        """
        Send webhook notification for face swap completion.
        
        Args:
            prompt_id: ComfyUI prompt ID
            success: Whether operation succeeded
            error: Optional error message
            
        Returns:
            True if webhook sent successfully, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            from services.webhook_service import WebhookEvent
            
            event = WebhookEvent(
                event_type="face_swap_completed",
                prompt_id=prompt_id,
                data={
                    "success": success
                }
            )
            
            if error:
                event.data["error"] = error
            
            await self.webhook_service.send_event(event)
            return True
            
        except Exception as e:
            logger.warning(f"Failed to send face swap webhook: {e}")
            return False

