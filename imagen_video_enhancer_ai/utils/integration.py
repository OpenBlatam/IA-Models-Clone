"""
Integration Utilities for Imagen Video Enhancer AI
==================================================

Utilities for integrating with external systems.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class IntegrationHelper:
    """
    Helper for integrating with external systems.
    
    Features:
    - Webhook integration
    - API integration
    - Data transformation
    - Error handling
    """
    
    @staticmethod
    def format_webhook_payload(
        event_type: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Format data for webhook payload.
        
        Args:
            event_type: Event type
            data: Event data
            metadata: Optional metadata
            
        Returns:
            Formatted payload
        """
        return {
            "event": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data,
            "metadata": metadata or {}
        }
    
    @staticmethod
    def format_api_response(
        success: bool,
        data: Any = None,
        message: Optional[str] = None,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format standard API response.
        
        Args:
            success: Success flag
            data: Response data
            message: Optional message
            error: Optional error message
            
        Returns:
            Formatted response
        """
        response = {
            "success": success,
            "timestamp": datetime.now().isoformat()
        }
        
        if data is not None:
            response["data"] = data
        
        if message:
            response["message"] = message
        
        if error:
            response["error"] = error
        
        return response
    
    @staticmethod
    def transform_task_result(
        result: Dict[str, Any],
        format: str = "standard"
    ) -> Dict[str, Any]:
        """
        Transform task result to different formats.
        
        Args:
            result: Task result
            format: Output format (standard, minimal, detailed)
            
        Returns:
            Transformed result
        """
        if format == "minimal":
            return {
                "task_id": result.get("task_id"),
                "status": result.get("status"),
                "success": result.get("status") == "completed"
            }
        elif format == "detailed":
            return {
                **result,
                "formatted_at": datetime.now().isoformat(),
                "enhancement_guide_length": len(result.get("enhancement_guide", "")),
                "tokens_used": result.get("tokens_used", 0),
                "model": result.get("model", "unknown")
            }
        else:  # standard
            return result
    
    @staticmethod
    def validate_webhook_signature(
        payload: str,
        signature: str,
        secret: str
    ) -> bool:
        """
        Validate webhook signature.
        
        Args:
            payload: Payload string
            signature: Signature to validate
            secret: Secret key
            
        Returns:
            True if valid
        """
        import hmac
        import hashlib
        
        expected = hmac.new(
            secret.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Handle different signature formats
        if signature.startswith("sha256="):
            signature = signature[7:]
        
        return hmac.compare_digest(expected, signature)




