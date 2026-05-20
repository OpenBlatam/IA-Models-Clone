"""
Metrics Tracker
===============
Tracks metrics for clothing change operations
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Tracks metrics for clothing change operations.
    """
    
    def __init__(self, metrics_service: Optional[Any] = None):
        """
        Initialize metrics tracker.
        
        Args:
            metrics_service: Optional metrics service instance
        """
        self.metrics_service = metrics_service
        self.enabled = metrics_service is not None
    
    def track_operation(
        self,
        operation_type: str,
        success: bool,
        openrouter_used: bool = False,
        truthgpt_used: bool = False,
        duration: Optional[float] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Track a clothing change operation.
        
        Args:
            operation_type: Type of operation ("clothing_change" or "face_swap")
            success: Whether operation succeeded
            openrouter_used: Whether OpenRouter was used
            truthgpt_used: Whether TruthGPT was used
            duration: Optional operation duration in seconds
            error: Optional error message
        """
        if not self.enabled:
            return
        
        try:
            metrics_data = {
                "operation_type": operation_type,
                "success": success,
                "openrouter_used": openrouter_used,
                "truthgpt_used": truthgpt_used
            }
            
            if duration is not None:
                metrics_data["duration"] = duration
            
            if error:
                metrics_data["error"] = error
            
            # Record metric
            if hasattr(self.metrics_service, 'record'):
                self.metrics_service.record("clothing_change_operation", metrics_data)
            elif hasattr(self.metrics_service, 'increment'):
                self.metrics_service.increment(f"clothing_change.{operation_type}.{'success' if success else 'failure'}")
                
        except Exception as e:
            logger.warning(f"Failed to track metrics: {e}")
    
    def get_analytics(self) -> Dict[str, Any]:
        """
        Get analytics data.
        
        Returns:
            Analytics dictionary
        """
        if not self.enabled:
            return {
                "enabled": False,
                "message": "Metrics service not available"
            }
        
        try:
            if hasattr(self.metrics_service, 'get_analytics'):
                return self.metrics_service.get_analytics()
            elif hasattr(self.metrics_service, 'get_stats'):
                return self.metrics_service.get_stats()
            else:
                return {
                    "enabled": True,
                    "message": "Analytics not available from metrics service"
                }
        except Exception as e:
            logger.warning(f"Failed to get analytics: {e}")
            return {
                "enabled": True,
                "error": str(e)
            }

