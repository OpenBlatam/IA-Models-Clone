"""
PDF Variantes Performance Service
Performance monitoring and optimization service
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional
from fastapi import Request, Response
from ..utils.config import Settings

logger = logging.getLogger(__name__)

class PerformanceService:
    """Performance monitoring service for PDF Variantes API"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.request_metrics: Dict[str, Any] = {}
        self.active_requests: Dict[str, float] = {}
    
    async def initialize(self):
        """Initialize performance service"""
        try:
            logger.info("Performance Service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Performance Service: {e}")
            # Don't raise - performance monitoring is optional
    
    async def cleanup(self):
        """Cleanup performance service"""
        try:
            self.request_metrics.clear()
            self.active_requests.clear()
        except Exception as e:
            logger.error(f"Error cleaning up Performance Service: {e}")
    
    async def start_request_monitoring(self, request: Request):
        """Start monitoring a request"""
        try:
            request_id = id(request)
            self.active_requests[request_id] = time.time()
        except Exception as e:
            logger.error(f"Error starting request monitoring: {e}")
    
    async def end_request_monitoring(self, request: Request, response: Response):
        """End monitoring a request and collect metrics"""
        try:
            request_id = id(request)
            if request_id in self.active_requests:
                start_time = self.active_requests.pop(request_id)
                process_time = time.time() - start_time
                
                # Store metrics
                path = str(request.url.path)
                method = request.method
                
                key = f"{method}:{path}"
                if key not in self.request_metrics:
                    self.request_metrics[key] = {
                        "count": 0,
                        "total_time": 0.0,
                        "min_time": float('inf'),
                        "max_time": 0.0,
                        "avg_time": 0.0
                    }
                
                metrics = self.request_metrics[key]
                metrics["count"] += 1
                metrics["total_time"] += process_time
                metrics["min_time"] = min(metrics["min_time"], process_time)
                metrics["max_time"] = max(metrics["max_time"], process_time)
                metrics["avg_time"] = metrics["total_time"] / metrics["count"]
                
        except Exception as e:
            logger.error(f"Error ending request monitoring: {e}")
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            "request_metrics": self.request_metrics,
            "active_requests": len(self.active_requests)
        }





