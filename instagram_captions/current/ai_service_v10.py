"""
Instagram Captions API v10.0 - Refactored AI Service

Consolidates advanced AI capabilities from v9.0 into a clean, efficient service.
"""

import asyncio
import time
import logging
from typing import Any, List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

from .core_v10 import (
    RefactoredConfig, RefactoredCaptionRequest, RefactoredCaptionResponse,
    BatchRefactoredRequest, RefactoredAIEngine, Metrics, RefactoredUtils
)
from .config import get_config
from .utils import get_logger, PerformanceMonitor

# =============================================================================
# REFACTORED AI SERVICE
# =============================================================================

class RefactoredAIService:
    """Consolidated AI service with essential v9.0 capabilities."""
    
    def __init__(self, config: Optional[RefactoredConfig] = None) -> None:
        self.config = config or get_config()
        self.ai_engine = RefactoredAIEngine(self.config)
        self.metrics = Metrics()
        self.performance_monitor = PerformanceMonitor()
        
        # Initialize thread pool with configuration
        max_workers = getattr(self.config, 'AI_WORKERS', 4)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Initialize logger
        self.logger = get_logger(__name__)
        
        self.stats = {
            "service_started": time.time(),
            "total_processed": 0,
            "concurrent_requests": 0,
            "cache_hits": 0,
            "fallback_used": 0
        }
        
        self.logger.info("🚀 Refactored AI Service v10.0 initialized")
    
    async def generate_single_caption(self, request: RefactoredCaptionRequest) -> RefactoredCaptionResponse:
        """Generate single caption with advanced analysis."""
        start_time = time.time()
        
        try:
            self.stats["concurrent_requests"] += 1
            
            # Generate using refactored AI engine
            response = await self.ai_engine.generate_caption(request)
            
            # Record metrics
            processing_time = time.time() - start_time
            self.metrics.record_request(True, processing_time)
            self.performance_monitor.record_metric("single_caption_generation", processing_time)
            
            self.stats["total_processed"] += 1
            self.logger.info(f"✅ Caption generated in {processing_time:.3f}s")
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.metrics.record_request(False, processing_time)
            self.stats["fallback_used"] += 1
            self.logger.error(f"❌ Caption generation failed: {e}")
            
            # Return fallback response
            return RefactoredCaptionResponse(
                caption=f"Sharing this amazing {request.text} ✨",
                style=request.style,
                length=request.length,
                hashtags=["#beautiful", "#amazing", "#inspiration"],
                emojis=["✨", "🌟"],
                metadata={"fallback": True, "error": str(e)},
                processing_time=processing_time,
                model_used="emergency_fallback"
            )
        
        finally:
            self.stats["concurrent_requests"] -= 1
    
    async def generate_batch_captions(self, batch_request: BatchRefactoredRequest) -> Dict[str, Any]:
        """Generate multiple captions efficiently."""
        start_time = time.time()
        batch_size = len(batch_request.requests)
        
        # Validate batch size using configuration
        max_batch_size = getattr(self.config, 'MAX_BATCH_SIZE', 100)
        if batch_size > max_batch_size:
            raise ValueError(f"Batch size {batch_size} exceeds maximum {max_batch_size}")
        
        self.logger.info(f"🔄 Processing batch {batch_request.batch_id} with {batch_size} requests")
        
        try:
            # Process with controlled concurrency
            max_workers = getattr(self.config, 'AI_WORKERS', 4)
            semaphore = asyncio.Semaphore(max_workers)
            
            async def process_single_with_semaphore(req: RefactoredCaptionRequest):
                async with semaphore:
                    return await self.generate_single_caption(req)
            
            # Process all requests concurrently
            tasks = [process_single_with_semaphore(req) for req in batch_request.requests]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Request {i} failed: {result}")
                    # Create fallback response for failed requests
                    fallback_response = RefactoredCaptionResponse(
                        caption=f"Amazing content: {batch_request.requests[i].text} ✨",
                        style=batch_request.requests[i].style,
                        length=batch_request.requests[i].length,
                        hashtags=["#content", "#amazing"],
                        emojis=["✨"],
                        metadata={"fallback": True, "error": str(result)},
                        processing_time=0.1,
                        model_used="fallback"
                    )
                    processed_results.append(fallback_response)
                else:
                    processed_results.append(result)
            
            processing_time = time.time() - start_time
            
            # Record performance metrics
            self.performance_monitor.record_metric("batch_processing", processing_time)
            self.performance_monitor.record_metric("batch_size", batch_size)
            
            return {
                "batch_id": batch_request.batch_id,
                "total_requests": batch_size,
                "successful_requests": len([r for r in processed_results if not r.metadata.get("fallback")]),
                "failed_requests": len([r for r in processed_results if r.metadata.get("fallback")]),
                "results": processed_results,
                "processing_time": processing_time,
                "average_time_per_request": processing_time / batch_size
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ Batch processing failed: {e}")
            raise
    
    async def test_service(self) -> Dict[str, Any]:
        """Test the AI service functionality."""
        try:
            # Test request
            test_request = RefactoredCaptionRequest(
                text="Beautiful sunset at the beach",
                style="casual",
                length="medium"
            )
            
            # Test single generation
            single_result = await self.generate_single_caption(test_request)
            
            # Test batch generation
            batch_request = BatchRefactoredRequest(
                requests=[test_request, test_request]
            )
            batch_result = await self.generate_batch_captions(batch_request)
            
            return {
                "service_status": "operational",
                "single_generation": "success",
                "batch_generation": "success",
                "ai_engine_status": "available" if self.ai_engine.pipeline else "fallback",
                "test_timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Service test failed: {e}")
            return {
                "service_status": "error",
                "error": str(e),
                "test_timestamp": time.time()
            }
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics."""
        uptime = time.time() - self.stats["service_started"]
        
        return {
            "service_info": {
                "name": "Refactored AI Service v10.0",
                "version": "10.0.0",
                "uptime_seconds": uptime,
                "uptime_formatted": f"{uptime/3600:.1f} hours"
            },
            "performance_stats": {
                "total_processed": self.stats["total_processed"],
                "concurrent_requests": self.stats["concurrent_requests"],
                "cache_hits": self.stats["cache_hits"],
                "fallback_used": self.stats["fallback_used"],
                "requests_per_hour": round(self.stats["total_processed"] / max(uptime/3600, 1), 2)
            },
            "ai_engine_status": {
                "torch_available": hasattr(self.ai_engine, 'pipeline') and self.ai_engine.pipeline is not None,
                "model_name": self.config.AI_MODEL_NAME,
                "cache_size": self.config.CACHE_SIZE
            },
            "metrics": self.metrics.get_stats(),
            "performance_metrics": self.performance_monitor.get_all_statistics()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of the AI service."""
        try:
            # Basic health check
            health_status = {
                "service": "healthy",
                "timestamp": time.time(),
                "version": "10.0.0"
            }
            
            # Test AI engine
            try:
                test_request = RefactoredCaptionRequest(
                    text="test",
                    style="casual",
                    length="short"
                )
                await self.ai_engine.generate_caption(test_request)
                health_status["ai_engine"] = "healthy"
            except Exception as e:
                health_status["ai_engine"] = "unhealthy"
                health_status["ai_engine_error"] = str(e)
            
            # Check thread pool
            if self.executor._shutdown:
                health_status["thread_pool"] = "unhealthy"
            else:
                health_status["thread_pool"] = "healthy"
            
            # Overall status
            if health_status.get("ai_engine") == "unhealthy" or health_status.get("thread_pool") == "unhealthy":
                health_status["overall"] = "degraded"
            else:
                health_status["overall"] = "healthy"
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "service": "unhealthy",
                "error": str(e),
                "timestamp": time.time(),
                "version": "10.0.0"
            }
    
    def cleanup(self):
        """Cleanup resources."""
        if self.executor:
            self.executor.shutdown(wait=True)
        self.logger.info("🧹 AI Service cleanup completed")

# =============================================================================
# SERVICE INSTANCE
# =============================================================================

# Create global service instance
refactored_ai_service = RefactoredAIService()

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'RefactoredAIService',
    'refactored_ai_service'
]






