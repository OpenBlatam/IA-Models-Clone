from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import signal
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import argparse
import json
import time
from datetime import datetime
from onyx.utils.logger import setup_logger
from onyx.utils.threadpool_concurrency import ThreadSafeDict
from onyx.utils.timing import time_function
from onyx.utils.telemetry import TelemetryLogger
from onyx.utils.gpu_utils import get_gpu_info, is_gpu_available
from onyx.core.functions import format_response, handle_error
from onyx.db.engine import get_session_with_current_tenant
from .models import VideoRequest, VideoResponse, PluginConfig
from .core.exceptions import AIVideoError, PluginError, ValidationError
from .core.onyx_integration import OnyxIntegrationManager, onyx_integration
from .onyx_video_workflow import OnyxVideoWorkflow, onyx_video_generator
from .onyx_plugin_manager import OnyxPluginManager, onyx_plugin_manager, OnyxPluginContext
from typing import Any, List, Dict, Optional
import logging
"""
Onyx AI Video System - Main Entry Point

Main entry point for the Onyx-adapted AI Video system that integrates
all Onyx components and provides a unified interface for video generation.
"""


# Onyx imports

# Local imports

logger = setup_logger(__name__)


class OnyxAIVideoSystem:
    """
    Main Onyx AI Video System.
    
    Integrates all Onyx components and provides a unified interface
    for AI-powered video generation with enterprise-grade features.
    """
    
    def __init__(self) -> Any:
        self.logger = setup_logger("onyx_ai_video_system")
        self.telemetry = TelemetryLogger()
        self.cache: ThreadSafeDict[str, Any] = ThreadSafeDict()
        
        # System components
        self.integration_manager = onyx_integration
        self.video_workflow = onyx_video_generator
        self.plugin_manager = onyx_plugin_manager
        
        # System state
        self.is_initialized = False
        self.is_shutting_down = False
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        
        # Performance tracking
        self.performance_metrics: ThreadSafeDict[str, Any] = ThreadSafeDict()
    
    async def initialize(self) -> None:
        """Initialize the Onyx AI Video System."""
        try:
            self.logger.info("Initializing Onyx AI Video System")
            
            # Initialize Onyx integration
            await self.integration_manager.initialize()
            
            # Initialize video workflow
            await self.video_workflow.initialize()
            
            # Initialize plugin manager
            await self.plugin_manager.initialize()
            
            # Setup signal handlers
            self._setup_signal_handlers()
            
            # Start background tasks
            asyncio.create_task(self._background_cleanup_loop())
            
            self.is_initialized = True
            self.logger.info("Onyx AI Video System initialized successfully")
            
            # Log telemetry
            self.telemetry.log_info("system_initialized", {
                "gpu_available": is_gpu_available(),
                "gpu_info": get_gpu_info() if is_gpu_available() else None,
                "initialization_time": time.time() - self.start_time
            })
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            raise AIVideoError(f"System initialization failed: {e}")
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame) -> Any:
            self.logger.info(f"Received signal {signum}, initiating shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def generate_video(self, request: VideoRequest) -> VideoResponse:
        """Generate video using Onyx AI Video System."""
        if not self.is_initialized:
            raise AIVideoError("System not initialized")
        
        if self.is_shutting_down:
            raise AIVideoError("System is shutting down")
        
        start_time = time.time()
        self.request_count += 1
        
        try:
            self.logger.info(f"Processing video request: {request.request_id}")
            
            # Validate request
            await self._validate_request(request)
            
            # Create plugin context
            context = await self._create_plugin_context(request)
            
            # Execute plugins if specified
            plugin_results = {}
            if request.plugins:
                plugin_results = await self.plugin_manager.execute_plugins(context, request.plugins)
            
            # Generate video using workflow
            with time_function("onyx_video_generation"):
                response = await self.video_workflow.generate_video(request)
            
            # Add plugin results to response
            if plugin_results:
                response.metadata["plugin_results"] = plugin_results
            
            # Record performance metrics
            duration = time.time() - start_time
            self._record_performance_metrics("video_generation", duration, True)
            
            # Log telemetry
            self.telemetry.log_info("video_generated", {
                "request_id": request.request_id,
                "user_id": request.user_id,
                "duration": duration,
                "plugins_used": len(request.plugins) if request.plugins else 0,
                "gpu_used": context.gpu_available
            })
            
            self.logger.info(f"Video generation completed: {request.request_id}")
            return response
            
        except Exception as e:
            # Record error metrics
            duration = time.time() - start_time
            self.error_count += 1
            self._record_performance_metrics("video_generation", duration, False)
            
            self.logger.error(f"Video generation failed: {request.request_id} - {e}")
            self.telemetry.log_error("video_generation_failed", {
                "request_id": request.request_id,
                "error": str(e),
                "error_type": type(e).__name__
            })
            
            raise AIVideoError(f"Video generation failed: {e}")
    
    async def generate_video_with_vision(self, request: VideoRequest, image_data: bytes) -> VideoResponse:
        """Generate video with vision capabilities."""
        if not self.is_initialized:
            raise AIVideoError("System not initialized")
        
        try:
            self.logger.info(f"Processing vision video request: {request.request_id}")
            
            # Validate request
            await self._validate_request(request)
            
            # Generate video with vision
            with time_function("onyx_vision_video_generation"):
                response = await self.video_workflow.generate_video_with_vision(request, image_data)
            
            # Log telemetry
            self.telemetry.log_info("vision_video_generated", {
                "request_id": request.request_id,
                "user_id": request.user_id,
                "image_size": len(image_data)
            })
            
            self.logger.info(f"Vision video generation completed: {request.request_id}")
            return response
            
        except Exception as e:
            self.logger.error(f"Vision video generation failed: {request.request_id} - {e}")
            raise AIVideoError(f"Vision video generation failed: {e}")
    
    async async def _validate_request(self, request: VideoRequest) -> None:
        """Validate video request."""
        try:
            # Basic validation
            if not request.input_text.strip():
                raise ValidationError("Input text cannot be empty")
            
            if request.duration <= 0 or request.duration > 600:
                raise ValidationError("Duration must be between 1 and 600 seconds")
            
            if request.quality not in ["low", "medium", "high", "ultra"]:
                raise ValidationError("Invalid quality setting")
            
            # Validate user access using Onyx
            if not await self.integration_manager.security_manager.validate_access(
                request.user_id, request.request_id
            ):
                raise ValidationError("Access denied")
            
        except Exception as e:
            self.logger.error(f"Request validation failed: {e}")
            raise ValidationError(f"Request validation failed: {e}")
    
    async def _create_plugin_context(self, request: VideoRequest) -> OnyxPluginContext:
        """Create context for plugin execution."""
        try:
            # Get LLM instance
            llm = await self.integration_manager.llm_manager.get_default_llm()
            
            # Check GPU availability
            gpu_available = is_gpu_available()
            
            context = OnyxPluginContext(
                request=request,
                llm=llm,
                gpu_available=gpu_available,
                metadata={
                    "user_id": request.user_id,
                    "quality": request.quality,
                    "duration": request.duration,
                    "output_format": request.output_format
                }
            )
            
            return context
            
        except Exception as e:
            self.logger.error(f"Context creation failed: {e}")
            raise AIVideoError(f"Context creation failed: {e}")
    
    def _record_performance_metrics(self, operation: str, duration: float, success: bool) -> None:
        """Record performance metrics."""
        metrics = self.performance_metrics.get(operation, {
            "count": 0,
            "total_duration": 0.0,
            "success_count": 0,
            "error_count": 0,
            "avg_duration": 0.0
        })
        
        metrics["count"] += 1
        metrics["total_duration"] += duration
        metrics["avg_duration"] = metrics["total_duration"] / metrics["count"]
        
        if success:
            metrics["success_count"] += 1
        else:
            metrics["error_count"] += 1
        
        self.performance_metrics[operation] = metrics
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            # Get component statuses
            integration_status = await self.integration_manager.get_system_status()
            workflow_status = await self.video_workflow.get_generator_status()
            plugin_status = await self.plugin_manager.get_manager_status()
            
            # Calculate performance metrics
            performance_summary = {}
            for operation, metrics in self.performance_metrics.items():
                performance_summary[operation] = {
                    "total_requests": metrics["count"],
                    "success_rate": metrics["success_count"] / max(metrics["count"], 1),
                    "avg_duration": metrics["avg_duration"],
                    "error_rate": metrics["error_count"] / max(metrics["count"], 1)
                }
            
            status = {
                "system": "onyx_ai_video",
                "status": "operational" if self.is_initialized and not self.is_shutting_down else "degraded",
                "uptime": time.time() - self.start_time,
                "request_count": self.request_count,
                "error_count": self.error_count,
                "error_rate": self.error_count / max(self.request_count, 1),
                "gpu_available": is_gpu_available(),
                "gpu_info": get_gpu_info() if is_gpu_available() else None,
                "components": {
                    "integration": integration_status,
                    "workflow": workflow_status,
                    "plugins": plugin_status
                },
                "performance": performance_summary,
                "cache_size": len(self.cache),
                "timestamp": datetime.now().isoformat()
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Status check failed: {e}")
            return {"error": str(e)}
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics."""
        try:
            # Get Onyx metrics
            onyx_metrics = await self.integration_manager.get_metrics()
            
            # Add custom metrics
            metrics = {
                **onyx_metrics,
                "ai_video": {
                    "request_count": self.request_count,
                    "error_count": self.error_count,
                    "error_rate": self.error_count / max(self.request_count, 1),
                    "uptime": time.time() - self.start_time,
                    "performance_metrics": dict(self.performance_metrics)
                }
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Metrics collection failed: {e}")
            return {"error": str(e)}
    
    async def _background_cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while not self.is_shutting_down:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Cleanup caches
                self.cache.clear()
                
                # Cleanup performance metrics (keep last 1000 entries)
                if len(self.performance_metrics) > 1000:
                    # Keep only recent metrics
                    recent_metrics = dict(list(self.performance_metrics.items())[-1000:])
                    self.performance_metrics.clear()
                    self.performance_metrics.update(recent_metrics)
                
                self.logger.debug("Background cleanup completed")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Background cleanup error: {e}")
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the system."""
        if self.is_shutting_down:
            return
        
        self.logger.info("Starting system shutdown...")
        self.is_shutting_down = True
        
        try:
            # Shutdown components
            await self.video_workflow.cleanup()
            await self.plugin_manager.cleanup()
            
            # Clear caches
            self.cache.clear()
            self.performance_metrics.clear()
            
            self.logger.info("System shutdown completed successfully")
            
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")
            raise


# Global system instance
onyx_ai_video_system = OnyxAIVideoSystem()


async def get_system() -> OnyxAIVideoSystem:
    """Get the global Onyx AI Video system instance."""
    if not onyx_ai_video_system.is_initialized:
        await onyx_ai_video_system.initialize()
    
    return onyx_ai_video_system


async def shutdown_system() -> None:
    """Shutdown the global Onyx AI Video system instance."""
    await onyx_ai_video_system.shutdown()


def setup_signal_handlers(system: OnyxAIVideoSystem) -> None:
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame) -> Any:
        logger.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(system.shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main() -> None:
    """Main entry point for the Onyx AI Video system."""
    parser = argparse.ArgumentParser(description="Onyx AI Video System")
    parser.add_argument(
        "--config",
        default="config/onyx_ai_video.json",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show system status and exit"
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Show system metrics and exit"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run system tests and exit"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize system
        system = await get_system()
        
        # Setup signal handlers
        setup_signal_handlers(system)
        
        # Handle status/metrics requests
        if args.status:
            status = await system.get_system_status()
            print(json.dumps(status, indent=2))
            return
        
        if args.metrics:
            metrics = await system.get_metrics()
            print(json.dumps(metrics, indent=2))
            return
        
        if args.test:
            await run_system_tests(system)
            return
        
        # Keep system running
        logger.info("Onyx AI Video System is running...")
        
        # Run forever
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"System error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if 'system' in locals():
            await system.shutdown()


async def run_system_tests(system: OnyxAIVideoSystem) -> None:
    """Run system tests."""
    try:
        logger.info("Running system tests...")
        
        # Test 1: System status
        status = await system.get_system_status()
        assert status["status"] in ["operational", "degraded"], "Invalid system status"
        logger.info("✓ System status test passed")
        
        # Test 2: Video generation
        test_request = VideoRequest(
            input_text="Test video generation with Onyx",
            user_id="test_user",
            request_id="test_001"
        )
        
        response = await system.generate_video(test_request)
        assert response.status == "completed", "Video generation failed"
        logger.info("✓ Video generation test passed")
        
        # Test 3: Metrics collection
        metrics = await system.get_metrics()
        assert "ai_video" in metrics, "Metrics collection failed"
        logger.info("✓ Metrics collection test passed")
        
        logger.info("All system tests passed!")
        
    except Exception as e:
        logger.error(f"System test failed: {e}")
        raise


# API endpoints for integration
async async def api_generate_video(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """API endpoint for video generation."""
    try:
        # Create video request
        request = VideoRequest(**request_data)
        
        # Get system instance
        system = await get_system()
        
        # Generate video
        response = await system.generate_video(request)
        
        # Exponer todos los campos relevantes del modelo mejorado
        response_dict = {
            "request_id": response.request_id,
            "status": response.status,
            "output_url": response.output_url,
            "output_path": response.output_path,
            "thumbnail_url": response.thumbnail_url,
            "duration": response.duration,
            "file_size": response.file_size,
            "resolution": response.resolution,
            "fps": response.fps,
            "processing_time": response.processing_time,
            "steps_completed": response.steps_completed,
            "error_message": response.error_message,
            "plugin_results": response.plugin_results,
            "metadata": response.metadata,
            "created_at": response.created_at,
            "updated_at": response.updated_at,
            # Mejoras
            "processing_history": getattr(response, "processing_history", None),
            "outputs": getattr(response, "outputs", None),
            "quality_score": getattr(response, "quality_score", None),
            "user_rating": getattr(response, "user_rating", None)
        }
        # Eliminar claves None para respuesta limpia
        response_dict = {k: v for k, v in response_dict.items() if v is not None}
        return await format_response(response_dict)
        
    except Exception as e:
        return await handle_error(e, {"endpoint": "generate_video"})


async async def api_get_status() -> Dict[str, Any]:
    """API endpoint for system status."""
    try:
        system = await get_system()
        status = await system.get_system_status()
        
        return await format_response(status)
        
    except Exception as e:
        return await handle_error(e, {"endpoint": "get_status"})


async async def api_get_metrics() -> Dict[str, Any]:
    """API endpoint for system metrics."""
    try:
        system = await get_system()
        metrics = await system.get_metrics()
        
        return await format_response(metrics)
        
    except Exception as e:
        return await handle_error(e, {"endpoint": "get_metrics"})


match __name__:
    case "__main__":
    asyncio.run(main()) 