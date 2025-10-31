from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging
from ..core.exceptions import AIVideoError, ValidationError, SecurityError
from ..core.models import VideoRequest, VideoResponse, SystemStatus, PerformanceMetrics
from ..core.integration import OnyxIntegrationManager, onyx_integration
from ..workflows.video_workflow import OnyxVideoWorkflow
from ..plugins.plugin_manager import OnyxPluginManager
from ..config.config_manager import OnyxConfigManager, get_config
from ..utils.logger import OnyxLogger, get_logger, get_performance_logger
from ..utils.performance import PerformanceMonitor, get_performance_monitor
from ..utils.security import SecurityManager, get_security_manager
from typing import Any, List, Dict, Optional
"""
Onyx AI Video System - Main API

Main API module for the Onyx AI Video system with unified initialization,
request processing, and system management.
"""




@dataclass
class SystemMetrics:
    """System metrics data."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_processing_time: float = 0.0
    min_processing_time: Optional[float] = None
    max_processing_time: Optional[float] = None
    plugin_executions: Dict[str, int] = field(default_factory=dict)
    plugin_errors: Dict[str, int] = field(default_factory=dict)
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    gpu_usage: Optional[float] = None
    cache_hits: int = 0
    cache_misses: int = 0
    cache_size: int = 0
    period_start: datetime = field(default_factory=datetime.now)
    period_end: datetime = field(default_factory=datetime.now)


class OnyxAIVideoSystem:
    """
    Main Onyx AI Video system.
    
    Provides unified interface for video generation, plugin management,
    and system administration with full Onyx integration.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        
    """__init__ function."""
self.config_path = config_path
        self.config: Optional[Any] = None
        self.logger: Optional[OnyxLogger] = None
        self.performance_monitor: Optional[PerformanceMonitor] = None
        self.security_manager: Optional[SecurityManager] = None
        
        # Core components
        self.onyx_integration: Optional[OnyxIntegrationManager] = None
        self.video_workflow: Optional[OnyxVideoWorkflow] = None
        self.plugin_manager: Optional[OnyxPluginManager] = None
        self.config_manager: Optional[OnyxConfigManager] = None
        
        # System state
        self.initialized = False
        self.shutdown_requested = False
        self.metrics = SystemMetrics()
        self.active_requests: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.request_times: List[float] = []
        self.max_request_history = 1000
    
    async def initialize(self) -> None:
        """Initialize the AI Video system."""
        try:
            self.logger = get_logger()
            self.logger.info("Initializing Onyx AI Video System")
            
            # Load configuration
            self.config_manager = OnyxConfigManager(self.config_path)
            self.config = self.config_manager.load_config()
            
            # Setup logging
            self.logger = OnyxLogger(
                name="ai_video_system",
                level=self.config.logging.level,
                use_onyx_logging=self.config.logging.use_onyx_logging,
                log_file=self.config.logging.file_path
            )
            
            # Initialize performance monitoring
            self.performance_monitor = get_performance_monitor(
                enable_monitoring=self.config.performance.enable_monitoring,
                metrics_interval=self.config.performance.metrics_interval
            )
            
            # Initialize security manager
            security_config = self.config.security
            self.security_manager = get_security_manager(security_config)
            
            # Initialize Onyx integration
            self.onyx_integration = onyx_integration
            await self.onyx_integration.initialize()
            
            # Initialize video workflow
            self.video_workflow = OnyxVideoWorkflow(
                config=self.config,
                onyx_integration=self.onyx_integration
            )
            await self.video_workflow.initialize()
            
            # Initialize plugin manager
            self.plugin_manager = OnyxPluginManager(
                plugins_directory=self.config.plugins.plugins_directory,
                config=self.config,
                onyx_integration=self.onyx_integration
            )
            await self.plugin_manager.initialize()
            
            # Load plugins if auto-load is enabled
            if self.config.plugins.auto_load:
                await self.plugin_manager.load_plugins()
            
            self.initialized = True
            self.logger.info("Onyx AI Video System initialized successfully")
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            raise AIVideoError(f"Initialization failed: {e}")
    
    async def generate_video(self, request: VideoRequest) -> VideoResponse:
        """
        Generate video from request.
        
        Args:
            request: Video generation request
            
        Returns:
            Video generation response
        """
        if not self.initialized:
            raise AIVideoError("System not initialized")
        
        start_time = time.time()
        request_id = request.request_id
        
        try:
            # Validate access
            if not await self.security_manager.validate_access(request.user_id, request_id):
                raise SecurityError("Access denied")
            
            # Check rate limit
            allowed, rate_info = self.security_manager.check_rate_limit(request.user_id)
            if not allowed:
                raise SecurityError(f"Rate limit exceeded. Reset at {rate_info['reset_time']}")
            
            # Validate input
            is_valid, cleaned_text = self.security_manager.validate_input(
                request.input_text, 
                self.config.security.max_input_length
            )
            if not is_valid:
                raise ValidationError(f"Input validation failed: {cleaned_text}")
            
            # Update request with cleaned text
            request.input_text = cleaned_text
            
            # Track active request
            self.active_requests[request_id] = {
                "start_time": start_time,
                "user_id": request.user_id,
                "status": "processing"
            }
            
            # Start performance monitoring
            self.performance_monitor.start_operation(f"video_generation_{request_id}")
            
            # Generate video using workflow
            response = await self.video_workflow.generate_video(request)
            
            # Update metrics
            processing_time = time.time() - start_time
            self._update_metrics(True, processing_time)
            
            # Update active request
            self.active_requests[request_id]["status"] = "completed"
            self.active_requests[request_id]["end_time"] = time.time()
            
            # End performance monitoring
            self.performance_monitor.end_operation(f"video_generation_{request_id}")
            
            self.logger.info(f"Video generated successfully: {request_id}")
            return response
            
        except Exception as e:
            # Update metrics
            processing_time = time.time() - start_time
            self._update_metrics(False, processing_time)
            
            # Update active request
            if request_id in self.active_requests:
                self.active_requests[request_id]["status"] = "failed"
                self.active_requests[request_id]["error"] = str(e)
                self.active_requests[request_id]["end_time"] = time.time()
            
            # End performance monitoring
            self.performance_monitor.end_operation(f"video_generation_{request_id}")
            
            self.logger.error(f"Video generation failed: {request_id} - {e}")
            raise
    
    async def generate_video_with_vision(self, request: VideoRequest, image_data: bytes) -> VideoResponse:
        """
        Generate video with vision capabilities.
        
        Args:
            request: Video generation request
            image_data: Image data for vision processing
            
        Returns:
            Video generation response
        """
        if not self.initialized:
            raise AIVideoError("System not initialized")
        
        start_time = time.time()
        request_id = request.request_id
        
        try:
            # Validate access
            if not await self.security_manager.validate_access(request.user_id, request_id):
                raise SecurityError("Access denied")
            
            # Check rate limit
            allowed, rate_info = self.security_manager.check_rate_limit(request.user_id)
            if not allowed:
                raise SecurityError(f"Rate limit exceeded. Reset at {rate_info['reset_time']}")
            
            # Validate input
            is_valid, cleaned_text = self.security_manager.validate_input(
                request.input_text, 
                self.config.security.max_input_length
            )
            if not is_valid:
                raise ValidationError(f"Input validation failed: {cleaned_text}")
            
            # Update request with cleaned text
            request.input_text = cleaned_text
            
            # Track active request
            self.active_requests[request_id] = {
                "start_time": start_time,
                "user_id": request.user_id,
                "status": "processing",
                "vision": True
            }
            
            # Start performance monitoring
            self.performance_monitor.start_operation(f"video_generation_vision_{request_id}")
            
            # Generate video with vision using workflow
            response = await self.video_workflow.generate_video_with_vision(request, image_data)
            
            # Update metrics
            processing_time = time.time() - start_time
            self._update_metrics(True, processing_time)
            
            # Update active request
            self.active_requests[request_id]["status"] = "completed"
            self.active_requests[request_id]["end_time"] = time.time()
            
            # End performance monitoring
            self.performance_monitor.end_operation(f"video_generation_vision_{request_id}")
            
            self.logger.info(f"Video with vision generated successfully: {request_id}")
            return response
            
        except Exception as e:
            # Update metrics
            processing_time = time.time() - start_time
            self._update_metrics(False, processing_time)
            
            # Update active request
            if request_id in self.active_requests:
                self.active_requests[request_id]["status"] = "failed"
                self.active_requests[request_id]["error"] = str(e)
                self.active_requests[request_id]["end_time"] = time.time()
            
            # End performance monitoring
            self.performance_monitor.end_operation(f"video_generation_vision_{request_id}")
            
            self.logger.error(f"Video with vision generation failed: {request_id} - {e}")
            raise
    
    def _update_metrics(self, success: bool, processing_time: float):
        """Update system metrics."""
        self.metrics.total_requests += 1
        
        if success:
            self.metrics.successful_requests += 1
        else:
            self.metrics.failed_requests += 1
        
        # Update processing time metrics
        self.request_times.append(processing_time)
        if len(self.request_times) > self.max_request_history:
            self.request_times.pop(0)
        
        if self.request_times:
            self.metrics.avg_processing_time = sum(self.request_times) / len(self.request_times)
            self.metrics.min_processing_time = min(self.request_times)
            self.metrics.max_processing_time = max(self.request_times)
        
        # Update period end
        self.metrics.period_end = datetime.now()
    
    async def get_system_status(self) -> SystemStatus:
        """Get system status."""
        try:
            # Get system metrics
            system_metrics = self.performance_monitor.get_system_metrics()
            
            # Calculate error rate
            total_requests = self.metrics.total_requests
            error_rate = (self.metrics.failed_requests / total_requests * 100) if total_requests > 0 else 0
            
            # Get uptime
            uptime = (datetime.now() - self.metrics.period_start).total_seconds()
            
            # Get component status
            components = {
                "onyx_integration": {
                    "status": "active" if self.onyx_integration else "inactive",
                    "available": bool(self.onyx_integration)
                },
                "video_workflow": {
                    "status": "active" if self.video_workflow else "inactive",
                    "available": bool(self.video_workflow)
                },
                "plugin_manager": {
                    "status": "active" if self.plugin_manager else "inactive",
                    "available": bool(self.plugin_manager),
                    "total_plugins": len(self.plugin_manager.get_plugins()) if self.plugin_manager else 0,
                    "active_plugins": len([p for p in self.plugin_manager.get_plugins().values() if p.status == "active"]) if self.plugin_manager else 0
                },
                "security_manager": {
                    "status": "active" if self.security_manager else "inactive",
                    "available": bool(self.security_manager)
                },
                "performance_monitor": {
                    "status": "active" if self.performance_monitor else "inactive",
                    "available": bool(self.performance_monitor)
                }
            }
            
            status = SystemStatus(
                status="running" if self.initialized and not self.shutdown_requested else "stopped",
                version=self.config.system.version if self.config else "unknown",
                components=components,
                uptime=uptime,
                request_count=total_requests,
                error_count=self.metrics.failed_requests,
                error_rate=error_rate,
                cpu_usage=system_metrics.get("cpu_percent"),
                memory_usage=system_metrics.get("memory_percent"),
                gpu_usage=system_metrics.get("gpu_usage"),
                total_plugins=components["plugin_manager"]["total_plugins"],
                active_plugins=components["plugin_manager"]["active_plugins"],
                timestamp=datetime.now()
            )
            
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            raise AIVideoError(f"Status check failed: {e}")
    
    async def get_metrics(self) -> PerformanceMetrics:
        """Get performance metrics."""
        try:
            # Get plugin metrics
            plugin_executions = {}
            plugin_errors = {}
            
            if self.plugin_manager:
                plugins = self.plugin_manager.get_plugins()
                for plugin_name, plugin_info in plugins.items():
                    plugin_executions[plugin_name] = plugin_info.execution_count
                    plugin_errors[plugin_name] = plugin_info.error_count
            
            # Get cache metrics
            cache_hits = 0
            cache_misses = 0
            cache_size = 0
            
            if self.performance_monitor:
                cache_metrics = self.performance_monitor.get_performance_summary()
                cache_hits = cache_metrics.get("cache_hits", 0)
                cache_misses = cache_metrics.get("cache_misses", 0)
                cache_size = cache_metrics.get("cache_size", 0)
            
            # Get system metrics
            system_metrics = self.performance_monitor.get_system_metrics() if self.performance_monitor else {}
            
            metrics = PerformanceMetrics(
                total_requests=self.metrics.total_requests,
                successful_requests=self.metrics.successful_requests,
                failed_requests=self.metrics.failed_requests,
                avg_processing_time=self.metrics.avg_processing_time,
                min_processing_time=self.metrics.min_processing_time,
                max_processing_time=self.metrics.max_processing_time,
                plugin_executions=plugin_executions,
                plugin_errors=plugin_errors,
                memory_usage=system_metrics.get("memory_percent"),
                cpu_usage=system_metrics.get("cpu_percent"),
                gpu_usage=system_metrics.get("gpu_usage"),
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                cache_size=cache_size,
                period_start=self.metrics.period_start,
                period_end=self.metrics.period_end
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get metrics: {e}")
            raise AIVideoError(f"Metrics retrieval failed: {e}")
    
    async async def get_active_requests(self) -> Dict[str, Dict[str, Any]]:
        """Get active requests."""
        return self.active_requests.copy()
    
    async async def cancel_request(self, request_id: str, user_id: str) -> bool:
        """Cancel an active request."""
        try:
            if request_id not in self.active_requests:
                return False
            
            request_info = self.active_requests[request_id]
            if request_info["user_id"] != user_id:
                raise SecurityError("Cannot cancel another user's request")
            
            if request_info["status"] == "processing":
                request_info["status"] = "cancelled"
                request_info["end_time"] = time.time()
                
                # Cancel in workflow if possible
                if self.video_workflow:
                    await self.video_workflow.cancel_request(request_id)
                
                self.logger.info(f"Request cancelled: {request_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to cancel request: {e}")
            return False
    
    async def reload_config(self) -> None:
        """Reload system configuration."""
        try:
            self.logger.info("Reloading configuration")
            
            # Reload config
            self.config = self.config_manager.reload_config()
            
            # Update components with new config
            if self.video_workflow:
                await self.video_workflow.update_config(self.config)
            
            if self.plugin_manager:
                await self.plugin_manager.update_config(self.config)
            
            self.logger.info("Configuration reloaded successfully")
            
        except Exception as e:
            self.logger.error(f"Configuration reload failed: {e}")
            raise AIVideoError(f"Configuration reload failed: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown the system gracefully."""
        try:
            self.logger.info("Shutting down Onyx AI Video System")
            self.shutdown_requested = True
            
            # Cancel active requests
            for request_id in list(self.active_requests.keys()):
                if self.active_requests[request_id]["status"] == "processing":
                    await self.cancel_request(request_id, self.active_requests[request_id]["user_id"])
            
            # Shutdown components
            if self.video_workflow:
                await self.video_workflow.shutdown()
            
            if self.plugin_manager:
                await self.plugin_manager.shutdown()
            
            if self.performance_monitor:
                self.performance_monitor.stop_monitoring()
            
            if self.security_manager:
                self.security_manager.cleanup_expired_access()
            
            self.initialized = False
            self.logger.info("Onyx AI Video System shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Shutdown failed: {e}")
            raise AIVideoError(f"Shutdown failed: {e}")


# Global system instance
_system_instance: Optional[OnyxAIVideoSystem] = None


async def get_system(config_path: Optional[str] = None) -> OnyxAIVideoSystem:
    """Get global system instance."""
    global _system_instance
    
    if _system_instance is None:
        _system_instance = OnyxAIVideoSystem(config_path)
        await _system_instance.initialize()
    
    return _system_instance


async def shutdown_system() -> None:
    """Shutdown global system instance."""
    global _system_instance
    
    if _system_instance:
        await _system_instance.shutdown()
        _system_instance = None


# Convenience functions
async def generate_video(request: VideoRequest) -> VideoResponse:
    """Generate video using global system."""
    system = await get_system()
    return await system.generate_video(request)


async def generate_video_with_vision(request: VideoRequest, image_data: bytes) -> VideoResponse:
    """Generate video with vision using global system."""
    system = await get_system()
    return await system.generate_video_with_vision(request, image_data)


async def get_system_status() -> SystemStatus:
    """Get system status."""
    system = await get_system()
    return await system.get_system_status()


async def get_metrics() -> PerformanceMetrics:
    """Get system metrics."""
    system = await get_system()
    return await system.get_metrics()


async def reload_config() -> None:
    """Reload system configuration."""
    system = await get_system()
    await system.reload_config()


async async def cancel_request(request_id: str, user_id: str) -> bool:
    """Cancel a request."""
    system = await get_system()
    return await system.cancel_request(request_id, user_id)


async async def get_active_requests() -> Dict[str, Dict[str, Any]]:
    """Get active requests."""
    system = await get_system()
    return await system.get_active_requests()


# System management functions
async def initialize_system(config_path: Optional[str] = None) -> OnyxAIVideoSystem:
    """Initialize the system."""
    return await get_system(config_path)


async def restart_system(config_path: Optional[str] = None) -> OnyxAIVideoSystem:
    """Restart the system."""
    await shutdown_system()
    return await get_system(config_path)


def is_system_initialized() -> bool:
    """Check if system is initialized."""
    return _system_instance is not None and _system_instance.initialized


def get_system_info() -> Dict[str, Any]:
    """Get system information."""
    if not _system_instance:
        return {"status": "not_initialized"}
    
    return {
        "status": "initialized" if _system_instance.initialized else "not_initialized",
        "config_path": _system_instance.config_path,
        "shutdown_requested": _system_instance.shutdown_requested,
        "active_requests_count": len(_system_instance.active_requests)
    } 