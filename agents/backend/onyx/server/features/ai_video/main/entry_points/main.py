from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
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
import orjson
from functools import lru_cache
from .core import (
from .config import ConfigManager, load_configuration
from .integrated_workflow import IntegratedWorkflow
from .video_workflow import VideoWorkflow
from .models import VideoRequest, VideoResponse, PluginConfig
from .plugins import PluginManager
from .state_repository import StateRepository
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
AI Video System - Main Entry Point

Production-ready main entry point for the AI Video system with comprehensive
integration of all core modules, advanced error handling, monitoring, and
enterprise-grade features.
"""


# Core imports
    # Core utilities
    AIVideoError, ConfigurationError, ValidationError, WorkflowError,
    SYSTEM_NAME, VERSION, DEFAULT_CONFIG_PATH, DEFAULT_LOG_LEVEL,
    
    # Performance
    performance_monitor, task_manager, default_cache,
    cleanup_performance_resources,
    
    # Security
    security_config, input_validator, encryption_manager,
    session_manager, security_auditor,
    cleanup_security_resources,
    
    # Async
    cleanup_async_resources,
    
    # Monitoring
    metrics_collector, health_checker, alert_manager,
    monitoring_dashboard, start_monitoring,
    cleanup_monitoring_resources,
    
    # Validation
    schema_validator, data_validator,
    
    # Logging
    main_logger, performance_logger, security_logger,
    setup_logging, cleanup_logging_resources,
    
    # Utilities
    ensure_directory, get_system_info, validate_json,
    measure_time, record_metric, log_event
)

# System imports


@lru_cache(maxsize=32)
def get_config_cached(config_path: str):
    
    """get_config_cached function."""
with open(config_path, 'rb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        return orjson.loads(f.read())
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")


class AIVideoSystem:
    """
    Main AI Video System class.
    
    Provides a comprehensive, production-ready interface for AI video generation
    with advanced features including monitoring, security, performance optimization,
    and enterprise-grade error handling.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the AI Video System."""
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.config_manager = None
        self.plugin_manager = None
        self.workflow = None
        self.state_repository = None
        self.is_initialized = False
        self.is_shutting_down = False
        
        # Performance tracking
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        
        # Setup logging
        setup_logging()
        main_logger.info(f"Initializing {SYSTEM_NAME} v{VERSION}")
    
    async def initialize(self) -> None:
        """Initialize the system components."""
        try:
            main_logger.info("Starting system initialization...")
            
            # Load configuration
            config_data = get_config_cached(self.config_path)
            self.config_manager = ConfigManager(config_data)
            
            # Initialize components
            await self._initialize_components()
            
            # Start monitoring
            await self._start_monitoring()
            
            # Validate system
            await self._validate_system()
            
            self.is_initialized = True
            main_logger.info("System initialization completed successfully")
            
            # Record initialization metrics
            initialization_time = time.time() - self.start_time
            record_metric("system_initialization_time", initialization_time)
            log_event("system_initialized", {
                "initialization_time": initialization_time,
                "version": VERSION
            })
            
        except Exception as e:
            main_logger.error(f"System initialization failed: {e}", exc_info=True)
            await self._handle_initialization_error(e)
            raise
    
    async def _initialize_components(self) -> None:
        """Initialize system components."""
        main_logger.info("Initializing system components...")
        
        try:
            # Initialize plugin manager
            self.plugin_manager = PluginManager(self.config_manager)
            await self.plugin_manager.initialize()
            
            # Initialize state repository
            self.state_repository = StateRepository(self.config_manager)
            await self.state_repository.initialize()
            
            # Initialize workflow
            self.workflow = IntegratedWorkflow(
                self.config_manager,
                self.plugin_manager,
                self.state_repository
            )
            await self.workflow.initialize()
            
            main_logger.info("System components initialized successfully")
            
        except Exception as e:
            main_logger.error(f"Component initialization failed: {e}")
            raise
    
    async def _start_monitoring(self) -> None:
        """Start monitoring services."""
        main_logger.info("Starting monitoring services...")
        
        try:
            # Start health monitoring
            await health_checker.start_monitoring()
            
            # Start metrics collection
            await start_monitoring()
            
            # Start cache cleanup
            asyncio.create_task(self._cache_cleanup_loop())
            
            main_logger.info("Monitoring services started successfully")
            
        except Exception as e:
            main_logger.error(f"Monitoring startup failed: {e}")
            raise
    
    async def _validate_system(self) -> None:
        """Validate system health and readiness."""
        main_logger.info("Validating system...")
        
        try:
            # Run health checks
            health_results = await health_checker.run_all_health_checks()
            
            # Check for critical failures
            critical_failures = [
                name for name, result in health_results.items()
                if result.status == 'unhealthy'
            ]
            
            if critical_failures:
                raise ValidationError(f"Critical health check failures: {critical_failures}")
            
            # Validate plugins
            plugin_status = await self.plugin_manager.get_status()
            if not plugin_status['ready']:
                raise ValidationError(f"Plugin system not ready: {plugin_status['errors']}")
            
            # Validate workflow
            workflow_status = await self.workflow.get_status()
            if not workflow_status['ready']:
                raise ValidationError(f"Workflow not ready: {workflow_status['errors']}")
            
            main_logger.info("System validation completed successfully")
            
        except Exception as e:
            main_logger.error(f"System validation failed: {e}")
            raise
    
    async def _handle_initialization_error(self, error: Exception) -> None:
        """Handle initialization errors."""
        # Create critical alert
        alert_manager.create_alert(
            severity='critical',
            title='System Initialization Failed',
            message=str(error),
            source='system_initialization',
            metadata={'error_type': type(error).__name__, 'error': str(error)}
        )
        
        # Log security event
        security_logger.log_security_threat(
            'system_initialization_failure',
            {'error': str(error), 'error_type': type(error).__name__},
            'critical'
        )
    
    async def generate_video(self, request: VideoRequest) -> VideoResponse:
        """
        Generate video using the AI Video system with early returns.
        
        Args:
            request: Video generation request
            
        Returns:
            VideoResponse with generation results
            
        Raises:
            AIVideoError: If video generation fails
            ValidationError: If request validation fails
            SecurityError: If security checks fail
        """
        start_time = time.time()
        self.request_count += 1
        
        # Early validation and error handling
        if not self.is_initialized:
            main_logger.error("System not initialized")
            raise AIVideoError("System not initialized")
        
        if self.is_shutting_down:
            main_logger.error("System is shutting down")
            raise AIVideoError("System is shutting down")
        
        if not request or not request.request_id:
            main_logger.error("Invalid request provided")
            raise ValidationError("Invalid request: missing request_id")
        
        # Log request
        main_logger.info(f"Processing video generation request: {request.request_id}")
        log_event("video_generation_started", {
            "request_id": request.request_id,
            "user_id": request.user_id,
            "input_length": len(request.input_text)
        })
        
        try:
            # Security validation
            await self._validate_request_security(request)
            
            # Input validation
            await self._validate_request_input(request)
            
            # Serialización rápida de metadata
            metadata_bytes = orjson.dumps(request.model_dump())
            
            # Ejemplo de uso de asyncio para tareas concurrentes
            await asyncio.gather(
                self.workflow.process_video(request),
                self._start_monitoring(),
            )
            
            # Record success metrics
            duration = time.time() - start_time
            record_metric("video_generation_duration", duration)
            record_metric("video_generation_success", 1)
            
            # Log success
            main_logger.info(f"Video generation completed: {request.request_id}")
            log_event("video_generation_completed", {
                "request_id": request.request_id,
                "duration": duration,
                "output_url": response.output_url
            })
            
            return response
            
        except Exception as e:
            # Record error metrics
            duration = time.time() - start_time
            self.error_count += 1
            record_metric("video_generation_duration", duration)
            record_metric("video_generation_error", 1)
            
            # Log error
            main_logger.error(f"Video generation failed: {request.request_id} - {e}", exc_info=True)
            log_event("video_generation_failed", {
                "request_id": request.request_id,
                "error": str(e),
                "error_type": type(e).__name__,
                "duration": duration
            })
            
            # Create alert for critical errors
            if isinstance(e, (SecurityError, ValidationError)):
                alert_manager.create_alert(
                    severity='warning',
                    title='Video Generation Error',
                    message=str(e),
                    source='video_generation',
                    metadata={
                        'request_id': request.request_id,
                        'error_type': type(e).__name__,
                        'error': str(e)
                    }
                )
            
            raise
    
    async async def _validate_request_security(self, request: VideoRequest) -> None:
        """Validate request security."""
        # Check rate limiting
        user_key = f"user_{request.user_id}"
        if not await task_manager.rate_limiter.acquire(user_key):
            raise SecurityError("Rate limit exceeded")
        
        # Validate input for security threats
        validation_result = input_validator.validate_string(
            request.input_text,
            max_length=security_config.max_input_length,
            allow_html=False
        )
        
        if not validation_result[0]:
            raise SecurityError(f"Input validation failed: {validation_result[1]}")
        
        # Check for suspicious patterns
        if input_validator._sanitize_html(request.input_text) != request.input_text:
            raise SecurityError("Suspicious content detected in input")
        
        # Log security event
        security_logger.log_data_access(
            request.user_id,
            "video_generation",
            "create",
            request.request_id
        )
    
    async async def _validate_request_input(self, request: VideoRequest) -> None:
        """Validate request input."""
        # Validate request schema
        request_data = {
            "input_text": request.input_text,
            "output_format": request.output_format,
            "duration": request.duration,
            "quality": request.quality,
            "plugins": request.plugins
        }
        
        validation_result = schema_validator.validate_data("video_generation", request_data)
        if not validation_result.is_valid:
            raise ValidationError(f"Request validation failed: {validation_result.errors}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status with early returns."""
        # Early return for uninitialized system
        if not self.is_initialized:
            return {
                "status": "not_initialized",
                "message": "System not initialized",
                "timestamp": datetime.now().isoformat()
            }
        
        # Early return for system in shutdown
        if self.is_shutting_down:
            return {
                "status": "shutting_down",
                "message": "System is shutting down",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            # Get component statuses
            plugin_status = await self.plugin_manager.get_status()
            workflow_status = await self.workflow.get_status()
            health_status = health_checker.get_overall_health()
            
            # Early return for unhealthy system
            if health_status['status'] != 'healthy':
                return {
                    "status": "degraded",
                    "message": health_status['message'],
                    "timestamp": datetime.now().isoformat(),
                    "version": VERSION,
                    "health_status": health_status
                }
            
            # Get performance metrics
            performance_stats = {
                "uptime": time.time() - self.start_time,
                "request_count": self.request_count,
                "error_count": self.error_count,
                "error_rate": self.error_count / max(self.request_count, 1),
                "system_metrics": metrics_collector.get_system_metrics()
            }
            
            # Get active alerts
            active_alerts = alert_manager.get_active_alerts()
            
            return {
                "status": "operational",
                "message": "System is operational",
                "timestamp": datetime.now().isoformat(),
                "version": VERSION,
                "components": {
                    "plugins": plugin_status,
                    "workflow": workflow_status,
                    "health": health_status
                },
                "performance": performance_stats,
                "alerts": {
                    "active_count": len(active_alerts),
                    "recent_alerts": [
                        {
                            "severity": alert.severity,
                            "title": alert.title,
                            "timestamp": alert.timestamp.isoformat()
                        }
                        for alert in active_alerts[:5]  # Last 5 alerts
                    ]
                }
            }
            
        except Exception as e:
            main_logger.error(f"Failed to get system status: {e}")
            return {
                "status": "error",
                "message": f"Failed to get status: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics."""
        return {
            "system_metrics": metrics_collector.get_system_metrics(),
            "operation_stats": {
                "video_generation": metrics_collector.get_metric_stats("video_generation_duration"),
                "plugin_execution": metrics_collector.get_metric_stats("plugin_execution_duration"),
                "file_processing": metrics_collector.get_metric_stats("file_processing_duration")
            },
            "performance_metrics": {
                "uptime": time.time() - self.start_time,
                "request_count": self.request_count,
                "error_count": self.error_count,
                "error_rate": self.error_count / max(self.request_count, 1)
            }
        }
    
    async def _cache_cleanup_loop(self) -> None:
        """Periodic cache cleanup loop."""
        while not self.is_shutting_down:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Cleanup caches
                clear_all_caches()
                
                # Cleanup task manager
                await task_manager.cleanup_completed_tasks()
                
                # Cleanup default cache
                await default_cache.cleanup_expired()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                main_logger.error(f"Cache cleanup error: {e}")
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the system."""
        if self.is_shutting_down:
            return
        
        main_logger.info("Starting system shutdown...")
        self.is_shutting_down = True
        
        try:
            # Stop accepting new requests
            main_logger.info("Stopping request processing...")
            
            # Cancel running tasks
            cancelled_count = await task_manager.cancel_all_tasks()
            main_logger.info(f"Cancelled {cancelled_count} running tasks")
            
            # Shutdown components
            if self.workflow:
                await self.workflow.shutdown()
            
            if self.plugin_manager:
                await self.plugin_manager.shutdown()
            
            if self.state_repository:
                await self.state_repository.shutdown()
            
            # Stop monitoring
            await health_checker.stop_monitoring()
            
            # Cleanup resources
            await cleanup_performance_resources()
            await cleanup_security_resources()
            await cleanup_async_resources()
            await cleanup_monitoring_resources()
            await cleanup_logging_resources()
            
            main_logger.info("System shutdown completed successfully")
            
        except Exception as e:
            main_logger.error(f"Shutdown error: {e}", exc_info=True)
            raise


# Global system instance
_system_instance: Optional[AIVideoSystem] = None


async def get_system() -> AIVideoSystem:
    """Get the global system instance."""
    global _system_instance
    
    if _system_instance is None:
        _system_instance = AIVideoSystem()
        await _system_instance.initialize()
    
    return _system_instance


async def shutdown_system() -> None:
    """Shutdown the global system instance."""
    global _system_instance
    
    if _system_instance:
        await _system_instance.shutdown()
        _system_instance = None


def setup_signal_handlers(system: AIVideoSystem) -> None:
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame) -> Any:
        main_logger.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(system.shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main() -> None:
    """Main entry point for the AI Video system."""
    parser = argparse.ArgumentParser(description=f"{SYSTEM_NAME} v{VERSION}")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--log-level",
        default=DEFAULT_LOG_LEVEL,
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
    
    args = parser.parse_args()
    
    try:
        # Initialize system
        system = AIVideoSystem(args.config)
        await system.initialize()
        
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
        
        # Keep system running
        main_logger.info(f"{SYSTEM_NAME} v{VERSION} is running...")
        
        # Run forever
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        main_logger.info("Received keyboard interrupt")
    except Exception as e:
        main_logger.error(f"System error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if 'system' in locals():
            await system.shutdown()


match __name__:
    case "__main__":
    asyncio.run(main()) 