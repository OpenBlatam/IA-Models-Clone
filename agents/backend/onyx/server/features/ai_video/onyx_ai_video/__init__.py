from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from .core.integration import OnyxIntegrationManager, onyx_integration
from .core.exceptions import AIVideoError, PluginError, ValidationError
from .core.models import VideoRequest, VideoResponse, PluginConfig
from .workflows.video_workflow import OnyxVideoWorkflow, onyx_video_generator
from .workflows.workflow_manager import WorkflowManager
from .plugins.plugin_manager import OnyxPluginManager, onyx_plugin_manager
from .plugins.plugin_base import OnyxPluginBase, OnyxPluginContext
from .config.config_manager import OnyxConfigManager, get_config, save_config
from .config.settings import OnyxAIVideoConfig
from .utils.logger import OnyxLogger
from .utils.performance import PerformanceMonitor
from .utils.security import SecurityManager
from .api.main import OnyxAIVideoSystem, get_system, shutdown_system
from .api.endpoints import create_api_router
from .cli.main import OnyxAIVideoCLI, main as cli_main
    import asyncio
    import asyncio
from typing import Any, List, Dict, Optional
import logging
"""
Onyx AI Video System - Modular Package

A comprehensive, modular AI video generation system fully integrated with Onyx.
Provides video generation, plugin management, workflow orchestration, and more.
"""

__version__ = "1.0.0"
__author__ = "Onyx Team"
__description__ = "Modular AI Video Generation System for Onyx"

# Core imports

# Workflow imports

# Plugin imports

# Configuration imports

# Utility imports

# API imports

# CLI imports

# Main system
__all__ = [
    # Core
    "OnyxIntegrationManager",
    "onyx_integration",
    "AIVideoError",
    "PluginError", 
    "ValidationError",
    "VideoRequest",
    "VideoResponse",
    "PluginConfig",
    
    # Workflows
    "OnyxVideoWorkflow",
    "onyx_video_generator",
    "WorkflowManager",
    
    # Plugins
    "OnyxPluginManager",
    "onyx_plugin_manager",
    "OnyxPluginBase",
    "OnyxPluginContext",
    
    # Configuration
    "OnyxConfigManager",
    "get_config",
    "save_config",
    "OnyxAIVideoConfig",
    
    # Utilities
    "OnyxLogger",
    "PerformanceMonitor",
    "SecurityManager",
    
    # API
    "OnyxAIVideoSystem",
    "get_system",
    "shutdown_system",
    "create_api_router",
    
    # CLI
    "OnyxAIVideoCLI",
    "cli_main",
    
    # Version
    "__version__",
    "__author__",
    "__description__"
]

# Convenience functions
async def initialize_system(config_path: str = None) -> OnyxAIVideoSystem:
    """Initialize the Onyx AI Video system."""
    system = await get_system()
    return system

async def generate_video(request: VideoRequest) -> VideoResponse:
    """Generate video using the Onyx AI Video system."""
    system = await get_system()
    return await system.generate_video(request)

async def generate_video_with_vision(request: VideoRequest, image_data: bytes) -> VideoResponse:
    """Generate video with vision capabilities."""
    system = await get_system()
    return await system.generate_video_with_vision(request, image_data)

def get_system_status() -> dict:
    """Get system status."""
    return asyncio.run(get_system().get_system_status())

def get_system_metrics() -> dict:
    """Get system metrics."""
    return asyncio.run(get_system().get_metrics())

# Quick start function
async def quick_start(input_text: str, user_id: str = "default_user") -> VideoResponse:
    """
    Quick start function for simple video generation.
    
    Args:
        input_text: Text to generate video from
        user_id: User identifier
        
    Returns:
        VideoResponse with generated video information
    """
    request = VideoRequest(
        input_text=input_text,
        user_id=user_id,
        request_id=f"quick_{int(time.time())}"
    )
    
    return await generate_video(request)

# Export convenience functions
__all__.extend([
    "initialize_system",
    "generate_video", 
    "generate_video_with_vision",
    "get_system_status",
    "get_system_metrics",
    "quick_start"
]) 