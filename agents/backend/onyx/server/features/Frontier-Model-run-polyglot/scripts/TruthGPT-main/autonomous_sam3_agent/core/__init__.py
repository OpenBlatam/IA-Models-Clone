"""
Core components for Autonomous SAM3 Agent
==========================================

Enhanced with dynamic scaling, health monitoring, and extensibility.
"""

from .agent_core import AutonomousSAM3Agent
from .task_manager import TaskManager
from .parallel_executor import ParallelExecutor
from .validators import (
    ValidationError,
    ImageValidator,
    PromptValidator,
    TaskValidator,
)

# New components for 24/7 operation
from .auto_scaler import AutoScaler, ScalingConfig
from .health_monitor import HealthMonitor, HealthConfig, HealthStatus
from .task_scheduler import TaskScheduler, ScheduledTask
from .plugin_system import PluginManager, Plugin, PluginInfo, HookType
from .distributed_coordinator import DistributedCoordinator, NodeInfo, NodeStatus

__all__ = [
    # Core agent
    "AutonomousSAM3Agent",
    "TaskManager",
    "ParallelExecutor",
    
    # Validation
    "ValidationError",
    "ImageValidator",
    "PromptValidator",
    "TaskValidator",
    
    # Auto-scaling
    "AutoScaler",
    "ScalingConfig",
    
    # Health monitoring
    "HealthMonitor",
    "HealthConfig",
    "HealthStatus",
    
    # Scheduling
    "TaskScheduler",
    "ScheduledTask",
    
    # Plugins
    "PluginManager",
    "Plugin",
    "PluginInfo",
    "HookType",
    
    # Distributed
    "DistributedCoordinator",
    "NodeInfo",
    "NodeStatus",
]
