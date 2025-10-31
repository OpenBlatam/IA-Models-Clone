"""
Edge Computing Package
======================

Edge computing and fog computing capabilities for distributed processing.
"""

from .manager import EdgeManager, FogManager, EdgeNodeManager
from .nodes import EdgeNode, FogNode, IoTDevice, EdgeCluster
from .orchestration import EdgeOrchestrator, TaskScheduler, ResourceManager
from .types import (
    NodeType, ComputeTier, TaskType, ResourceType, 
    EdgeTask, EdgeResource, NetworkTopology
)

__all__ = [
    "EdgeManager",
    "FogManager", 
    "EdgeNodeManager",
    "EdgeNode",
    "FogNode",
    "IoTDevice",
    "EdgeCluster",
    "EdgeOrchestrator",
    "TaskScheduler",
    "ResourceManager",
    "NodeType",
    "ComputeTier",
    "TaskType",
    "ResourceType",
    "EdgeTask",
    "EdgeResource",
    "NetworkTopology"
]
