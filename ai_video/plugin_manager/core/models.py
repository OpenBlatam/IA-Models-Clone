from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
from onyx.llm.interfaces import LLM
from ..models import VideoRequest
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Onyx Plugin Manager - Core Models

Data models and structures for the Onyx plugin management system.
"""




@dataclass
class OnyxPluginInfo:
    """Information about an Onyx plugin."""
    name: str
    version: str
    description: str
    author: str
    category: str
    enabled: bool = True
    priority: int = 0
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    gpu_required: bool = False
    timeout: int = 60
    max_workers: int = 1


@dataclass
class OnyxPluginContext:
    """Context for plugin execution."""
    request: VideoRequest
    llm: Optional[LLM] = None
    gpu_available: bool = False
    shared_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PluginStatus:
    """Status information for a plugin."""
    loaded: bool = False
    initialized: bool = False
    enabled: bool = True
    error: Optional[str] = None
    last_execution: Optional[datetime] = None
    execution_count: int = 0
    total_execution_time: float = 0.0


@dataclass
class PluginExecutionResult:
    """Result of plugin execution."""
    plugin_name: str
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PluginManagerStatus:
    """Status of the plugin manager."""
    manager: str = "onyx_plugin_manager"
    total_plugins: int = 0
    enabled_plugins: int = 0
    initialized_plugins: int = 0
    plugin_directories: List[str] = field(default_factory=list)
    gpu_available: bool = False
    gpu_info: Optional[Dict[str, Any]] = None
    plugins: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now) 