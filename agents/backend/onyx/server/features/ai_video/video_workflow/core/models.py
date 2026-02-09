from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
from onyx.llm.interfaces import LLM
from ...models import VideoRequest, VideoResponse
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Onyx Video Workflow - Core Models

Data models and structures for the Onyx video workflow system.
"""




@dataclass
class OnyxVideoStep:
    """Represents a step in the Onyx video workflow."""
    name: str
    description: str
    llm_prompt: str
    required: bool = True
    timeout: int = 60
    retry_attempts: int = 3
    dependencies: List[str] = field(default_factory=list)


@dataclass
class OnyxVideoContext:
    """Context for Onyx video generation."""
    request: VideoRequest
    llm: Optional[LLM] = None
    vision_llm: Optional[LLM] = None
    gpu_available: bool = False
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowExecutionResult:
    """Result of workflow execution."""
    success: bool
    results: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time: float = 0.0
    steps_completed: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class StepExecutionResult:
    """Result of individual step execution."""
    step_name: str
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class WorkflowStatus:
    """Status of the video workflow."""
    workflow: str = "onyx_video"
    steps: List[str] = field(default_factory=list)
    gpu_available: bool = False
    gpu_info: Optional[Dict[str, Any]] = None
    cache_size: int = 0
    onyx_integration: bool = True
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class GeneratorStatus:
    """Status of the video generator."""
    generator: str = "onyx_video"
    workflow_status: Optional[WorkflowStatus] = None
    telemetry_enabled: bool = True
    timestamp: datetime = field(default_factory=datetime.now) 