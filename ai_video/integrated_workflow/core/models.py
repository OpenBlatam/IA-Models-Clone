from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from ...video_workflow import VideoWorkflow, WorkflowState, WorkflowStatus, WorkflowHooks
from ...web_extract import WebContentExtractor, ExtractedContent
from ...suggestions import ContentSuggestions, SuggestionEngine
from ...video_generator import VideoGenerator, VideoGenerationResult
from ...state_repository import StateRepository, FileStateRepository
from ...models import AIVideo
from ...plugins import BasePlugin
from typing import Any, List, Dict, Optional
import asyncio
"""
Integrated Workflow - Core Models

Data models and structures for the integrated AI video workflow system.
"""


# Import existing components

# Import plugin system

logger = logging.getLogger(__name__)


class IntegratedWorkflowStatus(Enum):
    """Enhanced workflow status with plugin management."""
    INITIALIZING = "initializing"
    PLUGINS_LOADING = "plugins_loading"
    PLUGINS_READY = "plugins_ready"
    EXTRACTING = "extracting"
    SUGGESTING = "suggesting"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PLUGIN_ERROR = "plugin_error"


@dataclass
class PluginWorkflowState:
    """Enhanced workflow state with plugin information."""
    workflow_id: str
    source_url: str
    status: IntegratedWorkflowStatus
    avatar: Optional[str] = None
    
    # Content and processing results
    content: Optional[ExtractedContent] = None
    suggestions: Optional[ContentSuggestions] = None
    video_url: Optional[str] = None
    ai_video: Optional[AIVideo] = None
    
    # Plugin information
    loaded_plugins: List[str] = field(default_factory=list)
    active_plugins: List[str] = field(default_factory=list)
    plugin_errors: Dict[str, str] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Performance tracking
    plugin_load_time: Optional[float] = None
    extraction_time: Optional[float] = None
    suggestions_time: Optional[float] = None
    generation_time: Optional[float] = None
    total_time: Optional[float] = None
    
    # Error handling
    error: Optional[str] = None
    error_stage: Optional[str] = None
    
    # User customizations
    user_edits: Dict[str, Any] = field(default_factory=dict)
    
    # Plugin configuration
    plugin_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegratedWorkflowHooks:
    """Enhanced hooks with plugin events."""
    # Original workflow hooks
    on_extraction_complete: Optional[Callable[[ExtractedContent], None]] = None
    on_suggestions_complete: Optional[Callable[[ContentSuggestions], None]] = None
    on_generation_complete: Optional[Callable[[VideoGenerationResult], None]] = None
    on_workflow_complete: Optional[Callable[[PluginWorkflowState], None]] = None
    on_workflow_failed: Optional[Callable[[PluginWorkflowState, Exception], None]] = None
    
    # Plugin-specific hooks
    on_plugin_loaded: Optional[Callable[[str, BasePlugin], None]] = None
    on_plugin_error: Optional[Callable[[str, Exception], None]] = None
    on_plugins_ready: Optional[Callable[[List[str]], None]] = None


@dataclass
class PluginCategory:
    """Plugin category information."""
    name: str
    description: str
    plugins: List[str] = field(default_factory=list)
    count: int = 0


@dataclass
class WorkflowConfiguration:
    """Configuration for the integrated workflow."""
    plugin_auto_discover: bool = True
    plugin_auto_load: bool = True
    plugin_validation_level: str = "standard"
    enable_plugin_events: bool = True
    enable_plugin_metrics: bool = True
    
    # Workflow settings
    max_extraction_retries: int = 3
    max_suggestion_retries: int = 3
    max_generation_retries: int = 3
    timeout_seconds: int = 300
    
    # Performance settings
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    enable_parallel_processing: bool = True
    
    # Error handling
    fail_fast: bool = False
    continue_on_plugin_error: bool = True


@dataclass
class WorkflowStatistics:
    """Statistics for the integrated workflow."""
    total_workflows: int = 0
    successful_workflows: int = 0
    failed_workflows: int = 0
    average_execution_time: float = 0.0
    
    # Plugin statistics
    total_plugins: int = 0
    active_plugins: int = 0
    plugin_errors: int = 0
    
    # Category statistics
    extractors_count: int = 0
    suggestion_engines_count: int = 0
    generators_count: int = 0
    
    # Performance metrics
    total_extraction_time: float = 0.0
    total_suggestions_time: float = 0.0
    total_generation_time: float = 0.0
    
    # Timestamps
    last_workflow_start: Optional[datetime] = None
    last_workflow_complete: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class HealthReport:
    """Health report for the integrated workflow system."""
    status: str = "unknown"
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Component health
    plugin_manager_healthy: bool = False
    workflow_initialized: bool = False
    state_repository_available: bool = False
    
    # Plugin health
    plugins_loaded: int = 0
    plugins_healthy: int = 0
    plugins_failed: int = 0
    
    # System resources
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    disk_usage_percent: Optional[float] = None
    
    # Error information
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list) 