from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from abc import ABC, abstractmethod
from typing import Protocol, Optional, Dict, Any, List
from datetime import datetime
from .types import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Interfaces Module - Abstract Protocol Definitions

This module defines the core interfaces for the modular AI video workflow system
using Python's Protocol classes for structural typing and better flexibility.
"""


    ExtractedContent,
    ContentSuggestions,
    VideoGenerationResult,
    WorkflowState
)


class ExtractorInterface(Protocol):
    """Interface for web content extractors."""
    
    name: str
    priority: int
    
    async def extract(self, url: str) -> Optional[ExtractedContent]:
        """Extract content from a URL."""
        ...
    
    def get_stats(self) -> Dict[str, Any]:
        """Get extractor performance statistics."""
        ...
    
    def is_available(self) -> bool:
        """Check if extractor is available for use."""
        ...


class SuggestionInterface(Protocol):
    """Interface for AI suggestion engines."""
    
    name: str
    
    async def generate_suggestions(
        self, 
        content: ExtractedContent,
        context: Optional[Dict[str, Any]] = None
    ) -> ContentSuggestions:
        """Generate AI suggestions for content."""
        ...
    
    def get_capabilities(self) -> List[str]:
        """Get list of capabilities this engine supports."""
        ...
    
    def is_available(self) -> bool:
        """Check if suggestion engine is available."""
        ...


class GeneratorInterface(Protocol):
    """Interface for video generators."""
    
    name: str
    
    async def generate_video(
        self,
        content: ExtractedContent,
        suggestions: ContentSuggestions,
        avatar: Optional[str] = None,
        user_edits: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> VideoGenerationResult:
        """Generate a video from content and suggestions."""
        ...
    
    def get_supported_avatars(self) -> List[str]:
        """Get list of supported avatars."""
        ...
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported output formats."""
        ...
    
    def is_available(self) -> bool:
        """Check if generator is available."""
        ...


class StateRepositoryInterface(Protocol):
    """Interface for workflow state persistence."""
    
    async def save(self, state: WorkflowState) -> None:
        """Save workflow state."""
        ...
    
    async def load(self, workflow_id: str) -> Optional[WorkflowState]:
        """Load workflow state by ID."""
        ...
    
    async def delete(self, workflow_id: str) -> bool:
        """Delete workflow state by ID."""
        ...
    
    async def list_workflows(
        self, 
        status: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[WorkflowState]:
        """List workflows with optional filtering."""
        ...
    
    async def cleanup_old_states(self, days: int) -> int:
        """Clean up states older than specified days."""
        ...


class MetricsInterface(Protocol):
    """Interface for metrics collection and reporting."""
    
    def record_extraction(
        self, 
        extractor_name: str, 
        success: bool, 
        duration: float, 
        domain: str = "unknown"
    ) -> None:
        """Record extraction metrics."""
        ...
    
    def record_generation(
        self, 
        generator_name: str, 
        success: bool, 
        duration: float, 
        quality_score: Optional[float] = None
    ) -> None:
        """Record generation metrics."""
        ...
    
    def record_workflow(
        self, 
        success: bool, 
        total_duration: float, 
        stage_timings: Dict[str, float]
    ) -> None:
        """Record workflow metrics."""
        ...
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        ...
    
    def get_realtime_stats(self) -> Dict[str, Any]:
        """Get real-time statistics."""
        ...
    
    async def cleanup_old_metrics(self, days: int) -> None:
        """Clean up old metrics data."""
        ...


class WorkflowInterface(Protocol):
    """Interface for workflow orchestration."""
    
    async def execute(
        self,
        url: str,
        workflow_id: str,
        avatar: Optional[str] = None,
        user_edits: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> WorkflowState:
        """Execute the complete workflow."""
        ...
    
    async def resume(self, workflow_id: str) -> WorkflowState:
        """Resume an interrupted workflow."""
        ...
    
    async def cancel(self, workflow_id: str) -> bool:
        """Cancel a running workflow."""
        ...
    
    def get_status(self, workflow_id: str) -> Optional[WorkflowState]:
        """Get current workflow status."""
        ...
    
    def get_available_components(self) -> Dict[str, List[str]]:
        """Get list of available components by type."""
        ...


class PluginInterface(Protocol):
    """Base interface for all plugins."""
    
    name: str
    version: str
    description: str
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the plugin."""
        ...
    
    def cleanup(self) -> None:
        """Clean up plugin resources."""
        ...
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Get configuration schema for the plugin."""
        ...
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate plugin configuration."""
        ...


class EventHandlerInterface(Protocol):
    """Interface for workflow event handlers."""
    
    async def on_extraction_start(self, url: str, workflow_id: str) -> None:
        """Called when extraction starts."""
        ...
    
    async def on_extraction_complete(
        self, 
        content: ExtractedContent, 
        workflow_id: str
    ) -> None:
        """Called when extraction completes."""
        ...
    
    async def on_suggestions_start(self, workflow_id: str) -> None:
        """Called when suggestions generation starts."""
        ...
    
    async def on_suggestions_complete(
        self, 
        suggestions: ContentSuggestions, 
        workflow_id: str
    ) -> None:
        """Called when suggestions generation completes."""
        ...
    
    async def on_generation_start(self, workflow_id: str) -> None:
        """Called when video generation starts."""
        ...
    
    async def on_generation_complete(
        self, 
        result: VideoGenerationResult, 
        workflow_id: str
    ) -> None:
        """Called when video generation completes."""
        ...
    
    async def on_workflow_complete(self, state: WorkflowState) -> None:
        """Called when workflow completes successfully."""
        ...
    
    async def on_workflow_failed(
        self, 
        state: WorkflowState, 
        error: Exception
    ) -> None:
        """Called when workflow fails."""
        ...


class ConfigProviderInterface(Protocol):
    """Interface for configuration providers."""
    
    def get_config(self, key: str, default: Any = None) -> Optional[Dict[str, Any]]:
        """Get configuration value."""
        ...
    
    def set_config(self, key: str, value: Any) -> None:
        """Set configuration value."""
        ...
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        ...
    
    def reload(self) -> None:
        """Reload configuration from source."""
        ...
    
    def validate(self) -> bool:
        """Validate current configuration."""
        ...


class LoggerInterface(Protocol):
    """Interface for logging systems."""
    
    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        ...
    
    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        ...
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        ...
    
    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        ...
    
    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message."""
        ...
    
    def set_level(self, level: str) -> None:
        """Set logging level."""
        ... 