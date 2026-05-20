from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import logging
import json
from typing import Optional, Dict, Any
from pathlib import Path
from ..plugins import ManagerConfig, ValidationLevel
from .core.workflow import IntegratedVideoWorkflow
from .core.models import PluginWorkflowState, WorkflowConfiguration
from typing import Any, List, Dict, Optional
"""
Integrated Workflow - Utilities

Utility functions and convenience methods for the integrated workflow system.
"""


# Import plugin system

# Import integrated workflow

logger = logging.getLogger(__name__)


async def create_integrated_workflow(
    plugin_config: Optional[ManagerConfig] = None,
    workflow_config: Optional[WorkflowConfiguration] = None
) -> IntegratedVideoWorkflow:
    """
    Create and initialize an integrated workflow.
    
    Args:
        plugin_config: Plugin manager configuration
        workflow_config: Workflow configuration
        
    Returns:
        Initialized IntegratedVideoWorkflow instance
    """
    workflow = IntegratedVideoWorkflow(plugin_config, workflow_config)
    success = await workflow.initialize()
    
    if not success:
        raise Exception("Failed to initialize integrated workflow")
    
    return workflow


async def quick_video_generation(
    url: str,
    avatar: Optional[str] = None,
    plugin_config: Optional[Dict[str, Any]] = None
) -> PluginWorkflowState:
    """
    Quick video generation with default settings.
    
    Args:
        url: Source URL
        avatar: Avatar to use
        plugin_config: Optional plugin configuration
        
    Returns:
        Workflow state with results
    """
    workflow = await create_integrated_workflow()
    
    try:
        result = await workflow.execute_workflow(url, avatar=avatar, plugin_config=plugin_config)
        return result
    finally:
        await workflow.shutdown()


def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        return {}


def save_config_to_file(config: Dict[str, Any], config_path: str):
    """
    Save configuration to a JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    try:
        with open(config_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(config, f, indent=2)
        logger.info(f"Configuration saved to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save configuration to {config_path}: {e}")


def create_default_plugin_config() -> ManagerConfig:
    """
    Create a default plugin configuration.
    
    Returns:
        Default ManagerConfig instance
    """
    return ManagerConfig(
        auto_discover=True,
        auto_load=True,
        validation_level=ValidationLevel.STANDARD,
        enable_events=True,
        enable_metrics=True
    )


def create_default_workflow_config() -> WorkflowConfiguration:
    """
    Create a default workflow configuration.
    
    Returns:
        Default WorkflowConfiguration instance
    """
    return WorkflowConfiguration(
        plugin_auto_discover=True,
        plugin_auto_load=True,
        plugin_validation_level="standard",
        enable_plugin_events=True,
        enable_plugin_metrics=True,
        max_extraction_retries=3,
        max_suggestion_retries=3,
        max_generation_retries=3,
        timeout_seconds=300,
        enable_caching=True,
        cache_ttl_seconds=3600,
        enable_parallel_processing=True,
        fail_fast=False,
        continue_on_plugin_error=True
    )


async def batch_video_generation(
    urls: list[str],
    avatar: Optional[str] = None,
    plugin_config: Optional[Dict[str, Any]] = None,
    max_concurrent: int = 3
) -> list[PluginWorkflowState]:
    """
    Generate videos for multiple URLs concurrently.
    
    Args:
        urls: List of source URLs
        avatar: Avatar to use for all videos
        plugin_config: Optional plugin configuration
        max_concurrent: Maximum number of concurrent workflows
        
    Returns:
        List of workflow states
    """
    workflow = await create_integrated_workflow()
    results = []
    
    try:
        # Process URLs in batches
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_url(url: str) -> PluginWorkflowState:
            async with semaphore:
                return await workflow.execute_workflow(url, avatar=avatar, plugin_config=plugin_config)
        
        # Create tasks for all URLs
        tasks = [process_url(url) for url in urls]
        
        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process URL {urls[i]}: {result}")
            else:
                valid_results.append(result)
        
        return valid_results
        
    finally:
        await workflow.shutdown()


def validate_workflow_state(state: PluginWorkflowState) -> bool:
    """
    Validate a workflow state.
    
    Args:
        state: Workflow state to validate
        
    Returns:
        True if state is valid
    """
    if not state.workflow_id:
        return False
    
    if not state.source_url:
        return False
    
    if state.status.value not in [status.value for status in state.status.__class__]:
        return False
    
    return True


def get_workflow_summary(state: PluginWorkflowState) -> Dict[str, Any]:
    """
    Get a summary of workflow state.
    
    Args:
        state: Workflow state
        
    Returns:
        Summary dictionary
    """
    return {
        'workflow_id': state.workflow_id,
        'status': state.status.value,
        'source_url': state.source_url,
        'total_time': state.total_time,
        'extraction_time': state.extraction_time,
        'suggestions_time': state.suggestions_time,
        'generation_time': state.generation_time,
        'video_url': state.video_url,
        'error': state.error,
        'error_stage': state.error_stage,
        'plugins_loaded': len(state.loaded_plugins),
        'plugins_active': len(state.active_plugins),
        'plugins_errors': len(state.plugin_errors)
    }


def format_duration(seconds: Optional[float]) -> str:
    """
    Format duration in seconds to human readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds is None:
        return "N/A"
    
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def create_workflow_report(states: list[PluginWorkflowState]) -> Dict[str, Any]:
    """
    Create a comprehensive report from multiple workflow states.
    
    Args:
        states: List of workflow states
        
    Returns:
        Report dictionary
    """
    if not states:
        return {"message": "No workflows to report"}
    
    total_workflows = len(states)
    successful_workflows = len([s for s in states if s.status.value == "completed"])
    failed_workflows = len([s for s in states if s.status.value == "failed"])
    
    total_time = sum(s.total_time or 0 for s in states)
    avg_time = total_time / total_workflows if total_workflows > 0 else 0
    
    extraction_times = [s.extraction_time for s in states if s.extraction_time]
    suggestions_times = [s.suggestions_time for s in states if s.suggestions_time]
    generation_times = [s.generation_time for s in states if s.generation_time]
    
    return {
        'summary': {
            'total_workflows': total_workflows,
            'successful_workflows': successful_workflows,
            'failed_workflows': failed_workflows,
            'success_rate': (successful_workflows / total_workflows * 100) if total_workflows > 0 else 0
        },
        'timing': {
            'total_time': format_duration(total_time),
            'average_time': format_duration(avg_time),
            'extraction_time': format_duration(sum(extraction_times) / len(extraction_times)) if extraction_times else "N/A",
            'suggestions_time': format_duration(sum(suggestions_times) / len(suggestions_times)) if suggestions_times else "N/A",
            'generation_time': format_duration(sum(generation_times) / len(generation_times)) if generation_times else "N/A"
        },
        'errors': {
            'total_errors': len([s for s in states if s.error]),
            'error_stages': list(set(s.error_stage for s in states if s.error_stage))
        },
        'plugins': {
            'total_plugins_loaded': sum(len(s.loaded_plugins) for s in states),
            'total_plugins_active': sum(len(s.active_plugins) for s in states),
            'total_plugin_errors': sum(len(s.plugin_errors) for s in states)
        }
    } 