"""
Utility Functions - Helper functions for common operations
Provides convenient shortcuts for layered architecture
"""

from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
import logging

from .model_layer import ModelBuilder, ModelRegistry
from .inference_layer import InferenceEngine
from .data_layer import DataPipeline, NormalizationProcessor
from .service_layer import ServiceContainer
from .integration import WorkflowBuilder

logger = logging.getLogger(__name__)


# ============================================================================
# Model Utilities
# ============================================================================

def quick_model(
    model_type: str,
    config: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
    use_mixed_precision: bool = True
) -> nn.Module:
    """Quick model creation"""
    builder = ModelBuilder()
    if config:
        builder.with_config(**config)
    if device:
        builder.with_device(device)
    builder.with_mixed_precision(use_mixed_precision)
    return builder.build(model_type)


def quick_inference_engine(
    model: nn.Module,
    device: Optional[torch.device] = None,
    use_mixed_precision: bool = True
) -> InferenceEngine:
    """Quick inference engine creation"""
    return InferenceEngine(model, device, use_mixed_precision)


# ============================================================================
# Data Utilities
# ============================================================================

def quick_data_pipeline(
    normalize: bool = True,
    pad: bool = False,
    max_length: Optional[int] = None
) -> DataPipeline:
    """Quick data pipeline creation"""
    pipeline = DataPipeline()
    
    if normalize:
        pipeline.add_processor(NormalizationProcessor())
    
    if pad and max_length:
        from .data_layer import PaddingProcessor
        pipeline.add_processor(PaddingProcessor(max_length=max_length))
    
    return pipeline


# ============================================================================
# Service Utilities
# ============================================================================

def quick_service_container() -> ServiceContainer:
    """Quick service container creation"""
    return ServiceContainer()


# ============================================================================
# Workflow Utilities
# ============================================================================

def quick_workflow(
    model_type: str,
    model_config: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None
) -> 'WorkflowBuilder':
    """Quick workflow builder creation"""
    builder = WorkflowBuilder()
    builder.with_model(model_type, model_config, device)
    return builder


# ============================================================================
# Device Utilities
# ============================================================================

def get_optimal_device() -> torch.device:
    """Get optimal device (CUDA if available, else CPU)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_mixed_precision(device: torch.device) -> bool:
    """Check if mixed precision should be used"""
    return device.type == "cuda"


# ============================================================================
# Configuration Utilities
# ============================================================================

def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple configuration dictionaries"""
    result = {}
    for config in configs:
        if config:
            result.update(config)
    return result


def validate_config(config: Dict[str, Any], required_keys: List[str]) -> bool:
    """Validate configuration has required keys"""
    return all(key in config for key in required_keys)


# Export main components
__all__ = [
    "quick_model",
    "quick_inference_engine",
    "quick_data_pipeline",
    "quick_service_container",
    "quick_workflow",
    "get_optimal_device",
    "setup_mixed_precision",
    "merge_configs",
    "validate_config",
]



