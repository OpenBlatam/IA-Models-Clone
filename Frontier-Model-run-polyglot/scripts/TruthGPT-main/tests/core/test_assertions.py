"""
Custom Assertions
Enhanced assertion functions for tests
"""

import torch
import torch.nn as nn
from typing import Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def assert_model_equivalent(model1: nn.Module, model2: nn.Module, rtol: float = 1e-5, atol: float = 1e-8):
    """Assert two models are equivalent"""
    assert model1 is not None, "Model 1 is None"
    assert model2 is not None, "Model 2 is None"
    
    # Check parameter count
    params1 = sum(p.numel() for p in model1.parameters())
    params2 = sum(p.numel() for p in model2.parameters())
    assert params1 == params2, f"Parameter count mismatch: {params1} vs {params2}"
    
    # Check parameter values
    for (name1, param1), (name2, param2) in zip(
        model1.named_parameters(),
        model2.named_parameters()
    ):
        assert name1 == name2, f"Parameter name mismatch: {name1} vs {name2}"
        assert torch.allclose(param1, param2, rtol=rtol, atol=atol), \
            f"Parameter values differ for {name1}"

def assert_output_shape(output: torch.Tensor, expected_shape: Tuple[int, ...]):
    """Assert output has expected shape"""
    assert output is not None, "Output is None"
    assert isinstance(output, torch.Tensor), "Output is not a tensor"
    assert output.shape == expected_shape, \
        f"Shape mismatch: {output.shape} vs {expected_shape}"

def assert_gradients_exist(model: nn.Module):
    """Assert model has gradients"""
    has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_gradients, "Model has no gradients"

def assert_no_gradients(model: nn.Module):
    """Assert model has no gradients"""
    has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert not has_gradients, "Model has gradients when it shouldn't"

def assert_model_in_train_mode(model: nn.Module):
    """Assert model is in training mode"""
    assert model.training, "Model is not in training mode"

def assert_model_in_eval_mode(model: nn.Module):
    """Assert model is in evaluation mode"""
    assert not model.training, "Model is not in evaluation mode"

def assert_device_match(tensor: torch.Tensor, device: str):
    """Assert tensor is on specified device"""
    assert tensor.device.type == device, \
        f"Tensor is on {tensor.device.type}, expected {device}"

def assert_dtype_match(tensor: torch.Tensor, dtype: torch.dtype):
    """Assert tensor has specified dtype"""
    assert tensor.dtype == dtype, \
        f"Tensor dtype is {tensor.dtype}, expected {dtype}"

def assert_loss_decreasing(losses: list, tolerance: float = 0.0):
    """Assert losses are generally decreasing"""
    assert len(losses) > 1, "Need at least 2 losses to check decrease"
    
    decreasing_count = sum(1 for i in range(1, len(losses)) if losses[i] <= losses[i-1] + tolerance)
    decreasing_ratio = decreasing_count / (len(losses) - 1)
    
    assert decreasing_ratio > 0.5, \
        f"Losses are not decreasing (only {decreasing_ratio:.1%} decreasing)"

def assert_metrics_valid(metrics: dict, required_keys: list = None):
    """Assert metrics dictionary is valid"""
    assert metrics is not None, "Metrics is None"
    assert isinstance(metrics, dict), "Metrics is not a dictionary"
    
    if required_keys:
        for key in required_keys:
            assert key in metrics, f"Missing required key: {key}"
            assert metrics[key] is not None, f"Key {key} is None"

def assert_config_valid(config: Any, required_attrs: list):
    """Assert config has required attributes"""
    assert config is not None, "Config is None"
    for attr in required_attrs:
        assert hasattr(config, attr), f"Config missing attribute: {attr}"
        assert getattr(config, attr) is not None, f"Config attribute {attr} is None"

def assert_file_exists(filepath: str):
    """Assert file exists"""
    import os
    assert os.path.exists(filepath), f"File does not exist: {filepath}"

def assert_file_not_exists(filepath: str):
    """Assert file does not exist"""
    import os
    assert not os.path.exists(filepath), f"File exists when it shouldn't: {filepath}"

def assert_directory_exists(dirpath: str):
    """Assert directory exists"""
    import os
    assert os.path.isdir(dirpath), f"Directory does not exist: {dirpath}"

def assert_performance_acceptable(
    duration: float,
    max_duration: float,
    operation: str = "operation"
):
    """Assert performance is acceptable"""
    assert duration < max_duration, \
        f"{operation} took {duration:.2f}s, expected < {max_duration}s"

def assert_memory_usage_acceptable(
    usage_mb: float,
    max_mb: float,
    operation: str = "operation"
):
    """Assert memory usage is acceptable"""
    assert usage_mb < max_mb, \
        f"{operation} used {usage_mb:.2f}MB, expected < {max_mb}MB"








