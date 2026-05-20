"""
Logging utilities for TruthGPT Optimization Core
Provides comprehensive logging setup and utilities
"""

import logging
import sys
import time
import torch
from typing import Optional, Dict, Any, Union
from pathlib import Path
from datetime import datetime

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    include_timestamp: bool = True
) -> logging.Logger:
    """
    Setup comprehensive logging for the optimization core.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        log_format: Custom log format
        include_timestamp: Whether to include timestamp in logs
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger("truthgpt_optimization_core")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    if log_format is None:
        if include_timestamp:
            log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        else:
            log_format = "%(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"truthgpt_optimization_core.{name}")

def log_performance_metrics(
    logger: logging.Logger,
    metrics: Dict[str, Any],
    prefix: str = "Performance"
) -> None:
    """
    Log performance metrics in a structured format.
    
    Args:
        logger: Logger instance
        metrics: Dictionary of performance metrics
        prefix: Prefix for log messages
    """
    logger.info(f"{prefix} Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")

def log_model_info(
    logger: logging.Logger,
    model: torch.nn.Module,
    input_shape: Optional[tuple] = None
) -> None:
    """
    Log comprehensive model information.
    
    Args:
        logger: Logger instance
        model: PyTorch model
        input_shape: Optional input shape for parameter counting
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model Information:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Model size estimation
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / 1024 / 1024
    
    logger.info(f"  Model size: {model_size_mb:.2f} MB")
    
    # Model architecture
    logger.info(f"  Model class: {model.__class__.__name__}")
    
    # Input shape (if provided)
    if input_shape:
        logger.info(f"  Input shape: {input_shape}")

def log_training_progress(
    logger: logging.Logger,
    epoch: int,
    step: int,
    loss: float,
    learning_rate: float,
    additional_metrics: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log training progress in a structured format.
    
    Args:
        logger: Logger instance
        epoch: Current epoch
        step: Current step
        loss: Current loss value
        learning_rate: Current learning rate
        additional_metrics: Optional additional metrics
    """
    logger.info(f"Epoch {epoch}, Step {step}:")
    logger.info(f"  Loss: {loss:.6f}")
    logger.info(f"  Learning Rate: {learning_rate:.2e}")
    
    if additional_metrics:
        for key, value in additional_metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.6f}")
            else:
                logger.info(f"  {key}: {value}")

def log_optimization_results(
    logger: logging.Logger,
    results: Dict[str, Any],
    optimization_type: str = "Optimization"
) -> None:
    """
    Log optimization results in a structured format.
    
    Args:
        logger: Logger instance
        results: Dictionary of optimization results
        optimization_type: Type of optimization performed
    """
    logger.info(f"{optimization_type} Results:")
    
    for key, value in results.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.6f}")
        elif isinstance(value, dict):
            logger.info(f"  {key}:")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, float):
                    logger.info(f"    {sub_key}: {sub_value:.6f}")
                else:
                    logger.info(f"    {sub_key}: {sub_value}")
        else:
            logger.info(f"  {key}: {value}")

def log_error_with_context(
    logger: logging.Logger,
    error: Exception,
    context: Dict[str, Any],
    level: str = "ERROR"
) -> None:
    """
    Log error with additional context information.
    
    Args:
        logger: Logger instance
        error: Exception that occurred
        context: Additional context information
        level: Logging level
    """
    log_level = getattr(logging, level.upper())
    
    logger.log(log_level, f"Error occurred: {str(error)}")
    logger.log(log_level, f"Error type: {type(error).__name__}")
    
    if context:
        logger.log(log_level, "Context information:")
        for key, value in context.items():
            logger.log(log_level, f"  {key}: {value}")

def log_memory_usage(
    logger: logging.Logger,
    device: Optional[torch.device] = None
) -> None:
    """
    Log current memory usage.
    
    Args:
        logger: Logger instance
        device: Device to check memory for (default: current device)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / 1024 / 1024
        reserved = torch.cuda.memory_reserved(device) / 1024 / 1024
        logger.info(f"GPU Memory Usage ({device}):")
        logger.info(f"  Allocated: {allocated:.2f} MB")
        logger.info(f"  Reserved: {reserved:.2f} MB")
    else:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        logger.info(f"CPU Memory Usage:")
        logger.info(f"  RSS: {memory_mb:.2f} MB")

def log_configuration(
    logger: logging.Logger,
    config: Dict[str, Any],
    config_name: str = "Configuration"
) -> None:
    """
    Log configuration in a structured format.
    
    Args:
        logger: Logger instance
        config: Configuration dictionary
        config_name: Name of the configuration
    """
    logger.info(f"{config_name}:")
    
    def log_dict(d: Dict[str, Any], indent: int = 0) -> None:
        for key, value in d.items():
            if isinstance(value, dict):
                logger.info("  " * indent + f"{key}:")
                log_dict(value, indent + 1)
            else:
                logger.info("  " * indent + f"{key}: {value}")
    
    log_dict(config)

def create_log_file_path(
    base_dir: str = "logs",
    prefix: str = "truthgpt",
    include_timestamp: bool = True
) -> str:
    """
    Create a log file path with timestamp.
    
    Args:
        base_dir: Base directory for logs
        prefix: Prefix for log file
        include_timestamp: Whether to include timestamp
        
    Returns:
        Log file path
    """
    base_path = Path(base_dir)
    base_path.mkdir(exist_ok=True)
    
    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.log"
    else:
        filename = f"{prefix}.log"
    
    return str(base_path / filename)

def setup_experiment_logging(
    experiment_name: str,
    base_dir: str = "logs",
    level: str = "INFO"
) -> logging.Logger:
    """
    Setup logging for an experiment.
    
    Args:
        experiment_name: Name of the experiment
        base_dir: Base directory for logs
        level: Logging level
        
    Returns:
        Configured logger
    """
    log_file = create_log_file_path(base_dir, experiment_name)
    return setup_logging(level, log_file)