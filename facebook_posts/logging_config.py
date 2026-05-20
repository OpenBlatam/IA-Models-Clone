#!/usr/bin/env python3
"""
Centralized Logging Configuration for Gradient Clipping & NaN Handling System
Provides comprehensive logging setup for training progress, errors, and system monitoring.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Any, Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        
        # Add color to message for certain levels
        if record.levelno >= logging.ERROR:
            record.msg = f"{self.COLORS['ERROR']}{record.msg}{self.COLORS['RESET']}"
        elif record.levelno >= logging.WARNING:
            record.msg = f"{self.COLORS['WARNING']}{record.msg}{self.COLORS['RESET']}"
        
        return super().format(record)


class TrainingProgressFilter(logging.Filter):
    """Filter for training progress logs."""
    
    def filter(self, record):
        return (
            'training' in record.getMessage().lower() or
            'step' in record.getMessage().lower() or
            'epoch' in record.getMessage().lower() or
            'loss' in record.getMessage().lower() or
            'accuracy' in record.getMessage().lower() or
            'gradient' in record.getMessage().lower() or
            'stability' in record.getMessage().lower()
        )


class ErrorFilter(logging.Filter):
    """Filter for error logs."""
    
    def filter(self, record):
        return record.levelno >= logging.WARNING


class NumericalStabilityFilter(logging.Filter):
    """Filter for numerical stability related logs."""
    
    def filter(self, record):
        message = record.getMessage().lower()
        return any(keyword in message for keyword in [
            'nan', 'inf', 'overflow', 'clipping', 'gradient', 'stability',
            'numerical', 'convergence', 'divergence'
        ])


def setup_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
    enable_file_logging: bool = True,
    enable_console_logging: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    enable_json_logging: bool = True
) -> Dict[str, logging.Logger]:
    """
    Set up comprehensive logging configuration.
    
    Args:
        log_dir: Directory to store log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_file_logging: Whether to enable file logging
        enable_console_logging: Whether to enable console logging
        max_file_size: Maximum size of log files before rotation
        backup_count: Number of backup log files to keep
        enable_json_logging: Whether to enable JSON formatted logging
    
    Returns:
        Dictionary of configured loggers
    """
    
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Set log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    colored_formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    json_formatter = logging.Formatter(
        '{"timestamp": "%(asctime)s", "logger": "%(name)s", "level": "%(levelname)s", '
        '"function": "%(funcName)s", "line": %(lineno)d, "message": "%(message)s"}'
    )
    
    # Create loggers
    loggers = {}
    
    # Main application logger
    main_logger = logging.getLogger('gradient_clipping_system')
    main_logger.setLevel(numeric_level)
    main_logger.propagate = False
    
    # Training progress logger
    training_logger = logging.getLogger('training_progress')
    training_logger.setLevel(numeric_level)
    training_logger.propagate = False
    
    # Error logger
    error_logger = logging.getLogger('errors')
    error_logger.setLevel(logging.WARNING)
    error_logger.propagate = False
    
    # Numerical stability logger
    stability_logger = logging.getLogger('numerical_stability')
    stability_logger.setLevel(numeric_level)
    stability_logger.propagate = False
    
    # System logger
    system_logger = logging.getLogger('system')
    system_logger.setLevel(numeric_level)
    system_logger.propagate = False
    
    # Add handlers based on configuration
    if enable_console_logging:
        # Console handler with colored output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(colored_formatter)
        
        # Add to all loggers
        for logger in [main_logger, training_logger, error_logger, stability_logger, system_logger]:
            logger.addHandler(console_handler)
    
    if enable_file_logging:
        # Main log file
        main_handler = logging.handlers.RotatingFileHandler(
            log_path / "main.log",
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        main_handler.setLevel(numeric_level)
        main_handler.setFormatter(detailed_formatter)
        main_logger.addHandler(main_handler)
        
        # Training progress log file
        training_handler = logging.handlers.RotatingFileHandler(
            log_path / "training_progress.log",
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        training_handler.setLevel(numeric_level)
        training_handler.setFormatter(detailed_formatter)
        training_handler.addFilter(TrainingProgressFilter())
        training_logger.addHandler(training_handler)
        
        # Error log file
        error_handler = logging.handlers.RotatingFileHandler(
            log_path / "errors.log",
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.WARNING)
        error_handler.setFormatter(detailed_formatter)
        error_handler.addFilter(ErrorFilter())
        error_logger.addHandler(error_handler)
        
        # Numerical stability log file
        stability_handler = logging.handlers.RotatingFileHandler(
            log_path / "numerical_stability.log",
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        stability_handler.setLevel(numeric_level)
        stability_handler.setFormatter(detailed_formatter)
        stability_handler.addFilter(NumericalStabilityFilter())
        stability_logger.addHandler(stability_handler)
        
        # System log file
        system_handler = logging.handlers.RotatingFileHandler(
            log_path / "system.log",
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        system_handler.setLevel(numeric_level)
        system_handler.setFormatter(detailed_formatter)
        system_logger.addHandler(system_handler)
    
    if enable_json_logging:
        # JSON log file for machine processing
        json_handler = logging.handlers.RotatingFileHandler(
            log_path / "system.json",
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        json_handler.setLevel(numeric_level)
        json_handler.setFormatter(json_formatter)
        
        # Add to main logger
        main_logger.addHandler(json_handler)
    
    # Store loggers
    loggers['main'] = main_logger
    loggers['training'] = training_logger
    loggers['errors'] = error_logger
    loggers['stability'] = stability_logger
    loggers['system'] = system_logger
    
    return loggers


def get_logger(name: str) -> logging.Logger:
    """Get a logger by name."""
    return logging.getLogger(name)


def log_training_step(
    logger: logging.Logger,
    step: int,
    epoch: int,
    loss: float,
    accuracy: Optional[float] = None,
    learning_rate: Optional[float] = None,
    gradient_norm: Optional[float] = None,
    stability_score: Optional[float] = None,
    **kwargs
) -> None:
    """
    Log training step information.
    
    Args:
        logger: Logger instance
        step: Current training step
        epoch: Current epoch
        loss: Current loss value
        accuracy: Current accuracy (if applicable)
        learning_rate: Current learning rate
        gradient_norm: Current gradient norm
        stability_score: Current stability score
        **kwargs: Additional training metrics
    """
    
    # Basic training info
    log_data = {
        'step': step,
        'epoch': epoch,
        'loss': f"{loss:.6f}",
        'timestamp': datetime.now().isoformat()
    }
    
    # Optional metrics
    if accuracy is not None:
        log_data['accuracy'] = f"{accuracy:.4f}"
    if learning_rate is not None:
        log_data['learning_rate'] = f"{learning_rate:.6f}"
    if gradient_norm is not None:
        log_data['gradient_norm'] = f"{gradient_norm:.6f}"
    if stability_score is not None:
        log_data['stability_score'] = f"{stability_score:.4f}"
    
    # Additional metrics
    for key, value in kwargs.items():
        if value is not None:
            if isinstance(value, float):
                log_data[key] = f"{value:.6f}"
            else:
                log_data[key] = str(value)
    
    # Log the training step
    logger.info(f"Training Step: {json.dumps(log_data, indent=2)}")


def log_numerical_issue(
    logger: logging.Logger,
    issue_type: str,
    severity: str,
    location: str,
    details: Dict[str, Any],
    recovery_action: Optional[str] = None
) -> None:
    """
    Log numerical stability issues.
    
    Args:
        logger: Logger instance
        issue_type: Type of issue (NaN, Inf, Overflow, etc.)
        severity: Severity level (low, medium, high, critical)
        location: Where the issue occurred
        details: Additional details about the issue
        recovery_action: Action taken to recover
    """
    
    log_data = {
        'issue_type': issue_type,
        'severity': severity,
        'location': location,
        'details': details,
        'timestamp': datetime.now().isoformat()
    }
    
    if recovery_action:
        log_data['recovery_action'] = recovery_action
    
    if severity == 'critical':
        logger.critical(f"Critical Numerical Issue: {json.dumps(log_data, indent=2)}")
    elif severity == 'high':
        logger.error(f"High Severity Numerical Issue: {json.dumps(log_data, indent=2)}")
    elif severity == 'medium':
        logger.warning(f"Medium Severity Numerical Issue: {json.dumps(log_data, indent=2)}")
    else:
        logger.info(f"Low Severity Numerical Issue: {json.dumps(log_data, indent=2)}")


def log_system_event(
    logger: logging.Logger,
    event_type: str,
    description: str,
    details: Optional[Dict[str, Any]] = None,
    level: str = "info"
) -> None:
    """
    Log system events.
    
    Args:
        logger: Logger instance
        event_type: Type of system event
        description: Description of the event
        details: Additional details
        level: Log level (debug, info, warning, error, critical)
    """
    
    log_data = {
        'event_type': event_type,
        'description': description,
        'timestamp': datetime.now().isoformat()
    }
    
    if details:
        log_data['details'] = details
    
    log_method = getattr(logger, level.lower(), logger.info)
    log_method(f"System Event: {json.dumps(log_data, indent=2)}")


def log_error_with_context(
    logger: logging.Logger,
    error: Exception,
    operation: str,
    context: Optional[Dict[str, Any]] = None,
    recovery_attempted: bool = False
) -> None:
    """
    Log errors with full context and recovery information.
    
    Args:
        logger: Logger instance
        error: The exception that occurred
        operation: Operation being performed when error occurred
        context: Additional context information
        recovery_attempted: Whether recovery was attempted
    """
    
    import traceback
    
    log_data = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'operation': operation,
        'timestamp': datetime.now().isoformat(),
        'traceback': traceback.format_exc(),
        'recovery_attempted': recovery_attempted
    }
    
    if context:
        log_data['context'] = context
    
    logger.error(f"Error Details: {json.dumps(log_data, indent=2)}")


def log_performance_metrics(
    logger: logging.Logger,
    metrics: Dict[str, Any],
    operation: str,
    duration: Optional[float] = None
) -> None:
    """
    Log performance metrics.
    
    Args:
        logger: Logger instance
        metrics: Dictionary of performance metrics
        operation: Operation being measured
        duration: Duration of the operation in seconds
    """
    
    log_data = {
        'operation': operation,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    if duration is not None:
        log_data['duration_seconds'] = f"{duration:.4f}"
    
    logger.info(f"Performance Metrics: {json.dumps(log_data, indent=2)}")


# Convenience function to get all loggers
def get_all_loggers() -> Dict[str, logging.Logger]:
    """Get all configured loggers."""
    return {
        'main': get_logger('gradient_clipping_system'),
        'training': get_logger('training_progress'),
        'errors': get_logger('errors'),
        'stability': get_logger('numerical_stability'),
        'system': get_logger('system')
    }


if __name__ == "__main__":
    # Example usage
    loggers = setup_logging(
        log_dir="logs",
        log_level="DEBUG",
        enable_file_logging=True,
        enable_console_logging=True
    )
    
    # Test logging
    main_logger = loggers['main']
    training_logger = loggers['training']
    error_logger = loggers['errors']
    stability_logger = loggers['stability']
    system_logger = loggers['system']
    
    main_logger.info("Logging system initialized successfully")
    
    # Test training logging
    log_training_step(
        training_logger,
        step=1,
        epoch=1,
        loss=0.123456,
        accuracy=0.85,
        learning_rate=0.001,
        gradient_norm=1.234567,
        stability_score=0.95
    )
    
    # Test numerical issue logging
    log_numerical_issue(
        stability_logger,
        issue_type="NaN",
        severity="medium",
        location="layer_2",
        details={"tensor_shape": [32, 64], "value_count": 2},
        recovery_action="gradient_zeroing"
    )
    
    # Test system event logging
    log_system_event(
        system_logger,
        event_type="model_initialization",
        description="Neural network model created successfully",
        details={"input_dim": 784, "hidden_dim": 256, "output_dim": 10}
    )
    
    print("Logging system test completed. Check the logs/ directory for output files.")






