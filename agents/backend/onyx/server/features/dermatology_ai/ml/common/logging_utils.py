"""
Logging Utilities
Enhanced logging for ML workflows
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import torch


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
):
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        format_string: Optional custom format string
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    if format_string is None:
        format_string = (
            '%(asctime)s - %(name)s - %(levelname)s - '
            '%(message)s'
        )
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers
    )


def log_model_info(model: torch.nn.Module, logger: logging.Logger):
    """Log model information"""
    try:
        from ml.common.utils import count_parameters, get_model_size
        
        params = count_parameters(model)
        size_mb = get_model_size(model)
        
        logger.info(f"Model: {model.__class__.__name__}")
        logger.info(f"Total parameters: {params['total']:,}")
        logger.info(f"Trainable parameters: {params['trainable']:,}")
        logger.info(f"Model size: {size_mb:.2f} MB")
        
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        logger.warning(f"Could not log model info: {e}")


def log_training_summary(
    history: Dict[str, Any],
    logger: logging.Logger,
    num_epochs: int
):
    """Log training summary"""
    logger.info("=" * 50)
    logger.info("Training Summary")
    logger.info("=" * 50)
    logger.info(f"Total epochs: {num_epochs}")
    
    if history.get('train_loss'):
        final_train_loss = history['train_loss'][-1]
        logger.info(f"Final train loss: {final_train_loss:.4f}")
    
    if history.get('val_loss'):
        final_val_loss = history['val_loss'][-1]
        best_val_loss = min(history['val_loss'])
        best_epoch = history['val_loss'].index(best_val_loss) + 1
        logger.info(f"Final val loss: {final_val_loss:.4f}")
        logger.info(f"Best val loss: {best_val_loss:.4f} (epoch {best_epoch})")
    
    if history.get('learning_rates'):
        initial_lr = history['learning_rates'][0]
        final_lr = history['learning_rates'][-1]
        logger.info(f"Initial LR: {initial_lr:.6f}")
        logger.info(f"Final LR: {final_lr:.6f}")
    
    logger.info("=" * 50)


class TrainingLogger:
    """Context manager for training logging"""
    
    def __init__(self, logger: logging.Logger, experiment_name: str):
        self.logger = logger
        self.experiment_name = experiment_name
    
    def __enter__(self):
        self.logger.info(f"Starting experiment: {self.experiment_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.logger.info(f"Completed experiment: {self.experiment_name}")
        else:
            self.logger.error(
                f"Experiment {self.experiment_name} failed: {exc_val}",
                exc_info=True
            )
        return False
