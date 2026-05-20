"""
TensorBoard Logger Module
=========================

TensorBoard integration for training visualization.

Author: BUL System
Date: 2024
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logger.warning("TensorBoard not available. Install with: pip install tensorboard")


class TensorBoardLogger:
    """
    TensorBoard logger for training visualization.
    
    Provides integration with TensorBoard for real-time
    visualization of training metrics.
    
    Example:
        >>> logger = TensorBoardLogger("./logs")
        >>> logger.log_metric("loss", 0.5, step=100)
        >>> logger.close()
    """
    
    def __init__(self, log_dir: Path, enabled: bool = True):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory for TensorBoard logs
            enabled: Whether logging is enabled
        """
        if not TENSORBOARD_AVAILABLE:
            logger.warning("TensorBoard not available")
            enabled = False
        
        self.log_dir = Path(log_dir)
        self.enabled = enabled
        self.writer: Optional[Any] = None
        
        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(self.log_dir))
            logger.info(f"TensorBoard logging enabled at: {self.log_dir}")
    
    def log_metric(self, name: str, value: float, step: int) -> None:
        """
        Log a metric to TensorBoard.
        
        Args:
            name: Metric name
            value: Metric value
            step: Step number
        """
        if self.enabled and self.writer:
            self.writer.add_scalar(name, value, step)
    
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """
        Log multiple metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Step number
        """
        if self.enabled and self.writer:
            for name, value in metrics.items():
                self.writer.add_scalar(name, value, step)
    
    def log_histogram(self, name: str, values: Any, step: int) -> None:
        """
        Log a histogram.
        
        Args:
            name: Histogram name
            values: Values for histogram
            step: Step number
        """
        if self.enabled and self.writer:
            self.writer.add_histogram(name, values, step)
    
    def close(self) -> None:
        """Close the TensorBoard writer."""
        if self.writer:
            self.writer.close()
            logger.info("TensorBoard logger closed")

