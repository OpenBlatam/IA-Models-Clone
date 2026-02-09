"""
TensorBoard Integration
TensorBoard integration for experiment tracking
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


class TensorBoardIntegration:
    """
    TensorBoard integration for experiment tracking
    """
    
    def __init__(self, log_dir: Path = Path("runs")):
        """
        Initialize TensorBoard integration
        
        Args:
            log_dir: Log directory
        """
        if not TENSORBOARD_AVAILABLE:
            raise ImportError("TensorBoard is not installed. Install with: pip install tensorboard")
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(str(self.log_dir))
        logger.info(f"Initialized TensorBoard: log_dir={log_dir}")
    
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """
        Log scalar value
        
        Args:
            tag: Tag name
            value: Scalar value
            step: Step number
        """
        self.writer.add_scalar(tag, value, step)
    
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """
        Log multiple metrics
        
        Args:
            metrics: Dictionary of metrics
            step: Step number
        """
        for tag, value in metrics.items():
            self.log_scalar(tag, value, step)
    
    def log_model_graph(self, model, input_shape: tuple) -> None:
        """
        Log model graph
        
        Args:
            model: Model
            input_shape: Input shape
        """
        import torch
        dummy_input = torch.randn(input_shape)
        self.writer.add_graph(model, dummy_input)
    
    def close(self) -> None:
        """Close TensorBoard writer"""
        self.writer.close()
        logger.info("Closed TensorBoard writer")



