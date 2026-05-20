"""
TensorBoard Tracker
Integration with TensorBoard for experiment tracking
"""

from typing import Dict, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logger.warning("TensorBoard not available")

from .base_tracker import BaseExperimentTracker


class TensorBoardTracker(BaseExperimentTracker):
    """TensorBoard experiment tracker"""
    
    def __init__(
        self,
        experiment_name: str,
        project_name: Optional[str] = None,
        log_dir: str = "./logs"
    ):
        super().__init__(experiment_name, project_name)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = None
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize TensorBoard writer"""
        if not TENSORBOARD_AVAILABLE:
            logger.warning("TensorBoard not available, tracker disabled")
            return False
        
        try:
            run_dir = self.log_dir / self.experiment_name
            self.writer = SummaryWriter(log_dir=str(run_dir))
            self.initialized = True
            
            # Log hyperparameters
            if config:
                self.log_params(config)
            
            logger.info(f"TensorBoard tracker initialized: {self.experiment_name}")
            return True
        except Exception as e:
            logger.error(f"Error initializing TensorBoard: {str(e)}")
            return False
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to TensorBoard"""
        if not self.initialized or self.writer is None:
            return
        
        try:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, global_step=step or 0)
        except Exception as e:
            logger.error(f"Error logging metrics to TensorBoard: {str(e)}")
    
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters to TensorBoard"""
        if not self.initialized or self.writer is None:
            return
        
        try:
            self.writer.add_hparams(params, {})
        except Exception as e:
            logger.error(f"Error logging params to TensorBoard: {str(e)}")
    
    def log_model(self, model_path: str, model_name: str = "model"):
        """Log model to TensorBoard (as text)"""
        if not self.initialized or self.writer is None:
            return
        
        try:
            # TensorBoard doesn't directly support model artifacts
            # Log model path as text
            self.writer.add_text("model_path", model_path)
        except Exception as e:
            logger.error(f"Error logging model to TensorBoard: {str(e)}")
    
    def finish(self):
        """Close TensorBoard writer"""
        if self.writer is not None:
            self.writer.close()



