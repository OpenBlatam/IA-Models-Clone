"""
Experiment Tracking System
Integration with wandb, tensorboard, and custom tracking
"""

from typing import Dict, Any, Optional, List
import logging
from pathlib import Path
from datetime import datetime
import json

logger = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class ExperimentTracker:
    """
    Unified experiment tracking with support for:
    - Weights & Biases (wandb)
    - TensorBoard
    - Custom JSON logging
    """
    
    def __init__(
        self,
        experiment_name: str,
        use_wandb: bool = False,
        use_tensorboard: bool = True,
        log_dir: str = "./logs",
        project_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Weights & Biases
        self.wandb_run = None
        if use_wandb and WANDB_AVAILABLE:
            try:
                self.wandb_run = wandb.init(
                    project=project_name or "music_analyzer_ai",
                    name=experiment_name,
                    config=config or {}
                )
                logger.info("Initialized Weights & Biases tracking")
            except Exception as e:
                logger.warning(f"Could not initialize wandb: {str(e)}")
        
        # TensorBoard
        self.tensorboard_writer = None
        if use_tensorboard and TENSORBOARD_AVAILABLE:
            try:
                tb_dir = self.log_dir / "tensorboard" / experiment_name
                self.tensorboard_writer = SummaryWriter(str(tb_dir))
                logger.info(f"Initialized TensorBoard at {tb_dir}")
            except Exception as e:
                logger.warning(f"Could not initialize TensorBoard: {str(e)}")
        
        # Custom JSON logging
        self.json_log_path = self.log_dir / f"{experiment_name}_metrics.jsonl"
        self.metrics_history: List[Dict[str, Any]] = []
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to all enabled trackers"""
        step = step or metrics.get("epoch", len(self.metrics_history))
        
        # WandB
        if self.wandb_run:
            try:
                self.wandb_run.log(metrics, step=step)
            except Exception as e:
                logger.warning(f"WandB logging error: {str(e)}")
        
        # TensorBoard
        if self.tensorboard_writer:
            try:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self.tensorboard_writer.add_scalar(key, value, step)
            except Exception as e:
                logger.warning(f"TensorBoard logging error: {str(e)}")
        
        # JSON logging
        try:
            metrics_with_step = {**metrics, "step": step, "timestamp": datetime.now().isoformat()}
            self.metrics_history.append(metrics_with_step)
            
            with open(self.json_log_path, "a") as f:
                f.write(json.dumps(metrics_with_step) + "\n")
        except Exception as e:
            logger.warning(f"JSON logging error: {str(e)}")
    
    def log_model(self, model, input_sample):
        """Log model architecture"""
        if self.tensorboard_writer:
            try:
                self.tensorboard_writer.add_graph(model, input_sample)
            except Exception as e:
                logger.warning(f"Model logging error: {str(e)}")
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """Log hyperparameters"""
        if self.wandb_run:
            try:
                self.wandb_run.config.update(hyperparams)
            except Exception:
                pass
        
        # Save to file
        config_path = self.log_dir / f"{self.experiment_name}_config.json"
        with open(config_path, "w") as f:
            json.dump(hyperparams, f, indent=2)
    
    def finish(self):
        """Finish experiment tracking"""
        if self.wandb_run:
            try:
                self.wandb_run.finish()
            except Exception:
                pass
        
        if self.tensorboard_writer:
            try:
                self.tensorboard_writer.close()
            except Exception:
                pass
        
        logger.info(f"Experiment tracking finished for {self.experiment_name}")
    
    def get_metrics(self) -> List[Dict[str, Any]]:
        """Get logged metrics"""
        return self.metrics_history.copy()

