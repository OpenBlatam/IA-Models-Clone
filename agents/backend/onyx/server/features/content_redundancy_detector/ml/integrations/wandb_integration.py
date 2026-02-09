"""
Weights & Biases Integration
WandB integration for experiment tracking
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("WandB not available. Install with: pip install wandb")


class WandBIntegration:
    """
    WandB integration for experiment tracking
    """
    
    def __init__(
        self,
        project_name: str,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize WandB integration
        
        Args:
            project_name: Project name
            run_name: Run name (optional)
            config: Configuration dictionary (optional)
        """
        if not WANDB_AVAILABLE:
            raise ImportError("WandB is not installed. Install with: pip install wandb")
        
        self.project_name = project_name
        self.run_name = run_name
        self.config = config or {}
        
        wandb.init(
            project=project_name,
            name=run_name,
            config=config,
        )
        logger.info(f"Initialized WandB: project={project_name}, run={run_name}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics
        
        Args:
            metrics: Dictionary of metrics
            step: Step number (optional)
        """
        if step is not None:
            wandb.log(metrics, step=step)
        else:
            wandb.log(metrics)
    
    def log_model(self, model, artifact_name: str = "model") -> None:
        """
        Log model as artifact
        
        Args:
            model: Model to log
            artifact_name: Artifact name
        """
        import torch
        import tempfile
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / f"{artifact_name}.pth"
            torch.save(model.state_dict(), model_path)
            artifact = wandb.Artifact(artifact_name, type="model")
            artifact.add_file(str(model_path))
            wandb.log_artifact(artifact)
    
    def finish(self) -> None:
        """Finish WandB run"""
        wandb.finish()
        logger.info("Finished WandB run")



