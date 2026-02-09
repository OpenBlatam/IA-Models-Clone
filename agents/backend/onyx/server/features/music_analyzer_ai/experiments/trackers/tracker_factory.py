"""
Experiment Tracker Factory
Creates trackers based on configuration
"""

from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

from .base_tracker import BaseExperimentTracker
from .wandb_tracker import WandBTracker
from .tensorboard_tracker import TensorBoardTracker


class TrackerFactory:
    """Factory for creating experiment trackers"""
    
    @staticmethod
    def create(
        tracker_type: str,
        experiment_name: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Optional[BaseExperimentTracker]:
        """
        Create experiment tracker
        
        Args:
            tracker_type: Type of tracker ("wandb", "tensorboard", "mlflow")
            experiment_name: Name of experiment
            config: Tracker configuration
        
        Returns:
            Tracker instance or None
        """
        config = config or {}
        tracker_type = tracker_type.lower()
        
        if tracker_type == "wandb":
            tracker = WandBTracker(
                experiment_name=experiment_name,
                project_name=config.get("project_name"),
                entity=config.get("entity")
            )
        elif tracker_type == "tensorboard":
            tracker = TensorBoardTracker(
                experiment_name=experiment_name,
                project_name=config.get("project_name"),
                log_dir=config.get("log_dir", "./logs")
            )
        elif tracker_type == "mlflow":
            try:
                from .mlflow_tracker import MLflowTracker
                tracker = MLflowTracker(
                    experiment_name=experiment_name,
                    tracking_uri=config.get("tracking_uri")
                )
            except ImportError:
                logger.warning("MLflow not available")
                return None
        else:
            logger.warning(f"Unknown tracker type: {tracker_type}")
            return None
        
        # Initialize tracker
        if tracker.initialize(config):
            return tracker
        else:
            return None


def create_tracker(
    tracker_type: str,
    experiment_name: str,
    config: Optional[Dict[str, Any]] = None
) -> Optional[BaseExperimentTracker]:
    """Convenience function for creating trackers"""
    return TrackerFactory.create(tracker_type, experiment_name, config)



