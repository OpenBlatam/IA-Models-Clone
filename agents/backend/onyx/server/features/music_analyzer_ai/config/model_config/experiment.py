"""
Experiment Configuration Module

Experiment tracking configuration dataclasses.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking"""
    experiment_name: str = "music_analysis"
    project_name: str = "music_analyzer_ai"
    use_wandb: bool = False
    use_tensorboard: bool = True
    use_mlflow: bool = False
    log_dir: str = "./logs"
    tags: List[str] = field(default_factory=list)



