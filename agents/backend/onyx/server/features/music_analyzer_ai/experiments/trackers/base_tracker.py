"""
Base Experiment Tracker
Abstract base for experiment tracking
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseExperimentTracker(ABC):
    """
    Abstract base class for experiment tracking
    """
    
    def __init__(self, experiment_name: str, project_name: Optional[str] = None):
        self.experiment_name = experiment_name
        self.project_name = project_name
        self.initialized = False
    
    @abstractmethod
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize tracker"""
        pass
    
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics"""
        pass
    
    @abstractmethod
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters"""
        pass
    
    @abstractmethod
    def log_model(self, model_path: str, model_name: str = "model"):
        """Log model artifact"""
        pass
    
    def finish(self):
        """Finish tracking session"""
        pass



