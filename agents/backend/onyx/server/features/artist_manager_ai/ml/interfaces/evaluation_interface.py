"""
Evaluation Service Interface
=============================

Interface for evaluation services.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class IEvaluationService(ABC):
    """
    Interface for evaluation services.
    
    All evaluation services should implement:
    - evaluate(): Evaluate model
    - get_metrics(): Get evaluation metrics
    """
    
    @abstractmethod
    def evaluate(self, model, dataloader) -> Dict[str, float]:
        """
        Evaluate model.
        
        Args:
            model: Model to evaluate
            dataloader: DataLoader for evaluation
        
        Returns:
            Evaluation metrics
        """
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """
        Get evaluation metrics.
        
        Returns:
            Metrics dictionary
        """
        pass




