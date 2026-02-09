"""
Training Service Interface
==========================

Interface for training services.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class ITrainingService(ABC):
    """
    Interface for training services.
    
    All training services should implement:
    - train(): Train model
    - resume(): Resume training
    - get_status(): Get training status
    """
    
    @abstractmethod
    def train(
        self,
        config: Dict[str, Any],
        data_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train model.
        
        Args:
            config: Training configuration
            data_path: Path to training data
        
        Returns:
            Training results
        """
        pass
    
    @abstractmethod
    def resume(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Resume training from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
        
        Returns:
            Training results
        """
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        Get training status.
        
        Returns:
            Status dictionary
        """
        pass




