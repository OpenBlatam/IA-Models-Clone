"""
Base Trainer Interface
======================

Abstract base class for LLM trainers.
Provides interface for extensibility.

Author: BUL System
Date: 2024
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
from pathlib import Path


class BaseLLMTrainer(ABC):
    """
    Abstract base class for LLM trainers.
    
    Defines the interface that all trainer implementations must follow.
    This allows for different trainer implementations while maintaining
    a consistent API.
    
    Example:
        >>> class MyCustomTrainer(BaseLLMTrainer):
        ...     def train(self):
        ...         # Implementation
        ...         pass
    """
    
    @abstractmethod
    def train(self, resume_from_checkpoint: Optional[Union[str, bool]] = None) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            resume_from_checkpoint: Checkpoint to resume from
            
        Returns:
            Dictionary with training results
        """
        pass
    
    @abstractmethod
    def evaluate(self, eval_dataset=None) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            eval_dataset: Optional evaluation dataset
            
        Returns:
            Dictionary with evaluation metrics
        """
        pass
    
    @abstractmethod
    def predict(self, prompts: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
        """
        Generate predictions.
        
        Args:
            prompts: Input prompts
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text(s)
        """
        pass
    
    @abstractmethod
    def save_model(self, output_dir: Union[str, Path]) -> None:
        """
        Save the model.
        
        Args:
            output_dir: Directory to save model
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary with model information
        """
        pass
    
    @abstractmethod
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get training summary.
        
        Returns:
            Dictionary with training summary
        """
        pass

