"""
Model Interfaces - Define contracts for all models
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn


class IMusicModel(ABC, nn.Module):
    """
    Base interface for all music models
    """
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        pass
    
    @abstractmethod
    def predict(self, features: Any) -> Dict[str, Any]:
        """Make prediction"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        pass


class IMusicClassifier(IMusicModel):
    """
    Interface for classification models
    """
    
    @abstractmethod
    def classify(self, features: Any) -> Dict[str, Any]:
        """Classify input"""
        pass
    
    @abstractmethod
    def get_num_classes(self) -> int:
        """Get number of classes"""
        pass


class IMusicEncoder(IMusicModel):
    """
    Interface for encoder models
    """
    
    @abstractmethod
    def encode(self, features: Any) -> torch.Tensor:
        """Encode features to embeddings"""
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        pass


class IMultiTaskModel(IMusicModel):
    """
    Interface for multi-task models
    """
    
    @abstractmethod
    def predict_all_tasks(self, features: Any) -> Dict[str, Any]:
        """Predict all tasks simultaneously"""
        pass
    
    @abstractmethod
    def get_task_names(self) -> List[str]:
        """Get list of task names"""
        pass













