"""
Prediction Service Interface
=============================

Interface for prediction services.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class PredictionResult:
    """Prediction result."""
    value: float
    confidence: float
    metadata: Dict[str, Any]


class IPredictionService(ABC):
    """
    Interface for prediction services.
    
    All prediction services should implement:
    - predict(): Make prediction
    - batch_predict(): Batch predictions
    """
    
    @abstractmethod
    def predict(self, input_data: Dict[str, Any]) -> PredictionResult:
        """
        Make prediction.
        
        Args:
            input_data: Input data
        
        Returns:
            Prediction result
        """
        pass
    
    @abstractmethod
    def batch_predict(self, input_data_list: List[Dict[str, Any]]) -> List[PredictionResult]:
        """
        Make batch predictions.
        
        Args:
            input_data_list: List of input data
        
        Returns:
            List of prediction results
        """
        pass




