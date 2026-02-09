"""
Model Training Application Service

Application service that orchestrates model training use cases.
"""

import logging
from typing import Dict, Any, Optional

from ..use_cases import TrainModelUseCase

logger = logging.getLogger(__name__)


class ModelTrainingApplicationService:
    """
    Application service for model training operations.
    
    This service coordinates model training use cases and provides
    a high-level interface for training operations.
    """
    
    def __init__(self, train_model_use_case: TrainModelUseCase):
        """
        Initialize application service.
        
        Args:
            train_model_use_case: Use case for training models
        """
        self.train_model_use_case = train_model_use_case
    
    def train_model(
        self,
        model_type: str,
        training_config: Dict[str, Any],
        data_path: Optional[str] = None,
    ) -> dict:
        """
        Train a model.
        
        Args:
            model_type: Type of model to train
            training_config: Training configuration
            data_path: Optional path to training data
        
        Returns:
            Training results
        """
        return self.train_model_use_case.execute(
            model_type, training_config, data_path
        )
    
    def train_autoencoder(
        self,
        training_config: Dict[str, Any],
        data_path: Optional[str] = None,
    ) -> dict:
        """
        Train an autoencoder model.
        
        Args:
            training_config: Training configuration
            data_path: Optional path to training data
        
        Returns:
            Training results
        """
        return self.train_model("autoencoder", training_config, data_path)
    
    def train_classifier(
        self,
        training_config: Dict[str, Any],
        data_path: Optional[str] = None,
    ) -> dict:
        """
        Train a classifier model.
        
        Args:
            training_config: Training configuration
            data_path: Optional path to training data
        
        Returns:
            Training results
        """
        return self.train_model("classifier", training_config, data_path)
    
    def train_diffusion(
        self,
        training_config: Dict[str, Any],
        data_path: Optional[str] = None,
    ) -> dict:
        """
        Train a diffusion model.
        
        Args:
            training_config: Training configuration
            data_path: Optional path to training data
        
        Returns:
            Training results
        """
        return self.train_model("diffusion", training_config, data_path)



