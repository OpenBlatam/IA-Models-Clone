"""
Train Model Use Case

Use case for training ML models.
"""

import logging
from typing import Optional, Dict, Any

from ...domain.exceptions import ModelException, ModelTrainingException

logger = logging.getLogger(__name__)


class TrainModelUseCase:
    """
    Use case for training ML models.
    
    This use case:
    1. Validates training configuration
    2. Loads training data
    3. Trains the model
    4. Saves the trained model
    5. Returns training results
    """
    
    def __init__(
        self,
        model_trainer=None,  # Will be injected from infrastructure
        model_repository=None,  # Will be injected from infrastructure
    ):
        """
        Initialize use case.
        
        Args:
            model_trainer: Infrastructure adapter for model training
            model_repository: Infrastructure adapter for model storage
        """
        self.model_trainer = model_trainer
        self.model_repository = model_repository
    
    def execute(
        self,
        model_type: str,
        training_config: Dict[str, Any],
        data_path: Optional[str] = None
    ) -> dict:
        """
        Execute train model use case.
        
        Args:
            model_type: Type of model to train ('autoencoder', 'classifier', etc.)
            training_config: Training configuration
            data_path: Optional path to training data
        
        Returns:
            Dictionary with training results
        
        Raises:
            ModelTrainingException: If training fails
        """
        try:
            if not self.model_trainer:
                raise ModelTrainingException(
                    model_type, "Model trainer not available"
                )
            
            # Validate model type
            valid_types = ['autoencoder', 'classifier', 'diffusion']
            if model_type not in valid_types:
                raise ModelTrainingException(
                    model_type, f"Invalid model type. Must be one of: {valid_types}"
                )
            
            # Start training
            logger.info(f"Starting training for {model_type} model")
            
            training_result = self.model_trainer.train(
                model_type=model_type,
                config=training_config,
                data_path=data_path,
            )
            
            # Save model if repository available
            if self.model_repository and training_result.get('model_path'):
                model_id = self.model_repository.save_model(
                    model_type=model_type,
                    model_path=training_result['model_path'],
                    metadata=training_result.get('metadata', {}),
                )
                training_result['model_id'] = model_id
            
            logger.info(
                f"Training completed for {model_type} model: "
                f"epochs={training_result.get('epochs', 'N/A')}, "
                f"final_loss={training_result.get('final_loss', 'N/A')}"
            )
            
            return training_result
        
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}", exc_info=True)
            if isinstance(e, ModelTrainingException):
                raise
            raise ModelTrainingException(model_type, str(e))



