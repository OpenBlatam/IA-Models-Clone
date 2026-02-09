"""
ML Model Loader Adapter

Adapter for loading and managing ML models.
"""

import logging
from typing import Optional, Dict, Any
from pathlib import Path
import torch

from ...domain.exceptions import ModelException, ModelLoadException

logger = logging.getLogger(__name__)


class MLModelLoader:
    """
    Adapter for loading and managing ML models.
    
    Handles model loading, caching, and device management.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize model loader.
        
        Args:
            device: Device for model inference ('cuda', 'cpu', or None for auto)
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self._model_cache = {}  # model_id -> model instance
        logger.info(f"MLModelLoader initialized with device: {self.device}")
    
    def load_model(
        self,
        model_path: str,
        model_type: str,
        model_id: Optional[str] = None,
    ) -> Any:
        """
        Load a model from file.
        
        Args:
            model_path: Path to model file
            model_path: Type of model ('autoencoder', 'classifier', etc.)
            model_id: Optional model ID for caching
        
        Returns:
            Loaded model instance
        
        Raises:
            ModelLoadException: If loading fails
        """
        try:
            # Check cache
            if model_id and model_id in self._model_cache:
                logger.info(f"Loading model {model_id} from cache")
                return self._model_cache[model_id]
            
            # Load model based on type
            if model_type == "autoencoder":
                model = self._load_autoencoder(model_path)
            elif model_type == "classifier":
                model = self._load_classifier(model_path)
            elif model_type == "diffusion":
                model = self._load_diffusion(model_path)
            else:
                raise ModelLoadException(
                    model_type, f"Unsupported model type: {model_type}"
                )
            
            # Move to device
            model = model.to(self.device)
            model.eval()  # Set to evaluation mode
            
            # Cache if ID provided
            if model_id:
                self._model_cache[model_id] = model
            
            logger.info(f"Model loaded: type={model_type}, device={self.device}")
            return model
        
        except ModelLoadException:
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}", exc_info=True)
            raise ModelLoadException(model_type, str(e))
    
    def _load_autoencoder(self, model_path: str) -> Any:
        """Load autoencoder model."""
        # This would load the actual model from core.models
        # Placeholder implementation
        try:
            from ...core.models import create_autoencoder
            # In real implementation, would load from checkpoint
            # For now, create new model
            model = create_autoencoder(
                input_channels=3,
                latent_dim=128,
                input_size=(224, 224),
                device=self.device
            )
            return model
        except Exception as e:
            raise ModelLoadException("autoencoder", f"Failed to load: {str(e)}")
    
    def _load_classifier(self, model_path: str) -> Any:
        """Load classifier model."""
        try:
            from ...core.models import create_defect_classifier
            # In real implementation, would load from checkpoint
            model = create_defect_classifier(
                num_classes=10,
                pretrained=True
            )
            return model
        except Exception as e:
            raise ModelLoadException("classifier", f"Failed to load: {str(e)}")
    
    def _load_diffusion(self, model_path: str) -> Any:
        """Load diffusion model."""
        try:
            from ...core.models import create_diffusion_detector
            # In real implementation, would load from checkpoint
            model = create_diffusion_detector(
                image_size=224,
                in_channels=3
            )
            return model
        except Exception as e:
            raise ModelLoadException("diffusion", f"Failed to load: {str(e)}")
    
    def unload_model(self, model_id: str) -> bool:
        """
        Unload a model from cache.
        
        Args:
            model_id: Model ID
        
        Returns:
            True if unloaded, False if not found
        """
        if model_id in self._model_cache:
            del self._model_cache[model_id]
            logger.info(f"Model {model_id} unloaded from cache")
            return True
        return False
    
    def clear_cache(self):
        """Clear model cache."""
        self._model_cache.clear()
        logger.info("Model cache cleared")



