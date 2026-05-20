"""
Model Manager Submodule
Aggregates model management components.
"""

from typing import Dict, Any, Optional, List
from .manager import ModelManager as BaseModelManager
from .creation import ModelCreationMixin
from .persistence import ModelPersistenceMixin
from .inference import ModelInferenceMixin


class ModelManager(BaseModelManager):
    """
    Complete model manager combining all functionality.
    Uses composition to combine specialized managers.
    """
    
    def __init__(self, device=None, model_dir="./models"):
        # Initialize base
        super().__init__(device=device, model_dir=model_dir)
        
        # Initialize specialized managers
        self._creation_mixin = ModelCreationMixin()
        self._persistence_mixin = ModelPersistenceMixin()
        self._inference_mixin = ModelInferenceMixin()
    
    # Delegate creation methods
    def create_model(
        self,
        model_name: str,
        model_type: str,
        config: Optional[Dict[str, Any]] = None,
        compile_model: bool = True
    ):
        """Create and register a model."""
        return self._creation_mixin.create_model(
            self, model_name, model_type, config, compile_model
        )
    
    # Delegate persistence methods
    def load_model(
        self,
        model_name: str,
        checkpoint_path: str,
        model_type: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Load model from checkpoint."""
        return self._persistence_mixin.load_model(
            self, model_name, checkpoint_path, model_type, config
        )
    
    def save_model(
        self,
        model_name: str,
        checkpoint_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Save model to checkpoint."""
        return self._persistence_mixin.save_model(
            self, model_name, checkpoint_path, metadata
        )
    
    # Delegate inference methods
    def create_inference_pipeline(
        self,
        pipeline_name: str,
        model_name: str,
        preprocess_fn: Optional[Any] = None,
        postprocess_fn: Optional[Any] = None
    ):
        """Create inference pipeline for a model."""
        return self._inference_mixin.create_inference_pipeline(
            self, pipeline_name, model_name, preprocess_fn, postprocess_fn
        )
    
    def predict(
        self,
        pipeline_name: str,
        input_data: Any
    ) -> Dict[str, Any]:
        """Run inference using a pipeline."""
        return self._inference_mixin.predict(self, pipeline_name, input_data)


__all__ = [
    "ModelManager",
    "ModelCreationMixin",
    "ModelPersistenceMixin",
    "ModelInferenceMixin",
]
