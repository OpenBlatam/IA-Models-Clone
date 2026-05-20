"""
Modular Model Manager
Manages model lifecycle using modular components
"""

from typing import Dict, Any, Optional, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

from ..factories.unified_factory import get_factory
from ..utils.device_manager import get_device_manager
from ..utils.initialization import initialize_weights
from ..inference.pipelines import StandardInferencePipeline


class ModelManager:
    """
    Manages model lifecycle using modular components
    Handles creation, loading, saving, and inference
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        model_dir: str = "./models"
    ):
        self.factory = get_factory()
        self.device_manager = get_device_manager(device)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.models: Dict[str, nn.Module] = {}
        self.inference_pipelines: Dict[str, StandardInferencePipeline] = {}
    
    def create_model(
        self,
        model_name: str,
        model_type: str,
        config: Optional[Dict[str, Any]] = None,
        compile_model: bool = True
    ) -> nn.Module:
        """
        Create and register a model
        
        Args:
            model_name: Name for the model
            model_type: Type of model (must be registered)
            config: Model configuration
            compile_model: Whether to compile model
        
        Returns:
            Model instance
        """
        # Create model using factory
        model = self.factory.create_model(
            model_type=model_type,
            config=config or {},
            device=self.device_manager.get_device()
        )
        
        # Initialize weights
        initialize_weights(model, strategy="xavier")
        
        # Compile if requested
        if compile_model:
            model = self.device_manager.compile_model(model)
        
        # Register model
        self.models[model_name] = model
        
        logger.info(f"Created and registered model: {model_name}")
        return model
    
    def load_model(
        self,
        model_name: str,
        checkpoint_path: str,
        model_type: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> nn.Module:
        """
        Load model from checkpoint
        
        Args:
            model_name: Name for the model
            checkpoint_path: Path to checkpoint
            model_type: Model type (required if not in checkpoint)
            config: Model configuration (required if not in checkpoint)
        
        Returns:
            Loaded model instance
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device_manager.get_device())
        
        # Get model type and config from checkpoint or parameters
        if "model_type" in checkpoint:
            model_type = checkpoint["model_type"]
        if "config" in checkpoint:
            config = checkpoint["config"]
        
        if model_type is None:
            raise ValueError("model_type must be provided or in checkpoint")
        
        # Create model
        model = self.factory.create_model(
            model_type=model_type,
            config=config or {},
            device=self.device_manager.get_device()
        )
        
        # Load weights
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        
        # Register model
        self.models[model_name] = model
        
        logger.info(f"Loaded and registered model: {model_name}")
        return model
    
    def save_model(
        self,
        model_name: str,
        checkpoint_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Save model to checkpoint
        
        Args:
            model_name: Name of registered model
            checkpoint_path: Path to save checkpoint
            metadata: Additional metadata to save
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if checkpoint_path is None:
            checkpoint_path = self.model_dir / f"{model_name}.pt"
        else:
            checkpoint_path = Path(checkpoint_path)
        
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "metadata": metadata or {}
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved model {model_name} to {checkpoint_path}")
    
    def get_model(self, model_name: str) -> Optional[nn.Module]:
        """Get registered model"""
        return self.models.get(model_name)
    
    def create_inference_pipeline(
        self,
        pipeline_name: str,
        model_name: str,
        preprocess_fn: Optional[Any] = None,
        postprocess_fn: Optional[Any] = None
    ) -> StandardInferencePipeline:
        """
        Create inference pipeline for a model
        
        Args:
            pipeline_name: Name for the pipeline
            model_name: Name of registered model
            preprocess_fn: Preprocessing function
            postprocess_fn: Postprocessing function
        
        Returns:
            Inference pipeline instance
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        pipeline = self.factory.create_inference_pipeline(
            model=model,
            config={
                "device": self.device_manager.get_device(),
                "use_mixed_precision": self.device_manager.enable_mixed_precision(),
                "preprocess_fn": preprocess_fn,
                "postprocess_fn": postprocess_fn
            }
        )
        
        self.inference_pipelines[pipeline_name] = pipeline
        
        logger.info(f"Created inference pipeline: {pipeline_name}")
        return pipeline
    
    def predict(
        self,
        pipeline_name: str,
        input_data: Any
    ) -> Dict[str, Any]:
        """
        Run inference using a pipeline
        
        Args:
            pipeline_name: Name of inference pipeline
            input_data: Input data
        
        Returns:
            Prediction results
        """
        if pipeline_name not in self.inference_pipelines:
            raise ValueError(f"Pipeline {pipeline_name} not found")
        
        pipeline = self.inference_pipelines[pipeline_name]
        return pipeline.predict(input_data)
    
    def list_models(self) -> List[str]:
        """List all registered models"""
        return list(self.models.keys())
    
    def list_pipelines(self) -> List[str]:
        """List all inference pipelines"""
        return list(self.inference_pipelines.keys())



