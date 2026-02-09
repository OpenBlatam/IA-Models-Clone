"""
Inference Pipeline
High-level inference pipeline
"""

import torch
import logging
from typing import Union, List, Dict, Any, Optional
from pathlib import Path

from ..inference import ModelPredictor, ImagePreprocessor, PredictionPostprocessor
from ..models.mobilenet.factory import MobileNetFactory
from ..models.mobilenet.config import MobileNetConfig
from ..utils import ConfigLoader

logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    High-level inference pipeline
    Handles complete inference workflow
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        model: Optional[torch.nn.Module] = None,
        config_path: Optional[Path] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize inference pipeline
        
        Args:
            model_path: Path to model checkpoint
            model: Model instance (optional)
            config_path: Path to config file
            device: Device for inference
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Load config if provided
        if config_path:
            self.config = ConfigLoader.load_yaml(config_path)
        else:
            self.config = {}
        
        # Load or create model
        if model is not None:
            self.model = model
        elif model_path:
            self.model = self._load_model(model_path)
        else:
            # Create from config
            model_config = MobileNetConfig.from_dict(
                self.config.get('model', {})
            )
            self.model = MobileNetFactory.create_model(model_config, device=self.device)
        
        # Setup components
        self.predictor = ModelPredictor(
            self.model,
            self.device,
            use_mixed_precision=self.config.get('device', {}).get('use_mixed_precision', False)
        )
        
        self.preprocessor = ImagePreprocessor(
            image_size=self.config.get('data', {}).get('image_size', 224)
        )
        
        self.postprocessor = PredictionPostprocessor(
            class_names=self.config.get('model', {}).get('class_names')
        )
    
    def _load_model(self, model_path: Path) -> torch.nn.Module:
        """Load model from checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)
        model_config = MobileNetConfig.from_dict(self.config.get('model', {}))
        model = MobileNetFactory.create_model(model_config, device=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def predict(
        self,
        image: Union[str, Path, torch.Tensor],
        return_probabilities: bool = True,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Predict on single image
        
        Args:
            image: Input image
            return_probabilities: Return probabilities
            top_k: Return top k predictions
            
        Returns:
            Prediction results
        """
        # Preprocess
        if isinstance(image, torch.Tensor):
            tensor = image
        else:
            tensor = self.preprocessor.preprocess(image)
        
        # Predict
        result = self.predictor.predict(
            tensor,
            return_probabilities=return_probabilities,
            top_k=top_k
        )
        
        # Postprocess
        postprocessed = self.postprocessor.postprocess(
            result['predictions'],
            result.get('probabilities')
        )
        
        return postprocessed
    
    def predict_batch(
        self,
        images: List[Union[str, Path, torch.Tensor]],
        batch_size: int = 32,
    ) -> List[Dict[str, Any]]:
        """
        Predict on batch of images
        
        Args:
            images: List of images
            batch_size: Batch size
            
        Returns:
            List of prediction results
        """
        # Preprocess
        tensors = []
        for img in images:
            if isinstance(img, torch.Tensor):
                tensors.append(img)
            else:
                tensors.append(self.preprocessor.preprocess(img))
        
        batch_tensor = torch.stack(tensors)
        
        # Predict
        results = self.predictor.predict_batch(
            batch_tensor,
            batch_size=batch_size,
            return_probabilities=True
        )
        
        # Postprocess
        postprocessed = []
        for result in results:
            postprocessed.append(
                self.postprocessor.postprocess(
                    result.get('prediction'),
                    result.get('probability')
                )
            )
        
        return postprocessed



