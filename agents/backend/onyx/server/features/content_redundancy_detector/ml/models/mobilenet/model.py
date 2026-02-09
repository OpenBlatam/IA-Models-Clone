"""
MobileNet Model Wrapper
Wrapper class that implements BaseModel interface
"""

import logging
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional

from ..base import BaseModel
from .factory import MobileNetFactory
from .config import MobileNetConfig, MobileNetVariant

logger = logging.getLogger(__name__)


class MobileNetModel(BaseModel):
    """
    MobileNet Model Wrapper
    Implements BaseModel interface with proper GPU support and mixed precision
    """
    
    def __init__(
        self,
        model_name: str = "mobilenet_v2",
        num_classes: int = 1000,
        width_mult: float = 1.0,
        device: Optional[torch.device] = None,
        use_mixed_precision: bool = False,
        pretrained: bool = False,
        config: Optional[MobileNetConfig] = None,
    ):
        """
        Initialize MobileNet model
        
        Args:
            model_name: Model variant ('mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small')
            num_classes: Number of output classes
            width_mult: Width multiplier for model scaling
            device: PyTorch device
            use_mixed_precision: Use mixed precision
            pretrained: Load pretrained weights
            config: Optional configuration object
        """
        super().__init__(model_name, device, use_mixed_precision)
        self.num_classes = num_classes
        self.width_mult = width_mult
        self.pretrained = pretrained
        
        # Parse variant from model_name
        if model_name == "mobilenet_v2":
            self.variant = MobileNetVariant.MOBILENET_V2
        elif model_name == "mobilenet_v3_large":
            self.variant = MobileNetVariant.MOBILENET_V3_LARGE
        elif model_name == "mobilenet_v3_small":
            self.variant = MobileNetVariant.MOBILENET_V3_SMALL
        else:
            self.variant = MobileNetVariant.MOBILENET_V2
            logger.warning(f"Unknown model name {model_name}, defaulting to MobileNetV2")
        
        # Use provided config or create from parameters
        if config is None:
            self.config = MobileNetConfig(
                variant=self.variant,
                num_classes=num_classes,
                width_mult=width_mult,
                pretrained=pretrained,
            )
        else:
            self.config = config
    
    async def load(self) -> None:
        """Load MobileNet model using factory"""
        if self.is_loaded:
            return
        
        try:
            logger.info(f"Loading MobileNet model: {self.variant.value}")
            
            # Use factory to create model
            self.model = MobileNetFactory.create_model(
                config=self.config,
                device=self.device,
            )
            
            self.model.eval()
            self.is_loaded = True
            logger.info(f"Successfully loaded {self.variant.value}")
        except Exception as e:
            logger.error(f"Error loading MobileNet model: {e}", exc_info=True)
            raise
    
    async def predict(
        self,
        inputs: torch.Tensor,
        return_features: bool = False,
    ) -> Dict[str, Any]:
        """
        Run inference on input tensor
        
        Args:
            inputs: Input tensor (B, C, H, W)
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary with predictions and optionally features
        """
        if not self.is_loaded:
            await self.load()
        
        if not isinstance(inputs, torch.Tensor):
            raise TypeError(f"Inputs must be torch.Tensor, got {type(inputs)}")
        
        # Move inputs to device
        inputs = inputs.to(self.device)
        
        try:
            with torch.no_grad():
                if self.use_mixed_precision and self.device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        if return_features:
                            features = self.model.features(inputs)
                            features = (
                                self.model.avgpool(features)
                                if hasattr(self.model, 'avgpool')
                                else F.adaptive_avg_pool2d(features, (1, 1))
                            )
                            features = torch.flatten(features, 1)
                else:
                    outputs = self.model(inputs)
                    if return_features:
                        features = self.model.features(inputs)
                        features = (
                            self.model.avgpool(features)
                            if hasattr(self.model, 'avgpool')
                            else F.adaptive_avg_pool2d(features, (1, 1))
                        )
                        features = torch.flatten(features, 1)
            
            # Apply softmax for probabilities
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            result = {
                "logits": outputs.cpu().numpy().tolist(),
                "probabilities": probs.cpu().numpy().tolist(),
                "predictions": preds.cpu().numpy().tolist(),
            }
            
            if return_features:
                result["features"] = features.cpu().numpy().tolist()
            
            return result
        except Exception as e:
            logger.error(f"Error in MobileNet prediction: {e}", exc_info=True)
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if not self.is_loaded:
            return {"status": "not_loaded"}
        
        from .utils import count_parameters, get_model_size_mb
        
        total_params = count_parameters(self.model)
        trainable_params = count_parameters(self.model, trainable_only=True)
        model_size_mb = get_model_size_mb(self.model)
        
        return {
            "model_variant": self.variant.value,
            "num_classes": self.num_classes,
            "width_mult": self.width_mult,
            "device": str(self.device),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": model_size_mb,
            "pretrained": self.pretrained,
        }



