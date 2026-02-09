"""
Vision Transformer (ViT) for Defect Classification
"""

import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig, ViTForImageClassification
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class DefectViTClassifier(nn.Module):
    """
    Vision Transformer for defect classification in quality control
    
    Uses pre-trained ViT from HuggingFace Transformers library
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        model_name: str = "google/vit-base-patch16-224",
        pretrained: bool = True,
        image_size: int = 224,
        patch_size: int = 16
    ):
        """
        Initialize ViT classifier
        
        Args:
            num_classes: Number of defect classes
            model_name: HuggingFace model name
            pretrained: Whether to use pretrained weights
            image_size: Input image size
            patch_size: Patch size for ViT
        """
        super(DefectViTClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.image_size = image_size
        self.patch_size = patch_size
        
        try:
            if pretrained:
                # Use pre-trained ViT for image classification
                self.vit = ViTForImageClassification.from_pretrained(
                    model_name,
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True
                )
                logger.info(f"Loaded pretrained ViT: {model_name}")
            else:
                # Create ViT from scratch
                config = ViTConfig(
                    image_size=image_size,
                    patch_size=patch_size,
                    num_labels=num_classes
                )
                self.vit = ViTForImageClassification(config)
                logger.info(f"Created ViT from scratch")
            
            # Get the base ViT model for feature extraction
            self.vit_model = self.vit.vit
            
        except Exception as e:
            logger.warning(f"Could not load ViT model: {e}. Using fallback CNN.")
            self.vit = None
            self.vit_model = None
            # Fallback to simple CNN
            self._create_fallback_model(num_classes)
        
        logger.info(f"DefectViTClassifier initialized with {num_classes} classes")
    
    def _create_fallback_model(self, num_classes: int):
        """Create fallback CNN model if ViT is not available"""
        self.fallback_model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [B, C, H, W] normalized to [0, 1]
            
        Returns:
            Logits [B, num_classes]
        """
        if self.vit is not None:
            # ViT expects inputs in range [0, 1] and normalized
            # Transform to [0, 255] then normalize
            if x.max() <= 1.0:
                x = x * 255.0
            
            # Normalize using ImageNet stats
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
            x = (x - mean) / std
            
            outputs = self.vit(pixel_values=x)
            return outputs.logits
        else:
            # Fallback CNN
            return self.fallback_model(x)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input (for transfer learning)
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Features [B, feature_dim]
        """
        if self.vit_model is not None:
            if x.max() <= 1.0:
                x = x * 255.0
            
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
            x = (x - mean) / std
            
            outputs = self.vit_model(pixel_values=x)
            # Get [CLS] token
            return outputs.last_hidden_state[:, 0, :]
        else:
            # Fallback: use intermediate features
            features = self.fallback_model[:-3](x)  # All but last 3 layers
            return features
    
    def predict(self, x: torch.Tensor, return_probs: bool = True) -> Dict:
        """
        Predict defect classes
        
        Args:
            x: Input tensor [B, C, H, W]
            return_probs: Whether to return probabilities
            
        Returns:
            Dictionary with predictions
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            
            result = {
                "predictions": preds.cpu().numpy(),
                "logits": logits.cpu().numpy()
            }
            
            if return_probs:
                result["probabilities"] = probs.cpu().numpy()
            
            return result


def create_defect_classifier(
    num_classes: int = 10,
    model_name: str = "google/vit-base-patch16-224",
    pretrained: bool = True,
    device: Optional[torch.device] = None
) -> DefectViTClassifier:
    """
    Factory function to create defect classifier
    
    Args:
        num_classes: Number of defect classes
        model_name: HuggingFace model name
        pretrained: Whether to use pretrained weights
        device: Device to place model on
        
    Returns:
        Initialized classifier model
    """
    model = DefectViTClassifier(
        num_classes=num_classes,
        model_name=model_name,
        pretrained=pretrained
    )
    
    if device is not None:
        model = model.to(device)
    
    return model

