"""
Grad-CAM Implementation
Gradient-weighted Class Activation Mapping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class GradCAM:
    """
    Grad-CAM for visualizing model attention
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
    ):
        """
        Initialize Grad-CAM
        
        Args:
            model: Model to analyze
            target_layer: Target layer for visualization
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self) -> None:
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate class activation map
        
        Args:
            input_tensor: Input tensor
            target_class: Target class (None = use predicted class)
            
        Returns:
            Class activation map
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Compute CAM
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=0)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam / cam.max() if cam.max() > 0 else cam
        
        return cam.detach().cpu().numpy()


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ implementation
    """
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate class activation map using Grad-CAM++
        
        Args:
            input_tensor: Input tensor
            target_class: Target class
            
        Returns:
            Class activation map
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()
        
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        # Grad-CAM++ weights
        alpha = torch.sum(gradients, dim=(1, 2), keepdim=True)
        alpha = F.relu(alpha)
        alpha = alpha / (torch.sum(alpha, dim=0, keepdim=True) + 1e-7)
        
        weights = torch.sum(alpha * F.relu(gradients), dim=(1, 2), keepdim=True)
        
        # Weighted combination
        cam = torch.sum(weights * activations, dim=0)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam / cam.max() if cam.max() > 0 else cam
        
        return cam.detach().cpu().numpy()



