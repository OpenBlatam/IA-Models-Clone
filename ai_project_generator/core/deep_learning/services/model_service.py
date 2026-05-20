"""
Model Service - High-level Model Management
===========================================

Service for managing models:
- Model creation
- Model loading/saving
- Model analysis
- Model optimization
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn

from ..core.base import BaseComponent
from ..models import create_model, BaseModel
from ..utils import get_device, optimize_model_for_inference
from ..utils.model_analysis import (
    analyze_model_complexity,
    check_model_health,
    get_layer_output_shapes
)
from ..utils.checkpoint_utils import CheckpointManager

logger = logging.getLogger(__name__)


class ModelService(BaseComponent):
    """
    High-level service for model management.
    
    Provides unified interface for model operations.
    """
    
    def _initialize(self) -> None:
        """Initialize service."""
        self.device = get_device()
        self.checkpoint_manager: Optional[CheckpointManager] = None
    
    def create_model(
        self,
        model_type: str,
        config: Dict[str, Any],
        device: Optional[torch.device] = None
    ) -> BaseModel:
        """
        Create model.
        
        Args:
            model_type: Type of model
            config: Model configuration
            device: Target device
            
        Returns:
            Created model
        """
        model = create_model(model_type, config)
        model = model.to(device or self.device)
        
        # Analyze model
        complexity = analyze_model_complexity(model)
        logger.info(f"Model created: {complexity['total_parameters']:,} parameters")
        
        return model
    
    def load_model(
        self,
        checkpoint_path: Path,
        model_class: type,
        device: Optional[torch.device] = None
    ) -> BaseModel:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            model_class: Model class
            device: Target device
            
        Returns:
            Loaded model
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = checkpoint.get('config', {})
        
        model = model_class(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device or self.device)
        
        logger.info(f"Model loaded from {checkpoint_path}")
        return model
    
    def analyze_model(self, model: nn.Module) -> Dict[str, Any]:
        """
        Analyze model comprehensively.
        
        Args:
            model: PyTorch model
            
        Returns:
            Analysis results
        """
        complexity = analyze_model_complexity(model)
        is_healthy, issues = check_model_health(model)
        
        return {
            'complexity': complexity,
            'health': {
                'is_healthy': is_healthy,
                'issues': issues
            }
        }
    
    def optimize_for_inference(self, model: nn.Module) -> nn.Module:
        """
        Optimize model for inference.
        
        Args:
            model: PyTorch model
            
        Returns:
            Optimized model
        """
        return optimize_model_for_inference(model)
    
    def setup_checkpoint_manager(self, checkpoint_dir: Path) -> None:
        """
        Setup checkpoint manager.
        
        Args:
            checkpoint_dir: Checkpoint directory
        """
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        logger.info(f"Checkpoint manager setup: {checkpoint_dir}")



