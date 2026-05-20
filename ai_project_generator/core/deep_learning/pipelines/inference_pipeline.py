"""
Inference Pipeline - Complete Inference Workflow
================================================

High-level pipeline for model inference:
- Model loading
- Preprocessing
- Inference
- Postprocessing
- Batch processing
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..core.base import BaseComponent
from ..inference import InferenceEngine
from ..utils import get_device

logger = logging.getLogger(__name__)


class InferencePipeline(BaseComponent):
    """
    Complete inference pipeline.
    
    Handles model loading, preprocessing, inference, and postprocessing.
    """
    
    def _initialize(self) -> None:
        """Initialize pipeline."""
        self.device = get_device()
        self.engine = None
        self.model = None
        self.preprocess_fn: Optional[Callable] = None
        self.postprocess_fn: Optional[Callable] = None
    
    def setup(
        self,
        model: nn.Module,
        preprocess_fn: Optional[Callable] = None,
        postprocess_fn: Optional[Callable] = None,
        use_mixed_precision: bool = True
    ) -> 'InferencePipeline':
        """
        Setup inference pipeline.
        
        Args:
            model: PyTorch model
            preprocess_fn: Preprocessing function
            postprocess_fn: Postprocessing function
            use_mixed_precision: Use mixed precision
            
        Returns:
            Self for method chaining
        """
        self.model = model
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.engine = InferenceEngine(
            model=self.model,
            device=self.device,
            use_mixed_precision=use_mixed_precision
        )
        
        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn
        
        return self
    
    def load_from_checkpoint(
        self,
        checkpoint_path: Path,
        model_class: type,
        preprocess_fn: Optional[Callable] = None,
        postprocess_fn: Optional[Callable] = None
    ) -> 'InferencePipeline':
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            model_class: Model class
            preprocess_fn: Preprocessing function
            postprocess_fn: Postprocessing function
            
        Returns:
            Self for method chaining
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Create model
        config = checkpoint.get('config', {})
        self.model = model_class(**config)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Setup engine
        self.engine = InferenceEngine(self.model, device=self.device)
        
        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn
        
        logger.info(f"Model loaded from {checkpoint_path}")
        
        return self
    
    def predict(
        self,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor], List[Any]],
        return_probabilities: bool = False,
        top_k: Optional[int] = None
    ) -> Any:
        """
        Run inference on inputs.
        
        Args:
            inputs: Input data
            return_probabilities: Return probabilities
            top_k: Return top-k predictions
            
        Returns:
            Predictions
        """
        if self.engine is None:
            raise RuntimeError("Pipeline not setup. Call setup() or load_from_checkpoint() first.")
        
        # Preprocess
        if self.preprocess_fn:
            inputs = self.preprocess_fn(inputs)
        
        # Inference
        predictions = self.engine.predict(
            inputs,
            return_probabilities=return_probabilities,
            top_k=top_k
        )
        
        # Postprocess
        if self.postprocess_fn:
            predictions = self.postprocess_fn(predictions)
        
        return predictions
    
    def predict_batch(
        self,
        dataloader: DataLoader,
        max_batches: Optional[int] = None
    ) -> List[Any]:
        """
        Run batch inference.
        
        Args:
            dataloader: DataLoader with batches
            max_batches: Maximum number of batches
            
        Returns:
            List of predictions
        """
        if self.engine is None:
            raise RuntimeError("Pipeline not setup.")
        
        all_predictions = []
        
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            
            # Preprocess
            if self.preprocess_fn:
                batch = self.preprocess_fn(batch)
            
            # Inference
            predictions = self.engine.predict(batch)
            
            # Postprocess
            if self.postprocess_fn:
                predictions = self.postprocess_fn(predictions)
            
            all_predictions.append(predictions)
        
        return all_predictions



