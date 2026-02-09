"""
Inference Engine - Model Inference Utilities
=============================================

Provides utilities for running model inference with proper error handling.
"""

import logging
from typing import Dict, Any, Optional, List, Callable, Union
import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Engine for running model inference with error handling and optimization.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        use_mixed_precision: bool = True
    ):
        """
        Initialize inference engine.
        
        Args:
            model: PyTorch model
            device: Device to run inference on
            use_mixed_precision: Use mixed precision for inference
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_mixed_precision = use_mixed_precision and torch.cuda.is_available()
        
        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Inference engine initialized on {self.device}")
    
    def predict(
        self,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor], List[Any]],
        return_probabilities: bool = False,
        top_k: Optional[int] = None
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Run inference on inputs.
        
        Args:
            inputs: Input data (tensor, dict, or list)
            return_probabilities: Whether to return probabilities
            top_k: Return top-k predictions (for classification)
            
        Returns:
            Model predictions
        """
        try:
            # Prepare inputs
            inputs = self._prepare_inputs(inputs)
            
            # Run inference
            with torch.no_grad():
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self._forward(inputs)
                else:
                    outputs = self._forward(inputs)
            
            # Post-process outputs
            if return_probabilities or top_k is not None:
                if outputs.dim() > 1:
                    probs = torch.softmax(outputs, dim=-1)
                    if top_k is not None:
                        top_probs, top_indices = torch.topk(probs, k=min(top_k, probs.size(-1)))
                        return {
                            'predictions': top_indices,
                            'probabilities': top_probs,
                            'raw_outputs': outputs
                        }
                    return {
                        'probabilities': probs,
                        'raw_outputs': outputs
                    }
                else:
                    probs = torch.sigmoid(outputs)
                    return {
                        'probabilities': probs,
                        'raw_outputs': outputs
                    }
            
            return outputs
            
        except Exception as e:
            logger.error(f"Error during inference: {e}", exc_info=True)
            raise
    
    def batch_predict(
        self,
        dataloader: torch.utils.data.DataLoader,
        max_batches: Optional[int] = None
    ) -> List[torch.Tensor]:
        """
        Run batch inference on a DataLoader.
        
        Args:
            dataloader: DataLoader with batches
            max_batches: Maximum number of batches to process
            
        Returns:
            List of predictions for each batch
        """
        all_predictions = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                
                try:
                    batch = self._prepare_inputs(batch)
                    
                    if self.use_mixed_precision:
                        with torch.cuda.amp.autocast():
                            outputs = self._forward(batch)
                    else:
                        outputs = self._forward(batch)
                    
                    all_predictions.append(outputs)
                    
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {e}")
                    continue
        
        return all_predictions
    
    def _forward(self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Forward pass through model."""
        if isinstance(inputs, dict):
            return self.model(**inputs)
        elif isinstance(inputs, (tuple, list)):
            return self.model(*inputs)
        else:
            return self.model(inputs)
    
    def _prepare_inputs(
        self,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor], List[Any]]
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Prepare inputs for model."""
        if isinstance(inputs, dict):
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                   for k, v in inputs.items()}
        elif isinstance(inputs, (tuple, list)):
            return tuple(v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for v in inputs)
        elif isinstance(inputs, torch.Tensor):
            return inputs.to(self.device)
        else:
            # Try to convert to tensor
            try:
                tensor = torch.tensor(inputs)
                return tensor.to(self.device)
            except Exception as e:
                raise ValueError(f"Cannot prepare inputs: {e}")


def batch_inference(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: Optional[torch.device] = None
) -> List[torch.Tensor]:
    """
    Convenience function for batch inference.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader with batches
        device: Device to run inference on
        
    Returns:
        List of predictions
    """
    engine = InferenceEngine(model, device)
    return engine.batch_predict(dataloader)



