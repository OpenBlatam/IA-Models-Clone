"""
Model Predictor
Modular inference utilities
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelPredictor:
    """
    Model predictor for single and batch inference
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        use_mixed_precision: bool = False,
    ):
        """
        Initialize predictor
        
        Args:
            model: Model to use for prediction
            device: Device for inference
            use_mixed_precision: Use mixed precision
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.use_mixed_precision = use_mixed_precision
    
    def predict(
        self,
        inputs: torch.Tensor,
        return_probabilities: bool = True,
        return_features: bool = False,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Predict on single input
        
        Args:
            inputs: Input tensor
            return_probabilities: Return probabilities
            return_features: Return intermediate features
            top_k: Return top k predictions
            
        Returns:
            Dictionary with predictions
        """
        inputs = inputs.to(self.device)
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)
        
        with torch.no_grad():
            if self.use_mixed_precision and self.device.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
            else:
                outputs = self.model(inputs)
        
        result = self._process_outputs(
            outputs,
            return_probabilities,
            return_features,
            top_k
        )
        
        return result
    
    def predict_batch(
        self,
        inputs: torch.Tensor,
        batch_size: int = 32,
        return_probabilities: bool = True,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Predict on batch of inputs
        
        Args:
            inputs: Input tensor batch
            batch_size: Batch size for processing
            return_probabilities: Return probabilities
            top_k: Return top k predictions
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        num_samples = inputs.size(0)
        
        for i in range(0, num_samples, batch_size):
            batch = inputs[i:i + batch_size].to(self.device)
            
            with torch.no_grad():
                if self.use_mixed_precision and self.device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch)
                else:
                    outputs = self.model(batch)
            
            batch_results = self._process_outputs(
                outputs,
                return_probabilities,
                False,
                top_k
            )
            
            # Convert to list of individual predictions
            if isinstance(batch_results['predictions'], list):
                for j, pred in enumerate(batch_results['predictions']):
                    result = {'prediction': pred}
                    if return_probabilities and 'probabilities' in batch_results:
                        result['probability'] = batch_results['probabilities'][j]
                    if top_k and 'top_k' in batch_results:
                        result['top_k'] = batch_results['top_k'][j]
                    results.append(result)
            else:
                results.append(batch_results)
        
        return results
    
    def _process_outputs(
        self,
        outputs: torch.Tensor,
        return_probabilities: bool,
        return_features: bool,
        top_k: Optional[int],
    ) -> Dict[str, Any]:
        """Process model outputs"""
        import torch.nn.functional as F
        
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)
        
        result = {
            'predictions': preds.cpu().numpy().tolist(),
        }
        
        if return_probabilities:
            result['probabilities'] = probs.cpu().numpy().tolist()
        
        if top_k:
            top_probs, top_indices = torch.topk(probs, k=min(top_k, outputs.size(1)), dim=1)
            result['top_k'] = {
                'indices': top_indices.cpu().numpy().tolist(),
                'probabilities': top_probs.cpu().numpy().tolist(),
            }
        
        return result


class BatchPredictor:
    """
    Efficient batch prediction with batching and async support
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        batch_size: int = 32,
        use_mixed_precision: bool = False,
    ):
        """
        Initialize batch predictor
        
        Args:
            model: Model to use
            device: Device for inference
            batch_size: Batch size
            use_mixed_precision: Use mixed precision
        """
        self.predictor = ModelPredictor(model, device, use_mixed_precision)
        self.batch_size = batch_size
    
    def predict(
        self,
        inputs: Union[torch.Tensor, List[torch.Tensor]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Predict on inputs with automatic batching
        
        Args:
            inputs: Input tensor or list of tensors
            **kwargs: Additional prediction arguments
            
        Returns:
            List of predictions
        """
        if isinstance(inputs, list):
            # Stack into batch
            inputs = torch.stack(inputs)
        
        return self.predictor.predict_batch(
            inputs,
            batch_size=self.batch_size,
            **kwargs
        )



