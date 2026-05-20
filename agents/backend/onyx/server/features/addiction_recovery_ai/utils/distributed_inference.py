"""
Distributed Inference for Multi-GPU
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class DistributedInference:
    """Distributed inference across multiple GPUs"""
    
    def __init__(
        self,
        model: nn.Module,
        device_ids: Optional[List[int]] = None
    ):
        """
        Initialize distributed inference
        
        Args:
            model: Model to distribute
            device_ids: List of device IDs
        """
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        
        if len(device_ids) == 0:
            raise ValueError("No GPUs available for distributed inference")
        
        self.device_ids = device_ids
        self.models = []
        
        # Create model copies for each GPU
        for device_id in device_ids:
            model_copy = model.to(f"cuda:{device_id}")
            model_copy.eval()
            self.models.append(model_copy)
        
        logger.info(f"DistributedInference initialized on {len(device_ids)} GPUs")
    
    def predict_batch(
        self,
        inputs: torch.Tensor,
        batch_size_per_gpu: Optional[int] = None
    ) -> torch.Tensor:
        """
        Predict batch across GPUs
        
        Args:
            inputs: Input tensor
            batch_size_per_gpu: Batch size per GPU
        
        Returns:
            Predictions
        """
        if batch_size_per_gpu is None:
            batch_size_per_gpu = inputs.shape[0] // len(self.device_ids)
        
        # Split batch across GPUs
        results = []
        for i, model in enumerate(self.models):
            start_idx = i * batch_size_per_gpu
            end_idx = start_idx + batch_size_per_gpu if i < len(self.models) - 1 else inputs.shape[0]
            
            batch = inputs[start_idx:end_idx].to(f"cuda:{self.device_ids[i]}")
            
            with torch.no_grad():
                output = model(batch)
            
            results.append(output.cpu())
        
        # Concatenate results
        return torch.cat(results, dim=0)
    
    def predict_parallel(
        self,
        inputs_list: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Predict multiple inputs in parallel
        
        Args:
            inputs_list: List of input tensors
        
        Returns:
            List of predictions
        """
        import concurrent.futures
        
        def predict_one(model, inputs, device_id):
            inputs = inputs.to(f"cuda:{device_id}")
            with torch.no_grad():
                return model(inputs).cpu()
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.models)) as executor:
            futures = []
            for i, inputs in enumerate(inputs_list):
                model_idx = i % len(self.models)
                future = executor.submit(
                    predict_one,
                    self.models[model_idx],
                    inputs,
                    self.device_ids[model_idx]
                )
                futures.append(future)
            
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        
        return results

