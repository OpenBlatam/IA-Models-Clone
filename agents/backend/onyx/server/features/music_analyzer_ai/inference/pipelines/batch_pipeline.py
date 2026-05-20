"""
Batch Inference Pipeline

Optimized pipeline for batch inference with proper
batching, device management, and memory optimization.
"""

from typing import List, Any, Optional, Dict, Callable
import logging
import torch
from torch.utils.data import DataLoader

from .base_pipeline import BaseInferencePipeline

logger = logging.getLogger(__name__)


class BatchInferencePipeline(BaseInferencePipeline):
    """
    Batch inference pipeline with optimizations.
    
    Features:
    - Efficient batching
    - Automatic device management
    - Memory optimization
    - Progress tracking
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
        batch_size: int = 32,
        use_mixed_precision: bool = True,
        num_workers: int = 0
    ):
        """
        Initialize batch inference pipeline.
        
        Args:
            model: Model for inference
            device: Device to use (auto-detect if None)
            batch_size: Batch size for inference
            use_mixed_precision: Enable mixed precision
            num_workers: Number of worker processes
        """
        super().__init__(model)
        self.device = device or self._get_device()
        self.batch_size = batch_size
        self.use_mixed_precision = (
            use_mixed_precision and 
            self.device.type == "cuda"
        )
        self.num_workers = num_workers
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def _get_device(self) -> torch.device:
        """Get the appropriate device."""
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def predict_batch(
        self,
        inputs: List[Any],
        collate_fn: Optional[Callable] = None,
        **kwargs
    ) -> List[Any]:
        """
        Make batch predictions.
        
        Args:
            inputs: List of input samples
            collate_fn: Optional collate function
            **kwargs: Additional arguments
            
        Returns:
            List of predictions
        """
        # Create dataset and loader
        from torch.utils.data import Dataset, DataLoader
        
        class SimpleDataset(Dataset):
            def __init__(self, data):
                self.data = data
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        dataset = SimpleDataset(inputs)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )
        
        all_predictions = []
        
        with torch.no_grad():
            for batch in loader:
                # Move batch to device
                if isinstance(batch, (dict, list, tuple)):
                    batch = self._move_to_device(batch)
                else:
                    batch = batch.to(self.device)
                
                # Forward pass
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch)
                else:
                    outputs = self.model(batch)
                
                # Collect predictions
                if isinstance(outputs, torch.Tensor):
                    all_predictions.append(outputs.cpu())
                elif isinstance(outputs, (list, tuple)):
                    all_predictions.extend([o.cpu() for o in outputs])
                else:
                    all_predictions.append(outputs)
        
        # Concatenate if tensors
        if all_predictions and isinstance(all_predictions[0], torch.Tensor):
            return torch.cat(all_predictions, dim=0)
        else:
            # Flatten list
            result = []
            for pred in all_predictions:
                if isinstance(pred, (list, tuple)):
                    result.extend(pred)
                else:
                    result.append(pred)
            return result
    
    def _move_to_device(self, obj: Any) -> Any:
        """Move object(s) to device."""
        if isinstance(obj, torch.Tensor):
            return obj.to(self.device, non_blocking=True)
        elif isinstance(obj, dict):
            return {k: self._move_to_device(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self._move_to_device(item) for item in obj)
        return obj
    
    def predict(
        self,
        inputs: Any,
        **kwargs
    ) -> Any:
        """
        Make single prediction (wraps predict_batch).
        
        Args:
            inputs: Input sample
            **kwargs: Additional arguments
            
        Returns:
            Prediction
        """
        results = self.predict_batch([inputs], **kwargs)
        return results[0] if results else None


__all__ = [
    "BatchInferencePipeline",
]



