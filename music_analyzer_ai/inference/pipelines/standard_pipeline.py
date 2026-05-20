"""
Standard Inference Pipeline
Single sample inference pipeline
"""

from typing import Dict, Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

from .base_pipeline import BaseInferencePipeline


class StandardInferencePipeline(BaseInferencePipeline):
    """
    Standard inference pipeline for single samples
    """
    
    def __init__(
        self,
        model,
        preprocess_fn: Optional[Callable] = None,
        postprocess_fn: Optional[Callable] = None,
        device: str = "cuda",
        use_mixed_precision: bool = True
    ):
        super().__init__(model, preprocess_fn, postprocess_fn, device)
        self.use_mixed_precision = use_mixed_precision and device == "cuda"
    
    def predict(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """Run inference on single sample"""
        try:
            # Preprocess
            processed_input = self.preprocess(input_data)
            
            # Convert to tensor if needed
            if isinstance(processed_input, np.ndarray):
                processed_input = torch.from_numpy(processed_input).float()
            
            # Add batch dimension if needed
            if processed_input.dim() == 1:
                processed_input = processed_input.unsqueeze(0)
            
            # Move to device
            processed_input = processed_input.to(self.device)
            
            # Inference
            with torch.no_grad():
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        output = self.model(processed_input, **kwargs)
                else:
                    output = self.model(processed_input, **kwargs)
            
            # Postprocess
            if isinstance(output, torch.Tensor):
                output = output.cpu().numpy()
            
            output = self.postprocess(output)
            
            return {
                "success": True,
                "output": output
            }
        
        except Exception as e:
            logger.error(f"Inference error: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }



