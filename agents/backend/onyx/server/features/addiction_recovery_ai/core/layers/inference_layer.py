"""
Inference Layer - Ultra Modular Inference Components
Separates prediction logic into reusable components
"""

from typing import Optional, Dict, Any, List, Callable
import torch
import torch.nn as nn
import logging
from abc import ABC, abstractmethod

from .interfaces import IPredictor, IInferenceEngine, BasePredictor

logger = logging.getLogger(__name__)


# ============================================================================
# Inference Engine - Core inference processing
# ============================================================================

class InferenceEngine:
    """
    Modular inference engine
    Handles model inference with optimizations
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        use_mixed_precision: bool = True
    ):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_mixed_precision = use_mixed_precision and self.device.type == "cuda"
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Enable optimizations
        if hasattr(torch, 'compile') and self.device.type == "cuda":
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")
    
    def process(self, inputs: Any, **kwargs) -> Any:
        """Process inference request"""
        # Move inputs to device
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(self.device)
        
        # Inference with mixed precision
        with torch.inference_mode():
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs, **kwargs)
            else:
                outputs = self.model(inputs, **kwargs)
        
        return outputs
    
    def optimize(self):
        """Optimize model for inference"""
        # Additional optimizations can be added here
        pass


# ============================================================================
# Batch Processor - Handle batch inference
# ============================================================================

class BatchProcessor:
    """Process batches efficiently"""
    
    def __init__(
        self,
        engine: InferenceEngine,
        batch_size: int = 32
    ):
        self.engine = engine
        self.batch_size = batch_size
    
    def process_batch(self, inputs: List[Any], **kwargs) -> List[Any]:
        """Process batch of inputs"""
        results = []
        
        for i in range(0, len(inputs), self.batch_size):
            batch = inputs[i:i + self.batch_size]
            
            # Convert to tensor if needed
            if isinstance(batch[0], torch.Tensor):
                batch_tensor = torch.stack(batch)
            else:
                batch_tensor = torch.tensor(batch)
            
            # Process batch
            batch_results = self.engine.process(batch_tensor, **kwargs)
            
            # Convert back to list
            if isinstance(batch_results, torch.Tensor):
                results.extend(batch_results.cpu().tolist())
            else:
                results.extend(batch_results)
        
        return results


# ============================================================================
# Predictor Factory
# ============================================================================

class PredictorFactory:
    """Factory for creating predictors"""
    
    _registry: Dict[str, Callable] = {}
    
    @classmethod
    def register(cls, name: str, predictor_class: type):
        """Register predictor class"""
        cls._registry[name] = predictor_class
        logger.info(f"Registered predictor: {name}")
    
    @classmethod
    def create(
        cls,
        predictor_type: str,
        model: nn.Module,
        **kwargs
    ) -> IPredictor:
        """Create predictor"""
        if predictor_type not in cls._registry:
            raise ValueError(f"Unknown predictor type: {predictor_type}")
        
        predictor_class = cls._registry[predictor_type]
        return predictor_class(model, **kwargs)


# ============================================================================
# Inference Pipeline - Composable inference pipeline
# ============================================================================

class InferencePipeline:
    """
    Composable inference pipeline
    Chains preprocessing, inference, and postprocessing
    """
    
    def __init__(self, engine: InferenceEngine):
        self.engine = engine
        self.preprocessors: List[Callable] = []
        self.postprocessors: List[Callable] = []
    
    def add_preprocessor(self, preprocessor: Callable) -> 'InferencePipeline':
        """Add preprocessing step"""
        self.preprocessors.append(preprocessor)
        return self
    
    def add_postprocessor(self, postprocessor: Callable) -> 'InferencePipeline':
        """Add postprocessing step"""
        self.postprocessors.append(postprocessor)
        return self
    
    def process(self, inputs: Any, **kwargs) -> Any:
        """Process through pipeline"""
        # Preprocessing
        processed = inputs
        for preprocessor in self.preprocessors:
            processed = preprocessor(processed)
        
        # Inference
        outputs = self.engine.process(processed, **kwargs)
        
        # Postprocessing
        result = outputs
        for postprocessor in self.postprocessors:
            result = postprocessor(result)
        
        return result


# Export main components
__all__ = [
    "InferenceEngine",
    "BatchProcessor",
    "PredictorFactory",
    "InferencePipeline",
]



