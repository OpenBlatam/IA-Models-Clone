"""
Base Model Interface
Abstract base classes for all models
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all recovery models
    Provides common interface and functionality
    """
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        use_mixed_precision: bool = True
    ):
        """
        Initialize base model
        
        Args:
            device: PyTorch device
            use_mixed_precision: Use mixed precision
        """
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_mixed_precision = use_mixed_precision and self.device.type == "cuda"
        self._is_compiled = False
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass - must be implemented by subclasses"""
        pass
    
    def compile(self, mode: str = "reduce-overhead") -> 'BaseModel':
        """
        Compile model for faster inference
        
        Args:
            mode: Compilation mode
            
        Returns:
            Compiled model
        """
        if hasattr(torch, 'compile') and self.device.type == "cuda":
            try:
                self._is_compiled = True
                return torch.compile(self, mode=mode, fullgraph=True)
            except Exception as e:
                logger.warning(f"Compilation failed: {e}")
        return self
    
    def to_device(self, device: Optional[torch.device] = None) -> 'BaseModel':
        """Move model to device"""
        self.device = device or self.device
        return self.to(self.device)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device),
            "dtype": str(next(self.parameters()).dtype) if len(list(self.parameters())) > 0 else "unknown",
            "is_compiled": self._is_compiled,
            "use_mixed_precision": self.use_mixed_precision
        }


class BasePredictor(BaseModel):
    """
    Base class for prediction models
    """
    
    @abstractmethod
    def predict(self, inputs: Any, **kwargs) -> Any:
        """
        Make prediction
        
        Args:
            inputs: Input data
            **kwargs: Additional arguments
            
        Returns:
            Predictions
        """
        pass
    
    @torch.inference_mode()
    def predict_batch(self, inputs: List[Any], batch_size: int = 32, **kwargs) -> List[Any]:
        """
        Batch prediction
        
        Args:
            inputs: List of inputs
            batch_size: Batch size
            **kwargs: Additional arguments
            
        Returns:
            List of predictions
        """
        results = []
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            batch_results = self._process_batch(batch, **kwargs)
            results.extend(batch_results)
        return results
    
    @abstractmethod
    def _process_batch(self, batch: List[Any], **kwargs) -> List[Any]:
        """Process batch - must be implemented by subclasses"""
        pass


class BaseGenerator(BaseModel):
    """
    Base class for generation models (LLM, Diffusion, etc.)
    """
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Any:
        """
        Generate output from prompt
        
        Args:
            prompt: Input prompt
            **kwargs: Generation parameters
            
        Returns:
            Generated output
        """
        pass
    
    def generate_batch(self, prompts: List[str], **kwargs) -> List[Any]:
        """
        Batch generation
        
        Args:
            prompts: List of prompts
            **kwargs: Generation parameters
            
        Returns:
            List of generated outputs
        """
        return [self.generate(prompt, **kwargs) for prompt in prompts]


class BaseAnalyzer(BaseModel):
    """
    Base class for analysis models (sentiment, etc.)
    """
    
    @abstractmethod
    def analyze(self, inputs: Any, **kwargs) -> Dict[str, Any]:
        """
        Analyze inputs
        
        Args:
            inputs: Input data
            **kwargs: Analysis parameters
            
        Returns:
            Analysis results
        """
        pass
    
    def analyze_batch(self, inputs: List[Any], batch_size: int = 32, **kwargs) -> List[Dict[str, Any]]:
        """
        Batch analysis
        
        Args:
            inputs: List of inputs
            batch_size: Batch size
            **kwargs: Analysis parameters
            
        Returns:
            List of analysis results
        """
        results = []
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            batch_results = [self.analyze(item, **kwargs) for item in batch]
            results.extend(batch_results)
        return results













