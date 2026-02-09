"""
Model loading and initialization
"""

import logging
from typing import Dict, Any, Optional, Tuple
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles model and tokenizer loading"""
    
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        max_length: int,
        max_target_length: int
    ):
        """Initialize model loader
        
        Args:
            model_name: Name of the model to load
            device: Device to load model on
            max_length: Maximum input length
            max_target_length: Maximum target length
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.max_target_length = max_target_length
        self.model: Optional[AutoModelForSeq2SeqLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self._initialized = False
    
    def load(self) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
        """Load model and tokenizer
        
        Returns:
            Tuple of (model, tokenizer)
            
        Raises:
            RuntimeError: If loading fails
        """
        if self._initialized and self.model is not None and self.tokenizer is not None:
            return self.model, self.tokenizer
        
        try:
            logger.info(f"Loading tokenizer for {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True
            )
            
            logger.info(f"Loading model {self.model_name}...")
            model_kwargs: Dict[str, Any] = {
                "torch_dtype": torch.float16 if self.device.type == "cuda" else torch.float32,
                "low_cpu_mem_usage": True,
            }
            
            if self.device.type == "cuda":
                model_kwargs["device_map"] = "auto"
            
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            if self.device.type == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            # Enable optimizations if available
            if hasattr(torch, "compile") and self.device.type == "cuda":
                try:
                    self.model = torch.compile(self.model)
                    logger.info("Model compiled with torch.compile for better performance")
                except Exception as compile_error:
                    logger.warning(f"Could not compile model: {compile_error}")
            
            self._initialized = True
            
            num_params = sum(p.numel() for p in self.model.parameters())
            logger.info(
                f"Model loaded - name: {self.model_name}, "
                f"device: {self.device}, params: {num_params:,}, "
                f"max_length: {self.max_length}, max_target_length: {self.max_target_length}"
            )
            
            return self.model, self.tokenizer
            
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            self.model = None
            self.tokenizer = None
            self._initialized = False
            raise RuntimeError(f"Failed to load model: {e}") from e
    
    def save(self, path: str) -> None:
        """Save model and tokenizer
        
        Args:
            path: Path to save to
            
        Raises:
            RuntimeError: If model not initialized or save fails
        """
        if not self._initialized or self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not initialized")
        
        try:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to save model: {e}") from e

