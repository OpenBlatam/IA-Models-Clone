"""
Text Summarization Model
Uses BART or other transformer models for abstractive summarization
"""

import logging
import torch
from typing import Dict, Any, Optional
from transformers import pipeline

from .base import BaseModel

logger = logging.getLogger(__name__)


class SummarizationModel(BaseModel):
    """
    Transformer-based text summarization model
    """
    
    def __init__(
        self,
        model_name: str = "facebook/bart-large-cnn",
        device: Optional[torch.device] = None,
        use_mixed_precision: bool = False,
    ):
        """
        Initialize summarization model
        
        Args:
            model_name: HuggingFace model identifier
            device: PyTorch device
            use_mixed_precision: Use mixed precision for inference
        """
        super().__init__(model_name, device, use_mixed_precision)
        self.transformer_model_name = model_name
    
    async def load(self) -> None:
        """Load summarization pipeline"""
        if self.is_loaded:
            return
        
        try:
            logger.info(f"Loading summarization model: {self.transformer_model_name}")
            self.model = pipeline(
                "summarization",
                model=self.transformer_model_name,
                device=0 if self.device.type == "cuda" else -1,
            )
            self.is_loaded = True
            logger.info(f"Successfully loaded summarization model: {self.transformer_model_name}")
        except Exception as e:
            logger.error(f"Error loading summarization model: {e}")
            raise
    
    async def predict(
        self,
        inputs: str,
        max_length: int = 150,
        min_length: int = 30,
        do_sample: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate summary of input text
        
        Args:
            inputs: Text to summarize
            max_length: Maximum length of summary
            min_length: Minimum length of summary
            do_sample: Whether to use sampling
            
        Returns:
            Dictionary with summary and metadata
        """
        if not self.is_loaded:
            await self.load()
        
        try:
            # Truncate text if too long (BART has token limits)
            original_text = inputs
            if len(inputs) > 1024:
                inputs = inputs[:1024]
                logger.warning("Text truncated to 1024 characters for summarization")
            
            summary_result = self.model(
                inputs,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
            )
            
            summary_text = summary_result[0]["summary_text"]
            
            return {
                "summary": summary_text,
                "original_length": len(original_text),
                "summary_length": len(summary_text),
                "compression_ratio": len(summary_text) / len(original_text) if original_text else 0.0,
            }
        except Exception as e:
            logger.error(f"Error in summarization: {e}")
            raise



