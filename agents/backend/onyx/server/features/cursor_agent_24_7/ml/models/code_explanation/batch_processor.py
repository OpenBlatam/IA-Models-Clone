"""
Batch processing for code explanations
"""

import logging
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig

from .validator import InputValidator
from .prompt_builder import PromptBuilder
from .cache import ExplanationCache

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Processes batches of code explanations"""
    
    def __init__(
        self,
        model: AutoModelForSeq2SeqLM,
        tokenizer: AutoTokenizer,
        device: torch.device,
        max_length: int,
        max_target_length: int,
        cache: Optional[ExplanationCache] = None
    ):
        """Initialize batch processor
        
        Args:
            model: Model instance
            tokenizer: Tokenizer instance
            device: Device to run on
            max_length: Maximum input length
            max_target_length: Maximum target length
            cache: Cache instance (optional)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.max_target_length = max_target_length
        self.cache = cache
        self.validator = InputValidator()
        self.prompt_builder = PromptBuilder()
    
    def process(
        self,
        codes: List[str],
        max_length: Optional[int] = None,
        temperature: float = 0.7,
        num_beams: int = 4,
        do_sample: bool = True,
        **kwargs: Any
    ) -> List[str]:
        """Process batch of codes
        
        Args:
            codes: List of codes to explain
            max_length: Maximum generation length
            temperature: Sampling temperature
            num_beams: Number of beams
            **kwargs: Additional generation parameters
            
        Returns:
            List of explanations
            
        Raises:
            ValueError: If no valid codes provided
            RuntimeError: If processing fails
        """
        # Validate and filter codes (validación mejorada con min_valid)
        valid_codes = self.validator.validate_batch_codes(codes, min_valid=1)
        
        # Build prompts
        prompts = self.prompt_builder.build_batch(valid_codes)
        
        try:
            # Tokenize batch
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True
            ).to(self.device)
            
            max_gen_length = max_length or self.max_target_length
            
            # Configuración de generación
            generation_config = GenerationConfig(
                max_length=max_gen_length,
                temperature=temperature,
                num_beams=num_beams if not do_sample else 1,
                do_sample=do_sample,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **{k: v for k, v in kwargs.items() 
                   if k not in ["max_length", "temperature", "num_beams", "do_sample"]}
            )
            
            # Generate batch
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config
                )
            
            # Decode batch
            explanations = [
                self.tokenizer.decode(output, skip_special_tokens=True).strip()
                for output in outputs
            ]
            
            # Cache results
            if self.cache and self.cache.enabled:
                for code, explanation in zip(valid_codes, explanations):
                    if explanation:
                        self.cache.set(
                            code,
                            explanation,
                            max_length=max_length,
                            temperature=temperature,
                            num_beams=num_beams,
                            do_sample=do_sample,
                            **kwargs
                        )
            
            avg_len = sum(len(e) for e in explanations) / len(explanations) if explanations else 0
            logger.debug(
                f"Batch processed - size: {len(valid_codes)}, "
                f"avg_explanation_length: {avg_len:.1f}"
            )
            
            return explanations
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}", exc_info=True)
            raise RuntimeError(f"Failed to process batch: {e}") from e

