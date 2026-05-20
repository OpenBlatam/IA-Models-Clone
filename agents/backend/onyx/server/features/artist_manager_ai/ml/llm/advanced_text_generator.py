"""
Advanced Text Generator
=======================

Advanced text generation with better tokenization and attention handling.
"""

import torch
import logging
from typing import List, Dict, Any, Optional, Union
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    GenerationConfig,
    pipeline
)

logger = logging.getLogger(__name__)


class AdvancedTextGenerator:
    """
    Advanced text generator with:
    - Better tokenization
    - Attention visualization
    - Custom generation strategies
    - Temperature scheduling
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        model_type: str = "causal",
        device: Optional[str] = None,
        use_flash_attention: bool = False
    ):
        """
        Initialize advanced text generator.
        
        Args:
            model_name: HuggingFace model name
            model_type: "causal" or "seq2seq"
            device: Device
            use_flash_attention: Use flash attention (if available)
        """
        self.model_name = model_name
        self.model_type = model_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_flash_attention = use_flash_attention
        
        logger.info(f"Loading model: {model_name} on {self.device}")
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            if self.model_type == "causal":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
            else:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
            
            if self.device != "cuda" or not hasattr(self.model, 'device'):
                self.model.to(self.device)
            
            self.model.eval()
            
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def generate_with_strategy(
        self,
        prompt: str,
        strategy: str = "greedy",
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_beams: int = 5,
        **kwargs
    ) -> List[str]:
        """
        Generate with different strategies.
        
        Args:
            prompt: Input prompt
            strategy: Generation strategy ("greedy", "sampling", "beam", "nucleus")
            max_length: Maximum length
            temperature: Temperature
            top_p: Nucleus sampling
            top_k: Top-k sampling
            num_beams: Number of beams
            **kwargs: Additional parameters
        
        Returns:
            Generated texts
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        generation_config = GenerationConfig(
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=strategy != "greedy",
            num_beams=num_beams if strategy == "beam" else 1,
            **kwargs
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                generation_config=generation_config
            )
        
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return generated_texts
    
    def generate_with_temperature_schedule(
        self,
        prompt: str,
        temperature_schedule: List[float],
        max_length: int = 100
    ) -> str:
        """
        Generate with temperature scheduling.
        
        Args:
            prompt: Input prompt
            temperature_schedule: List of temperatures for each step
            max_length: Maximum length
        
        Returns:
            Generated text
        """
        # This is a simplified version
        # In practice, you'd need to modify the generation loop
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        generated_ids = inputs.input_ids.clone()
        
        for step in range(max_length - inputs.input_ids.size(1)):
            if step < len(temperature_schedule):
                temp = temperature_schedule[step]
            else:
                temp = temperature_schedule[-1] if temperature_schedule else 0.7
            
            with torch.no_grad():
                outputs = self.model(generated_ids)
                logits = outputs.logits[:, -1, :] / temp
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
        
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)




