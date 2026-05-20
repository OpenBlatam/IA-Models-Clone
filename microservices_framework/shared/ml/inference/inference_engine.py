"""
Inference Engine
Optimized inference with batching, caching, and GPU optimization.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union
from functools import lru_cache
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    High-performance inference engine with optimizations.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: Optional[str] = None,
        use_amp: bool = True,
        max_batch_size: int = 32,
        compile_model: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp and self.device == "cuda"
        self.max_batch_size = max_batch_size
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Compile model for PyTorch 2.0+ (optional)
        if compile_model and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
    
    @torch.no_grad()
    def generate(
        self,
        prompts: Union[str, List[str]],
        max_length: int = 100,
        max_new_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate text with optimized batching.
        
        Args:
            prompts: Single prompt or list of prompts
            max_length: Maximum sequence length
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
            do_sample: Whether to sample
            num_return_sequences: Number of sequences to return
            
        Returns:
            Generated text(s)
        """
        is_single = isinstance(prompts, str)
        if is_single:
            prompts = [prompts]
        
        # Tokenize
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        
        # Generate
        with self._autocast_context():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                **kwargs
            )
        
        # Decode
        generated_texts = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
        )
        
        # Remove input prompts from generated text
        if num_return_sequences == 1:
            generated_texts = [
                text[len(prompt):].strip()
                for text, prompt in zip(generated_texts, prompts)
            ]
        else:
            # Handle multiple sequences per prompt
            result = []
            for i, prompt in enumerate(prompts):
                prompt_texts = generated_texts[i * num_return_sequences:(i + 1) * num_return_sequences]
                result.append([text[len(prompt):].strip() for text in prompt_texts])
            generated_texts = result
        
        return generated_texts[0] if is_single else generated_texts
    
    @torch.no_grad()
    def get_embeddings(
        self,
        texts: List[str],
        normalize: bool = True,
        pooling: str = "mean",
    ) -> torch.Tensor:
        """
        Get embeddings for texts.
        
        Args:
            texts: List of input texts
            normalize: Whether to normalize embeddings
            pooling: Pooling strategy (mean, cls, max)
            
        Returns:
            Embeddings tensor [batch_size, hidden_size]
        """
        # Tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        
        # Forward pass
        with self._autocast_context():
            outputs = self.model(**inputs)
        
        # Extract embeddings
        if hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state
        elif hasattr(outputs, "hidden_states"):
            hidden_states = outputs.hidden_states[-1]
        else:
            hidden_states = outputs[0]
        
        # Pooling
        if pooling == "mean":
            # Mean pooling with attention mask
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                embeddings = (hidden_states * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
            else:
                embeddings = hidden_states.mean(dim=1)
        elif pooling == "cls":
            embeddings = hidden_states[:, 0, :]
        elif pooling == "max":
            embeddings = hidden_states.max(dim=1)[0]
        else:
            embeddings = hidden_states.mean(dim=1)
        
        # Normalize
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu()
    
    @contextmanager
    def _autocast_context(self):
        """Context manager for mixed precision."""
        if self.use_amp:
            with torch.cuda.amp.autocast():
                yield
        else:
            yield
    
    def batch_generate(
        self,
        prompts: List[str],
        batch_size: Optional[int] = None,
        **generation_kwargs
    ) -> List[str]:
        """
        Generate text for multiple prompts with batching.
        
        Args:
            prompts: List of prompts
            batch_size: Batch size (defaults to max_batch_size)
            **generation_kwargs: Generation parameters
            
        Returns:
            List of generated texts
        """
        batch_size = batch_size or self.max_batch_size
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_results = self.generate(batch_prompts, **generation_kwargs)
            results.extend(batch_results)
        
        return results



