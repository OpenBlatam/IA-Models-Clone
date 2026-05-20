"""
Optimized Inference Utilities

Fast inference using best practices and optimizations.
"""

import torch
import logging
from typing import List, Optional, Dict, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@contextmanager
def inference_mode():
    """
    Context manager for optimized inference.
    
    Disables gradient computation and enables optimizations.
    """
    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            yield


def generate_text_optimized(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    use_cache: bool = True
) -> str:
    """
    Generate text with optimizations.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Input prompt
        max_length: Maximum generation length
        temperature: Sampling temperature
        top_p: Nucleus sampling
        do_sample: Whether to sample
        use_cache: Use KV cache
        
    Returns:
        Generated text
    """
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate with optimizations
    with inference_mode():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            use_cache=use_cache,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text


def batch_inference(
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    batch_size: int = 8,
    **generation_kwargs
) -> List[str]:
    """
    Batch inference for multiple prompts.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        prompts: List of prompts
        batch_size: Batch size
        **generation_kwargs: Additional generation parameters
        
    Returns:
        List of generated texts
    """
    results = []
    
    # Process in batches
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Move to device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with inference_mode():
            outputs = model.generate(**inputs, **generation_kwargs)
        
        # Decode
        batch_results = tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )
        
        results.extend(batch_results)
    
    return results


def compile_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Compile model with torch.compile for faster inference.
    
    Requires PyTorch 2.0+.
    
    Args:
        model: PyTorch model
        
    Returns:
        Compiled model
    """
    try:
        if hasattr(torch, 'compile'):
            compiled_model = torch.compile(model, mode="max-autotune")
            logger.info("Model compiled with torch.compile")
            return compiled_model
        else:
            logger.warning("torch.compile not available (requires PyTorch 2.0+)")
            return model
    except Exception as e:
        logger.warning(f"Model compilation failed: {e}")
        return model













