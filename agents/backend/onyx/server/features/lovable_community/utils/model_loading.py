"""
Optimized Model Loading

Fast and efficient model loading using best practices.
"""

import logging
from typing import Optional, Dict, Any
import torch

logger = logging.getLogger(__name__)


def load_model_optimized(
    model_name: str,
    device: Optional[str] = None,
    use_fast_tokenizer: bool = True,
    torch_dtype: Optional[torch.dtype] = None
) -> tuple[Any, Any]:
    """
    Load model and tokenizer with optimizations.
    
    Args:
        model_name: Model name or path
        device: Device to load on (auto-detect if None)
        use_fast_tokenizer: Use Rust-based tokenizer
        torch_dtype: Data type for model weights
        
    Returns:
        Tuple of (model, tokenizer)
    """
    from transformers import AutoModel, AutoTokenizer
    
    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Auto-detect dtype
    if torch_dtype is None:
        if device == "cuda":
            torch_dtype = torch.float16  # Faster on GPU
        else:
            torch_dtype = torch.float32
    
    logger.info(f"Loading model {model_name} on {device} with dtype {torch_dtype}")
    
    # Load tokenizer (fast Rust-based)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=use_fast_tokenizer
    )
    
    # Load model with optimizations
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    if device != "cuda":
        model = model.to(device)
    
    model.eval()  # Set to evaluation mode
    
    logger.info(f"Model loaded successfully on {device}")
    
    return model, tokenizer


def load_model_quantized(
    model_name: str,
    quantization: str = "8bit",
    device: Optional[str] = None
) -> tuple[Any, Any]:
    """
    Load model with quantization for memory efficiency.
    
    Args:
        model_name: Model name or path
        quantization: Quantization type (4bit, 8bit)
        device: Device to load on
        
    Returns:
        Tuple of (model, tokenizer)
    """
    from transformers import AutoTokenizer, BitsAndBytesConfig
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device != "cuda":
        logger.warning("Quantization requires CUDA, falling back to normal loading")
        return load_model_optimized(model_name, device)
    
    # Configure quantization
    if quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
    elif quantization == "8bit":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    else:
        raise ValueError(f"Unsupported quantization: {quantization}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    # Load model with quantization
    from transformers import AutoModelForCausalLM
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    logger.info(f"Model loaded with {quantization} quantization")
    
    return model, tokenizer













