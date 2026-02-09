"""
LoRA (Low-Rank Adaptation) - Efficient Fine-tuning
==================================================

Implementation of LoRA for efficient fine-tuning of large models.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import logging

try:
    from peft import LoraConfig as PEFT_LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logging.warning("PEFT not available. Install with: pip install peft")

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


@dataclass
class LoraConfig:
    """
    Configuration for LoRA (Low-Rank Adaptation).
    
    LoRA reduces the number of trainable parameters by using low-rank matrices.
    """
    r: int = 8  # Rank of adaptation
    lora_alpha: int = 16  # Scaling factor
    target_modules: Optional[List[str]] = None  # Modules to apply LoRA to
    lora_dropout: float = 0.1  # Dropout for LoRA layers
    bias: str = "none"  # none, all, lora_only
    task_type: str = "FEATURE_EXTRACTION"  # Task type


def apply_lora(
    model,
    config: Optional[LoraConfig] = None,
    target_modules: Optional[List[str]] = None
):
    """
    Apply LoRA to a model.
    
    Args:
        model: Model to apply LoRA to
        config: LoRA configuration
        target_modules: Target modules (if not in config)
    
    Returns:
        Model with LoRA applied
    
    Raises:
        ImportError: If PEFT is not available
    """
    if not PEFT_AVAILABLE:
        raise ImportError(
            "PEFT library is required for LoRA. Install with: pip install peft"
        )
    
    if config is None:
        config = LoraConfig()
    
    # Determine target modules
    if config.target_modules is None:
        if target_modules is None:
            # Default modules for transformers
            config.target_modules = ["query", "key", "value", "dense"]
        else:
            config.target_modules = target_modules
    
    # Create PEFT config
    peft_config = PEFT_LoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias=config.bias,
        task_type=getattr(TaskType, config.task_type, TaskType.FEATURE_EXTRACTION),
    )
    
    # Apply LoRA
    try:
        model = get_peft_model(model, peft_config)
        logger.info(
            f"✅ LoRA applied: r={config.r}, alpha={config.lora_alpha}, "
            f"target_modules={config.target_modules}"
        )
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Trainable parameters: {trainable_params:,} / {total_params:,} "
            f"({100 * trainable_params / total_params:.2f}%)"
        )
        
        return model
        
    except Exception as e:
        logger.error(f"❌ Error applying LoRA: {e}")
        raise


def print_trainable_parameters(model) -> Dict[str, int]:
    """
    Print information about trainable parameters.
    
    Args:
        model: Model to analyze
    
    Returns:
        Dictionary with parameter counts
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    info = {
        "trainable": trainable_params,
        "total": total_params,
        "non_trainable": total_params - trainable_params,
        "percentage": 100.0 * trainable_params / total_params if total_params > 0 else 0.0
    }
    
    logger.info(
        f"Trainable parameters: {trainable_params:,} || "
        f"Total parameters: {total_params:,} || "
        f"Trainable: {info['percentage']:.2f}%"
    )
    
    return info



