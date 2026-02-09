"""Model factory for creating models from configurations."""
from typing import Any, Dict

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, PreTrainedModel

from models.deepseek_v3 import create_deepseek_v3_model
from scripts.types import DeepSeekConfig

AUTO_DEVICE_MAP = "auto"

PARALLEL_CONFIG_KEYS = ['attention', 'mlp', 'layernorm', 'embedding', 'residual', 'ffn']


def _get_parallel_config_value(config: DeepSeekConfig, name: str, config_type: str) -> bool:
    """Get parallel configuration value for a given name and type.
    
    Args:
        config: DeepSeek configuration
        name: Configuration name (e.g., 'attention', 'mlp')
        config_type: Type of configuration ('base', 'input', 'output')
        
    Returns:
        Boolean value indicating if parallel config is enabled
    """
    if config_type == 'base':
        attr_name = f'use_parallel_{name}'
    else:
        attr_name = f'use_parallel_{name}_{config_type}'
    return getattr(config, attr_name, False)


def _apply_parallel_configs(
    model: PreTrainedModel,
    config: DeepSeekConfig,
    config_type: str
) -> None:
    """Apply parallel configuration settings to model.
    
    Args:
        model: The model to configure
        config: DeepSeek configuration
        config_type: Type of configuration ('base', 'input', 'output')
        
    Raises:
        ValueError: If config_type is invalid
    """
    if config_type not in ('base', 'input', 'output'):
        raise ValueError(f"Invalid config_type: {config_type}. Must be 'base', 'input', or 'output'")
    
    if not isinstance(model, PreTrainedModel):
        raise TypeError(f"model must be a PreTrainedModel, got {type(model)}")
    
    suffix = f'_{config_type}' if config_type != 'base' else ''
    
    for name in PARALLEL_CONFIG_KEYS:
        if _get_parallel_config_value(config, name, config_type):
            attr_name = f'use_parallel_{name}{suffix}'
            setattr(model.config, attr_name, True)
            logger.info(f"Enabled parallel {name}{suffix.replace('_', ' ')}")


def setup_deepseek_optimizations(model: PreTrainedModel, config: DeepSeekConfig) -> None:
    """Setup DeepSeek-specific optimizations for the model.
    
    Args:
        model: The model to optimize
        config: DeepSeek configuration
        
    Raises:
        TypeError: If model is not a PreTrainedModel
    """
    if not isinstance(model, PreTrainedModel):
        raise TypeError(f"model must be a PreTrainedModel, got {type(model)}")
    
    if config.use_flash_attention_2:
        model.config.use_flash_attention_2 = True
        logger.info("Enabled Flash Attention 2")
    
    if config.use_sliding_window:
        if config.sliding_window_size <= 0:
            raise ValueError(f"sliding_window_size must be positive, got {config.sliding_window_size}")
        model.config.use_sliding_window = True
        model.config.sliding_window_size = config.sliding_window_size
        logger.info(f"Enabled sliding window attention with size {config.sliding_window_size}")
    
    _apply_parallel_configs(model, config, 'base')
    _apply_parallel_configs(model, config, 'output')
    _apply_parallel_configs(model, config, 'input')


def build_model_config(deepseek_config: DeepSeekConfig) -> Dict[str, Any]:
    """Build model configuration dictionary from DeepSeekConfig."""
    return {
        'hidden_size': deepseek_config.hidden_size,
        'num_hidden_layers': deepseek_config.num_hidden_layers,
        'num_attention_heads': deepseek_config.num_attention_heads,
        'num_key_value_heads': deepseek_config.num_key_value_heads,
        'vocab_size': deepseek_config.vocab_size,
        'layer_norm_eps': deepseek_config.layer_norm_eps,
        'rope_theta': deepseek_config.rope_theta,
        'max_position_embeddings': deepseek_config.max_position_embeddings,
        'q_lora_rank': deepseek_config.q_lora_rank,
        'kv_lora_rank': deepseek_config.kv_lora_rank,
        'qk_rope_head_dim': deepseek_config.qk_rope_head_dim,
        'v_head_dim': deepseek_config.v_head_dim,
        'qk_nope_head_dim': deepseek_config.qk_nope_head_dim,
        'n_routed_experts': deepseek_config.n_routed_experts,
        'n_shared_experts': deepseek_config.n_shared_experts,
        'n_activated_experts': deepseek_config.n_activated_experts,
        'moe_intermediate_size': deepseek_config.moe_intermediate_size,
        'shared_intermediate_size': deepseek_config.shared_intermediate_size,
        'use_fp8': deepseek_config.use_fp8,
        'original_seq_len': deepseek_config.original_seq_len,
        'rope_factor': deepseek_config.rope_factor,
        'beta_fast': deepseek_config.beta_fast,
        'beta_slow': deepseek_config.beta_slow,
        'mscale': deepseek_config.mscale
    }


def create_model_from_config(
    deepseek_config: DeepSeekConfig,
    use_native: bool,
    fp16: bool,
    bf16: bool,
    use_deepspeed: bool
) -> PreTrainedModel:
    """Create model from DeepSeek configuration.
    
    Args:
        deepseek_config: DeepSeek configuration
        use_native: Whether to use native implementation
        fp16: Whether to use float16 precision
        bf16: Whether to use bfloat16 precision
        use_deepspeed: Whether to use DeepSpeed
        
    Returns:
        Configured model instance
        
    Raises:
        ValueError: If both fp16 and bf16 are True, or if model_name is empty
    """
    if fp16 and bf16:
        raise ValueError("Cannot use both fp16 and bf16 simultaneously")
    
    if not deepseek_config.model_name:
        raise ValueError("model_name cannot be empty")
    
    if use_native:
        model_config = build_model_config(deepseek_config)
        model = create_deepseek_v3_model(model_config)
        
        if bf16:
            model = model.to(torch.bfloat16)
        elif fp16:
            model = model.to(torch.float16)
        
        logger.info("Using native DeepSeek-V3 implementation")
        return model
    else:
        torch_dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32)
        model = AutoModelForCausalLM.from_pretrained(
            deepseek_config.model_name,
            torch_dtype=torch_dtype,
            device_map=AUTO_DEVICE_MAP if use_deepspeed else None,
            trust_remote_code=True
        )
        logger.info("Using HuggingFace DeepSeek implementation")
        return model

