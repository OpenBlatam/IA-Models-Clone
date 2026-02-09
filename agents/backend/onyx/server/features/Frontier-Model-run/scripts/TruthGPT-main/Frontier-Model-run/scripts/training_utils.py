"""Training utilities and helper functions."""
import logging
import os
from typing import Any, Dict

import torch

from .types import KFGRPOScriptArguments

DEFAULT_CUDA_DEVICE = 0


def setup_environment() -> None:
    """Setup the training environment with necessary configurations."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.set_device(DEFAULT_CUDA_DEVICE)
        torch.cuda.synchronize()
        logging.info(f"Using GPU: {torch.cuda.get_device_name(DEFAULT_CUDA_DEVICE)}")
    else:
        logging.warning("CUDA not available, using CPU")


def setup_distributed_training(args: KFGRPOScriptArguments) -> None:
    """Setup distributed training environment."""
    if args.distributed_world_size > 1:
        os.environ['MASTER_ADDR'] = args.distributed_master_addr
        os.environ['MASTER_PORT'] = args.distributed_master_port
        torch.distributed.init_process_group(
            backend=args.distributed_backend,
            world_size=args.distributed_world_size,
            rank=args.distributed_rank
        )
        logging.info(f"Distributed training initialized with {args.distributed_world_size} processes")


def _extract_deepseek_config_dict(deepseek_config: Any) -> Dict[str, Any]:
    """Extract DeepSeek configuration as dictionary."""
    return {
        'model_type': deepseek_config.model_type,
        'model_name': deepseek_config.model_name,
        'use_native_implementation': deepseek_config.use_native_implementation,
        'max_position_embeddings': deepseek_config.max_position_embeddings,
        'hidden_size': deepseek_config.hidden_size,
        'num_hidden_layers': deepseek_config.num_hidden_layers,
        'num_attention_heads': deepseek_config.num_attention_heads,
        'num_key_value_heads': deepseek_config.num_key_value_heads,
        'vocab_size': deepseek_config.vocab_size,
        'intermediate_size': deepseek_config.intermediate_size,
        'hidden_dropout_prob': deepseek_config.hidden_dropout_prob,
        'attention_dropout_prob': deepseek_config.attention_dropout_prob,
        'layer_norm_eps': deepseek_config.layer_norm_eps,
        'rope_theta': deepseek_config.rope_theta,
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
        'mscale': deepseek_config.mscale,
        'use_rotary_embeddings': deepseek_config.use_rotary_embeddings,
        'use_alibi': deepseek_config.use_alibi,
        'use_flash_attention_2': deepseek_config.use_flash_attention_2,
        'use_sliding_window': deepseek_config.use_sliding_window,
        'sliding_window_size': deepseek_config.sliding_window_size
    }


def setup_training_config(args: KFGRPOScriptArguments) -> Dict[str, Any]:
    """Setup training configuration from arguments."""
    return {
        'training': {
            'batch_size': args.train_batch_size,
            'gradient_accumulation_steps': args.gradient_accumulation_steps,
            'max_grad_norm': args.max_grad_norm,
            'warmup_ratio': args.warmup_ratio,
            'weight_decay': args.weight_decay,
            'lr_scheduler_type': args.lr_scheduler_type,
            'num_cycles': args.num_cycles
        },
        'optimization': {
            'use_amp': args.use_amp,
            'use_gradient_checkpointing': args.use_gradient_checkpointing,
            'use_flash_attention': args.use_flash_attention,
            'use_8bit_optimizer': args.use_8bit_optimizer,
            'use_cpu_offload': args.use_cpu_offload,
            'use_activation_checkpointing': args.use_activation_checkpointing,
            'use_attention_slicing': args.use_attention_slicing,
            'use_sequence_parallelism': args.use_sequence_parallelism
        },
        'performance': {
            'use_cudnn_benchmark': args.use_cudnn_benchmark,
            'use_tf32': args.use_tf32,
            'use_channels_last': args.use_channels_last,
            'use_compile': args.use_compile
        },
        'deepspeed': {
            'use_deepspeed': args.use_deepspeed,
            'deepspeed_config': args.deepspeed_config,
            'zero_stage': args.zero_stage,
            'offload_optimizer': args.offload_optimizer,
            'offload_param': args.offload_param,
            'gradient_clipping': args.gradient_clipping,
            'fp16': args.fp16,
            'bf16': args.bf16
        },
        'deepseek': _extract_deepseek_config_dict(args.deepseek_config)
    }

