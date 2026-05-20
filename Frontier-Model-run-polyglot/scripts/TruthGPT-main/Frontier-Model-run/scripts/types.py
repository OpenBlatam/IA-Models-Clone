"""Type definitions and dataclasses for training scripts."""
from dataclasses import dataclass, field
from typing import List, Optional

from trl import ScriptArguments


@dataclass
class DeepSeekConfig:
    """Configuration specific to DeepSeek V3 model."""
    model_type: str = "deepseek"
    model_name: str = "deepseek-ai/deepseek-v3"
    use_native_implementation: bool = True
    
    max_position_embeddings: int = 8192
    hidden_size: int = 4096
    num_hidden_layers: int = 30
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = None
    vocab_size: int = 102400
    intermediate_size: int = 11008
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    qk_nope_head_dim: int = 128
    
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    moe_intermediate_size: int = 1407
    shared_intermediate_size: int = 1024
    
    use_fp8: bool = False
    
    original_seq_len: int = 4096
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0
    
    use_rotary_embeddings: bool = True
    use_alibi: bool = False
    use_flash_attention_2: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 4096
    use_parallel_attention: bool = True
    use_parallel_mlp: bool = True
    use_parallel_layernorm: bool = True
    use_parallel_embedding: bool = True
    use_parallel_output: bool = True
    use_parallel_residual: bool = True
    use_parallel_ffn: bool = True
    use_parallel_attention_output: bool = True
    use_parallel_mlp_output: bool = True
    use_parallel_layernorm_output: bool = True
    use_parallel_embedding_output: bool = True
    use_parallel_residual_output: bool = True
    use_parallel_ffn_output: bool = True
    use_parallel_attention_input: bool = True
    use_parallel_mlp_input: bool = True
    use_parallel_layernorm_input: bool = True
    use_parallel_embedding_input: bool = True
    use_parallel_residual_input: bool = True
    use_parallel_ffn_input: bool = True


@dataclass
class KFGRPOScriptArguments(ScriptArguments):
    """Script arguments for the KF-GRPO training script with DeepSeek R1 optimizations."""
    deepseek_config: DeepSeekConfig = field(default_factory=DeepSeekConfig)
    use_deepseek_optimizations: bool = field(default=True, metadata={"help": "Use DeepSeek-specific optimizations"})
    use_deepseek_attention: bool = field(default=True, metadata={"help": "Use DeepSeek attention optimizations"})
    use_deepseek_mlp: bool = field(default=True, metadata={"help": "Use DeepSeek MLP optimizations"})
    use_deepseek_layernorm: bool = field(default=True, metadata={"help": "Use DeepSeek LayerNorm optimizations"})
    use_deepseek_embedding: bool = field(default=True, metadata={"help": "Use DeepSeek embedding optimizations"})
    use_deepseek_residual: bool = field(default=True, metadata={"help": "Use DeepSeek residual optimizations"})
    use_deepseek_ffn: bool = field(default=True, metadata={"help": "Use DeepSeek FFN optimizations"})
    
    reward_funcs: List[str] = field(
        default_factory=lambda: ["accuracy", "format", "tag_count"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', tag_count', 'code', 'code_format'"
        },
    )
    process_noise: float = field(default=0.01, metadata={"help": "Process noise covariance (Q)"})
    measurement_noise: float = field(default=0.1, metadata={"help": "Measurement noise covariance (R)"})
    kalman_memory_size: int = field(default=1000, metadata={"help": "Size of Kalman filter memory buffer"})
    
    use_amp: bool = field(default=True, metadata={"help": "Use automatic mixed precision"})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Number of steps to accumulate gradients"})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Maximum gradient norm for clipping"})
    warmup_ratio: float = field(default=0.1, metadata={"help": "Ratio of warmup steps"})
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay for optimizer"})
    lr_scheduler_type: str = field(default="cosine", metadata={"help": "Learning rate scheduler type"})
    num_cycles: int = field(default=1, metadata={"help": "Number of cycles for cosine scheduler"})
    
    use_gradient_checkpointing: bool = field(default=True, metadata={"help": "Use gradient checkpointing"})
    use_flash_attention: bool = field(default=True, metadata={"help": "Use flash attention"})
    use_8bit_optimizer: bool = field(default=False, metadata={"help": "Use 8-bit optimizer"})
    
    distributed_backend: str = field(default="nccl", metadata={"help": "Distributed backend"})
    distributed_world_size: int = field(default=-1, metadata={"help": "Number of distributed processes"})
    distributed_rank: int = field(default=-1, metadata={"help": "Process rank"})
    distributed_master_addr: str = field(default="localhost", metadata={"help": "Master address"})
    distributed_master_port: str = field(default="29500", metadata={"help": "Master port"})
    
    use_cpu_offload: bool = field(default=False, metadata={"help": "Use CPU offloading"})
    use_activation_checkpointing: bool = field(default=True, metadata={"help": "Use activation checkpointing"})
    use_attention_slicing: bool = field(default=True, metadata={"help": "Use attention slicing"})
    use_sequence_parallelism: bool = field(default=False, metadata={"help": "Use sequence parallelism"})
    
    use_cudnn_benchmark: bool = field(default=True, metadata={"help": "Use cuDNN benchmark"})
    use_tf32: bool = field(default=True, metadata={"help": "Use TF32 precision"})
    use_channels_last: bool = field(default=True, metadata={"help": "Use channels last memory format"})
    use_compile: bool = field(default=True, metadata={"help": "Use torch.compile"})
    
    use_deepspeed: bool = field(default=False, metadata={"help": "Use DeepSpeed for training"})
    deepspeed_config: Optional[str] = field(default=None, metadata={"help": "Path to DeepSpeed config file"})
    zero_stage: int = field(default=2, metadata={"help": "DeepSpeed ZeRO stage (0, 1, 2, 3)"})
    offload_optimizer: bool = field(default=True, metadata={"help": "Offload optimizer states to CPU"})
    offload_param: bool = field(default=False, metadata={"help": "Offload parameters to CPU"})
    gradient_clipping: float = field(default=1.0, metadata={"help": "Gradient clipping value"})
    train_batch_size: int = field(default=8, metadata={"help": "Training batch size"})
    fp16: bool = field(default=True, metadata={"help": "Use FP16 precision"})
    bf16: bool = field(default=False, metadata={"help": "Use BF16 precision"})







