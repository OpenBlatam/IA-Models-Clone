"""Configuration parser for training scripts."""
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml
from trl import ModelConfig

from .types import DeepSeekConfig, KFGRPOScriptArguments

DEFAULT_OUTPUT_DIR = './output'
DEFAULT_SEED = 42
DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_NUM_EPOCHS = 3
DEFAULT_BATCH_SIZE = 8
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 1
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_LR_SCHEDULER_TYPE = 'cosine'
DEFAULT_LOGGING_STEPS = 10
DEFAULT_SAVE_STEPS = 500
DEFAULT_EVAL_STRATEGY = 'no'
DEFAULT_MODEL_NAME = 'deepseek-ai/deepseek-v3'
DEFAULT_MODEL_REVISION = 'main'
DEFAULT_ATTN_IMPLEMENTATION = 'flash_attention_2'


class ConfigParser:
    """Parser for YAML configuration files."""
    
    @staticmethod
    def _get_section(config: Dict[str, Any], section: str, default: Dict[str, Any] = None) -> Dict[str, Any]:
        """Safely get a configuration section."""
        return config.get(section, default or {})
    
    @staticmethod
    def _get_value(config: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Safely get a configuration value."""
        return config.get(key, default)
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not config_path:
            raise ValueError("Configuration file path cannot be empty")
        
        config_path_obj = Path(config_path)
        if not config_path_obj.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        if not config_path_obj.is_file():
            raise ValueError(f"Path is not a file: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if config is None:
                raise ValueError(f"Configuration file is empty: {config_path}")
            
            if not isinstance(config, dict):
                raise ValueError(f"Configuration must be a dictionary, got {type(config)}")
            
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration from {config_path}: {e}")
    
    @staticmethod
    def _extract_dataset_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract dataset configuration."""
        return {
            'dataset_name': ConfigParser._get_value(config, 'dataset_name'),
            'dataset_config': ConfigParser._get_value(config, 'dataset_config'),
            'dataset_train_split': ConfigParser._get_value(config, 'dataset_train_split'),
            'dataset_test_split': ConfigParser._get_value(config, 'dataset_test_split'),
            'output_dir': ConfigParser._get_value(config, 'output_dir')
        }
    
    @staticmethod
    def _extract_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract model configuration."""
        model_section = ConfigParser._get_section(config, 'model')
        return {
            'model_name': ConfigParser._get_value(model_section, 'name'),
            'use_deepspeed': ConfigParser._get_value(model_section, 'use_deepspeed', False),
            'fp16': ConfigParser._get_value(model_section, 'fp16', False),
            'bf16': ConfigParser._get_value(model_section, 'bf16', False)
        }
    
    @staticmethod
    def _extract_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract training configuration."""
        training_section = ConfigParser._get_section(config, 'training')
        return {
            'train_batch_size': ConfigParser._get_value(training_section, 'batch_size'),
            'gradient_accumulation_steps': ConfigParser._get_value(training_section, 'gradient_accumulation_steps'),
            'learning_rate': ConfigParser._get_value(training_section, 'learning_rate'),
            'max_steps': ConfigParser._get_value(training_section, 'max_steps'),
            'warmup_ratio': ConfigParser._get_value(training_section, 'warmup_ratio'),
            'weight_decay': ConfigParser._get_value(training_section, 'weight_decay'),
            'max_grad_norm': ConfigParser._get_value(training_section, 'max_grad_norm'),
            'lr_scheduler_type': ConfigParser._get_value(training_section, 'lr_scheduler_type'),
            'num_cycles': ConfigParser._get_value(training_section, 'num_cycles')
        }
    
    @staticmethod
    def _extract_optimization_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract optimization configuration."""
        optimization_section = ConfigParser._get_section(config, 'optimization')
        return {
            'use_amp': ConfigParser._get_value(optimization_section, 'use_amp', False),
            'use_gradient_checkpointing': ConfigParser._get_value(optimization_section, 'use_gradient_checkpointing', False),
            'use_flash_attention': ConfigParser._get_value(optimization_section, 'use_flash_attention', False),
            'use_8bit_optimizer': ConfigParser._get_value(optimization_section, 'use_8bit_optimizer', False),
            'use_cpu_offload': ConfigParser._get_value(optimization_section, 'use_cpu_offload', False),
            'use_activation_checkpointing': ConfigParser._get_value(optimization_section, 'use_activation_checkpointing', False),
            'use_attention_slicing': ConfigParser._get_value(optimization_section, 'use_attention_slicing', False),
            'use_sequence_parallelism': ConfigParser._get_value(optimization_section, 'use_sequence_parallelism', False)
        }
    
    @staticmethod
    def _extract_performance_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract performance configuration."""
        performance_section = ConfigParser._get_section(config, 'performance')
        return {
            'use_cudnn_benchmark': ConfigParser._get_value(performance_section, 'use_cudnn_benchmark', False),
            'use_tf32': ConfigParser._get_value(performance_section, 'use_tf32', False),
            'use_channels_last': ConfigParser._get_value(performance_section, 'use_channels_last', False),
            'use_compile': ConfigParser._get_value(performance_section, 'use_compile', False)
        }
    
    @staticmethod
    def _extract_deepspeed_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract DeepSpeed configuration."""
        deepspeed_section = ConfigParser._get_section(config, 'deepspeed')
        use_deepspeed = ConfigParser._get_value(deepspeed_section, 'use_deepspeed', False)
        
        return {
            'deepspeed_config': {
                'zero_stage': ConfigParser._get_value(deepspeed_section, 'zero_stage'),
                'offload_optimizer': ConfigParser._get_value(deepspeed_section, 'offload_optimizer', False),
                'offload_param': ConfigParser._get_value(deepspeed_section, 'offload_param', False),
                'gradient_clipping': ConfigParser._get_value(deepspeed_section, 'gradient_clipping')
            } if use_deepspeed else None
        }
    
    @staticmethod
    def _extract_deepseek_config(config: Dict[str, Any]) -> DeepSeekConfig:
        """Extract DeepSeek configuration."""
        deepseek_section = ConfigParser._get_section(config, 'deepseek')
        default_config = DeepSeekConfig()
        
        return DeepSeekConfig(
            model_type=ConfigParser._get_value(deepseek_section, 'model_type', default_config.model_type),
            use_native_implementation=ConfigParser._get_value(deepseek_section, 'use_native_implementation', default_config.use_native_implementation),
            max_position_embeddings=ConfigParser._get_value(deepseek_section, 'max_position_embeddings', default_config.max_position_embeddings),
            hidden_size=ConfigParser._get_value(deepseek_section, 'hidden_size', default_config.hidden_size),
            num_hidden_layers=ConfigParser._get_value(deepseek_section, 'num_hidden_layers', default_config.num_hidden_layers),
            num_attention_heads=ConfigParser._get_value(deepseek_section, 'num_attention_heads', default_config.num_attention_heads),
            num_key_value_heads=ConfigParser._get_value(deepseek_section, 'num_key_value_heads', default_config.num_key_value_heads),
            vocab_size=ConfigParser._get_value(deepseek_section, 'vocab_size', default_config.vocab_size),
            intermediate_size=ConfigParser._get_value(deepseek_section, 'intermediate_size', default_config.intermediate_size),
            hidden_dropout_prob=ConfigParser._get_value(deepseek_section, 'hidden_dropout_prob', default_config.hidden_dropout_prob),
            attention_dropout_prob=ConfigParser._get_value(deepseek_section, 'attention_dropout_prob', default_config.attention_dropout_prob),
            layer_norm_eps=ConfigParser._get_value(deepseek_section, 'layer_norm_eps', default_config.layer_norm_eps),
            rope_theta=ConfigParser._get_value(deepseek_section, 'rope_theta', default_config.rope_theta),
            q_lora_rank=ConfigParser._get_value(deepseek_section, 'q_lora_rank', default_config.q_lora_rank),
            kv_lora_rank=ConfigParser._get_value(deepseek_section, 'kv_lora_rank', default_config.kv_lora_rank),
            qk_rope_head_dim=ConfigParser._get_value(deepseek_section, 'qk_rope_head_dim', default_config.qk_rope_head_dim),
            v_head_dim=ConfigParser._get_value(deepseek_section, 'v_head_dim', default_config.v_head_dim),
            qk_nope_head_dim=ConfigParser._get_value(deepseek_section, 'qk_nope_head_dim', default_config.qk_nope_head_dim),
            n_routed_experts=ConfigParser._get_value(deepseek_section, 'n_routed_experts', default_config.n_routed_experts),
            n_shared_experts=ConfigParser._get_value(deepseek_section, 'n_shared_experts', default_config.n_shared_experts),
            n_activated_experts=ConfigParser._get_value(deepseek_section, 'n_activated_experts', default_config.n_activated_experts),
            moe_intermediate_size=ConfigParser._get_value(deepseek_section, 'moe_intermediate_size', default_config.moe_intermediate_size),
            shared_intermediate_size=ConfigParser._get_value(deepseek_section, 'shared_intermediate_size', default_config.shared_intermediate_size),
            use_fp8=ConfigParser._get_value(deepseek_section, 'use_fp8', default_config.use_fp8),
            original_seq_len=ConfigParser._get_value(deepseek_section, 'original_seq_len', default_config.original_seq_len),
            rope_factor=ConfigParser._get_value(deepseek_section, 'rope_factor', default_config.rope_factor),
            beta_fast=ConfigParser._get_value(deepseek_section, 'beta_fast', default_config.beta_fast),
            beta_slow=ConfigParser._get_value(deepseek_section, 'beta_slow', default_config.beta_slow),
            mscale=ConfigParser._get_value(deepseek_section, 'mscale', default_config.mscale),
            use_rotary_embeddings=ConfigParser._get_value(deepseek_section, 'use_rotary_embeddings', default_config.use_rotary_embeddings),
            use_alibi=ConfigParser._get_value(deepseek_section, 'use_alibi', default_config.use_alibi),
            use_flash_attention_2=ConfigParser._get_value(deepseek_section, 'use_flash_attention_2', default_config.use_flash_attention_2),
            use_sliding_window=ConfigParser._get_value(deepseek_section, 'use_sliding_window', default_config.use_sliding_window),
            sliding_window_size=ConfigParser._get_value(deepseek_section, 'sliding_window_size', default_config.sliding_window_size),
            use_parallel_attention=ConfigParser._get_value(deepseek_section, 'use_parallel_attention', default_config.use_parallel_attention),
            use_parallel_mlp=ConfigParser._get_value(deepseek_section, 'use_parallel_mlp', default_config.use_parallel_mlp),
            use_parallel_layernorm=ConfigParser._get_value(deepseek_section, 'use_parallel_layernorm', default_config.use_parallel_layernorm),
            use_parallel_embedding=ConfigParser._get_value(deepseek_section, 'use_parallel_embedding', default_config.use_parallel_embedding),
            use_parallel_output=ConfigParser._get_value(deepseek_section, 'use_parallel_output', default_config.use_parallel_output),
            use_parallel_residual=ConfigParser._get_value(deepseek_section, 'use_parallel_residual', default_config.use_parallel_residual),
            use_parallel_ffn=ConfigParser._get_value(deepseek_section, 'use_parallel_ffn', default_config.use_parallel_ffn),
            use_parallel_attention_output=ConfigParser._get_value(deepseek_section, 'use_parallel_attention_output', default_config.use_parallel_attention_output),
            use_parallel_mlp_output=ConfigParser._get_value(deepseek_section, 'use_parallel_mlp_output', default_config.use_parallel_mlp_output),
            use_parallel_layernorm_output=ConfigParser._get_value(deepseek_section, 'use_parallel_layernorm_output', default_config.use_parallel_layernorm_output),
            use_parallel_embedding_output=ConfigParser._get_value(deepseek_section, 'use_parallel_embedding_output', default_config.use_parallel_embedding_output),
            use_parallel_residual_output=ConfigParser._get_value(deepseek_section, 'use_parallel_residual_output', default_config.use_parallel_residual_output),
            use_parallel_ffn_output=ConfigParser._get_value(deepseek_section, 'use_parallel_ffn_output', default_config.use_parallel_ffn_output),
            use_parallel_attention_input=ConfigParser._get_value(deepseek_section, 'use_parallel_attention_input', default_config.use_parallel_attention_input),
            use_parallel_mlp_input=ConfigParser._get_value(deepseek_section, 'use_parallel_mlp_input', default_config.use_parallel_mlp_input),
            use_parallel_layernorm_input=ConfigParser._get_value(deepseek_section, 'use_parallel_layernorm_input', default_config.use_parallel_layernorm_input),
            use_parallel_embedding_input=ConfigParser._get_value(deepseek_section, 'use_parallel_embedding_input', default_config.use_parallel_embedding_input),
            use_parallel_residual_input=ConfigParser._get_value(deepseek_section, 'use_parallel_residual_input', default_config.use_parallel_residual_input),
            use_parallel_ffn_input=ConfigParser._get_value(deepseek_section, 'use_parallel_ffn_input', default_config.use_parallel_ffn_input)
        )
    
    @staticmethod
    def _extract_other_configs(config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract other configuration sections."""
        parallel_section = ConfigParser._get_section(config, 'parallel')
        kalman_section = ConfigParser._get_section(config, 'kalman')
        distributed_section = ConfigParser._get_section(config, 'distributed')
        
        return {
            'parallel_config': parallel_section,
            'kalman_config': {
                'process_noise': ConfigParser._get_value(kalman_section, 'process_noise'),
                'measurement_noise': ConfigParser._get_value(kalman_section, 'measurement_noise'),
                'memory_size': ConfigParser._get_value(kalman_section, 'memory_size')
            },
            'reward_funcs': ConfigParser._get_value(config, 'reward_funcs', []),
            'distributed_config': {
                'backend': ConfigParser._get_value(distributed_section, 'backend'),
                'world_size': ConfigParser._get_value(distributed_section, 'world_size'),
                'rank': ConfigParser._get_value(distributed_section, 'rank'),
                'master_addr': ConfigParser._get_value(distributed_section, 'master_addr'),
                'master_port': ConfigParser._get_value(distributed_section, 'master_port')
            }
        }
    
    @staticmethod
    def _create_training_model_config(config: Dict[str, Any]) -> ModelConfig:
        """Create ModelConfig for training arguments from YAML config."""
        training_section = ConfigParser._get_section(config, 'training')
        model_section = ConfigParser._get_section(config, 'model')
        optimization_section = ConfigParser._get_section(config, 'optimization')
        
        return ModelConfig(
            output_dir=ConfigParser._get_value(config, 'output_dir', DEFAULT_OUTPUT_DIR),
            seed=ConfigParser._get_value(training_section, 'seed', DEFAULT_SEED),
            learning_rate=ConfigParser._get_value(training_section, 'learning_rate', DEFAULT_LEARNING_RATE),
            num_train_epochs=ConfigParser._get_value(training_section, 'num_epochs', DEFAULT_NUM_EPOCHS),
            per_device_train_batch_size=ConfigParser._get_value(training_section, 'batch_size', DEFAULT_BATCH_SIZE),
            gradient_accumulation_steps=ConfigParser._get_value(training_section, 'gradient_accumulation_steps', DEFAULT_GRADIENT_ACCUMULATION_STEPS),
            warmup_ratio=ConfigParser._get_value(training_section, 'warmup_ratio', DEFAULT_WARMUP_RATIO),
            weight_decay=ConfigParser._get_value(training_section, 'weight_decay', DEFAULT_WEIGHT_DECAY),
            max_grad_norm=ConfigParser._get_value(training_section, 'max_grad_norm', DEFAULT_MAX_GRAD_NORM),
            lr_scheduler_type=ConfigParser._get_value(training_section, 'lr_scheduler_type', DEFAULT_LR_SCHEDULER_TYPE),
            logging_steps=ConfigParser._get_value(training_section, 'logging_steps', DEFAULT_LOGGING_STEPS),
            save_steps=ConfigParser._get_value(training_section, 'save_steps', DEFAULT_SAVE_STEPS),
            eval_strategy=ConfigParser._get_value(training_section, 'eval_strategy', DEFAULT_EVAL_STRATEGY),
            fp16=ConfigParser._get_value(model_section, 'fp16', False),
            bf16=ConfigParser._get_value(model_section, 'bf16', False),
            gradient_checkpointing=ConfigParser._get_value(optimization_section, 'use_gradient_checkpointing', True),
            push_to_hub=ConfigParser._get_value(training_section, 'push_to_hub', False),
            report_to=ConfigParser._get_value(training_section, 'report_to', [])
        )
    
    @staticmethod
    def _create_model_model_config(config: Dict[str, Any]) -> ModelConfig:
        """Create ModelConfig for model arguments from YAML config."""
        model_section = ConfigParser._get_section(config, 'model')
        optimization_section = ConfigParser._get_section(config, 'optimization')
        
        return ModelConfig(
            model_name_or_path=ConfigParser._get_value(model_section, 'name', DEFAULT_MODEL_NAME),
            trust_remote_code=ConfigParser._get_value(model_section, 'trust_remote_code', True),
            torch_dtype=ConfigParser._get_value(model_section, 'torch_dtype', 'auto'),
            attn_implementation=ConfigParser._get_value(model_section, 'attn_implementation', DEFAULT_ATTN_IMPLEMENTATION),
            model_revision=ConfigParser._get_value(model_section, 'revision', DEFAULT_MODEL_REVISION),
            use_cache=not ConfigParser._get_value(optimization_section, 'use_gradient_checkpointing', True)
        )
    
    @staticmethod
    def convert_to_args(config: Dict[str, Any]) -> KFGRPOScriptArguments:
        """Convert YAML config to script arguments."""
        args_dict = {}
        args_dict.update(ConfigParser._extract_dataset_config(config))
        args_dict.update(ConfigParser._extract_model_config(config))
        args_dict.update(ConfigParser._extract_training_config(config))
        args_dict.update(ConfigParser._extract_optimization_config(config))
        args_dict.update(ConfigParser._extract_performance_config(config))
        args_dict.update(ConfigParser._extract_deepspeed_config(config))
        args_dict.update({
            'deepseek_config': ConfigParser._extract_deepseek_config(config)
        })
        args_dict.update(ConfigParser._extract_other_configs(config))
        
        return KFGRPOScriptArguments(**args_dict)
    
    @staticmethod
    def convert_to_all_args(config: Dict[str, Any]) -> Tuple[KFGRPOScriptArguments, ModelConfig, ModelConfig]:
        """Convert YAML config to all required arguments."""
        script_args = ConfigParser.convert_to_args(config)
        training_args = ConfigParser._create_training_model_config(config)
        model_args = ConfigParser._create_model_model_config(config)
        return script_args, training_args, model_args

