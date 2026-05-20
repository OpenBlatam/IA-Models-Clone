"""
Configuration Management for ML Services
Supports YAML configuration files and environment variables.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ModelConfig(BaseModel):
    """Model configuration."""
    name: str
    type: str = Field(..., description="causal_lm, seq2seq, encoder, diffusion")
    path: Optional[str] = None
    use_fp16: bool = True
    device_map: str = "auto"
    trust_remote_code: bool = False


class TrainingConfig(BaseModel):
    """Training configuration."""
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    save_steps: int = 500
    eval_steps: Optional[int] = None
    logging_steps: int = 10
    fp16: bool = True
    bf16: bool = False
    dataloader_num_workers: int = 0
    remove_unused_columns: bool = True


class LoRAConfig(BaseModel):
    """LoRA configuration."""
    enabled: bool = False
    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    target_modules: Optional[list] = None
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


class GenerationConfig(BaseModel):
    """Text generation configuration."""
    max_length: int = 100
    max_new_tokens: Optional[int] = None
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    do_sample: bool = True
    num_return_sequences: int = 1
    pad_token_id: Optional[int] = None


class DiffusionConfig(BaseModel):
    """Diffusion model configuration."""
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    scheduler: str = "DPMSolverMultistep"
    width: int = 512
    height: int = 512
    strength: float = 0.8


class MLServiceSettings(BaseSettings):
    """ML Service settings."""
    # Device settings
    device: str = "cuda"
    use_fp16: bool = True
    cuda_visible_devices: Optional[str] = None
    
    # Model cache
    model_cache_dir: str = "./models"
    max_cached_models: int = 5
    cache_clear_on_exit: bool = True
    
    # Training
    output_dir: str = "./trained_models"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    # Experiment tracking
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    use_tensorboard: bool = True
    tensorboard_log_dir: str = "./logs/tensorboard"
    
    # Performance
    enable_xformers: bool = True
    enable_flash_attention: bool = False
    compile_model: bool = False  # PyTorch 2.0+ compile
    
    class Config:
        env_file = ".env"
        env_prefix = "ML_"


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = os.getenv("ML_CONFIG_PATH", "config.yaml")
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        return {}
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config or {}


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save config
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


# Default configuration
DEFAULT_CONFIG = {
    "model": {
        "use_fp16": True,
        "device_map": "auto",
    },
    "training": {
        "num_epochs": 3,
        "batch_size": 4,
        "learning_rate": 5e-5,
        "fp16": True,
    },
    "generation": {
        "max_length": 100,
        "temperature": 1.0,
        "top_p": 0.9,
    },
    "diffusion": {
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
    },
}



