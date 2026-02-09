"""
Configuration Manager

Provides robust configuration management with:
- YAML config loading
- Environment variable support
- Validation
- Type checking
- Default values
"""

import logging
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
import yaml
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    """Configuration for a specific model"""
    name: str
    path: Optional[str] = None
    version: Optional[str] = None
    device: Optional[str] = None
    batch_size: int = 32
    max_length: int = 512
    dtype: str = "float32"  # float32, float16, int8
    
    @validator('dtype')
    def validate_dtype(cls, v):
        valid = ['float32', 'float16', 'int8']
        if v not in valid:
            raise ValueError(f'dtype must be one of {valid}')
        return v


class TrainingConfig(BaseModel):
    """Configuration for training"""
    learning_rate: float = 2e-5
    batch_size: int = 32
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    use_mixed_precision: bool = True
    early_stopping_patience: int = 5
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100


class LoRAConfig(BaseModel):
    """Configuration for LoRA fine-tuning"""
    r: int = 16
    alpha: int = 32
    dropout: float = 0.1
    target_modules: List[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])


class AIConfig(BaseModel):
    """Complete AI configuration"""
    enabled: bool = True
    use_gpu: bool = True
    device: Optional[str] = None
    model_cache_dir: str = "./models_cache"
    
    # Model configurations
    embedding: ModelConfig
    sentiment: ModelConfig
    moderation: ModelConfig
    text_generation: ModelConfig
    diffusion: Optional[ModelConfig] = None
    
    # Training
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    
    class Config:
        extra = "allow"


class ConfigManager:
    """
    Manages AI configuration with validation and environment variable support
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config manager
        
        Args:
            config_path: Path to YAML config file
        """
        self.config_path = config_path or "model_config.yaml"
        self.config: Optional[AIConfig] = None
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from file and environment"""
        # Load from YAML if exists
        config_dict = {}
        if Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    config_dict = yaml.safe_load(f) or {}
                logger.info(f"Loaded config from {self.config_path}")
            except Exception as e:
                logger.warning(f"Error loading config file: {e}")
        
        # Override with environment variables
        config_dict = self._merge_env_vars(config_dict)
        
        # Create config object with validation
        try:
            self.config = AIConfig(**config_dict)
            logger.info("Configuration loaded and validated successfully")
        except Exception as e:
            logger.error(f"Error validating config: {e}", exc_info=True)
            # Use defaults
            self.config = AIConfig(
                embedding=ModelConfig(name="sentence-transformers/all-MiniLM-L6-v2"),
                sentiment=ModelConfig(name="cardiffnlp/twitter-roberta-base-sentiment-latest"),
                moderation=ModelConfig(name="unitary/toxic-bert"),
                text_generation=ModelConfig(name="gpt2")
            )
    
    def _merge_env_vars(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Merge environment variables into config"""
        env_mappings = {
            "AI_ENABLED": ("enabled", bool),
            "USE_GPU": ("use_gpu", bool),
            "DEVICE": ("device", str),
            "MODEL_CACHE_DIR": ("model_cache_dir", str),
            "EMBEDDING_MODEL": ("embedding", "name", str),
            "SENTIMENT_MODEL": ("sentiment", "name", str),
            "MODERATION_MODEL": ("moderation", "name", str),
            "TEXT_GENERATION_MODEL": ("text_generation", "name", str),
            "BATCH_SIZE": ("training", "batch_size", int),
            "LEARNING_RATE": ("training", "learning_rate", float),
            "NUM_EPOCHS": ("training", "num_epochs", int),
        }
        
        for env_key, (path, *rest) in env_mappings.items():
            value = os.getenv(env_key)
            if value is not None:
                if len(rest) == 1:
                    # Simple path
                    config_dict[path] = rest[0](value) if rest[0] != str else value
                else:
                    # Nested path
                    if path not in config_dict:
                        config_dict[path] = {}
                    config_dict[path][rest[0]] = rest[1](value) if len(rest) > 1 and rest[1] != str else value
        
        return config_dict
    
    def get_model_config(self, model_type: str) -> Optional[ModelConfig]:
        """
        Get configuration for a specific model type
        
        Args:
            model_type: Type of model (embedding, sentiment, etc.)
            
        Returns:
            ModelConfig or None
        """
        if not self.config:
            return None
        
        return getattr(self.config, model_type, None)
    
    def get_training_config(self) -> Optional[TrainingConfig]:
        """Get training configuration"""
        return self.config.training if self.config else None
    
    def get_lora_config(self) -> Optional[LoRAConfig]:
        """Get LoRA configuration"""
        return self.config.lora if self.config else None
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration
        
        Args:
            updates: Dictionary with updates
        """
        if not self.config:
            return
        
        for key, value in updates.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def save_config(self, save_path: Optional[str] = None) -> None:
        """
        Save current configuration to file
        
        Args:
            save_path: Path to save (defaults to config_path)
        """
        if not self.config:
            return
        
        save_path = save_path or self.config_path
        
        # Convert to dict
        config_dict = self.config.dict()
        
        # Save to YAML
        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {save_path}")


# Global config manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """
    Get or create global config manager
    
    Args:
        config_path: Path to config file
        
    Returns:
        ConfigManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager















