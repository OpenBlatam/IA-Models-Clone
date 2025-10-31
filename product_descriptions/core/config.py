from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import torch
from pathlib import Path
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Configuration module for Product Descriptions Generator
=======================================================

Defines configuration classes and settings for the AI model.
"""



@dataclass
class ModelConfig:
    """Configuration for the transformer model."""
    
    # Model architecture
    model_name: str = "microsoft/DialoGPT-medium"
    max_length: int = 512
    min_length: int = 50
    num_beams: int = 4
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    
    # Device and performance
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    
    # Fine-tuning parameters
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0


@dataclass
class DataConfig:
    """Configuration for data processing."""
    
    # Tokenization
    max_input_length: int = 256
    max_target_length: int = 512
    padding: str = "max_length"
    truncation: bool = True
    
    # Data paths
    train_data_path: Optional[str] = None
    val_data_path: Optional[str] = None
    test_data_path: Optional[str] = None
    
    # Preprocessing
    remove_special_chars: bool = True
    lowercase: bool = False
    min_word_count: int = 3


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    
    # Generation parameters
    max_new_tokens: int = 300
    min_new_tokens: int = 50
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    length_penalty: float = 1.0
    
    # Output formatting
    add_special_tokens: bool = True
    clean_up_tokenization_spaces: bool = True
    
    # Language and style
    languages: List[str] = field(default_factory=lambda: ["en", "es", "fr"])
    tones: List[str] = field(default_factory=lambda: [
        "professional", "casual", "luxury", "technical", "creative"
    ])
    
    # SEO optimization
    include_keywords: bool = True
    keyword_density: float = 0.02
    meta_description_length: int = 160


@dataclass
class ProductDescriptionConfig:
    """Main configuration class combining all settings."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    
    # Application settings
    app_name: str = "Product Description Generator"
    version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    
    # Cache settings
    cache_enabled: bool = True
    cache_ttl: int = 3600
    cache_max_size: int = 1000
    
    # API settings
    api_timeout: int = 30
    max_requests_per_minute: int = 100
    
    # File paths
    model_cache_dir: str = "./models_cache"
    logs_dir: str = "./logs"
    output_dir: str = "./outputs"
    
    def __post_init__(self) -> Any:
        """Validate configuration after initialization."""
        self._create_directories()
        self._validate_config()
    
    def _create_directories(self) -> Any:
        """Create necessary directories."""
        for directory in [self.model_cache_dir, self.logs_dir, self.output_dir]:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _validate_config(self) -> bool:
        """Validate configuration parameters."""
        if self.model.max_length < self.model.min_length:
            raise ValueError("max_length must be greater than min_length")
        
        if self.generation.max_new_tokens < self.generation.min_new_tokens:
            raise ValueError("max_new_tokens must be greater than min_new_tokens")
        
        if not 0 <= self.model.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        
        if not 0 <= self.model.top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            "model": self.model.__dict__,
            "data": self.data.__dict__, 
            "generation": self.generation.__dict__,
            "app_name": self.app_name,
            "version": self.version,
            "debug": self.debug
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> "ProductDescriptionConfig":
        """Create configuration from dictionary."""
        config = cls()
        
        if "model" in config_dict:
            for key, value in config_dict["model"].items():
                setattr(config.model, key, value)
        
        if "data" in config_dict:
            for key, value in config_dict["data"].items():
                setattr(config.data, key, value)
        
        if "generation" in config_dict:
            for key, value in config_dict["generation"].items():
                setattr(config.generation, key, value)
        
        # Update other attributes
        for key in ["app_name", "version", "debug", "log_level"]:
            if key in config_dict:
                setattr(config, key, config_dict[key])
        
        return config


# Predefined configurations for different use cases
ECOMMERCE_CONFIG = ProductDescriptionConfig(
    model=ModelConfig(
        model_name="microsoft/DialoGPT-medium",
        max_length=400,
        temperature=0.7,
        num_beams=3
    ),
    generation=GenerationConfig(
        max_new_tokens=200,
        temperature=0.7,
        include_keywords=True,
        tones=["professional", "persuasive", "friendly"]
    )
)

LUXURY_CONFIG = ProductDescriptionConfig(
    model=ModelConfig(
        model_name="microsoft/DialoGPT-large",
        max_length=600,
        temperature=0.8,
        num_beams=5
    ),
    generation=GenerationConfig(
        max_new_tokens=350,
        temperature=0.8,
        tones=["luxury", "sophisticated", "exclusive"],
        length_penalty=1.2
    )
)

TECHNICAL_CONFIG = ProductDescriptionConfig(
    model=ModelConfig(
        model_name="microsoft/DialoGPT-medium",
        max_length=800,
        temperature=0.6,
        num_beams=4
    ),
    generation=GenerationConfig(
        max_new_tokens=500,
        temperature=0.6,
        tones=["technical", "detailed", "informative"],
        include_keywords=True,
        keyword_density=0.03
    )
) 