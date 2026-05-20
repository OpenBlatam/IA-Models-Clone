"""
🧠 Model Configuration

YAML-based configuration for model architectures and parameters.
Implements the key convention: "Use configuration files (e.g., YAML) for hyperparameters and model settings."
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """
    Comprehensive model configuration class.
    
    This class defines all model-related parameters that can be configured
    via YAML files, including architecture, optimization, and device settings.
    """
    
    # Model Architecture
    model_type: str = "classification"  # classification, regression, generation, transformer, diffusion
    architecture: str = "resnet"  # resnet, vgg, bert, gpt, stable_diffusion, etc.
    
    # Model Parameters
    input_size: Union[int, Tuple[int, ...]] = 224  # Input dimensions
    output_size: int = 10  # Number of classes or output dimensions
    hidden_size: int = 512  # Hidden layer size
    num_layers: int = 3  # Number of layers
    dropout_rate: float = 0.1  # Dropout probability
    
    # Advanced Architecture Parameters
    activation_function: str = "relu"  # relu, gelu, swish, tanh
    normalization: str = "batch_norm"  # batch_norm, layer_norm, group_norm
    attention_heads: int = 8  # For transformer models
    sequence_length: int = 512  # For sequence models
    
    # Pre-trained Model Settings
    pretrained: bool = False  # Use pre-trained weights
    pretrained_model_name: Optional[str] = None  # Name of pre-trained model
    freeze_backbone: bool = False  # Freeze pre-trained backbone
    freeze_layers: List[str] = field(default_factory=list)  # Specific layers to freeze
    
    # Model Initialization
    weight_init: str = "xavier_uniform"  # xavier_uniform, kaiming_normal, normal
    bias_init: str = "zeros"  # zeros, normal, uniform
    init_std: float = 0.02  # Standard deviation for weight initialization
    
    # Device and Precision Settings
    device: str = "auto"  # auto, cpu, cuda, mps
    mixed_precision: bool = False  # Enable mixed precision training
    compile_model: bool = False  # Use torch.compile (PyTorch 2.0+)
    
    # Memory Optimization
    gradient_checkpointing: bool = False  # Enable gradient checkpointing
    memory_efficient: bool = False  # Use memory-efficient implementations
    
    # Model-Specific Configurations
    classification_config: Dict[str, Any] = field(default_factory=dict)
    regression_config: Dict[str, Any] = field(default_factory=dict)
    transformer_config: Dict[str, Any] = field(default_factory=dict)
    diffusion_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and setup configuration after initialization."""
        self._validate_config()
        self._setup_device()
        self._setup_model_specific_config()
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        # Validate model type
        valid_model_types = ["classification", "regression", "generation", "transformer", "diffusion"]
        if self.model_type not in valid_model_types:
            raise ValueError(f"Invalid model_type: {self.model_type}. Must be one of {valid_model_types}")
        
        # Validate dropout rate
        if not 0.0 <= self.dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be between 0.0 and 1.0, got {self.dropout_rate}")
        
        # Validate output size
        if self.output_size <= 0:
            raise ValueError(f"output_size must be positive, got {self.output_size}")
        
        # Validate hidden size
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        
        logger.debug("Model configuration validation passed")
    
    def _setup_device(self) -> None:
        """Setup device configuration."""
        if self.device == "auto":
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        
        logger.debug(f"Device set to: {self.device}")
    
    def _setup_model_specific_config(self) -> None:
        """Setup model-specific configurations."""
        if self.model_type == "classification":
            default_classification = {
                "num_classes": self.output_size,
                "class_weights": None,
                "label_smoothing": 0.0
            }
            self.classification_config = {**default_classification, **self.classification_config}
        
        elif self.model_type == "regression":
            default_regression = {
                "output_activation": None,
                "loss_function": "mse"
            }
            self.regression_config = {**default_regression, **self.regression_config}
        
        elif self.model_type == "transformer":
            default_transformer = {
                "vocab_size": 30000,
                "max_position_embeddings": self.sequence_length,
                "num_attention_heads": self.attention_heads,
                "intermediate_size": self.hidden_size * 4,
                "attention_dropout": self.dropout_rate,
                "hidden_dropout": self.dropout_rate
            }
            self.transformer_config = {**default_transformer, **self.transformer_config}
        
        elif self.model_type == "diffusion":
            default_diffusion = {
                "timesteps": 1000,
                "beta_schedule": "linear",
                "prediction_type": "epsilon",
                "sample_size": 512
            }
            self.diffusion_config = {**default_diffusion, **self.diffusion_config}
    
    def get_model_summary(self) -> str:
        """
        Generate a summary of the model configuration.
        
        Returns:
            String summary of the model configuration
        """
        summary = f"""
Model Configuration Summary:
{'=' * 40}
Model Type: {self.model_type}
Architecture: {self.architecture}
Input Size: {self.input_size}
Output Size: {self.output_size}
Hidden Size: {self.hidden_size}
Number of Layers: {self.num_layers}
Dropout Rate: {self.dropout_rate}
Activation: {self.activation_function}
Normalization: {self.normalization}
Device: {self.device}
Pre-trained: {self.pretrained}
Mixed Precision: {self.mixed_precision}
        """.strip()
        
        # Add model-specific configurations
        if self.model_type == "transformer":
            summary += f"\n\nTransformer Configuration:"
            summary += f"\n  Attention Heads: {self.attention_heads}"
            summary += f"\n  Sequence Length: {self.sequence_length}"
            summary += f"\n  Vocab Size: {self.transformer_config.get('vocab_size', 'N/A')}"
        
        return summary
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        from dataclasses import asdict
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create ModelConfig from dictionary."""
        return cls(**config_dict)
    
    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown configuration key: {key}")
        
        # Re-validate after updates
        self._validate_config()
        self._setup_device()
        self._setup_model_specific_config()
    
    def clone(self) -> 'ModelConfig':
        """Create a copy of the configuration."""
        from copy import deepcopy
        return deepcopy(self)


# Predefined model configurations for common use cases
CLASSIFICATION_CONFIGS = {
    "resnet18": ModelConfig(
        model_type="classification",
        architecture="resnet18",
        input_size=(3, 224, 224),
        output_size=1000,
        pretrained=True,
        pretrained_model_name="resnet18"
    ),
    
    "simple_cnn": ModelConfig(
        model_type="classification",
        architecture="simple_cnn",
        input_size=(3, 32, 32),
        output_size=10,
        hidden_size=128,
        num_layers=3,
        dropout_rate=0.2
    ),
    
    "mlp": ModelConfig(
        model_type="classification",
        architecture="mlp",
        input_size=784,
        output_size=10,
        hidden_size=256,
        num_layers=3,
        dropout_rate=0.3
    )
}

TRANSFORMER_CONFIGS = {
    "bert_base": ModelConfig(
        model_type="transformer",
        architecture="bert",
        hidden_size=768,
        num_layers=12,
        attention_heads=12,
        sequence_length=512,
        dropout_rate=0.1,
        pretrained=True,
        pretrained_model_name="bert-base-uncased"
    ),
    
    "gpt2_small": ModelConfig(
        model_type="transformer",
        architecture="gpt2",
        hidden_size=768,
        num_layers=12,
        attention_heads=12,
        sequence_length=1024,
        dropout_rate=0.1,
        pretrained=True,
        pretrained_model_name="gpt2"
    ),
    
    "custom_transformer": ModelConfig(
        model_type="transformer",
        architecture="custom_transformer",
        hidden_size=512,
        num_layers=6,
        attention_heads=8,
        sequence_length=256,
        dropout_rate=0.1,
        transformer_config={
            "vocab_size": 30000,
            "intermediate_size": 2048,
            "attention_dropout": 0.1,
            "hidden_dropout": 0.1
        }
    )
}

DIFFUSION_CONFIGS = {
    "stable_diffusion": ModelConfig(
        model_type="diffusion",
        architecture="stable_diffusion",
        input_size=(3, 512, 512),
        hidden_size=1024,
        pretrained=True,
        pretrained_model_name="runwayml/stable-diffusion-v1-5",
        diffusion_config={
            "timesteps": 1000,
            "beta_schedule": "scaled_linear",
            "prediction_type": "epsilon",
            "sample_size": 512
        }
    ),
    
    "simple_diffusion": ModelConfig(
        model_type="diffusion",
        architecture="simple_diffusion",
        input_size=(3, 64, 64),
        hidden_size=256,
        num_layers=4,
        diffusion_config={
            "timesteps": 1000,
            "beta_schedule": "linear",
            "prediction_type": "epsilon",
            "sample_size": 64
        }
    )
}






