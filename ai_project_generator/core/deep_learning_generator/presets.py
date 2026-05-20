"""
Presets Module

Pre-configured generator presets for common use cases.
"""

from typing import Dict, Any
from .constants import SUPPORTED_FRAMEWORKS, SUPPORTED_MODEL_TYPES


class PresetManager:
    """
    Manages pre-configured presets for common deep learning scenarios.
    """
    
    @staticmethod
    def get_preset(preset_name: str) -> Dict[str, Any]:
        """
        Get a preset configuration by name.
        
        Args:
            preset_name: Name of the preset
            
        Returns:
            Configuration dictionary
            
        Raises:
            ValueError: If preset doesn't exist
        """
        presets = PresetManager._get_all_presets()
        if preset_name not in presets:
            raise ValueError(
                f"Preset '{preset_name}' not found. "
                f"Available presets: {', '.join(presets.keys())}"
            )
        return presets[preset_name].copy()
    
    @staticmethod
    def list_presets() -> list[str]:
        """List all available presets."""
        return list(PresetManager._get_all_presets().keys())
    
    @staticmethod
    def _get_all_presets() -> Dict[str, Dict[str, Any]]:
        """Get all available presets."""
        return {
            "transformer_pytorch": {
                "framework": "pytorch",
                "model_type": "transformer",
                "use_gpu": True,
                "mixed_precision": True,
                "batch_size": 32,
                "learning_rate": 1e-4,
                "num_epochs": 10,
                "early_stopping": True,
                "gradient_clipping": True,
                "checkpointing": True,
                "experiment_tracking": True
            },
            "transformer_tensorflow": {
                "framework": "tensorflow",
                "model_type": "transformer",
                "use_gpu": True,
                "mixed_precision": True,
                "batch_size": 32,
                "learning_rate": 1e-4,
                "num_epochs": 10,
                "early_stopping": True,
                "gradient_clipping": True,
                "checkpointing": True
            },
            "cnn_pytorch": {
                "framework": "pytorch",
                "model_type": "cnn",
                "use_gpu": True,
                "mixed_precision": True,
                "batch_size": 64,
                "learning_rate": 1e-3,
                "num_epochs": 20,
                "early_stopping": True,
                "gradient_clipping": False,
                "checkpointing": True
            },
            "llm_pytorch": {
                "framework": "pytorch",
                "model_type": "llm",
                "use_gpu": True,
                "mixed_precision": True,
                "batch_size": 8,
                "learning_rate": 5e-5,
                "num_epochs": 3,
                "early_stopping": True,
                "gradient_clipping": True,
                "gradient_clipping_max_norm": 1.0,
                "checkpointing": True,
                "experiment_tracking": True
            },
            "diffusion_pytorch": {
                "framework": "pytorch",
                "model_type": "diffusion",
                "use_gpu": True,
                "mixed_precision": True,
                "batch_size": 4,
                "learning_rate": 1e-4,
                "num_epochs": 50,
                "early_stopping": False,
                "gradient_clipping": True,
                "checkpointing": True,
                "experiment_tracking": True
            },
            "vision_transformer": {
                "framework": "pytorch",
                "model_type": "vision_transformer",
                "use_gpu": True,
                "mixed_precision": True,
                "batch_size": 16,
                "learning_rate": 1e-4,
                "num_epochs": 15,
                "early_stopping": True,
                "gradient_clipping": True,
                "checkpointing": True
            },
            "gan_pytorch": {
                "framework": "pytorch",
                "model_type": "gan",
                "use_gpu": True,
                "mixed_precision": False,
                "batch_size": 32,
                "learning_rate": 2e-4,
                "num_epochs": 100,
                "early_stopping": False,
                "gradient_clipping": False,
                "checkpointing": True
            },
            "vae_pytorch": {
                "framework": "pytorch",
                "model_type": "vae",
                "use_gpu": True,
                "mixed_precision": True,
                "batch_size": 64,
                "learning_rate": 1e-3,
                "num_epochs": 30,
                "early_stopping": True,
                "gradient_clipping": True,
                "checkpointing": True
            },
            "rnn_pytorch": {
                "framework": "pytorch",
                "model_type": "rnn",
                "use_gpu": True,
                "mixed_precision": False,
                "batch_size": 32,
                "learning_rate": 1e-3,
                "num_epochs": 20,
                "early_stopping": True,
                "gradient_clipping": True,
                "checkpointing": True
            },
            "production_ready": {
                "framework": "pytorch",
                "use_gpu": True,
                "mixed_precision": True,
                "batch_size": 32,
                "learning_rate": 1e-4,
                "num_epochs": 10,
                "early_stopping": True,
                "early_stopping_patience": 5,
                "gradient_clipping": True,
                "gradient_clipping_max_norm": 1.0,
                "checkpointing": True,
                "checkpoint_save_best": True,
                "experiment_tracking": True,
                "experiment_tracking_backend": "wandb"
            },
            "fast_prototyping": {
                "framework": "pytorch",
                "use_gpu": False,
                "mixed_precision": False,
                "batch_size": 16,
                "learning_rate": 1e-3,
                "num_epochs": 3,
                "early_stopping": False,
                "gradient_clipping": False,
                "checkpointing": False,
                "experiment_tracking": False
            }
        }


def get_preset(preset_name: str) -> Dict[str, Any]:
    """Get a preset configuration."""
    return PresetManager.get_preset(preset_name)


def list_presets() -> list[str]:
    """List all available presets."""
    return PresetManager.list_presets()















