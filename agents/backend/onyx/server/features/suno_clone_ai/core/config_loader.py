"""
Configuration Loader for Hyperparameters

Loads configuration from YAML files and provides easy access to hyperparameters.
"""

import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Load and manage configuration from YAML files.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config loader.
        
        Args:
            config_path: Path to YAML config file
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config: Dict[str, Any] = {}
        self._load_config()
    
    def _get_default_config_path(self) -> str:
        """Get default config path."""
        # Look for config in the config directory
        config_dir = Path(__file__).parent.parent / "config"
        config_file = config_dir / "hyperparameters.yaml"
        return str(config_file)
    
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                logger.warning(
                    f"Config file not found: {self.config_path}. "
                    "Using default configuration."
                )
                self.config = self._get_default_config()
                return
            
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f) or {}
            
            logger.info(f"Loaded configuration from {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error loading config: {e}", exc_info=True)
            logger.warning("Using default configuration")
            self.config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "model": {
                "vocab_size": 32000,
                "d_model": 512,
                "num_heads": 8,
                "num_layers": 6,
                "dropout": 0.1
            },
            "training": {
                "batch_size": 4,
                "num_epochs": 100,
                "learning_rate": 1.0e-4
            },
            "inference": {
                "temperature": 1.0,
                "top_k": 250,
                "top_p": 0.0
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports nested keys with dots).
        
        Args:
            key: Configuration key (e.g., "model.d_model" or "training.batch_size")
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config.get("model", {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.config.get("training", {})
    
    def get_inference_config(self) -> Dict[str, Any]:
        """Get inference configuration."""
        return self.config.get("inference", {})
    
    def get_lora_config(self) -> Dict[str, Any]:
        """Get LoRA configuration."""
        return self.config.get("lora", {})
    
    def update(self, key: str, value: Any) -> None:
        """
        Update configuration value.
        
        Args:
            key: Configuration key (supports nested keys with dots)
            value: New value
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        logger.info(f"Updated config: {key} = {value}")
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            path: Path to save config (if None, uses original path)
        """
        save_path = path or self.config_path
        
        try:
            with open(save_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Saved configuration to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving config: {e}", exc_info=True)
            raise


# Global config loader instance
_config_loader: Optional[ConfigLoader] = None


def get_config_loader(config_path: Optional[str] = None) -> ConfigLoader:
    """
    Get global config loader instance.
    
    Args:
        config_path: Path to config file (only used on first call)
        
    Returns:
        ConfigLoader instance
    """
    global _config_loader
    
    if _config_loader is None:
        _config_loader = ConfigLoader(config_path)
    
    return _config_loader


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    loader = get_config_loader(config_path)
    return loader.config



