"""
Configuration Loader
Load and manage YAML configurations
"""

import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import logging
import os

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Load and manage configurations from YAML files
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config loader
        
        Args:
            config_path: Path to config file
        """
        if config_path is None:
            # Default config path
            config_path = Path(__file__).parent.parent.parent / "config" / "model_config.yaml"
        
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.load()
    
    def load(self) -> Dict[str, Any]:
        """
        Load configuration from file
        
        Returns:
            Configuration dictionary
        """
        try:
            if not self.config_path.exists():
                logger.warning(f"Config file not found: {self.config_path}, using defaults")
                return self._get_default_config()
            
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f) or {}
            
            # Override with environment variables
            self._override_from_env()
            
            logger.info(f"Configuration loaded from {self.config_path}")
            return self.config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._get_default_config()
    
    def _override_from_env(self):
        """Override config with environment variables"""
        # Example: RECOVERY_AI_MODEL_DEVICE=cuda
        prefix = "RECOVERY_AI_"
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Convert RECOVERY_AI_MODEL_DEVICE -> model.device
                config_key = key[len(prefix):].lower().replace("_", ".")
                
                # Try to convert value type
                if value.lower() in ("true", "false"):
                    value = value.lower() == "true"
                elif value.isdigit():
                    value = int(value)
                elif self._is_float(value):
                    value = float(value)
                
                self._set_nested(self.config, config_key, value)
    
    def _set_nested(self, d: Dict, key: str, value: Any):
        """Set nested dictionary value"""
        keys = key.split(".")
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    
    def _is_float(self, s: str) -> bool:
        """Check if string is float"""
        try:
            float(s)
            return True
        except ValueError:
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value
        
        Args:
            key: Configuration key (supports dot notation, e.g., "model.device")
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration value
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        self._set_nested(self.config, key, value)
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get model-specific configuration
        
        Args:
            model_name: Name of model
            
        Returns:
            Model configuration
        """
        return self.get(f"models.{model_name}", {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return self.get("training", {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration"""
        return self.get("data", {})
    
    def get_inference_config(self) -> Dict[str, Any]:
        """Get inference configuration"""
        return self.get("inference", {})
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "models": {},
            "training": {
                "batch_size": 32,
                "num_epochs": 50,
                "learning_rate": 0.001
            },
            "data": {
                "batch_size": 32,
                "num_workers": 4
            },
            "inference": {
                "batch_size": 64
            }
        }
    
    def save(self, path: Optional[str] = None):
        """
        Save configuration to file
        
        Args:
            path: Path to save (defaults to original path)
        """
        save_path = Path(path) if path else self.config_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Configuration saved to {save_path}")


# Global config instance
_global_config: Optional[ConfigLoader] = None


def get_config(config_path: Optional[str] = None) -> ConfigLoader:
    """
    Get global configuration instance
    
    Args:
        config_path: Path to config file (only used on first call)
        
    Returns:
        ConfigLoader instance
    """
    global _global_config
    if _global_config is None:
        _global_config = ConfigLoader(config_path)
    return _global_config













