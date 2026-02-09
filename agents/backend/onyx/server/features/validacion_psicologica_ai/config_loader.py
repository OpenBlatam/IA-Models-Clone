"""
Configuration Loader
===================
Load and manage configuration from YAML files
"""

from typing import Dict, Any, Optional
import yaml
import os
from pathlib import Path
import structlog

logger = structlog.get_logger()


class ConfigLoader:
    """Loader for YAML configuration files"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config loader
        
        Args:
            config_path: Path to config file
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config: Dict[str, Any] = {}
        self._load_config()
    
    def _get_default_config_path(self) -> str:
        """Get default config path"""
        current_dir = Path(__file__).parent
        return str(current_dir / "config" / "dl_config.yaml")
    
    def _load_config(self) -> None:
        """Load configuration from YAML file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config = yaml.safe_load(f) or {}
                logger.info("Configuration loaded", path=self.config_path)
            else:
                logger.warning("Config file not found, using defaults", path=self.config_path)
                self.config = self._get_default_config()
        except Exception as e:
            logger.error("Error loading config", error=str(e))
            self.config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "models": {
                "embedding": {"name": "sentence-transformers/all-MiniLM-L6-v2"},
                "personality": {"name": "distilbert-base-uncased"},
                "sentiment": {"name": "cardiffnlp/twitter-roberta-base-sentiment-latest"}
            },
            "training": {
                "batch_size": 16,
                "learning_rate": 2e-5,
                "num_epochs": 3
            },
            "device": {
                "use_cuda": True,
                "cuda_device": 0
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value if value is not None else default
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """
        Get model configuration
        
        Args:
            model_type: Type of model (embedding, personality, sentiment, diffusion)
            
        Returns:
            Model configuration
        """
        return self.config.get("models", {}).get(model_type, {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return self.config.get("training", {})
    
    def get_device_config(self) -> Dict[str, Any]:
        """Get device configuration"""
        return self.config.get("device", {})


# Global config loader instance
config_loader = ConfigLoader()




