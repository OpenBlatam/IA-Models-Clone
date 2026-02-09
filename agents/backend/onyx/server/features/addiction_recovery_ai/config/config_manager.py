"""
Configuration Management for Recovery AI
"""

import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """Configuration manager for the system"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize config manager
        
        Args:
            config_file: Path to config file
        """
        self.config = {}
        self.config_file = config_file
        
        if config_file:
            self.load(config_file)
        else:
            self.load_defaults()
    
    def load_defaults(self):
        """Load default configuration"""
        self.config = {
            "model": {
                "use_gpu": True,
                "device": "auto",
                "batch_size": 32,
                "max_sequence_length": 30
            },
            "training": {
                "learning_rate": 1e-3,
                "batch_size": 32,
                "num_epochs": 10,
                "use_mixed_precision": True,
                "gradient_accumulation": 1
            },
            "optimization": {
                "use_jit": True,
                "use_quantization": True,
                "use_compile": True,
                "cache_size": 1000
            },
            "serving": {
                "max_batch_size": 32,
                "max_queue_size": 100,
                "num_workers": 2,
                "timeout": 10.0
            },
            "security": {
                "rate_limit": 100,
                "rate_window": 60.0,
                "enable_api_keys": True
            },
            "monitoring": {
                "enable_profiling": True,
                "log_level": "INFO",
                "metrics_file": "metrics.csv"
            }
        }
    
    def load(self, config_file: str):
        """
        Load configuration from file
        
        Args:
            config_file: Path to config file
        """
        path = Path(config_file)
        
        if not path.exists():
            logger.warning(f"Config file not found: {config_file}, using defaults")
            self.load_defaults()
            return
        
        try:
            if path.suffix in ['.yaml', '.yml']:
                with open(path, 'r') as f:
                    self.config = yaml.safe_load(f) or {}
            elif path.suffix == '.json':
                with open(path, 'r') as f:
                    self.config = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {path.suffix}")
            
            logger.info(f"Configuration loaded from {config_file}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}, using defaults")
            self.load_defaults()
    
    def save(self, config_file: str):
        """
        Save configuration to file
        
        Args:
            config_file: Path to config file
        """
        path = Path(config_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if path.suffix in ['.yaml', '.yml']:
                with open(path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
            elif path.suffix == '.json':
                with open(path, 'w') as f:
                    json.dump(self.config, f, indent=2)
            else:
                raise ValueError(f"Unsupported config file format: {path.suffix}")
            
            logger.info(f"Configuration saved to {config_file}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
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
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        logger.debug(f"Configuration updated: {key} = {value}")
    
    def update(self, updates: Dict[str, Any]):
        """
        Update configuration with dictionary
        
        Args:
            updates: Dictionary of updates
        """
        def deep_update(base, updates):
            for key, value in updates.items():
                if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                    deep_update(base[key], value)
                else:
                    base[key] = value
        
        deep_update(self.config, updates)
        logger.info("Configuration updated")


def load_config(config_file: Optional[str] = None) -> ConfigManager:
    """
    Load configuration
    
    Args:
        config_file: Optional config file path
    
    Returns:
        ConfigManager instance
    """
    return ConfigManager(config_file)

