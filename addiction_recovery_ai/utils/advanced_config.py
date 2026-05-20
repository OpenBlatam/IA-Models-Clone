"""
Advanced Configuration Management
"""

import json
import yaml
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
import os

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str
    model_type: str
    input_size: int
    hidden_size: int
    num_layers: int
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 10


@dataclass
class TrainingConfig:
    """Training configuration"""
    optimizer: str = "adam"
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 10
    early_stopping: bool = True
    validation_split: float = 0.2


@dataclass
class InferenceConfig:
    """Inference configuration"""
    batch_size: int = 32
    device: str = "auto"
    use_mixed_precision: bool = False
    max_queue_size: int = 100


class ConfigManager:
    """Advanced configuration manager"""
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize config manager
        
        Args:
            config_dir: Configuration directory
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.configs: Dict[str, Any] = {}
        logger.info(f"ConfigManager initialized: {config_dir}")
    
    def load_config(
        self,
        name: str,
        filepath: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load configuration from file
        
        Args:
            name: Configuration name
            filepath: Optional file path
        
        Returns:
            Configuration dictionary
        """
        if filepath:
            path = Path(filepath)
        else:
            # Try YAML first, then JSON
            yaml_path = self.config_dir / f"{name}.yaml"
            json_path = self.config_dir / f"{name}.json"
            
            if yaml_path.exists():
                path = yaml_path
            elif json_path.exists():
                path = json_path
            else:
                raise FileNotFoundError(f"Config file not found: {name}")
        
        with open(path, 'r') as f:
            if path.suffix == '.yaml':
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
        
        self.configs[name] = config
        logger.info(f"Config loaded: {name}")
        return config
    
    def save_config(
        self,
        name: str,
        config: Dict[str, Any],
        format: str = "yaml"
    ):
        """
        Save configuration to file
        
        Args:
            name: Configuration name
            config: Configuration dictionary
            format: File format (yaml or json)
        """
        if format == "yaml":
            filepath = self.config_dir / f"{name}.yaml"
            with open(filepath, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            filepath = self.config_dir / f"{name}.json"
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
        
        self.configs[name] = config
        logger.info(f"Config saved: {name}")
    
    def get_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Get configuration"""
        return self.configs.get(name)
    
    def update_config(
        self,
        name: str,
        updates: Dict[str, Any]
    ):
        """
        Update configuration
        
        Args:
            name: Configuration name
            updates: Updates to apply
        """
        if name not in self.configs:
            self.configs[name] = {}
        
        self.configs[name].update(updates)
        logger.info(f"Config updated: {name}")
    
    def load_from_env(self, prefix: str = "RECOVERY_AI_") -> Dict[str, Any]:
        """
        Load configuration from environment variables
        
        Args:
            prefix: Environment variable prefix
        
        Returns:
            Configuration dictionary
        """
        config = {}
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                # Try to parse as JSON, otherwise use as string
                try:
                    config[config_key] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    config[config_key] = value
        
        logger.info(f"Config loaded from environment: {len(config)} variables")
        return config
    
    def merge_configs(
        self,
        base_config: Dict[str, Any],
        override_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge configurations
        
        Args:
            base_config: Base configuration
            override_config: Override configuration
        
        Returns:
            Merged configuration
        """
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self.merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged

