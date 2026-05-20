"""
Configuration Utilities
Enhanced configuration management
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging
from dataclasses import dataclass, asdict
from copy import deepcopy

logger = logging.getLogger(__name__)


class ConfigManager:
    """Enhanced configuration manager"""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = Path(config_path) if config_path else None
        self.config: Dict[str, Any] = {}
        
        if self.config_path and self.config_path.exists():
            self.load(self.config_path)
    
    def load(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from file"""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        if path.suffix in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                self.config = yaml.safe_load(f) or {}
        elif path.suffix == '.json':
            with open(path, 'r') as f:
                self.config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
        
        logger.info(f"Configuration loaded from {path}")
        return self.config
    
    def save(self, path: Optional[Union[str, Path]] = None):
        """Save configuration to file"""
        path = Path(path or self.config_path)
        
        if not path:
            raise ValueError("No path specified for saving config")
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.suffix in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        elif path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(self.config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
        
        logger.info(f"Configuration saved to {path}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value (supports dot notation)"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any], deep: bool = True):
        """Update configuration"""
        if deep:
            self.config = deep_merge(self.config, updates)
        else:
            self.config.update(updates)
    
    def merge(self, other: 'ConfigManager'):
        """Merge another config manager"""
        self.config = deep_merge(self.config, other.config)
    
    def copy(self) -> Dict[str, Any]:
        """Get a copy of the configuration"""
        return deepcopy(self.config)
    
    def validate(self, schema: Optional[Dict[str, Any]] = None) -> bool:
        """Validate configuration against schema"""
        if schema is None:
            return True
        
        # Simple validation - check required keys
        required_keys = schema.get('required', [])
        for key in required_keys:
            if self.get(key) is None:
                logger.error(f"Missing required config key: {key}")
                return False
        
        return True


def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries"""
    result = deepcopy(base)
    
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    
    return result


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from file (convenience function)"""
    manager = ConfigManager()
    return manager.load(path)


def save_config(config: Dict[str, Any], path: Union[str, Path]):
    """Save configuration to file (convenience function)"""
    manager = ConfigManager()
    manager.config = config
    manager.save(path)


@dataclass
class TrainingConfig:
    """Training configuration dataclass"""
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    use_mixed_precision: bool = True
    gradient_clip_val: float = 1.0
    gradient_accumulation_steps: int = 1
    num_workers: int = 4
    pin_memory: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingConfig':
        """Create from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class ModelConfig:
    """Model configuration dataclass"""
    name: str = "vit_skin"
    num_conditions: int = 6
    num_metrics: int = 8
    input_size: tuple = (224, 224)
    use_pretrained: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """Create from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})













