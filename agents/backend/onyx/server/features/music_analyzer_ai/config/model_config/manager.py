"""
Config Manager Module

Configuration file management.
"""

from typing import Dict, Any, List
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging
import yaml
import json

logger = logging.getLogger(__name__)

from .architecture import ModelArchitectureConfig
from .training import TrainingConfig
from .data import DataConfig
from .experiment import ExperimentConfig


@dataclass
class ModelConfig:
    """Complete model configuration"""
    architecture: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    device: str = "cuda"
    seed: Any = None
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create config from dictionary"""
        return cls(
            architecture=ModelArchitectureConfig(**config_dict.get("architecture", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            data=DataConfig(**config_dict.get("data", {})),
            experiment=ExperimentConfig(**config_dict.get("experiment", {})),
            device=config_dict.get("device", "cuda"),
            seed=config_dict.get("seed", None)
        )
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ModelConfig':
        """Load config from YAML file"""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_json(cls, json_path: str) -> 'ModelConfig':
        """Load config from JSON file"""
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Config file not found: {json_path}")
        
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)
    
    def to_yaml(self, yaml_path: str):
        """Save config to YAML file"""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    def to_json(self, json_path: str):
        """Save config to JSON file"""
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class ConfigManager:
    """Manager for configuration files"""
    
    def __init__(self, config_dir: str = "./configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def save_config(self, config: ModelConfig, name: str, format: str = "yaml"):
        """Save configuration"""
        if format == "yaml":
            config.to_yaml(self.config_dir / f"{name}.yaml")
        elif format == "json":
            config.to_json(self.config_dir / f"{name}.json")
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def load_config(self, name: str, format: str = "yaml") -> ModelConfig:
        """Load configuration"""
        if format == "yaml":
            return ModelConfig.from_yaml(self.config_dir / f"{name}.yaml")
        elif format == "json":
            return ModelConfig.from_json(self.config_dir / f"{name}.json")
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def list_configs(self) -> List[str]:
        """List available configurations"""
        configs = []
        for path in self.config_dir.glob("*.yaml"):
            configs.append(path.stem)
        for path in self.config_dir.glob("*.json"):
            if path.stem not in configs:
                configs.append(path.stem)
        return configs



