"""
Configuration Manager for Enhanced Transformer Models

This module provides comprehensive configuration management
for the refactored transformer architecture.
"""

import json
import yaml
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from ..base import BaseConfigManager
from ...transformer_config import TransformerConfig


class EnhancedConfigManager(BaseConfigManager):
    """Enhanced configuration manager with advanced features."""
    
    def __init__(self):
        super().__init__()
        self.config_cache = {}
        self.config_validators = {}
        self._register_default_validators()
    
    def _register_default_validators(self):
        """Register default configuration validators."""
        self.register_validator("transformer", self._validate_transformer_config)
        self.register_validator("quantum", self._validate_quantum_config)
        self.register_validator("attention", self._validate_attention_config)
    
    def register_validator(self, config_type: str, validator_func):
        """Register a configuration validator."""
        self.config_validators[config_type] = validator_func
    
    def load_config(self, config_path: str) -> TransformerConfig:
        """Load configuration from file with caching."""
        config_path = Path(config_path)
        
        # Check cache first
        if str(config_path) in self.config_cache:
            return self.config_cache[str(config_path)]
        
        # Load from file
        if config_path.suffix == '.json':
            config_dict = self._load_json_config(config_path)
        elif config_path.suffix in ['.yml', '.yaml']:
            config_dict = self._load_yaml_config(config_path)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        # Create config object
        config = TransformerConfig(**config_dict)
        
        # Cache the config
        self.config_cache[str(config_path)] = config
        
        return config
    
    def _load_json_config(self, config_path: Path) -> Dict[str, Any]:
        """Load JSON configuration."""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _load_yaml_config(self, config_path: Path) -> Dict[str, Any]:
        """Load YAML configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def save_config(self, config: TransformerConfig, config_path: str, format: str = 'json') -> None:
        """Save configuration to file."""
        config_path = Path(config_path)
        config_dict = config.__dict__.copy()
        
        if format == 'json':
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif format in ['yml', 'yaml']:
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def validate_config(self, config: TransformerConfig) -> bool:
        """Validate configuration with registered validators."""
        try:
            # Basic validation
            if not super().validate_config(config):
                return False
            
            # Type-specific validation
            for config_type, validator in self.config_validators.items():
                if not validator(config):
                    return False
            
            return True
        except Exception:
            return False
    
    def _validate_transformer_config(self, config: TransformerConfig) -> bool:
        """Validate transformer-specific configuration."""
        try:
            assert config.hidden_size % config.num_attention_heads == 0
            assert config.intermediate_size >= config.hidden_size
            assert config.max_position_embeddings > 0
            return True
        except (AssertionError, AttributeError):
            return False
    
    def _validate_quantum_config(self, config: TransformerConfig) -> bool:
        """Validate quantum-specific configuration."""
        # Quantum config validation would go here
        return True
    
    def _validate_attention_config(self, config: TransformerConfig) -> bool:
        """Validate attention-specific configuration."""
        try:
            assert config.num_attention_heads > 0
            assert config.hidden_size % config.num_attention_heads == 0
            return True
        except (AssertionError, AttributeError):
            return False
    
    def create_config_template(self, config_type: str = "transformer") -> Dict[str, Any]:
        """Create a configuration template."""
        templates = {
            "transformer": {
                "vocab_size": 50257,
                "hidden_size": 768,
                "num_layers": 12,
                "num_attention_heads": 12,
                "intermediate_size": 3072,
                "max_position_embeddings": 1024,
                "dropout": 0.1,
                "layer_norm_eps": 1e-12
            },
            "quantum": {
                "vocab_size": 50257,
                "hidden_size": 768,
                "num_layers": 12,
                "num_attention_heads": 12,
                "intermediate_size": 3072,
                "max_position_embeddings": 1024,
                "dropout": 0.1,
                "layer_norm_eps": 1e-12,
                "quantum_level": 0.8,
                "quantum_dim": 1024
            },
            "sparse": {
                "vocab_size": 50257,
                "hidden_size": 768,
                "num_layers": 12,
                "num_attention_heads": 12,
                "intermediate_size": 3072,
                "max_position_embeddings": 1024,
                "dropout": 0.1,
                "layer_norm_eps": 1e-12,
                "sparsity_pattern": "strided",
                "sparsity_ratio": 0.1
            }
        }
        
        return templates.get(config_type, templates["transformer"])
    
    def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configuration dictionaries."""
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self.merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def get_config_diff(self, config1: TransformerConfig, config2: TransformerConfig) -> Dict[str, Any]:
        """Get differences between two configurations."""
        dict1 = config1.__dict__
        dict2 = config2.__dict__
        
        diff = {}
        all_keys = set(dict1.keys()) | set(dict2.keys())
        
        for key in all_keys:
            if key not in dict1:
                diff[key] = {"type": "added", "value": dict2[key]}
            elif key not in dict2:
                diff[key] = {"type": "removed", "value": dict1[key]}
            elif dict1[key] != dict2[key]:
                diff[key] = {
                    "type": "changed",
                    "old_value": dict1[key],
                    "new_value": dict2[key]
                }
        
        return diff
    
    def clear_cache(self):
        """Clear configuration cache."""
        self.config_cache.clear()
    
    def get_cached_configs(self) -> List[str]:
        """Get list of cached configuration paths."""
        return list(self.config_cache.keys())


class ConfigBuilder:
    """Builder pattern for creating configurations."""
    
    def __init__(self):
        self.config_dict = {}
    
    def set_vocab_size(self, vocab_size: int) -> 'ConfigBuilder':
        """Set vocabulary size."""
        self.config_dict['vocab_size'] = vocab_size
        return self
    
    def set_hidden_size(self, hidden_size: int) -> 'ConfigBuilder':
        """Set hidden size."""
        self.config_dict['hidden_size'] = hidden_size
        return self
    
    def set_num_layers(self, num_layers: int) -> 'ConfigBuilder':
        """Set number of layers."""
        self.config_dict['num_layers'] = num_layers
        return self
    
    def set_num_attention_heads(self, num_attention_heads: int) -> 'ConfigBuilder':
        """Set number of attention heads."""
        self.config_dict['num_attention_heads'] = num_attention_heads
        return self
    
    def set_intermediate_size(self, intermediate_size: int) -> 'ConfigBuilder':
        """Set intermediate size."""
        self.config_dict['intermediate_size'] = intermediate_size
        return self
    
    def set_max_position_embeddings(self, max_position_embeddings: int) -> 'ConfigBuilder':
        """Set maximum position embeddings."""
        self.config_dict['max_position_embeddings'] = max_position_embeddings
        return self
    
    def set_dropout(self, dropout: float) -> 'ConfigBuilder':
        """Set dropout rate."""
        self.config_dict['dropout'] = dropout
        return self
    
    def set_layer_norm_eps(self, layer_norm_eps: float) -> 'ConfigBuilder':
        """Set layer normalization epsilon."""
        self.config_dict['layer_norm_eps'] = layer_norm_eps
        return self
    
    def set_quantum_level(self, quantum_level: float) -> 'ConfigBuilder':
        """Set quantum level."""
        self.config_dict['quantum_level'] = quantum_level
        return self
    
    def set_sparsity_pattern(self, sparsity_pattern: str) -> 'ConfigBuilder':
        """Set sparsity pattern."""
        self.config_dict['sparsity_pattern'] = sparsity_pattern
        return self
    
    def build(self) -> TransformerConfig:
        """Build the configuration."""
        return TransformerConfig(**self.config_dict)
    
    def build_dict(self) -> Dict[str, Any]:
        """Build configuration as dictionary."""
        return self.config_dict.copy()

