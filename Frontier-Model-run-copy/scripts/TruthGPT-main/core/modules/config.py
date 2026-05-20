"""
Configuration Management
Centralized configuration system for modular architecture
"""

import json
import yaml
import os
import logging
from typing import Dict, Any, List, Optional, Union, Type, Callable
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import threading
import time

logger = logging.getLogger(__name__)

class ConfigFormat(Enum):
    """Supported configuration formats"""
    JSON = "json"
    YAML = "yaml"
    ENV = "env"
    PYTHON = "python"

@dataclass
class ConfigSchema:
    """Configuration schema definition"""
    name: str
    version: str
    description: str
    properties: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)
    defaults: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConfigSource:
    """Configuration source definition"""
    name: str
    path: str
    format: ConfigFormat
    priority: int = 0
    enabled: bool = True
    last_modified: float = 0

class ConfigValidator:
    """Validator for configuration data"""
    
    def __init__(self):
        self.schemas: Dict[str, ConfigSchema] = {}
        self.validators: Dict[str, Callable] = {}
    
    def register_schema(self, schema: ConfigSchema) -> None:
        """Register a configuration schema"""
        self.schemas[schema.name] = schema
        logger.info(f"Registered schema: {schema.name}")
    
    def register_validator(self, name: str, validator: Callable[[Any], bool]) -> None:
        """Register a custom validator"""
        self.validators[name] = validator
    
    def validate(self, config_name: str, config: Dict[str, Any]) -> List[str]:
        """Validate configuration against schema"""
        errors = []
        
        if config_name not in self.schemas:
            return errors
        
        schema = self.schemas[config_name]
        
        # Check required fields
        for field in schema.required:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # Validate properties
        for field, value in config.items():
            if field in schema.properties:
                prop_schema = schema.properties[field]
                field_errors = self._validate_field(field, value, prop_schema)
                errors.extend(field_errors)
        
        return errors
    
    def _validate_field(self, field: str, value: Any, schema: Dict[str, Any]) -> List[str]:
        """Validate a single field"""
        errors = []
        
        # Type validation
        expected_type = schema.get("type")
        if expected_type and not isinstance(value, expected_type):
            errors.append(f"Field {field} must be of type {expected_type.__name__}")
        
        # Range validation
        if "min" in schema and value < schema["min"]:
            errors.append(f"Field {field} must be >= {schema['min']}")
        if "max" in schema and value > schema["max"]:
            errors.append(f"Field {field} must be <= {schema['max']}")
        
        # Enum validation
        if "enum" in schema and value not in schema["enum"]:
            errors.append(f"Field {field} must be one of {schema['enum']}")
        
        # Custom validator
        validator_name = schema.get("validator")
        if validator_name and validator_name in self.validators:
            if not self.validators[validator_name](value):
                errors.append(f"Field {field} failed custom validation")
        
        return errors

class ConfigLoader:
    """Loader for configuration files"""
    
    def __init__(self):
        self.loaders = {
            ConfigFormat.JSON: self._load_json,
            ConfigFormat.YAML: self._load_yaml,
            ConfigFormat.ENV: self._load_env,
            ConfigFormat.PYTHON: self._load_python
        }
    
    def load(self, source: ConfigSource) -> Optional[Dict[str, Any]]:
        """Load configuration from source"""
        if not source.enabled:
            return None
        
        if source.format not in self.loaders:
            logger.error(f"Unsupported format: {source.format}")
            return None
        
        try:
            return self.loaders[source.format](source.path)
        except Exception as e:
            logger.error(f"Failed to load config from {source.path}: {e}")
            return None
    
    def _load_json(self, path: str) -> Dict[str, Any]:
        """Load JSON configuration"""
        with open(path, 'r') as f:
            return json.load(f)
    
    def _load_yaml(self, path: str) -> Dict[str, Any]:
        """Load YAML configuration"""
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_env(self, path: str) -> Dict[str, Any]:
        """Load environment variables"""
        config = {}
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    config[key] = value
        return config
    
    def _load_python(self, path: str) -> Dict[str, Any]:
        """Load Python configuration"""
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        config = {}
        for attr in dir(module):
            if not attr.startswith('_'):
                config[attr] = getattr(module, attr)
        return config

class ConfigManager:
    """Centralized configuration manager"""
    
    def __init__(self):
        self.configs: Dict[str, Dict[str, Any]] = {}
        self.sources: List[ConfigSource] = []
        self.validator = ConfigValidator()
        self.loader = ConfigLoader()
        self._lock = threading.Lock()
        self._watchers: Dict[str, List[Callable]] = {}
    
    def add_source(self, source: ConfigSource) -> None:
        """Add configuration source"""
        with self._lock:
            self.sources.append(source)
            self.sources.sort(key=lambda x: x.priority, reverse=True)
            logger.info(f"Added config source: {source.name}")
    
    def load_all(self) -> Dict[str, bool]:
        """Load all configuration sources"""
        results = {}
        
        for source in self.sources:
            try:
                config = self.loader.load(source)
                if config:
                    self.configs[source.name] = config
                    results[source.name] = True
                    logger.info(f"Loaded config: {source.name}")
                else:
                    results[source.name] = False
            except Exception as e:
                logger.error(f"Failed to load {source.name}: {e}")
                results[source.name] = False
        
        return results
    
    def get_config(self, name: str, key: Optional[str] = None) -> Any:
        """Get configuration value"""
        with self._lock:
            if name not in self.configs:
                return None
            
            config = self.configs[name]
            if key is None:
                return config
            
            return config.get(key)
    
    def set_config(self, name: str, key: str, value: Any) -> None:
        """Set configuration value"""
        with self._lock:
            if name not in self.configs:
                self.configs[name] = {}
            
            self.configs[name][key] = value
            self._notify_watchers(name, key, value)
    
    def update_config(self, name: str, updates: Dict[str, Any]) -> None:
        """Update configuration with multiple values"""
        with self._lock:
            if name not in self.configs:
                self.configs[name] = {}
            
            self.configs[name].update(updates)
            for key, value in updates.items():
                self._notify_watchers(name, key, value)
    
    def validate_config(self, name: str) -> List[str]:
        """Validate configuration"""
        if name not in self.configs:
            return [f"Configuration {name} not found"]
        
        return self.validator.validate(name, self.configs[name])
    
    def register_schema(self, schema: ConfigSchema) -> None:
        """Register configuration schema"""
        self.validator.register_schema(schema)
    
    def add_watcher(self, name: str, callback: Callable[[str, Any], None]) -> None:
        """Add configuration watcher"""
        with self._lock:
            if name not in self._watchers:
                self._watchers[name] = []
            self._watchers[name].append(callback)
    
    def _notify_watchers(self, name: str, key: str, value: Any) -> None:
        """Notify configuration watchers"""
        if name in self._watchers:
            for callback in self._watchers[name]:
                try:
                    callback(key, value)
                except Exception as e:
                    logger.error(f"Watcher callback error: {e}")
    
    def save_config(self, name: str, path: str, format: ConfigFormat = ConfigFormat.JSON) -> bool:
        """Save configuration to file"""
        if name not in self.configs:
            return False
        
        try:
            if format == ConfigFormat.JSON:
                with open(path, 'w') as f:
                    json.dump(self.configs[name], f, indent=2)
            elif format == ConfigFormat.YAML:
                with open(path, 'w') as f:
                    yaml.dump(self.configs[name], f, default_flow_style=False)
            else:
                logger.error(f"Unsupported save format: {format}")
                return False
            
            logger.info(f"Saved config {name} to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save config {name}: {e}")
            return False
    
    def reload_config(self, name: str) -> bool:
        """Reload configuration from source"""
        source = next((s for s in self.sources if s.name == name), None)
        if not source:
            return False
        
        try:
            config = self.loader.load(source)
            if config:
                self.configs[name] = config
                logger.info(f"Reloaded config: {name}")
                return True
        except Exception as e:
            logger.error(f"Failed to reload {name}: {e}")
        
        return False
    
    def list_configs(self) -> List[str]:
        """List available configurations"""
        with self._lock:
            return list(self.configs.keys())
    
    def get_config_info(self, name: str) -> Dict[str, Any]:
        """Get configuration information"""
        if name not in self.configs:
            return {}
        
        config = self.configs[name]
        return {
            "name": name,
            "keys": list(config.keys()),
            "size": len(str(config)),
            "last_modified": time.time()
        }

class ConfigBuilder:
    """Builder for configuration setup"""
    
    def __init__(self, manager: ConfigManager):
        self.manager = manager
        self.sources: List[ConfigSource] = []
        self.schemas: List[ConfigSchema] = []
    
    def add_json_source(self, name: str, path: str, priority: int = 0) -> 'ConfigBuilder':
        """Add JSON configuration source"""
        source = ConfigSource(name, path, ConfigFormat.JSON, priority)
        self.sources.append(source)
        return self
    
    def add_yaml_source(self, name: str, path: str, priority: int = 0) -> 'ConfigBuilder':
        """Add YAML configuration source"""
        source = ConfigSource(name, path, ConfigFormat.YAML, priority)
        self.sources.append(source)
        return self
    
    def add_env_source(self, name: str, path: str, priority: int = 0) -> 'ConfigBuilder':
        """Add environment configuration source"""
        source = ConfigSource(name, path, ConfigFormat.ENV, priority)
        self.sources.append(source)
        return self
    
    def add_schema(self, schema: ConfigSchema) -> 'ConfigBuilder':
        """Add configuration schema"""
        self.schemas.append(schema)
        return self
    
    def build(self) -> bool:
        """Build configuration system"""
        # Add sources to manager
        for source in self.sources:
            self.manager.add_source(source)
        
        # Register schemas
        for schema in self.schemas:
            self.manager.register_schema(schema)
        
        # Load all configurations
        results = self.manager.load_all()
        return all(results.values())

# Default configuration schemas
def create_default_schemas() -> List[ConfigSchema]:
    """Create default configuration schemas"""
    schemas = []
    
    # Optimization schema
    optimization_schema = ConfigSchema(
        name="optimization",
        version="1.0.0",
        description="Optimization configuration",
        properties={
            "level": {"type": str, "enum": ["basic", "enhanced", "advanced", "ultra", "supreme", "transcendent"]},
            "enable_adaptive_precision": {"type": bool},
            "enable_memory_optimization": {"type": bool},
            "enable_kernel_fusion": {"type": bool},
            "enable_quantization": {"type": bool},
            "enable_sparsity": {"type": bool},
            "enable_meta_learning": {"type": bool},
            "enable_neural_architecture_search": {"type": bool},
            "quantum_simulation": {"type": bool},
            "consciousness_simulation": {"type": bool},
            "temporal_optimization": {"type": bool}
        },
        required=["level"],
        defaults={
            "level": "enhanced",
            "enable_adaptive_precision": True,
            "enable_memory_optimization": True,
            "enable_kernel_fusion": True,
            "enable_quantization": False,
            "enable_sparsity": False,
            "enable_meta_learning": False,
            "enable_neural_architecture_search": False,
            "quantum_simulation": False,
            "consciousness_simulation": False,
            "temporal_optimization": False
        }
    )
    schemas.append(optimization_schema)
    
    # Model schema
    model_schema = ConfigSchema(
        name="model",
        version="1.0.0",
        description="Model configuration",
        properties={
            "model_type": {"type": str, "enum": ["transformer", "cnn", "rnn", "hybrid"]},
            "hidden_size": {"type": int, "min": 1, "max": 10000},
            "num_layers": {"type": int, "min": 1, "max": 100},
            "num_heads": {"type": int, "min": 1, "max": 100},
            "vocab_size": {"type": int, "min": 1, "max": 1000000},
            "device": {"type": str, "enum": ["auto", "cpu", "cuda", "mps"]},
            "precision": {"type": str, "enum": ["float32", "float16", "bfloat16"]}
        },
        required=["model_type"],
        defaults={
            "model_type": "transformer",
            "hidden_size": 768,
            "num_layers": 12,
            "num_heads": 12,
            "vocab_size": 50000,
            "device": "auto",
            "precision": "float32"
        }
    )
    schemas.append(model_schema)
    
    # Training schema
    training_schema = ConfigSchema(
        name="training",
        version="1.0.0",
        description="Training configuration",
        properties={
            "epochs": {"type": int, "min": 1, "max": 1000},
            "batch_size": {"type": int, "min": 1, "max": 1000},
            "learning_rate": {"type": float, "min": 1e-6, "max": 1.0},
            "optimizer": {"type": str, "enum": ["adam", "adamw", "sgd"]},
            "scheduler": {"type": str, "enum": ["cosine", "step", "plateau"]},
            "mixed_precision": {"type": bool},
            "gradient_clip": {"type": float, "min": 0.0, "max": 10.0},
            "early_stopping_patience": {"type": int, "min": 1, "max": 100}
        },
        required=["epochs", "batch_size", "learning_rate"],
        defaults={
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": 1e-3,
            "optimizer": "adam",
            "scheduler": "cosine",
            "mixed_precision": False,
            "gradient_clip": 1.0,
            "early_stopping_patience": 5
        }
    )
    schemas.append(training_schema)
    
    return schemas

