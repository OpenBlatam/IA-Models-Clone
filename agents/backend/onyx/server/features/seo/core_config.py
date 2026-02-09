#!/usr/bin/env python3
"""
Core Configuration Management System
Centralized configuration with dependency injection and validation
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, Type, TypeVar
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
import logging
from contextlib import contextmanager
import threading
from functools import wraps
import time
from collections import defaultdict

# Type variables for generic operations
T = TypeVar('T')
ConfigT = TypeVar('ConfigT', bound='BaseConfig')

class ConfigurationError(Exception):
    """Raised when configuration validation fails."""
    pass

class ConfigurationManager:
    """Centralized configuration management with hot-reload support."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.yaml"
        self._config: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._watchers: Dict[str, list] = defaultdict(list)
        self._last_modified = 0
        
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from file with automatic format detection."""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                self._create_default_config()
                return
            
            with open(config_file, 'r', encoding='utf-8') as f:
                if config_file.suffix.lower() in ['.yaml', '.yml']:
                    self._config = yaml.safe_load(f)
                elif config_file.suffix.lower() == '.json':
                    self._config = json.load(f)
                else:
                    raise ConfigurationError(f"Unsupported config format: {config_file.suffix}")
            
            self._last_modified = config_file.stat().st_mtime
            self._notify_watchers()
            
        except Exception as e:
            logging.error(f"Failed to load configuration: {e}")
            self._create_default_config()
    
    def _create_default_config(self) -> None:
        """Create default configuration if none exists."""
        self._config = {
            'system': {
                'debug': False,
                'log_level': 'INFO',
                'max_workers': os.cpu_count() or 4
            },
            'models': {
                'default_model': 'microsoft/DialoGPT-medium',
                'cache_enabled': True,
                'cache_ttl': 3600
            },
            'performance': {
                'batch_size': 4,
                'max_memory_usage': 0.8,
                'enable_mixed_precision': True,
                'enable_compilation': True
            },
            'monitoring': {
                'metrics_enabled': True,
                'profiling_enabled': False,
                'alerting_enabled': True
            }
        }
        
        # Save default config
        self.save_config()
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        try:
            config_file = Path(self.config_path)
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, default_flow_style=False, indent=2)
                
        except Exception as e:
            logging.error(f"Failed to save configuration: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        with self._lock:
            keys = key.split('.')
            value = self._config
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation."""
        with self._lock:
            keys = key.split('.')
            config = self._config
            
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            config[keys[-1]] = value
            self._notify_watchers()
    
    def watch(self, key: str, callback: callable) -> None:
        """Watch for configuration changes."""
        self._watchers[key].append(callback)
    
    def _notify_watchers(self) -> None:
        """Notify all watchers of configuration changes."""
        for key, callbacks in self._watchers.items():
            for callback in callbacks:
                try:
                    callback(self.get(key))
                except Exception as e:
                    logging.error(f"Watcher callback failed: {e}")
    
    def reload_if_changed(self) -> bool:
        """Reload configuration if file has changed."""
        config_file = Path(self.config_path)
        if config_file.exists():
            current_mtime = config_file.stat().st_mtime
            if current_mtime > self._last_modified:
                self.load_config()
                return True
        return False

class DependencyContainer:
    """Dependency injection container for clean architecture."""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._singletons: Dict[str, Any] = {}
        self._factories: Dict[str, callable] = {}
        self._lock = threading.RLock()
    
    def register(self, name: str, service: Any) -> None:
        """Register a service instance."""
        with self._lock:
            self._services[name] = service
    
    def register_singleton(self, name: str, service_class: Type[T], *args, **kwargs) -> None:
        """Register a singleton service."""
        with self._lock:
            self._factories[name] = lambda: service_class(*args, **kwargs)
    
    def register_factory(self, name: str, factory: callable) -> None:
        """Register a factory function."""
        with self._lock:
            self._factories[name] = factory
    
    def get(self, name: str) -> Any:
        """Get a service instance."""
        with self._lock:
            # Return existing singleton
            if name in self._singletons:
                return self._singletons[name]
            
            # Return registered service
            if name in self._services:
                return self._services[name]
            
            # Create singleton from factory
            if name in self._factories:
                instance = self._factories[name]()
                self._singletons[name] = instance
                return instance
            
            raise KeyError(f"Service '{name}' not found")
    
    def has(self, name: str) -> bool:
        """Check if a service is registered."""
        with self._lock:
            return name in self._services or name in self._factories
    
    def clear(self) -> None:
        """Clear all services and singletons."""
        with self._lock:
            self._services.clear()
            self._singletons.clear()
            self._factories.clear()

@dataclass
class BaseConfig:
    """Base configuration class with validation."""
    
    def validate(self) -> None:
        """Validate configuration values."""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls: Type[ConfigT], data: Dict[str, Any]) -> ConfigT:
        """Create instance from dictionary."""
        return cls(**data)
    
    def merge(self, other: 'BaseConfig') -> None:
        """Merge with another configuration."""
        for key, value in other.to_dict().items():
            if hasattr(self, key):
                setattr(self, key, value)

@dataclass
class SystemConfig(BaseConfig):
    """System-level configuration."""
    debug: bool = False
    log_level: str = "INFO"
    max_workers: int = 4
    temp_dir: str = "/tmp"
    data_dir: str = "./data"
    
    def validate(self) -> None:
        if self.max_workers < 1:
            raise ConfigurationError("max_workers must be at least 1")
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ConfigurationError("Invalid log_level")

@dataclass
class ModelConfig(BaseConfig):
    """Model configuration."""
    default_model: str = "microsoft/DialoGPT-medium"
    cache_enabled: bool = True
    cache_ttl: int = 3600
    model_cache_dir: str = "./models"
    max_model_size: int = 1024 * 1024 * 1024  # 1GB
    
    def validate(self) -> None:
        if self.cache_ttl < 0:
            raise ConfigurationError("cache_ttl must be non-negative")
        if self.max_model_size < 0:
            raise ConfigurationError("max_model_size must be non-negative")

@dataclass
class PerformanceConfig(BaseConfig):
    """Performance optimization configuration."""
    batch_size: int = 4
    max_memory_usage: float = 0.8
    enable_mixed_precision: bool = True
    enable_compilation: bool = True
    enable_gradient_checkpointing: bool = False
    num_accumulation_steps: int = 1
    
    def validate(self) -> None:
        if not 0 < self.max_memory_usage <= 1:
            raise ConfigurationError("max_memory_usage must be between 0 and 1")
        if self.batch_size < 1:
            raise ConfigurationError("batch_size must be at least 1")

@dataclass
class MonitoringConfig(BaseConfig):
    """Monitoring and observability configuration."""
    metrics_enabled: bool = True
    profiling_enabled: bool = False
    alerting_enabled: bool = True
    log_retention_days: int = 30
    metrics_export_interval: int = 60
    
    def validate(self) -> None:
        if self.log_retention_days < 1:
            raise ConfigurationError("log_retention_days must be at least 1")
        if self.metrics_export_interval < 1:
            raise ConfigurationError("metrics_export_interval must be at least 1")

@dataclass
class SEOConfig(BaseConfig):
    """Complete SEO system configuration."""
    system: SystemConfig = field(default_factory=SystemConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    def validate(self) -> None:
        """Validate all sub-configurations."""
        self.system.validate()
        self.models.validate()
        self.performance.validate()
        self.monitoring.validate()
    
    @classmethod
    def load_from_file(cls, config_path: str) -> 'SEOConfig':
        """Load configuration from file."""
        config_manager = ConfigurationManager(config_path)
        
        config = cls(
            system=SystemConfig.from_dict(config_manager.get('system', {})),
            models=ModelConfig.from_dict(config_manager.get('models', {})),
            performance=PerformanceConfig.from_dict(config_manager.get('performance', {})),
            monitoring=MonitoringConfig.from_dict(config_manager.get('monitoring', {}))
        )
        
        config.validate()
        return config
    
    def save_to_file(self, config_path: str) -> None:
        """Save configuration to file."""
        config_manager = ConfigurationManager(config_path)
        
        config_manager.set('system', self.system.to_dict())
        config_manager.set('models', self.models.to_dict())
        config_manager.set('performance', self.performance.to_dict())
        config_manager.set('monitoring', self.monitoring.to_dict())
        
        config_manager.save_config()

# Global configuration instance
config_manager = ConfigurationManager()
dependency_container = DependencyContainer()

def get_config() -> ConfigurationManager:
    """Get global configuration manager."""
    return config_manager

def get_container() -> DependencyContainer:
    """Get global dependency container."""
    return dependency_container

def config_section(section: str):
    """Decorator to inject configuration section."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            config = get_config().get(section, {})
            return func(*args, config=config, **kwargs)
        return wrapper
    return decorator

def inject_service(service_name: str):
    """Decorator to inject service dependency."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            service = get_container().get(service_name)
            return func(*args, service=service, **kwargs)
        return wrapper
    return decorator

@contextmanager
def config_context(config: SEOConfig):
    """Context manager for temporary configuration changes."""
    original_config = config_manager._config.copy()
    try:
        # Apply temporary config
        config_manager._config.update(config.to_dict())
        yield
    finally:
        # Restore original config
        config_manager._config = original_config


