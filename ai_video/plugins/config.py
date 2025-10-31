from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from .validator import ValidationLevel
    import sys
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Plugin System Configuration

This module provides comprehensive configuration management for the plugin system,
including environment variables, configuration files, and validation.
"""




class ConfigSource(Enum):
    """Configuration source types."""
    ENVIRONMENT = "environment"
    FILE = "file"
    DEFAULT = "default"


@dataclass
class PluginSystemConfig:
    """Complete configuration for the plugin system."""
    
    # Plugin discovery and loading
    auto_discover: bool = True
    auto_load: bool = False
    auto_initialize: bool = False
    
    # Validation and security
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    enable_security_checks: bool = True
    enable_performance_checks: bool = True
    
    # Plugin directories
    plugin_dirs: List[str] = field(default_factory=lambda: [
        "./plugins",
        "./ai_video/plugins",
        "./extensions",
        "~/.ai_video/plugins"
    ])
    
    # HTTP and networking
    http_timeout: int = 30
    max_retries: int = 3
    user_agent: str = "AI-Video-Plugin-System/1.0"
    
    # Performance and monitoring
    enable_metrics: bool = True
    enable_events: bool = True
    enable_logging: bool = True
    log_level: str = "INFO"
    
    # Storage and caching
    cache_enabled: bool = True
    cache_dir: str = "~/.ai_video/cache"
    cache_ttl: int = 3600  # 1 hour
    
    # Default plugin configurations
    default_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Advanced settings
    max_concurrent_plugins: int = 10
    plugin_load_timeout: int = 60
    enable_auto_recovery: bool = True
    enable_health_checks: bool = True
    health_check_interval: int = 300  # 5 minutes


class ConfigManager:
    """
    Configuration manager for the plugin system.
    
    Features:
    - Environment variable support
    - Configuration file loading (JSON, YAML)
    - Configuration validation
    - Default value management
    - Configuration merging
    """
    
    def __init__(self, config_file: Optional[str] = None):
        
    """__init__ function."""
self.config_file = config_file
        self.config = PluginSystemConfig()
        self.sources: Dict[str, ConfigSource] = {}
        
        # Load configuration from all sources
        self._load_configuration()
    
    def _load_configuration(self) -> Any:
        """Load configuration from all sources in order of precedence."""
        # 1. Load defaults
        self._load_defaults()
        
        # 2. Load from file
        if self.config_file:
            self._load_from_file(self.config_file)
        
        # 3. Load from environment variables
        self._load_from_environment()
        
        # 4. Validate configuration
        self._validate_configuration()
    
    def _load_defaults(self) -> Any:
        """Load default configuration values."""
        # Defaults are already set in the dataclass
        for field_name in self.config.__dataclass_fields__:
            self.sources[field_name] = ConfigSource.DEFAULT
    
    def _load_from_file(self, file_path: str):
        """Load configuration from file."""
        path = Path(file_path).expanduser()
        
        if not path.exists():
            logger.warning(f"Configuration file not found: {file_path}")
            return
        
        try:
            if path.suffix.lower() == '.json':
                with open(path, 'r', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    config_data = json.load(f)
            elif path.suffix.lower() in ['.yml', '.yaml']:
                with open(path, 'r', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    config_data = yaml.safe_load(f)
            else:
                logger.warning(f"Unsupported configuration file format: {path.suffix}")
                return
            
            # Update configuration
            for key, value in config_data.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                    self.sources[key] = ConfigSource.FILE
            
            logger.info(f"Configuration loaded from file: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration from file {file_path}: {e}")
    
    def _load_from_environment(self) -> Any:
        """Load configuration from environment variables."""
        env_mappings = {
            'AI_VIDEO_AUTO_DISCOVER': 'auto_discover',
            'AI_VIDEO_AUTO_LOAD': 'auto_load',
            'AI_VIDEO_AUTO_INITIALIZE': 'auto_initialize',
            'AI_VIDEO_VALIDATION_LEVEL': 'validation_level',
            'AI_VIDEO_ENABLE_SECURITY': 'enable_security_checks',
            'AI_VIDEO_ENABLE_PERFORMANCE': 'enable_performance_checks',
            'AI_VIDEO_PLUGIN_DIRS': 'plugin_dirs',
            'AI_VIDEO_HTTP_TIMEOUT': 'http_timeout',
            'AI_VIDEO_MAX_RETRIES': 'max_retries',
            'AI_VIDEO_USER_AGENT': 'user_agent',
            'AI_VIDEO_ENABLE_METRICS': 'enable_metrics',
            'AI_VIDEO_ENABLE_EVENTS': 'enable_events',
            'AI_VIDEO_ENABLE_LOGGING': 'enable_logging',
            'AI_VIDEO_LOG_LEVEL': 'log_level',
            'AI_VIDEO_CACHE_ENABLED': 'cache_enabled',
            'AI_VIDEO_CACHE_DIR': 'cache_dir',
            'AI_VIDEO_CACHE_TTL': 'cache_ttl',
            'AI_VIDEO_MAX_CONCURRENT': 'max_concurrent_plugins',
            'AI_VIDEO_LOAD_TIMEOUT': 'plugin_load_timeout',
            'AI_VIDEO_AUTO_RECOVERY': 'enable_auto_recovery',
            'AI_VIDEO_HEALTH_CHECKS': 'enable_health_checks',
            'AI_VIDEO_HEALTH_INTERVAL': 'health_check_interval',
        }
        
        for env_var, config_key in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert environment variable to appropriate type
                converted_value = self._convert_env_value(config_key, env_value)
                if converted_value is not None:
                    setattr(self.config, config_key, converted_value)
                    self.sources[config_key] = ConfigSource.ENVIRONMENT
    
    def _convert_env_value(self, config_key: str, env_value: str) -> Any:
        """Convert environment variable value to appropriate type."""
        try:
            if config_key in ['auto_discover', 'auto_load', 'auto_initialize', 
                             'enable_security_checks', 'enable_performance_checks',
                             'enable_metrics', 'enable_events', 'enable_logging',
                             'cache_enabled', 'enable_auto_recovery', 'enable_health_checks']:
                return env_value.lower() in ['true', '1', 'yes', 'on']
            
            elif config_key in ['http_timeout', 'max_retries', 'cache_ttl', 
                               'max_concurrent_plugins', 'plugin_load_timeout',
                               'health_check_interval']:
                return int(env_value)
            
            elif config_key == 'validation_level':
                return ValidationLevel(env_value.lower())
            
            elif config_key == 'plugin_dirs':
                return [dir.strip() for dir in env_value.split(',')]
            
            else:
                return env_value
                
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to convert environment variable {config_key}: {e}")
            return None
    
    def _validate_configuration(self) -> bool:
        """Validate the configuration."""
        errors = []
        
        # Validate timeouts
        if self.config.http_timeout < 1 or self.config.http_timeout > 300:
            errors.append("http_timeout must be between 1 and 300 seconds")
        
        if self.config.max_retries < 0 or self.config.max_retries > 10:
            errors.append("max_retries must be between 0 and 10")
        
        if self.config.plugin_load_timeout < 10 or self.config.plugin_load_timeout > 300:
            errors.append("plugin_load_timeout must be between 10 and 300 seconds")
        
        # Validate directories
        for plugin_dir in self.config.plugin_dirs:
            if not plugin_dir:
                errors.append("Plugin directory cannot be empty")
        
        # Validate log level
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.config.log_level.upper() not in valid_log_levels:
            errors.append(f"log_level must be one of: {valid_log_levels}")
        
        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors))
    
    def get_config(self) -> PluginSystemConfig:
        """Get the current configuration."""
        return self.config
    
    def get_config_source(self, key: str) -> Optional[ConfigSource]:
        """Get the source of a configuration value."""
        return self.sources.get(key)
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """Update configuration with new values."""
        try:
            for key, value in updates.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                    self.sources[key] = ConfigSource.FILE  # Mark as file source
                else:
                    logger.warning(f"Unknown configuration key: {key}")
            
            # Re-validate configuration
            self._validate_configuration()
            return True
            
        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            return False
    
    def save_config(self, file_path: Optional[str] = None) -> bool:
        """Save current configuration to file."""
        try:
            path = Path(file_path or self.config_file or "ai_video_config.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert configuration to dictionary
            config_dict = {}
            for field_name in self.config.__dataclass_fields__:
                value = getattr(self.config, field_name)
                if isinstance(value, ValidationLevel):
                    config_dict[field_name] = value.value
                else:
                    config_dict[field_name] = value
            
            # Save based on file extension
            if path.suffix.lower() == '.json':
                with open(path, 'w', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    json.dump(config_dict, f, indent=2, default=str)
            elif path.suffix.lower() in ['.yml', '.yaml']:
                with open(path, 'w', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    yaml.dump(config_dict, f, default_flow_style=False)
            else:
                # Default to JSON
                path = path.with_suffix('.json')
                with open(path, 'w', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    json.dump(config_dict, f, indent=2, default=str)
            
            logger.info(f"Configuration saved to: {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration."""
        summary = {
            'config_file': self.config_file,
            'sources': {},
            'values': {}
        }
        
        for field_name in self.config.__dataclass_fields__:
            value = getattr(self.config, field_name)
            source = self.sources.get(field_name, ConfigSource.DEFAULT)
            
            summary['sources'][field_name] = source.value
            summary['values'][field_name] = value
        
        return summary


# Convenience functions

def load_config(config_file: Optional[str] = None) -> PluginSystemConfig:
    """
    Load configuration from file and environment variables.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        PluginSystemConfig instance
    """
    manager = ConfigManager(config_file)
    return manager.get_config()


def create_default_config(file_path: str = "ai_video_config.json") -> bool:
    """
    Create a default configuration file.
    
    Args:
        file_path: Path to save the configuration file
        
    Returns:
        True if successful
    """
    manager = ConfigManager()
    return manager.save_config(file_path)


def get_config_from_env() -> PluginSystemConfig:
    """
    Load configuration from environment variables only.
    
    Returns:
        PluginSystemConfig instance
    """
    manager = ConfigManager()
    # Override to only use environment variables
    manager._load_defaults()
    manager._load_from_environment()
    manager._validate_configuration()
    return manager.get_config()


# Example configuration file template
CONFIG_TEMPLATE = {
    "auto_discover": True,
    "auto_load": False,
    "auto_initialize": False,
    "validation_level": "standard",
    "enable_security_checks": True,
    "enable_performance_checks": True,
    "plugin_dirs": [
        "./plugins",
        "./ai_video/plugins",
        "./extensions",
        "~/.ai_video/plugins"
    ],
    "http_timeout": 30,
    "max_retries": 3,
    "user_agent": "AI-Video-Plugin-System/1.0",
    "enable_metrics": True,
    "enable_events": True,
    "enable_logging": True,
    "log_level": "INFO",
    "cache_enabled": True,
    "cache_dir": "~/.ai_video/cache",
    "cache_ttl": 3600,
    "default_configs": {
        "web_extractor": {
            "timeout": 30,
            "max_retries": 3,
            "extraction_methods": ["newspaper3k", "trafilatura"]
        }
    },
    "max_concurrent_plugins": 10,
    "plugin_load_timeout": 60,
    "enable_auto_recovery": True,
    "enable_health_checks": True,
    "health_check_interval": 300
}


if __name__ == "__main__":
    # Create default configuration file
    
    if len(sys.argv) > 1 and sys.argv[1] == "--create-config":
        file_path = sys.argv[2] if len(sys.argv) > 2 else "ai_video_config.json"
        if create_default_config(file_path):
            print(f"✅ Default configuration created: {file_path}")
        else:
            print("❌ Failed to create configuration file")
    else:
        # Show current configuration
        config = load_config()
        print("Current configuration:")
        for field_name, value in config.__dict__.items():
            print(f"  {field_name}: {value}") 