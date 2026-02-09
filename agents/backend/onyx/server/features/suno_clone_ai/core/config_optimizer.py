"""
Configuration Optimizations

Optimizations for:
- Dynamic configuration
- Configuration validation
- Environment-specific configs
- Feature flags
- Configuration hot-reload
"""

import logging
import os
import json
from typing import Optional, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass, asdict

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    """Application configuration."""
    debug: bool = False
    workers: int = 1
    log_level: str = "INFO"
    enable_cache: bool = True
    enable_compression: bool = True
    pool_size: int = 10
    use_gpu: bool = True
    compile_mode: str = "reduce-overhead"
    use_mixed_precision: bool = True


class ConfigManager:
    """Optimized configuration management."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize config manager.
        
        Args:
            config_file: Optional config file path
        """
        self.config_file = config_file
        self.config: Dict[str, Any] = {}
        self.watchers: List[callable] = []
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from file and environment."""
        # Load from file if exists
        if self.config_file and Path(self.config_file).exists():
            with open(self.config_file) as f:
                if (self.config_file.endswith('.yaml') or self.config_file.endswith('.yml')):
                    if YAML_AVAILABLE:
                        self.config = yaml.safe_load(f) or {}
                    else:
                        logger.warning("YAML not available, using JSON")
                        self.config = json.load(f)
                else:
                    self.config = json.load(f)
        
        # Override with environment variables
        for key, value in os.environ.items():
            if key.startswith('SUNO_'):
                config_key = key[5:].lower()
                # Convert string values to appropriate types
                if value.lower() in ('true', 'false'):
                    self.config[config_key] = value.lower() == 'true'
                elif value.isdigit():
                    self.config[config_key] = int(value)
                else:
                    try:
                        self.config[config_key] = float(value)
                    except ValueError:
                        self.config[config_key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config[key] = value
        self._notify_watchers(key, value)
    
    def reload(self) -> None:
        """Reload configuration."""
        self._load_config()
        self._notify_watchers(None, None)
    
    def watch(self, callback: callable) -> None:
        """
        Watch for configuration changes.
        
        Args:
            callback: Callback function
        """
        self.watchers.append(callback)
    
    def _notify_watchers(self, key: Optional[str], value: Any) -> None:
        """Notify watchers of configuration changes."""
        for watcher in self.watchers:
            try:
                watcher(key, value)
            except Exception as e:
                logger.error(f"Config watcher error: {e}")


class FeatureFlags:
    """Feature flags management."""
    
    def __init__(self):
        """Initialize feature flags."""
        self.flags: Dict[str, bool] = {}
        self.conditions: Dict[str, callable] = {}
    
    def enable(self, flag: str, condition: Optional[callable] = None) -> None:
        """
        Enable feature flag.
        
        Args:
            flag: Flag name
            condition: Optional condition function
        """
        self.flags[flag] = True
        if condition:
            self.conditions[flag] = condition
    
    def disable(self, flag: str) -> None:
        """
        Disable feature flag.
        
        Args:
            flag: Flag name
        """
        self.flags[flag] = False
        if flag in self.conditions:
            del self.conditions[flag]
    
    def is_enabled(self, flag: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Check if feature flag is enabled.
        
        Args:
            flag: Flag name
            context: Optional context for conditional flags
            
        Returns:
            True if enabled
        """
        if flag not in self.flags:
            return False
        
        if not self.flags[flag]:
            return False
        
        # Check condition if exists
        if flag in self.conditions:
            try:
                return self.conditions[flag](context or {})
            except Exception:
                return False
        
        return True


class EnvironmentConfig:
    """Environment-specific configuration."""
    
    @staticmethod
    def get_config(env: Optional[str] = None) -> AppConfig:
        """
        Get configuration for environment.
        
        Args:
            env: Environment name (auto-detect if None)
            
        Returns:
            Application configuration
        """
        if env is None:
            env = os.environ.get('ENVIRONMENT', 'development')
        
        if env == 'production':
            return AppConfig(
                debug=False,
                workers=4,
                log_level="INFO",
                enable_cache=True,
                enable_compression=True,
                pool_size=20,
                use_gpu=True,
                compile_mode="max-autotune",
                use_mixed_precision=True
            )
        elif env == 'staging':
            return AppConfig(
                debug=False,
                workers=2,
                log_level="DEBUG",
                enable_cache=True,
                enable_compression=True,
                pool_size=10,
                use_gpu=True,
                compile_mode="reduce-overhead",
                use_mixed_precision=True
            )
        else:  # development
            return AppConfig(
                debug=True,
                workers=1,
                log_level="DEBUG",
                enable_cache=False,
                enable_compression=False,
                pool_size=5,
                use_gpu=True,
                compile_mode="default",
                use_mixed_precision=False
            )

