"""
Environment Module - Environment configuration and management.

Provides:
- Environment detection
- Environment-specific configs
- Feature toggles by environment
- Environment validation
"""

import logging
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class Environment(str, Enum):
    """Environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"
    LOCAL = "local"


@dataclass
class EnvironmentConfig:
    """Environment configuration."""
    name: Environment
    debug: bool = False
    log_level: str = "INFO"
    enable_profiling: bool = False
    enable_metrics: bool = True
    cache_enabled: bool = True
    rate_limiting_enabled: bool = True
    features: Dict[str, bool] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name.value,
            "debug": self.debug,
            "log_level": self.log_level,
            "enable_profiling": self.enable_profiling,
            "enable_metrics": self.enable_metrics,
            "cache_enabled": self.cache_enabled,
            "rate_limiting_enabled": self.rate_limiting_enabled,
            "features": self.features,
            "settings": self.settings,
        }


class EnvironmentManager:
    """Environment manager."""
    
    def __init__(self):
        """Initialize environment manager."""
        self.current_env = self._detect_environment()
        self.configs: Dict[Environment, EnvironmentConfig] = {}
        self._initialize_default_configs()
    
    def _detect_environment(self) -> Environment:
        """
        Detect current environment.
        
        Returns:
            Current environment
        """
        env_str = os.getenv("ENVIRONMENT", os.getenv("ENV", "development")).lower()
        
        env_map = {
            "dev": Environment.DEVELOPMENT,
            "development": Environment.DEVELOPMENT,
            "staging": Environment.STAGING,
            "prod": Environment.PRODUCTION,
            "production": Environment.PRODUCTION,
            "test": Environment.TESTING,
            "testing": Environment.TESTING,
            "local": Environment.LOCAL,
        }
        
        return env_map.get(env_str, Environment.DEVELOPMENT)
    
    def _initialize_default_configs(self) -> None:
        """Initialize default environment configurations."""
        # Development
        self.configs[Environment.DEVELOPMENT] = EnvironmentConfig(
            name=Environment.DEVELOPMENT,
            debug=True,
            log_level="DEBUG",
            enable_profiling=True,
            enable_metrics=True,
            cache_enabled=True,
            rate_limiting_enabled=False,
        )
        
        # Staging
        self.configs[Environment.STAGING] = EnvironmentConfig(
            name=Environment.STAGING,
            debug=False,
            log_level="INFO",
            enable_profiling=False,
            enable_metrics=True,
            cache_enabled=True,
            rate_limiting_enabled=True,
        )
        
        # Production
        self.configs[Environment.PRODUCTION] = EnvironmentConfig(
            name=Environment.PRODUCTION,
            debug=False,
            log_level="WARNING",
            enable_profiling=False,
            enable_metrics=True,
            cache_enabled=True,
            rate_limiting_enabled=True,
        )
        
        # Testing
        self.configs[Environment.TESTING] = EnvironmentConfig(
            name=Environment.TESTING,
            debug=True,
            log_level="DEBUG",
            enable_profiling=False,
            enable_metrics=False,
            cache_enabled=False,
            rate_limiting_enabled=False,
        )
        
        # Local
        self.configs[Environment.LOCAL] = EnvironmentConfig(
            name=Environment.LOCAL,
            debug=True,
            log_level="INFO",
            enable_profiling=True,
            enable_metrics=True,
            cache_enabled=True,
            rate_limiting_enabled=False,
        )
    
    def get_config(self, env: Optional[Environment] = None) -> EnvironmentConfig:
        """
        Get environment configuration.
        
        Args:
            env: Environment (defaults to current)
            
        Returns:
            Environment configuration
        """
        env = env or self.current_env
        return self.configs.get(env, self.configs[Environment.DEVELOPMENT])
    
    def set_environment(self, env: Environment) -> None:
        """
        Set current environment.
        
        Args:
            env: Environment to set
        """
        self.current_env = env
        logger.info(f"Environment set to: {env.value}")
    
    def is_production(self) -> bool:
        """Check if current environment is production."""
        return self.current_env == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        """Check if current environment is development."""
        return self.current_env == Environment.DEVELOPMENT
    
    def is_testing(self) -> bool:
        """Check if current environment is testing."""
        return self.current_env == Environment.TESTING
    
    def get_feature(self, feature_name: str, default: bool = False) -> bool:
        """
        Get feature flag for current environment.
        
        Args:
            feature_name: Feature name
            default: Default value
            
        Returns:
            Feature enabled status
        """
        config = self.get_config()
        return config.features.get(feature_name, default)
    
    def set_feature(self, feature_name: str, enabled: bool, env: Optional[Environment] = None) -> None:
        """
        Set feature flag for environment.
        
        Args:
            feature_name: Feature name
            enabled: Enabled status
            env: Environment (defaults to current)
        """
        config = self.get_config(env)
        config.features[feature_name] = enabled
        logger.info(f"Feature '{feature_name}' set to {enabled} for {config.name.value}")


# Global environment manager
env_manager = EnvironmentManager()












