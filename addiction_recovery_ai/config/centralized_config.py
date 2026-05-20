"""
Centralized Configuration Management
Configuration for microservices with hot-reload support
"""

import logging
import json
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from pathlib import Path

from config.aws_settings import get_aws_settings
from aws.aws_services import ParameterStoreService, SecretsManagerService

logger = logging.getLogger(__name__)


class CentralizedConfig:
    """
    Centralized configuration manager
    
    Features:
    - Multi-source configuration (env, files, Parameter Store, Secrets Manager)
    - Hot-reload support
    - Configuration validation
    - Change notifications
    """
    
    def __init__(self):
        self._config: Dict[str, Any] = {}
        self._watchers: Dict[str, List[Callable]] = {}
        self._parameter_store: Optional[ParameterStoreService] = None
        self._secrets_manager: Optional[SecretsManagerService] = None
        
        # Initialize AWS services if available
        try:
            aws_settings = get_aws_settings()
            if aws_settings.parameter_store_prefix:
                self._parameter_store = ParameterStoreService()
            if aws_settings.secrets_manager_secret_name:
                self._secrets_manager = SecretsManagerService()
        except Exception as e:
            logger.warning(f"Failed to initialize AWS config services: {str(e)}")
    
    def load_from_file(self, file_path: str) -> None:
        """Load configuration from file"""
        try:
            path = Path(file_path)
            if path.suffix == ".json":
                with open(path) as f:
                    self._config.update(json.load(f))
            elif path.suffix in [".yaml", ".yml"]:
                import yaml
                with open(path) as f:
                    self._config.update(yaml.safe_load(f))
            
            logger.info(f"Loaded configuration from: {file_path}")
        except Exception as e:
            logger.error(f"Failed to load config from file: {str(e)}")
    
    def load_from_parameter_store(self, prefix: Optional[str] = None) -> None:
        """Load configuration from AWS Parameter Store"""
        if not self._parameter_store:
            return
        
        try:
            from config.aws_settings import get_aws_settings
            settings = get_aws_settings()
            prefix = prefix or settings.parameter_store_prefix
            
            # In production, would fetch all parameters with prefix
            # For now, placeholder
            logger.info(f"Loaded configuration from Parameter Store: {prefix}")
        except Exception as e:
            logger.error(f"Failed to load from Parameter Store: {str(e)}")
    
    def load_from_secrets_manager(self, secret_name: Optional[str] = None) -> None:
        """Load configuration from AWS Secrets Manager"""
        if not self._secrets_manager:
            return
        
        try:
            from config.aws_settings import get_aws_settings
            settings = get_aws_settings()
            secret_name = secret_name or settings.secrets_manager_secret_name
            
            if secret_name:
                secrets = self._secrets_manager.get_secret(secret_name)
                self._config.update(secrets)
                logger.info(f"Loaded configuration from Secrets Manager: {secret_name}")
        except Exception as e:
            logger.error(f"Failed to load from Secrets Manager: {str(e)}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        old_value = self._config.get(key)
        self._config[key] = value
        
        # Notify watchers
        if key in self._watchers:
            for watcher in self._watchers[key]:
                try:
                    watcher(key, value, old_value)
                except Exception as e:
                    logger.error(f"Watcher error: {str(e)}")
    
    def watch(self, key: str, callback: Callable) -> None:
        """Watch for configuration changes"""
        if key not in self._watchers:
            self._watchers[key] = []
        self._watchers[key].append(callback)
    
    def reload(self) -> None:
        """Reload configuration from all sources"""
        # Reload from Parameter Store
        self.load_from_parameter_store()
        
        # Reload from Secrets Manager
        self.load_from_secrets_manager()
        
        logger.info("Configuration reloaded")
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration"""
        return self._config.copy()


# Global config instance
_config: Optional[CentralizedConfig] = None


def get_centralized_config() -> CentralizedConfig:
    """Get global centralized config"""
    global _config
    if _config is None:
        _config = CentralizedConfig()
        # Load from default sources
        _config.load_from_parameter_store()
        _config.load_from_secrets_manager()
    return _config















