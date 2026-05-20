"""
Config Manager
==============

Centralized configuration management.
"""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import json

# Default config structure
DEFAULT_CONFIG = {
    "openrouter": {
        "api_key": "",
        "base_url": "https://openrouter.ai/api/v1"
    },
    "truthgpt": {
        "endpoint": ""
    },
    "agent": {
        "max_workers": 4,
        "task_check_interval": 1.0
    }
}

logger = logging.getLogger(__name__)


class ConfigManager:
    """Centralized configuration manager."""
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize config manager.
        
        Args:
            config_file: Optional path to config file
        """
        self.config_file = config_file
        self._config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file and environment."""
        # Start with defaults
        self._config = DEFAULT_CONFIG.copy()
        
        # Load from file if provided
        if self.config_file and self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                    self._config.update(file_config)
                logger.info(f"Loaded config from {self.config_file}")
            except Exception as e:
                logger.warning(f"Error loading config file: {e}")
        
        # Override with environment variables
        self._load_from_env()
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        # OpenRouter config
        if os.getenv("OPENROUTER_API_KEY"):
            self._config["openrouter"]["api_key"] = os.getenv("OPENROUTER_API_KEY")
        
        if os.getenv("OPENROUTER_BASE_URL"):
            self._config["openrouter"]["base_url"] = os.getenv("OPENROUTER_BASE_URL")
        
        # TruthGPT config
        if os.getenv("TRUTHGPT_ENDPOINT"):
            self._config["truthgpt"]["endpoint"] = os.getenv("TRUTHGPT_ENDPOINT")
        
        # Agent config
        if os.getenv("MAX_WORKERS"):
            self._config["agent"]["max_workers"] = int(os.getenv("MAX_WORKERS"))
        
        if os.getenv("TASK_CHECK_INTERVAL"):
            self._config["agent"]["task_check_interval"] = float(os.getenv("TASK_CHECK_INTERVAL"))
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value by key (supports dot notation).
        
        Args:
            key: Config key (e.g., "openrouter.api_key")
            default: Default value if not found
            
        Returns:
            Config value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set config value by key (supports dot notation).
        
        Args:
            key: Config key (e.g., "openrouter.api_key")
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire config section.
        
        Args:
            section: Section name
            
        Returns:
            Section dictionary
        """
        return self._config.get(section, {})
    
    def update(self, updates: Dict[str, Any]):
        """
        Update config with dictionary.
        
        Args:
            updates: Dictionary of updates
        """
        self._config.update(updates)
    
    def save(self, config_file: Optional[Path] = None):
        """
        Save config to file.
        
        Args:
            config_file: Optional path to save (uses default if not provided)
        """
        file_path = config_file or self.config_file
        if not file_path:
            logger.warning("No config file path specified")
            return
        
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved config to {file_path}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get full config as dictionary.
        
        Returns:
            Full configuration dictionary
        """
        return self._config.copy()

