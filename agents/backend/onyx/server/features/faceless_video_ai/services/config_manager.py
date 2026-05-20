"""
Dynamic Configuration Manager
Manages dynamic configuration updates
"""

from typing import Dict, Any, Optional
from datetime import datetime
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages dynamic configuration"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = Path(config_file) if config_file else Path("/tmp/faceless_video/config.json")
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        self.config: Dict[str, Any] = {}
        self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
                logger.info("Configuration loaded")
            except Exception as e:
                logger.warning(f"Failed to load config: {str(e)}")
                self.config = self._get_default_config()
        else:
            self.config = self._get_default_config()
            self.save_config()
    
    def save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info("Configuration saved")
        except Exception as e:
            logger.error(f"Failed to save config: {str(e)}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "version": "1.0.0",
            "created_at": datetime.utcnow().isoformat(),
            "settings": {
                "max_concurrent_generations": 5,
                "default_resolution": "1920x1080",
                "default_fps": 30,
                "cache_enabled": True,
                "cloud_storage_enabled": True,
            },
            "limits": {
                "max_video_duration": 3600,
                "max_batch_size": 50,
                "max_file_size_mb": 500,
            },
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value if value is not None else default
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        self.save_config()
        logger.info(f"Configuration updated: {key} = {value}")
    
    def update(self, updates: Dict[str, Any]):
        """Update multiple configuration values"""
        for key, value in updates.items():
            self.set(key, value)
    
    def reset(self):
        """Reset to default configuration"""
        self.config = self._get_default_config()
        self.save_config()
        logger.info("Configuration reset to defaults")
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration"""
        return self.config.copy()


_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_file: Optional[str] = None) -> ConfigManager:
    """Get config manager instance (singleton)"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_file=config_file)
    return _config_manager

