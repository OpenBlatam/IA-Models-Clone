"""
Configuration Mixin

Contains configuration and preset management functionality.
"""

import logging
import json
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigurationMixin:
    """
    Mixin providing configuration and preset management.
    
    This mixin contains:
    - Configuration management
    - Preset creation and loading
    - Settings persistence
    - Configuration validation
    - Default configurations
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize configuration mixin."""
        super().__init__(*args, **kwargs)
        self._config = {}
        self._presets = {}
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return {
            "enable_cache": getattr(self, 'enable_cache', True),
            "cache_size": getattr(self.cache, 'max_size', 64) if hasattr(self, 'cache') and self.cache else 64,
            "cache_ttl": getattr(self.cache, 'ttl', 3600) if hasattr(self, 'cache') and self.cache else 3600,
            "max_workers": getattr(self.executor, '_max_workers', 4) if hasattr(self, 'executor') else 4,
            "validate_images": getattr(self, 'validate_images', True),
            "enhance_images": getattr(self, 'enhance_images', False),
            "auto_select_method": getattr(self, 'auto_select_method', False),
            "max_retries": getattr(self, 'max_retries', 3),
        }
    
    def update_config(self, **kwargs) -> Dict[str, Any]:
        """
        Update configuration.
        
        Args:
            **kwargs: Configuration parameters to update
            
        Returns:
            Updated configuration
        """
        config = self.get_config()
        config.update(kwargs)
        
        # Apply updates
        if 'enable_cache' in kwargs:
            self.enable_cache = kwargs['enable_cache']
        if 'validate_images' in kwargs:
            self.validate_images = kwargs['validate_images']
        if 'enhance_images' in kwargs:
            self.enhance_images = kwargs['enhance_images']
        if 'auto_select_method' in kwargs:
            self.auto_select_method = kwargs['auto_select_method']
        if 'max_retries' in kwargs:
            self.max_retries = kwargs['max_retries']
        
        logger.info(f"Configuration updated: {kwargs}")
        return self.get_config()
    
    def create_preset(
        self,
        preset_name: str,
        config: Dict[str, Any],
        description: str = ""
    ) -> bool:
        """
        Create a configuration preset.
        
        Args:
            preset_name: Name of the preset
            config: Configuration dictionary
            description: Optional description
            
        Returns:
            True if successful
        """
        if not hasattr(self, '_presets'):
            self._presets = {}
        
        self._presets[preset_name] = {
            "config": config,
            "description": description,
            "created_at": str(Path().cwd())  # Simple timestamp placeholder
        }
        
        logger.info(f"Preset '{preset_name}' created")
        return True
    
    def load_preset(self, preset_name: str) -> bool:
        """
        Load a configuration preset.
        
        Args:
            preset_name: Name of the preset to load
            
        Returns:
            True if successful
        """
        if not hasattr(self, '_presets') or preset_name not in self._presets:
            logger.warning(f"Preset '{preset_name}' not found")
            return False
        
        preset = self._presets[preset_name]
        self.update_config(**preset["config"])
        
        logger.info(f"Preset '{preset_name}' loaded")
        return True
    
    def list_presets(self) -> List[str]:
        """List available presets."""
        if not hasattr(self, '_presets'):
            return []
        return list(self._presets.keys())
    
    def get_preset_info(self, preset_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a preset."""
        if not hasattr(self, '_presets') or preset_name not in self._presets:
            return None
        return self._presets[preset_name]
    
    def delete_preset(self, preset_name: str) -> bool:
        """Delete a preset."""
        if not hasattr(self, '_presets') or preset_name not in self._presets:
            return False
        
        del self._presets[preset_name]
        logger.info(f"Preset '{preset_name}' deleted")
        return True
    
    def save_config(self, file_path: Union[str, Path]) -> bool:
        """
        Save configuration to file.
        
        Args:
            file_path: Path to save configuration
            
        Returns:
            True if successful
        """
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            config_data = {
                "config": self.get_config(),
                "presets": getattr(self, '_presets', {})
            }
            
            with open(file_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Configuration saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def load_config(self, file_path: Union[str, Path]) -> bool:
        """
        Load configuration from file.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            True if successful
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                logger.warning(f"Configuration file not found: {file_path}")
                return False
            
            with open(file_path, 'r') as f:
                config_data = json.load(f)
            
            if 'config' in config_data:
                self.update_config(**config_data['config'])
            
            if 'presets' in config_data:
                if not hasattr(self, '_presets'):
                    self._presets = {}
                self._presets.update(config_data['presets'])
            
            logger.info(f"Configuration loaded from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False
    
    def get_default_presets(self) -> Dict[str, Dict[str, Any]]:
        """Get default presets."""
        return {
            "fast": {
                "config": {
                    "enable_cache": True,
                    "cache_size": 32,
                    "max_workers": 2,
                    "validate_images": False,
                    "enhance_images": False,
                    "auto_select_method": True,
                },
                "description": "Fast processing with minimal validation"
            },
            "quality": {
                "config": {
                    "enable_cache": True,
                    "cache_size": 128,
                    "max_workers": 4,
                    "validate_images": True,
                    "enhance_images": True,
                    "auto_select_method": True,
                },
                "description": "High quality processing with full validation"
            },
            "balanced": {
                "config": {
                    "enable_cache": True,
                    "cache_size": 64,
                    "max_workers": 4,
                    "validate_images": True,
                    "enhance_images": False,
                    "auto_select_method": True,
                },
                "description": "Balanced processing"
            }
        }
    
    def initialize_default_presets(self):
        """Initialize default presets."""
        defaults = self.get_default_presets()
        for name, preset in defaults.items():
            self.create_preset(name, preset["config"], preset["description"])


