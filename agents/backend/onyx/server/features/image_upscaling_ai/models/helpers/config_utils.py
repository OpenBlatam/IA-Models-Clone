"""
Configuration Utilities
=======================

Utilities for managing upscaling configurations.
"""

import logging
import json
import time
from typing import Dict, Any, List, Optional, Union, Callable
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)


class ConfigUtils:
    """Utilities for configuration management."""
    
    @staticmethod
    def export_config(
        config_name: str,
        method: str,
        scale_factor: float,
        enhancement_steps: Optional[List[Dict[str, Any]]] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Export upscaling configuration for reuse.
        
        Args:
            config_name: Name for the configuration
            method: Upscaling method
            scale_factor: Scale factor
            enhancement_steps: List of enhancement steps
            output_path: Optional path to save config
            
        Returns:
            Configuration dictionary
        """
        config = {
            "name": config_name,
            "method": method,
            "scale_factor": scale_factor,
            "enhancement_steps": enhancement_steps or [],
            "created_at": time.time(),
        }
        
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Configuration exported to {output_path}")
        
        return config
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        return config
    
    @staticmethod
    def apply_config_steps(
        image: Image.Image,
        config: Dict[str, Any],
        step_processor: Callable[[Image.Image, Dict[str, Any]], Image.Image]
    ) -> Image.Image:
        """
        Apply configuration steps to an image.
        
        Args:
            image: Input image
            config: Configuration dictionary
            step_processor: Function to process each step
            
        Returns:
            Processed image
        """
        result = image
        
        for step in config.get("enhancement_steps", []):
            result = step_processor(result, step)
        
        return result

