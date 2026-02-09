"""
Configuration Validator for Piel Mejorador AI SAM3
==================================================

Advanced configuration validation and management.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, error: str):
        """Add validation error."""
        self.errors.append(error)
        self.valid = False
    
    def add_warning(self, warning: str):
        """Add validation warning."""
        self.warnings.append(warning)


class ConfigValidator:
    """
    Advanced configuration validator.
    
    Features:
    - Comprehensive validation
    - Environment variable loading
    - Default value management
    - Type checking
    - Range validation
    """
    
    @staticmethod
    def validate_config(config: Any) -> ValidationResult:
        """
        Validate configuration.
        
        Args:
            config: Configuration object
            
        Returns:
            ValidationResult
        """
        result = ValidationResult(valid=True)
        
        # Validate OpenRouter
        if not config.openrouter.api_key:
            result.add_error("OpenRouter API key is required")
        elif len(config.openrouter.api_key) < 10:
            result.add_warning("OpenRouter API key seems too short")
        
        if config.openrouter.timeout <= 0:
            result.add_error("OpenRouter timeout must be positive")
        elif config.openrouter.timeout < 10:
            result.add_warning("OpenRouter timeout is very low, may cause timeouts")
        
        # Validate TruthGPT (optional)
        if config.truthgpt.enabled:
            if not config.truthgpt.endpoint:
                result.add_warning("TruthGPT enabled but endpoint not configured")
        
        # Validate limits
        if config.max_parallel_tasks <= 0:
            result.add_error("max_parallel_tasks must be positive")
        elif config.max_parallel_tasks > 50:
            result.add_warning("max_parallel_tasks is very high, may cause resource issues")
        
        # Validate file size limits
        if config.max_image_size_mb <= 0:
            result.add_error("max_image_size_mb must be positive")
        if config.max_video_size_mb <= 0:
            result.add_error("max_video_size_mb must be positive")
        
        # Validate enhancement levels
        if not config.enhancement_levels:
            result.add_error("enhancement_levels cannot be empty")
        else:
            for level, config_data in config.enhancement_levels.items():
                intensity = config_data.get("intensity", 0)
                realism = config_data.get("realism", 0)
                
                if not 0 <= intensity <= 1:
                    result.add_error(f"Enhancement level '{level}' intensity must be 0-1")
                if not 0 <= realism <= 1:
                    result.add_error(f"Enhancement level '{level}' realism must be 0-1")
        
        # Validate output directory
        try:
            output_path = Path(config.output_dir)
            if output_path.exists() and not output_path.is_dir():
                result.add_error(f"Output directory path exists but is not a directory: {config.output_dir}")
        except Exception as e:
            result.add_warning(f"Cannot validate output directory: {e}")
        
        return result
    
    @staticmethod
    def load_from_env(config: Any) -> Dict[str, Any]:
        """
        Load configuration from environment variables.
        
        Args:
            config: Configuration object
            
        Returns:
            Dictionary of loaded values
        """
        loaded = {}
        
        # OpenRouter
        if os.getenv("OPENROUTER_API_KEY"):
            config.openrouter.api_key = os.getenv("OPENROUTER_API_KEY")
            loaded["openrouter.api_key"] = "***"
        
        if os.getenv("OPENROUTER_MODEL"):
            config.openrouter.model = os.getenv("OPENROUTER_MODEL")
            loaded["openrouter.model"] = config.openrouter.model
        
        if os.getenv("OPENROUTER_TIMEOUT"):
            try:
                config.openrouter.timeout = float(os.getenv("OPENROUTER_TIMEOUT"))
                loaded["openrouter.timeout"] = config.openrouter.timeout
            except ValueError:
                logger.warning(f"Invalid OPENROUTER_TIMEOUT: {os.getenv('OPENROUTER_TIMEOUT')}")
        
        # TruthGPT
        if os.getenv("TRUTHGPT_ENDPOINT"):
            config.truthgpt.endpoint = os.getenv("TRUTHGPT_ENDPOINT")
            loaded["truthgpt.endpoint"] = config.truthgpt.endpoint
        
        if os.getenv("TRUTHGPT_ENABLED"):
            config.truthgpt.enabled = os.getenv("TRUTHGPT_ENABLED").lower() == "true"
            loaded["truthgpt.enabled"] = config.truthgpt.enabled
        
        # General
        if os.getenv("PIEL_MEJORADOR_MAX_PARALLEL_TASKS"):
            try:
                config.max_parallel_tasks = int(os.getenv("PIEL_MEJORADOR_MAX_PARALLEL_TASKS"))
                loaded["max_parallel_tasks"] = config.max_parallel_tasks
            except ValueError:
                logger.warning(f"Invalid PIEL_MEJORADOR_MAX_PARALLEL_TASKS: {os.getenv('PIEL_MEJORADOR_MAX_PARALLEL_TASKS')}")
        
        if os.getenv("PIEL_MEJORADOR_OUTPUT_DIR"):
            config.output_dir = os.getenv("PIEL_MEJORADOR_OUTPUT_DIR")
            loaded["output_dir"] = config.output_dir
        
        if os.getenv("PIEL_MEJORADOR_DEBUG"):
            config.debug = os.getenv("PIEL_MEJORADOR_DEBUG").lower() == "true"
            loaded["debug"] = config.debug
        
        return loaded
    
    @staticmethod
    def get_recommended_config() -> Dict[str, Any]:
        """
        Get recommended configuration values.
        
        Returns:
            Dictionary with recommended values
        """
        return {
            "openrouter": {
                "timeout": 120.0,
                "max_retries": 3,
            },
            "max_parallel_tasks": 5,
            "max_image_size_mb": 50,
            "max_video_size_mb": 500,
            "enhancement_levels": {
                "low": {"intensity": 0.3, "realism": 0.5},
                "medium": {"intensity": 0.6, "realism": 0.7},
                "high": {"intensity": 0.9, "realism": 0.9},
                "ultra": {"intensity": 1.0, "realism": 1.0},
            }
        }




