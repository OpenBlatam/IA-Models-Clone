"""
Unified Configuration System for Color Grading AI
=================================================

Consolidates configuration services:
- DynamicConfig (dynamic configuration)
- ConfigValidator (configuration validation)
- ConfigManager (if exists)

Features:
- Unified configuration interface
- Hot-reloading
- Validation
- Multiple sources
- Change notifications
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum

from .dynamic_config import DynamicConfig, ConfigSource, ConfigChange
from .config_validator import ConfigValidator, ValidationResult, ValidationLevel

logger = logging.getLogger(__name__)


class ConfigMode(Enum):
    """Configuration modes."""
    STATIC = "static"  # Static configuration
    DYNAMIC = "dynamic"  # Dynamic with hot-reload
    VALIDATED = "validated"  # With validation
    FULL = "full"  # Dynamic + validated


@dataclass
class UnifiedConfigResult:
    """Unified configuration result."""
    success: bool
    config: Dict[str, Any]
    validation_passed: bool = True
    errors: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class UnifiedConfigSystem:
    """
    Unified configuration system.
    
    Consolidates:
    - DynamicConfig: Dynamic configuration
    - ConfigValidator: Configuration validation
    
    Features:
    - Unified configuration interface
    - Hot-reloading
    - Validation
    - Multiple sources
    """
    
    def __init__(
        self,
        config_file: Optional[Path] = None,
        config_mode: ConfigMode = ConfigMode.FULL,
        validation_level: ValidationLevel = ValidationLevel.STRICT
    ):
        """
        Initialize unified configuration system.
        
        Args:
            config_file: Optional config file path
            config_mode: Configuration mode
            validation_level: Validation level
        """
        self.config_mode = config_mode
        
        # Initialize components
        self.dynamic_config = DynamicConfig(config_file=config_file) if config_mode != ConfigMode.STATIC else None
        self.config_validator = ConfigValidator(validation_level=validation_level) if config_mode in [ConfigMode.VALIDATED, ConfigMode.FULL] else None
        
        self._static_config: Dict[str, Any] = {}
        
        logger.info(f"Initialized UnifiedConfigSystem (mode={config_mode.value})")
    
    def get(
        self,
        key: str,
        default: Any = None
    ) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value
            
        Returns:
            Configuration value
        """
        if self.dynamic_config:
            return self.dynamic_config.get(key, default)
        else:
            return self._static_config.get(key, default)
    
    def set(
        self,
        key: str,
        value: Any,
        source: ConfigSource = ConfigSource.MEMORY,
        validate: bool = True
    ) -> UnifiedConfigResult:
        """
        Set configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
            source: Configuration source
            validate: Whether to validate
            
        Returns:
            Unified configuration result
        """
        errors = []
        
        # Validate if enabled
        if validate and self.config_validator:
            # Try to find schema for this key
            # For now, just validate the value type
            if not isinstance(value, (str, int, float, bool, dict, list)):
                errors.append(f"Invalid type for {key}: {type(value)}")
        
        if errors:
            return UnifiedConfigResult(
                success=False,
                config={},
                validation_passed=False,
                errors=errors
            )
        
        # Set value
        if self.dynamic_config:
            self.dynamic_config.set(key, value, source=source)
        else:
            self._static_config[key] = value
        
        return UnifiedConfigResult(
            success=True,
            config={key: value},
            validation_passed=True
        )
    
    def validate_config(
        self,
        config_data: Dict[str, Any],
        schema_name: str
    ) -> ValidationResult:
        """
        Validate configuration.
        
        Args:
            config_data: Configuration data
            schema_name: Schema name
            
        Returns:
            Validation result
        """
        if not self.config_validator:
            return ValidationResult(
                status="success",
                errors=[],
                validated_config=config_data
            )
        
        return self.config_validator.validate_config(config_data, schema_name)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration."""
        if self.dynamic_config:
            return self.dynamic_config.get_all()
        else:
            return self._static_config.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get configuration statistics."""
        return {
            "config_mode": self.config_mode.value,
            "dynamic_config_enabled": self.dynamic_config is not None,
            "validation_enabled": self.config_validator is not None,
            "total_keys": len(self.get_all()),
        }


