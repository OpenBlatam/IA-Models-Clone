"""
Configuration-related exceptions.
"""

from .base import QualityControlException


class ConfigurationException(QualityControlException):
    """Exception raised for configuration-related errors."""
    pass


class InvalidConfigurationException(ConfigurationException):
    """Exception raised when configuration is invalid."""
    
    def __init__(self, config_key: str, reason: str):
        super().__init__(
            message=f"Invalid configuration for '{config_key}': {reason}",
            error_code="INVALID_CONFIGURATION",
            details={"config_key": config_key, "reason": reason}
        )


class MissingConfigurationException(ConfigurationException):
    """Exception raised when required configuration is missing."""
    
    def __init__(self, config_key: str):
        super().__init__(
            message=f"Missing required configuration: {config_key}",
            error_code="MISSING_CONFIGURATION",
            details={"config_key": config_key}
        )



