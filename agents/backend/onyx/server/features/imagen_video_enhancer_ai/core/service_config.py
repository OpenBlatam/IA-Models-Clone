"""
Service Configuration
=====================

Consolidated service configuration management.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ServiceType(Enum):
    """Service types."""
    ENHANCE_IMAGE = "enhance_image"
    ENHANCE_VIDEO = "enhance_video"
    UPSCALE = "upscale"
    DENOISE = "denoise"
    RESTORE = "restore"
    COLOR_CORRECTION = "color_correction"


@dataclass
class ServiceConfig:
    """Service configuration."""
    service_type: ServiceType
    system_prompt_key: str
    response_key: str
    temperature: float = 0.3
    max_tokens: int = 4000
    include_timestamp: bool = False
    timeout: float = 300.0
    retry_attempts: int = 3
    cache_enabled: bool = True
    cache_ttl: int = 3600
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(
        cls,
        service_type: ServiceType,
        system_prompt_key: str,
        response_key: str,
        **kwargs
    ) -> "ServiceConfig":
        """
        Create service configuration.
        
        Args:
            service_type: Service type
            system_prompt_key: System prompt key
            response_key: Response key
            **kwargs: Additional configuration
            
        Returns:
            Service configuration
        """
        return cls(
            service_type=service_type,
            system_prompt_key=system_prompt_key,
            response_key=response_key,
            **kwargs
        )
    
    @classmethod
    def for_enhance_image(cls, **kwargs) -> "ServiceConfig":
        """Create configuration for image enhancement."""
        defaults = {
            "temperature": 0.3,
            "max_tokens": 4000,
            "timeout": 300.0,
        }
        defaults.update(kwargs)
        return cls.create(
            ServiceType.ENHANCE_IMAGE,
            "enhance_image",
            "enhancement_guide",
            **defaults
        )
    
    @classmethod
    def for_enhance_video(cls, **kwargs) -> "ServiceConfig":
        """Create configuration for video enhancement."""
        defaults = {
            "temperature": 0.3,
            "max_tokens": 4000,
            "timeout": 600.0,
        }
        defaults.update(kwargs)
        return cls.create(
            ServiceType.ENHANCE_VIDEO,
            "enhance_video",
            "enhancement_guide",
            **defaults
        )
    
    @classmethod
    def for_upscale(cls, **kwargs) -> "ServiceConfig":
        """Create configuration for upscale."""
        defaults = {
            "temperature": 0.2,
            "max_tokens": 3000,
            "timeout": 300.0,
        }
        defaults.update(kwargs)
        return cls.create(
            ServiceType.UPSCALE,
            "upscale",
            "upscale_guide",
            **defaults
        )
    
    @classmethod
    def for_denoise(cls, **kwargs) -> "ServiceConfig":
        """Create configuration for denoise."""
        defaults = {
            "temperature": 0.2,
            "max_tokens": 3000,
            "timeout": 300.0,
        }
        defaults.update(kwargs)
        return cls.create(
            ServiceType.DENOISE,
            "denoise",
            "denoise_guide",
            **defaults
        )
    
    @classmethod
    def for_restore(cls, **kwargs) -> "ServiceConfig":
        """Create configuration for restore."""
        defaults = {
            "temperature": 0.3,
            "max_tokens": 4000,
            "timeout": 300.0,
        }
        defaults.update(kwargs)
        return cls.create(
            ServiceType.RESTORE,
            "restore",
            "restore_guide",
            **defaults
        )
    
    @classmethod
    def for_color_correction(cls, **kwargs) -> "ServiceConfig":
        """Create configuration for color correction."""
        defaults = {
            "temperature": 0.3,
            "max_tokens": 3000,
            "timeout": 300.0,
        }
        defaults.update(kwargs)
        return cls.create(
            ServiceType.COLOR_CORRECTION,
            "color_correction",
            "color_guide",
            **defaults
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "service_type": self.service_type.value,
            "system_prompt_key": self.system_prompt_key,
            "response_key": self.response_key,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "include_timestamp": self.include_timestamp,
            "timeout": self.timeout,
            "retry_attempts": self.retry_attempts,
            "cache_enabled": self.cache_enabled,
            "cache_ttl": self.cache_ttl,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ServiceConfig":
        """Create from dictionary."""
        return cls(
            service_type=ServiceType(data["service_type"]),
            system_prompt_key=data["system_prompt_key"],
            response_key=data["response_key"],
            temperature=data.get("temperature", 0.3),
            max_tokens=data.get("max_tokens", 4000),
            include_timestamp=data.get("include_timestamp", False),
            timeout=data.get("timeout", 300.0),
            retry_attempts=data.get("retry_attempts", 3),
            cache_enabled=data.get("cache_enabled", True),
            cache_ttl=data.get("cache_ttl", 3600),
            metadata=data.get("metadata", {})
        )


class ServiceConfigRegistry:
    """Registry for service configurations."""
    
    def __init__(self):
        """Initialize service config registry."""
        self.configs: Dict[ServiceType, ServiceConfig] = {}
        self._register_defaults()
    
    def _register_defaults(self):
        """Register default configurations."""
        self.configs[ServiceType.ENHANCE_IMAGE] = ServiceConfig.for_enhance_image()
        self.configs[ServiceType.ENHANCE_VIDEO] = ServiceConfig.for_enhance_video()
        self.configs[ServiceType.UPSCALE] = ServiceConfig.for_upscale()
        self.configs[ServiceType.DENOISE] = ServiceConfig.for_denoise()
        self.configs[ServiceType.RESTORE] = ServiceConfig.for_restore()
        self.configs[ServiceType.COLOR_CORRECTION] = ServiceConfig.for_color_correction()
    
    def register(self, service_type: ServiceType, config: ServiceConfig):
        """
        Register service configuration.
        
        Args:
            service_type: Service type
            config: Service configuration
        """
        self.configs[service_type] = config
        logger.debug(f"Registered config for {service_type.value}")
    
    def get(self, service_type: ServiceType) -> Optional[ServiceConfig]:
        """
        Get service configuration.
        
        Args:
            service_type: Service type
            
        Returns:
            Service configuration or None
        """
        return self.configs.get(service_type)
    
    def get_or_default(self, service_type: ServiceType) -> ServiceConfig:
        """
        Get service configuration or default.
        
        Args:
            service_type: Service type
            
        Returns:
            Service configuration
        """
        config = self.get(service_type)
        if config:
            return config
        
        # Create default based on service type
        factory_methods = {
            ServiceType.ENHANCE_IMAGE: ServiceConfig.for_enhance_image,
            ServiceType.ENHANCE_VIDEO: ServiceConfig.for_enhance_video,
            ServiceType.UPSCALE: ServiceConfig.for_upscale,
            ServiceType.DENOISE: ServiceConfig.for_denoise,
            ServiceType.RESTORE: ServiceConfig.for_restore,
            ServiceType.COLOR_CORRECTION: ServiceConfig.for_color_correction,
        }
        
        factory = factory_methods.get(service_type)
        if factory:
            config = factory()
            self.register(service_type, config)
            return config
        
        raise ValueError(f"No configuration available for {service_type}")




