"""
Core interfaces for the Blaze AI module.

This module provides the fundamental interfaces and data structures
used throughout the Blaze AI system.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

# =============================================================================
# Enums
# =============================================================================

class HealthStatus(Enum):
    """System health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class ComponentType(Enum):
    """Component type enumeration."""
    ENGINE = "engine"
    MANAGER = "manager"
    SERVICE = "service"
    UTILITY = "utility"
    INTERFACE = "interface"

# =============================================================================
# Core Configuration
# =============================================================================

@dataclass
class CoreConfig:
    """Core configuration for the Blaze AI system."""
    
    # System settings
    system_name: str = "Blaze AI"
    version: str = "1.0.0"
    environment: str = "development"
    debug_mode: bool = False
    log_level: str = "INFO"
    
    # Performance settings
    max_concurrent_requests: int = 100
    max_memory_usage: int = 8 * 1024 * 1024 * 1024  # 8GB
    max_gpu_memory_usage: int = 6 * 1024 * 1024 * 1024  # 6GB
    enable_profiling: bool = False
    enable_metrics: bool = True
    
    # Network settings
    host: str = "0.0.0.0"
    port: int = 8000
    max_connections: int = 1000
    connection_timeout: float = 30.0
    keep_alive_timeout: float = 60.0
    
    # Security settings
    enable_authentication: bool = False
    enable_authorization: bool = False
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 1000
    
    # Monitoring settings
    enable_health_checks: bool = True
    health_check_interval: float = 30.0
    enable_alerting: bool = True
    alert_threshold: float = 0.8
    
    # Storage settings
    data_directory: str = "./data"
    cache_directory: str = "./cache"
    log_directory: str = "./logs"
    max_disk_usage: int = 100 * 1024 * 1024 * 1024  # 100GB
    
    # Model settings
    default_model_device: str = "auto"
    default_model_precision: str = "float16"
    enable_model_caching: bool = True
    max_cached_models: int = 10
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        if self.max_concurrent_requests <= 0:
            errors.append("max_concurrent_requests must be positive")
        
        if self.max_memory_usage <= 0:
            errors.append("max_memory_usage must be positive")
        
        if self.port < 1 or self.port > 65535:
            errors.append("port must be between 1 and 65535")
        
        if self.connection_timeout <= 0:
            errors.append("connection_timeout must be positive")
        
        if self.health_check_interval <= 0:
            errors.append("health_check_interval must be positive")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "system_name": self.system_name,
            "version": self.version,
            "environment": self.environment,
            "debug_mode": self.debug_mode,
            "log_level": self.log_level,
            "max_concurrent_requests": self.max_concurrent_requests,
            "max_memory_usage": self.max_memory_usage,
            "max_gpu_memory_usage": self.max_gpu_memory_usage,
            "enable_profiling": self.enable_profiling,
            "enable_metrics": self.enable_metrics,
            "host": self.host,
            "port": self.port,
            "max_connections": self.max_connections,
            "connection_timeout": self.connection_timeout,
            "keep_alive_timeout": self.keep_alive_timeout,
            "enable_authentication": self.enable_authentication,
            "enable_authorization": self.enable_authorization,
            "enable_rate_limiting": self.enable_rate_limiting,
            "max_requests_per_minute": self.max_requests_per_minute,
            "enable_health_checks": self.enable_health_checks,
            "health_check_interval": self.health_check_interval,
            "enable_alerting": self.enable_alerting,
            "alert_threshold": self.alert_threshold,
            "data_directory": self.data_directory,
            "cache_directory": self.cache_directory,
            "log_directory": self.log_directory,
            "max_disk_usage": self.max_disk_usage,
            "default_model_device": self.default_model_device,
            "default_model_precision": self.default_model_precision,
            "enable_model_caching": self.enable_model_caching,
            "max_cached_models": self.max_cached_models
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CoreConfig:
        """Create configuration from dictionary."""
        return cls(**data)

# =============================================================================
# System Health
# =============================================================================

@dataclass
class ComponentHealth:
    """Health information for a single component."""
    
    component_id: str
    component_type: ComponentType
    status: HealthStatus
    message: str
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)
    
    def is_healthy(self) -> bool:
        """Check if component is healthy."""
        return self.status == HealthStatus.HEALTHY
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component_id": self.component_id,
            "component_type": self.component_type.value,
            "status": self.status.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "details": self.details
        }

@dataclass
class SystemHealth:
    """Overall system health information."""
    
    system_name: str = "Blaze AI"
    overall_status: HealthStatus = HealthStatus.UNKNOWN
    last_updated: float = field(default_factory=time.time)
    components: Dict[str, ComponentHealth] = field(default_factory=dict)
    
    def add_component(self, component_id: str, component_type: ComponentType, 
                     status: HealthStatus, message: str, details: Optional[Dict[str, Any]] = None):
        """Add or update component health information."""
        self.components[component_id] = ComponentHealth(
            component_id=component_id,
            component_type=component_type,
            status=status,
            message=message,
            details=details or {}
        )
        self.last_updated = time.time()
        self._update_overall_status()
    
    def update_component(self, component_id: str, status: HealthStatus, 
                        message: str, details: Optional[Dict[str, Any]] = None):
        """Update existing component health information."""
        if component_id in self.components:
            component = self.components[component_id]
            component.status = status
            component.message = message
            component.timestamp = time.time()
            if details:
                component.details.update(details)
            
            self.last_updated = time.time()
            self._update_overall_status()
    
    def remove_component(self, component_id: str):
        """Remove component health information."""
        if component_id in self.components:
            del self.components[component_id]
            self.last_updated = time.time()
            self._update_overall_status()
    
    def get_component_health(self, component_id: str) -> Optional[ComponentHealth]:
        """Get health information for a specific component."""
        return self.components.get(component_id)
    
    def get_healthy_components(self) -> List[ComponentHealth]:
        """Get list of healthy components."""
        return [comp for comp in self.components.values() if comp.is_healthy()]
    
    def get_unhealthy_components(self) -> List[ComponentHealth]:
        """Get list of unhealthy components."""
        return [comp for comp in self.components.values() if not comp.is_healthy()]
    
    def _update_overall_status(self):
        """Update overall system health status."""
        if not self.components:
            self.overall_status = HealthStatus.UNKNOWN
            return
        
        healthy_count = len(self.get_healthy_components())
        total_count = len(self.components)
        
        if healthy_count == total_count:
            self.overall_status = HealthStatus.HEALTHY
        elif healthy_count >= total_count * 0.8:  # 80% threshold
            self.overall_status = HealthStatus.DEGRADED
        else:
            self.overall_status = HealthStatus.UNHEALTHY
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary information."""
        total_components = len(self.components)
        healthy_components = len(self.get_healthy_components())
        unhealthy_components = len(self.get_unhealthy_components())
        
        return {
            "system_name": self.system_name,
            "overall_status": self.overall_status.value,
            "last_updated": self.last_updated,
            "total_components": total_components,
            "healthy_components": healthy_components,
            "unhealthy_components": unhealthy_components,
            "health_percentage": (healthy_components / total_components * 100) if total_components > 0 else 0.0,
            "components": {
                comp_id: comp.to_dict() for comp_id, comp in self.components.items()
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.get_health_summary()

# =============================================================================
# Abstract Base Classes
# =============================================================================

class HealthCheckable(ABC):
    """Abstract base class for components that can report health."""
    
    @abstractmethod
    def get_health_status(self) -> ComponentHealth:
        """Get the health status of this component."""
        pass
    
    @abstractmethod
    async def perform_health_check(self) -> bool:
        """Perform a health check and return success status."""
        pass

class Configurable(ABC):
    """Abstract base class for configurable components."""
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration."""
        pass
    
    @abstractmethod
    def update_config(self, config: Dict[str, Any]) -> bool:
        """Update the configuration."""
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate a configuration and return list of errors."""
        pass

class MetricsProvider(ABC):
    """Abstract base class for components that provide metrics."""
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get the current metrics."""
        pass
    
    @abstractmethod
    def reset_metrics(self):
        """Reset all metrics to default values."""
        pass

# =============================================================================
# Utility Functions
# =============================================================================

def create_default_config() -> CoreConfig:
    """Create a default configuration."""
    return CoreConfig()

def create_production_config() -> CoreConfig:
    """Create a production-ready configuration."""
    config = CoreConfig()
    config.environment = "production"
    config.debug_mode = False
    config.log_level = "WARNING"
    config.enable_profiling = False
    config.enable_metrics = True
    config.max_concurrent_requests = 500
    config.max_connections = 5000
    config.enable_authentication = True
    config.enable_authorization = True
    config.enable_rate_limiting = True
    config.max_requests_per_minute = 10000
    config.health_check_interval = 60.0
    return config

def create_development_config() -> CoreConfig:
    """Create a development configuration."""
    config = CoreConfig()
    config.environment = "development"
    config.debug_mode = True
    config.log_level = "DEBUG"
    config.enable_profiling = True
    config.enable_metrics = True
    config.max_concurrent_requests = 50
    config.max_connections = 100
    config.enable_authentication = False
    config.enable_authorization = False
    config.enable_rate_limiting = False
    config.health_check_interval = 10.0
    return config


