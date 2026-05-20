"""
Base Service
============
Base class for all services with common functionality
"""

import logging
import time
from typing import Dict, Any, Optional
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ServiceStatistics:
    """Base statistics for services"""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    total_duration: float = 0.0
    average_duration: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    
    def record_operation(self, success: bool, duration: float):
        """Record an operation"""
        self.total_operations += 1
        self.total_duration += duration
        
        if success:
            self.successful_operations += 1
        else:
            self.failed_operations += 1
        
        if self.total_operations > 0:
            self.average_duration = self.total_duration / self.total_operations
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_operations == 0:
            return 0.0
        return self.successful_operations / self.total_operations
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total_operations': self.total_operations,
            'successful_operations': self.successful_operations,
            'failed_operations': self.failed_operations,
            'success_rate': self.success_rate,
            'total_duration': self.total_duration,
            'average_duration': self.average_duration,
            'created_at': self.created_at.isoformat()
        }


class BaseService(ABC):
    """
    Base class for all services.
    
    Provides:
    - Logging
    - Statistics tracking
    - Common utility methods
    """
    
    def __init__(self, service_name: Optional[str] = None):
        """
        Initialize base service.
        
        Args:
            service_name: Name of the service (default: class name)
        """
        self.service_name = service_name or self.__class__.__name__
        self.logger = logging.getLogger(f"{__name__}.{self.service_name}")
        self._stats = ServiceStatistics()
        self._initialized_at = time.time()
    
    def log_info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(f"[{self.service_name}] {message}", **kwargs)
    
    def log_error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(f"[{self.service_name}] {message}", **kwargs)
    
    def log_warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(f"[{self.service_name}] {message}", **kwargs)
    
    def log_debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(f"[{self.service_name}] {message}", **kwargs)
    
    def record_operation(self, success: bool, duration: float):
        """Record operation in statistics"""
        self._stats.record_operation(success, duration)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get service statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = self._stats.to_dict()
        stats['service_name'] = self.service_name
        stats['uptime_seconds'] = time.time() - self._initialized_at
        return stats
    
    def reset_statistics(self):
        """Reset statistics"""
        self._stats = ServiceStatistics()
        self.log_debug("Statistics reset")
    
    def get_health(self) -> Dict[str, Any]:
        """
        Get service health status.
        
        Returns:
            Dictionary with health information
        """
        return {
            'service_name': self.service_name,
            'status': 'healthy',
            'uptime_seconds': time.time() - self._initialized_at,
            'statistics': self.get_statistics()
        }

