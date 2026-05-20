"""
Enhanced Base Service for Color Grading AI
===========================================

Enhanced base service with unified data management, statistics, and history tracking.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from .base_service import BaseService, ServiceConfig
from .data_manager import DataManager, StatisticsManager

logger = logging.getLogger(__name__)


class EnhancedBaseService(BaseService):
    """
    Enhanced base service with additional features.
    
    Adds:
    - Unified data management
    - Statistics collection
    - History tracking
    - Persistence
    """
    
    def __init__(self, name: str, config: Optional[ServiceConfig] = None):
        """
        Initialize enhanced base service.
        
        Args:
            name: Service name
            config: Optional service configuration
        """
        super().__init__(name, config)
        
        # Data management
        self.data_manager: Optional[DataManager] = None
        self.statistics_manager = StatisticsManager()
        
        # Enhanced stats
        self._enhanced_stats: Dict[str, Any] = {
            "operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "total_duration": 0.0,
            "avg_duration": 0.0,
        }
    
    def record_operation(
        self,
        operation_name: str,
        duration: float,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record operation.
        
        Args:
            operation_name: Operation name
            duration: Operation duration
            success: Whether operation was successful
            metadata: Optional metadata
        """
        # Update base stats
        self._record_call(success)
        
        # Update enhanced stats
        self._enhanced_stats["operations"] += 1
        if success:
            self._enhanced_stats["successful_operations"] += 1
        else:
            self._enhanced_stats["failed_operations"] += 1
        
        self._enhanced_stats["total_duration"] += duration
        self._enhanced_stats["avg_duration"] = (
            self._enhanced_stats["total_duration"] / self._enhanced_stats["operations"]
        )
        
        # Update statistics manager
        self.statistics_manager.increment(f"{operation_name}_count")
        self.statistics_manager.record_histogram(f"{operation_name}_duration", duration)
        
        if success:
            self.statistics_manager.increment(f"{operation_name}_success")
        else:
            self.statistics_manager.increment(f"{operation_name}_failure")
        
        # Record in data manager if available
        if self.data_manager:
            self.data_manager.add({
                "operation": operation_name,
                "duration": duration,
                "success": success
            }, metadata=metadata)
    
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """
        Get enhanced statistics.
        
        Returns:
            Enhanced statistics dictionary
        """
        base_stats = self.get_stats()
        stats_stats = self.statistics_manager.get_statistics()
        
        return {
            **base_stats,
            **self._enhanced_stats,
            "statistics": stats_stats,
            "success_rate": (
                self._enhanced_stats["successful_operations"] / self._enhanced_stats["operations"]
                if self._enhanced_stats["operations"] > 0 else 0.0
            ),
        }
    
    def get_history(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Any]:
        """
        Get operation history.
        
        Args:
            start_date: Optional start date
            end_date: Optional end date
            limit: Optional limit
            
        Returns:
            List of history entries
        """
        if not self.data_manager:
            return []
        
        return self.data_manager.get_history(
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
    
    def reset_enhanced_stats(self):
        """Reset enhanced statistics."""
        self.reset_stats()
        self._enhanced_stats = {
            "operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "total_duration": 0.0,
            "avg_duration": 0.0,
        }
        self.statistics_manager.reset()
        logger.debug(f"Reset enhanced stats for {self.name}")
    
    def setup_data_manager(self, data_manager: DataManager):
        """
        Setup data manager.
        
        Args:
            data_manager: Data manager instance
        """
        self.data_manager = data_manager
        logger.debug(f"Setup data manager for {self.name}")




