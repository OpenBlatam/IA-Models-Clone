#!/usr/bin/env python3
"""
Domain Repositories - Abstract repositories for data persistence
Implements Repository pattern for data access abstraction
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import uuid
from datetime import datetime, timezone

from .entities import OptimizationTask, ModelProfile, OptimizationResult, PerformanceScore, ResourceUsage
from .value_objects import OptimizationType, OptimizationStatus

class Repository(ABC):
    """Base repository interface."""
    
    @abstractmethod
    async def save(self, entity) -> None:
        """Save an entity."""
        pass
    
    @abstractmethod
    async def find_by_id(self, entity_id: str):
        """Find entity by ID."""
        pass
    
    @abstractmethod
    async def find_all(self, limit: int = 100, offset: int = 0) -> List:
        """Find all entities with pagination."""
        pass
    
    @abstractmethod
    async def delete(self, entity_id: str) -> bool:
        """Delete entity by ID."""
        pass
    
    @abstractmethod
    async def count(self) -> int:
        """Get total count of entities."""
        pass

class OptimizationRepository(Repository):
    """Repository for optimization tasks."""
    
    @abstractmethod
    async def find_by_status(self, status: OptimizationStatus) -> List[OptimizationTask]:
        """Find tasks by status."""
        pass
    
    @abstractmethod
    async def find_by_optimization_type(self, optimization_type: OptimizationType) -> List[OptimizationTask]:
        """Find tasks by optimization type."""
        pass
    
    @abstractmethod
    async def find_by_model_profile(self, model_profile_id: str) -> List[OptimizationTask]:
        """Find tasks by model profile."""
        pass
    
    @abstractmethod
    async def find_pending_tasks(self, limit: int = 10) -> List[OptimizationTask]:
        """Find pending tasks for processing."""
        pass
    
    @abstractmethod
    async def find_running_tasks(self) -> List[OptimizationTask]:
        """Find currently running tasks."""
        pass
    
    @abstractmethod
    async def find_completed_tasks(self, limit: int = 100) -> List[OptimizationTask]:
        """Find completed tasks."""
        pass
    
    @abstractmethod
    async def find_failed_tasks(self, limit: int = 100) -> List[OptimizationTask]:
        """Find failed tasks."""
        pass
    
    @abstractmethod
    async def find_by_priority(self, min_priority: int = 1) -> List[OptimizationTask]:
        """Find tasks by minimum priority."""
        pass
    
    @abstractmethod
    async def find_by_date_range(self, start_date: datetime, end_date: datetime) -> List[OptimizationTask]:
        """Find tasks by date range."""
        pass
    
    @abstractmethod
    async def get_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        pass

class ModelRepository(Repository):
    """Repository for model profiles."""
    
    @abstractmethod
    async def find_by_name(self, model_name: str) -> Optional[ModelProfile]:
        """Find model profile by name."""
        pass
    
    @abstractmethod
    async def find_by_complexity_range(self, min_complexity: float, max_complexity: float) -> List[ModelProfile]:
        """Find models by complexity range."""
        pass
    
    @abstractmethod
    async def find_by_parameter_range(self, min_params: int, max_params: int) -> List[ModelProfile]:
        """Find models by parameter range."""
        pass
    
    @abstractmethod
    async def find_by_memory_range(self, min_memory: float, max_memory: float) -> List[ModelProfile]:
        """Find models by memory usage range."""
        pass
    
    @abstractmethod
    async def find_suitable_for_optimization(self, optimization_type: OptimizationType) -> List[ModelProfile]:
        """Find models suitable for specific optimization type."""
        pass
    
    @abstractmethod
    async def find_recent_models(self, days: int = 7) -> List[ModelProfile]:
        """Find recently created models."""
        pass
    
    @abstractmethod
    async def get_model_statistics(self) -> Dict[str, Any]:
        """Get model statistics."""
        pass

class PerformanceRepository(Repository):
    """Repository for performance data."""
    
    @abstractmethod
    async def save_performance_score(self, task_id: str, score: PerformanceScore) -> None:
        """Save performance score for a task."""
        pass
    
    @abstractmethod
    async def save_resource_usage(self, task_id: str, usage: ResourceUsage) -> None:
        """Save resource usage for a task."""
        pass
    
    @abstractmethod
    async def find_performance_by_task(self, task_id: str) -> Optional[PerformanceScore]:
        """Find performance score by task ID."""
        pass
    
    @abstractmethod
    async def find_resource_usage_by_task(self, task_id: str) -> Optional[ResourceUsage]:
        """Find resource usage by task ID."""
        pass
    
    @abstractmethod
    async def find_performance_by_date_range(self, start_date: datetime, end_date: datetime) -> List[Tuple[str, PerformanceScore]]:
        """Find performance scores by date range."""
        pass
    
    @abstractmethod
    async def find_resource_usage_by_date_range(self, start_date: datetime, end_date: datetime) -> List[Tuple[str, ResourceUsage]]:
        """Find resource usage by date range."""
        pass
    
    @abstractmethod
    async def get_performance_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        pass
    
    @abstractmethod
    async def get_resource_usage_statistics(self) -> Dict[str, Any]:
        """Get resource usage statistics."""
        pass
    
    @abstractmethod
    async def find_top_performers(self, limit: int = 10) -> List[Tuple[str, PerformanceScore]]:
        """Find top performing tasks."""
        pass
    
    @abstractmethod
    async def find_most_efficient(self, limit: int = 10) -> List[Tuple[str, ResourceUsage]]:
        """Find most efficient tasks."""
        pass

class ResultRepository(Repository):
    """Repository for optimization results."""
    
    @abstractmethod
    async def find_by_task(self, task_id: str) -> List[OptimizationResult]:
        """Find results by task ID."""
        pass
    
    @abstractmethod
    async def find_successful_results(self, limit: int = 100) -> List[OptimizationResult]:
        """Find successful results."""
        pass
    
    @abstractmethod
    async def find_failed_results(self, limit: int = 100) -> List[OptimizationResult]:
        """Find failed results."""
        pass
    
    @abstractmethod
    async def find_by_improvement_range(self, min_improvement: float, max_improvement: float) -> List[OptimizationResult]:
        """Find results by improvement range."""
        pass
    
    @abstractmethod
    async def find_by_execution_time_range(self, min_time: float, max_time: float) -> List[OptimizationResult]:
        """Find results by execution time range."""
        pass
    
    @abstractmethod
    async def find_best_results(self, limit: int = 10) -> List[OptimizationResult]:
        """Find best performing results."""
        pass
    
    @abstractmethod
    async def find_most_efficient_results(self, limit: int = 10) -> List[OptimizationResult]:
        """Find most efficient results."""
        pass
    
    @abstractmethod
    async def get_result_statistics(self) -> Dict[str, Any]:
        """Get result statistics."""
        pass
    
    @abstractmethod
    async def find_by_date_range(self, start_date: datetime, end_date: datetime) -> List[OptimizationResult]:
        """Find results by date range."""
        pass

class RepositoryFactory:
    """Factory for creating repositories."""
    
    def __init__(self, database_service=None, cache_service=None):
        self.database_service = database_service
        self.cache_service = cache_service
    
    def create_optimization_repository(self) -> OptimizationRepository:
        """Create optimization repository."""
        # This would return a concrete implementation
        # For now, return a placeholder
        pass
    
    def create_model_repository(self) -> ModelRepository:
        """Create model repository."""
        # This would return a concrete implementation
        # For now, return a placeholder
        pass
    
    def create_performance_repository(self) -> PerformanceRepository:
        """Create performance repository."""
        # This would return a concrete implementation
        # For now, return a placeholder
        pass
    
    def create_result_repository(self) -> ResultRepository:
        """Create result repository."""
        # This would return a concrete implementation
        # For now, return a placeholder
        pass

class UnitOfWork:
    """Unit of Work pattern for transaction management."""
    
    def __init__(self, repositories: Dict[str, Repository]):
        self.repositories = repositories
        self._committed = False
        self._rolled_back = False
    
    async def __aenter__(self):
        """Enter the unit of work context."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the unit of work context."""
        if exc_type is None:
            await self.commit()
        else:
            await self.rollback()
    
    async def commit(self):
        """Commit all changes."""
        if self._committed or self._rolled_back:
            return
        
        try:
            # Commit all repositories
            for repository in self.repositories.values():
                if hasattr(repository, 'commit'):
                    await repository.commit()
            
            self._committed = True
        except Exception as e:
            await self.rollback()
            raise e
    
    async def rollback(self):
        """Rollback all changes."""
        if self._committed or self._rolled_back:
            return
        
        try:
            # Rollback all repositories
            for repository in self.repositories.values():
                if hasattr(repository, 'rollback'):
                    await repository.rollback()
            
            self._rolled_back = True
        except Exception as e:
            # Log error but don't raise to avoid masking original exception
            pass
    
    def get_repository(self, name: str) -> Repository:
        """Get repository by name."""
        return self.repositories.get(name)
    
    def add_repository(self, name: str, repository: Repository):
        """Add repository to unit of work."""
        self.repositories[name] = repository
