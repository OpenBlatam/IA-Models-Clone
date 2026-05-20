#!/usr/bin/env python3
"""
Application Services - Business logic orchestration
Implements Clean Architecture application services
"""

import torch
import torch.nn as nn
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import uuid
from datetime import datetime, timezone

from ..domain.entities import OptimizationTask, ModelProfile, OptimizationResult, PerformanceScore, ResourceUsage
from ..domain.value_objects import OptimizationType, OptimizationStatus
from ..domain.repositories import OptimizationRepository, ModelRepository, PerformanceRepository, ResultRepository
from .commands import Command, CreateOptimizationTask, StartOptimization, CompleteOptimization, CancelOptimization
from .queries import Query, GetOptimizationTask, GetModelProfile, GetPerformanceMetrics, GetOptimizationStatistics
from .handlers import CommandHandler, QueryHandler, EventHandler
from .events import ApplicationEvent, OptimizationStarted, OptimizationCompleted, OptimizationFailed
from .dto import OptimizationTaskDTO, ModelProfileDTO, PerformanceMetricsDTO, OptimizationResultDTO

class ApplicationService(ABC):
    """Base application service."""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    async def execute(self, command_or_query) -> Any:
        """Execute a command or query."""
        pass

class OptimizationApplicationService(ApplicationService):
    """Application service for optimization operations."""
    
    def __init__(self, 
                 optimization_repository: OptimizationRepository,
                 model_repository: ModelRepository,
                 performance_repository: PerformanceRepository,
                 result_repository: ResultRepository,
                 command_handlers: Dict[str, CommandHandler],
                 query_handlers: Dict[str, QueryHandler],
                 event_handlers: Dict[str, EventHandler],
                 logger: logging.Logger = None):
        super().__init__(logger)
        self.optimization_repository = optimization_repository
        self.model_repository = model_repository
        self.performance_repository = performance_repository
        self.result_repository = result_repository
        self.command_handlers = command_handlers
        self.query_handlers = query_handlers
        self.event_handlers = event_handlers
    
    async def execute(self, command_or_query) -> Any:
        """Execute a command or query."""
        try:
            if isinstance(command_or_query, Command):
                return await self._execute_command(command_or_query)
            elif isinstance(command_or_query, Query):
                return await self._execute_query(command_or_query)
            else:
                raise ValueError(f"Unknown type: {type(command_or_query)}")
        except Exception as e:
            self.logger.error(f"Execution failed: {e}")
            raise
    
    async def _execute_command(self, command: Command) -> Any:
        """Execute a command."""
        command_type = type(command).__name__
        handler = self.command_handlers.get(command_type)
        
        if not handler:
            raise ValueError(f"No handler found for command: {command_type}")
        
        return await handler.handle(command)
    
    async def _execute_query(self, query: Query) -> Any:
        """Execute a query."""
        query_type = type(query).__name__
        handler = self.query_handlers.get(query_type)
        
        if not handler:
            raise ValueError(f"No handler found for query: {query_type}")
        
        return await handler.handle(query)
    
    async def create_optimization_task(self, model: nn.Module, model_name: str,
                                      optimization_type: OptimizationType,
                                      target_improvement: float = 0.5,
                                      priority: int = 1) -> OptimizationTaskDTO:
        """Create a new optimization task."""
        try:
            # Create model profile
            model_profile = ModelProfile(model, model_name)
            await self.model_repository.save(model_profile)
            
            # Create optimization task
            task = OptimizationTask(
                model_profile=model_profile,
                optimization_type=optimization_type,
                target_improvement=target_improvement,
                priority=priority
            )
            
            await self.optimization_repository.save(task)
            
            # Publish event
            await self._publish_event(OptimizationStarted(task.id, model_name, optimization_type.value))
            
            # Convert to DTO
            return OptimizationTaskDTO.from_entity(task)
            
        except Exception as e:
            self.logger.error(f"Failed to create optimization task: {e}")
            raise
    
    async def start_optimization(self, task_id: str) -> OptimizationTaskDTO:
        """Start optimization for a task."""
        try:
            # Get task
            task = await self.optimization_repository.find_by_id(task_id)
            if not task:
                raise ValueError(f"Task not found: {task_id}")
            
            # Start task
            task.start()
            await self.optimization_repository.save(task)
            
            # Convert to DTO
            return OptimizationTaskDTO.from_entity(task)
            
        except Exception as e:
            self.logger.error(f"Failed to start optimization: {e}")
            raise
    
    async def complete_optimization(self, task_id: str, 
                                  performance_scores: PerformanceScore,
                                  resource_usage: ResourceUsage,
                                  applied_strategies: List[str]) -> OptimizationResultDTO:
        """Complete optimization for a task."""
        try:
            # Get task
            task = await self.optimization_repository.find_by_id(task_id)
            if not task:
                raise ValueError(f"Task not found: {task_id}")
            
            # Add applied strategies
            for strategy in applied_strategies:
                task.add_strategy(strategy)
            
            # Complete task
            task.complete(performance_scores, resource_usage)
            await self.optimization_repository.save(task)
            
            # Save performance data
            await self.performance_repository.save_performance_score(task_id, performance_scores)
            await self.performance_repository.save_resource_usage(task_id, resource_usage)
            
            # Create result
            result = OptimizationResult(
                task=task,
                success=True,
                improvement_score=performance_scores.overall_score,
                execution_time=resource_usage.execution_time,
                resource_usage=resource_usage
            )
            await self.result_repository.save(result)
            
            # Publish event
            await self._publish_event(OptimizationCompleted(task_id, performance_scores.overall_score))
            
            # Convert to DTO
            return OptimizationResultDTO.from_entity(result)
            
        except Exception as e:
            self.logger.error(f"Failed to complete optimization: {e}")
            raise
    
    async def fail_optimization(self, task_id: str, error_message: str) -> OptimizationTaskDTO:
        """Mark optimization as failed."""
        try:
            # Get task
            task = await self.optimization_repository.find_by_id(task_id)
            if not task:
                raise ValueError(f"Task not found: {task_id}")
            
            # Fail task
            task.fail(error_message)
            await self.optimization_repository.save(task)
            
            # Publish event
            await self._publish_event(OptimizationFailed(task_id, error_message))
            
            # Convert to DTO
            return OptimizationTaskDTO.from_entity(task)
            
        except Exception as e:
            self.logger.error(f"Failed to fail optimization: {e}")
            raise
    
    async def cancel_optimization(self, task_id: str) -> OptimizationTaskDTO:
        """Cancel optimization for a task."""
        try:
            # Get task
            task = await self.optimization_repository.find_by_id(task_id)
            if not task:
                raise ValueError(f"Task not found: {task_id}")
            
            # Cancel task
            task.cancel()
            await self.optimization_repository.save(task)
            
            # Convert to DTO
            return OptimizationTaskDTO.from_entity(task)
            
        except Exception as e:
            self.logger.error(f"Failed to cancel optimization: {e}")
            raise
    
    async def get_optimization_task(self, task_id: str) -> OptimizationTaskDTO:
        """Get optimization task by ID."""
        try:
            task = await self.optimization_repository.find_by_id(task_id)
            if not task:
                raise ValueError(f"Task not found: {task_id}")
            
            return OptimizationTaskDTO.from_entity(task)
            
        except Exception as e:
            self.logger.error(f"Failed to get optimization task: {e}")
            raise
    
    async def get_model_profile(self, model_profile_id: str) -> ModelProfileDTO:
        """Get model profile by ID."""
        try:
            model_profile = await self.model_repository.find_by_id(model_profile_id)
            if not model_profile:
                raise ValueError(f"Model profile not found: {model_profile_id}")
            
            return ModelProfileDTO.from_entity(model_profile)
            
        except Exception as e:
            self.logger.error(f"Failed to get model profile: {e}")
            raise
    
    async def get_performance_metrics(self, task_id: str) -> PerformanceMetricsDTO:
        """Get performance metrics for a task."""
        try:
            performance_score = await self.performance_repository.find_performance_by_task(task_id)
            resource_usage = await self.performance_repository.find_resource_usage_by_task(task_id)
            
            if not performance_score or not resource_usage:
                raise ValueError(f"Performance data not found for task: {task_id}")
            
            return PerformanceMetricsDTO(
                performance_score=performance_score,
                resource_usage=resource_usage
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {e}")
            raise
    
    async def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        try:
            # Get task statistics
            task_stats = await self.optimization_repository.get_statistics()
            
            # Get performance statistics
            performance_stats = await self.performance_repository.get_performance_statistics()
            resource_stats = await self.performance_repository.get_resource_usage_statistics()
            
            # Get result statistics
            result_stats = await self.result_repository.get_result_statistics()
            
            return {
                'tasks': task_stats,
                'performance': performance_stats,
                'resources': resource_stats,
                'results': result_stats
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get optimization statistics: {e}")
            raise
    
    async def _publish_event(self, event: ApplicationEvent):
        """Publish an application event."""
        try:
            event_type = type(event).__name__
            handlers = self.event_handlers.get(event_type, [])
            
            for handler in handlers:
                try:
                    await handler.handle(event)
                except Exception as e:
                    self.logger.error(f"Event handler failed: {e}")
                    
        except Exception as e:
            self.logger.error(f"Failed to publish event: {e}")

class PerformanceApplicationService(ApplicationService):
    """Application service for performance operations."""
    
    def __init__(self,
                 performance_repository: PerformanceRepository,
                 result_repository: ResultRepository,
                 query_handlers: Dict[str, QueryHandler],
                 logger: logging.Logger = None):
        super().__init__(logger)
        self.performance_repository = performance_repository
        self.result_repository = result_repository
        self.query_handlers = query_handlers
    
    async def execute(self, command_or_query) -> Any:
        """Execute a command or query."""
        try:
            if isinstance(command_or_query, Query):
                return await self._execute_query(command_or_query)
            else:
                raise ValueError(f"Unknown type: {type(command_or_query)}")
        except Exception as e:
            self.logger.error(f"Execution failed: {e}")
            raise
    
    async def _execute_query(self, query: Query) -> Any:
        """Execute a query."""
        query_type = type(query).__name__
        handler = self.query_handlers.get(query_type)
        
        if not handler:
            raise ValueError(f"No handler found for query: {query_type}")
        
        return await handler.handle(query)
    
    async def get_performance_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        try:
            performance_stats = await self.performance_repository.get_performance_statistics()
            resource_stats = await self.performance_repository.get_resource_usage_statistics()
            
            return {
                'performance': performance_stats,
                'resources': resource_stats
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get performance statistics: {e}")
            raise
    
    async def get_top_performers(self, limit: int = 10) -> List[Tuple[str, PerformanceScore]]:
        """Get top performing tasks."""
        try:
            return await self.performance_repository.find_top_performers(limit)
        except Exception as e:
            self.logger.error(f"Failed to get top performers: {e}")
            raise
    
    async def get_most_efficient(self, limit: int = 10) -> List[Tuple[str, ResourceUsage]]:
        """Get most efficient tasks."""
        try:
            return await self.performance_repository.find_most_efficient(limit)
        except Exception as e:
            self.logger.error(f"Failed to get most efficient: {e}")
            raise
