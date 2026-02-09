"""
Base Classes Module
"""

from .base_manager import BaseManager, BaseProcessor, BaseSystem
from .manager_registry import ManagerRegistry, manager_registry
from .manager_factory import ManagerFactory, manager_factory
from .pipeline_base import BasePipeline, PipelineStep, PipelineExecution, PipelineStatus
from .interfaces import (
    IExecutable, IProcessable, IConfigurable, IMonitorable,
    IRetryable, IObservable, IStateful, IValidatable,
    IExportable, IImportable, ISearchable, ICacheable, IQueueable,
    ExecutionStatus
)
from .common_types import (
    Status, Priority, ExecutionResult, OperationMetrics,
    ResourceUsage, HealthStatus, ErrorInfo, PaginationInfo,
    FilterCriteria, SortCriteria, QueryOptions
)
from .statistics_mixin import StatisticsMixin
from .entity_manager_mixin import EntityManagerMixin
from .task_manager_mixin import TaskManagerMixin, Task, TaskStatus

__all__ = [
    # Base Classes
    'BaseManager',
    'BaseProcessor',
    'BaseSystem',
    'BasePipeline',
    'PipelineStep',
    'PipelineExecution',
    'PipelineStatus',
    # Registry & Factory
    'ManagerRegistry',
    'manager_registry',
    'ManagerFactory',
    'manager_factory',
    # Interfaces
    'IExecutable',
    'IProcessable',
    'IConfigurable',
    'IMonitorable',
    'IRetryable',
    'IObservable',
    'IStateful',
    'IValidatable',
    'IExportable',
    'IImportable',
    'ISearchable',
    'ICacheable',
    'IQueueable',
    'ExecutionStatus',
    # Common Types
    'Status',
    'Priority',
    'ExecutionResult',
    'OperationMetrics',
    'ResourceUsage',
    'HealthStatus',
    'ErrorInfo',
    'PaginationInfo',
    'FilterCriteria',
    'SortCriteria',
    'QueryOptions',
    # Mixins
    'StatisticsMixin',
    'EntityManagerMixin',
    'TaskManagerMixin',
    'Task',
    'TaskStatus'
]
