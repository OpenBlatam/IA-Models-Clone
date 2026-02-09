"""
Common Types
============
Tipos comunes compartidos entre sistemas
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import time


class Status(Enum):
    """Estado común"""
    PENDING = "pending"
    ACTIVE = "active"
    INACTIVE = "inactive"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class Priority(Enum):
    """Prioridad común"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ExecutionResult:
    """Resultado de ejecución común"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class OperationMetrics:
    """Métricas de operación comunes"""
    operation_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_time: float = 0.0
    average_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    last_operation_time: Optional[float] = None
    
    def record_operation(self, execution_time: float, success: bool):
        """Registrar operación"""
        self.operation_count += 1
        self.total_time += execution_time
        self.last_operation_time = time.time()
        
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        
        if execution_time < self.min_time:
            self.min_time = execution_time
        if execution_time > self.max_time:
            self.max_time = execution_time
        
        if self.operation_count > 0:
            self.average_time = self.total_time / self.operation_count
    
    @property
    def success_rate(self) -> float:
        """Tasa de éxito"""
        if self.operation_count == 0:
            return 0.0
        return self.success_count / self.operation_count


@dataclass
class ResourceUsage:
    """Uso de recursos común"""
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    disk_mb: float = 0.0
    network_mb: float = 0.0
    gpu_percent: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class HealthStatus:
    """Estado de salud común"""
    healthy: bool
    status: str
    message: Optional[str] = None
    checks: Dict[str, bool] = None
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.checks is None:
            self.checks = {}


@dataclass
class ErrorInfo:
    """Información de error común"""
    error_type: str
    message: str
    code: Optional[str] = None
    details: Dict[str, Any] = None
    timestamp: float = 0.0
    stack_trace: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.details is None:
            self.details = {}


@dataclass
class PaginationInfo:
    """Información de paginación común"""
    page: int
    page_size: int
    total_items: int
    total_pages: int
    
    @property
    def has_next(self) -> bool:
        """Tiene siguiente página"""
        return self.page < self.total_pages
    
    @property
    def has_previous(self) -> bool:
        """Tiene página anterior"""
        return self.page > 1
    
    @property
    def offset(self) -> int:
        """Offset para queries"""
        return (self.page - 1) * self.page_size


@dataclass
class FilterCriteria:
    """Criterios de filtrado comunes"""
    field: str
    operator: str  # ==, !=, >, <, >=, <=, in, not_in, contains, starts_with, ends_with
    value: Any
    logic: str = "AND"  # AND, OR


@dataclass
class SortCriteria:
    """Criterios de ordenamiento comunes"""
    field: str
    direction: str = "asc"  # asc, desc


@dataclass
class QueryOptions:
    """Opciones de query comunes"""
    filters: Optional[List[FilterCriteria]] = None
    sorts: Optional[List[SortCriteria]] = None
    pagination: Optional[PaginationInfo] = None
    fields: Optional[List[str]] = None  # Campos a retornar
    
    def __post_init__(self):
        if self.filters is None:
            self.filters = []
        if self.sorts is None:
            self.sorts = []

