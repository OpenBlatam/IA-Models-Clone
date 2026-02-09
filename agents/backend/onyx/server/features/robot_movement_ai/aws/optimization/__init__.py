"""
Optimization Package
====================

Serverless optimizations, service mesh, and observability.
"""

from aws.optimization.serverless_optimizer import (
    ServerlessOptimizer,
    ConnectionPool,
    LazyLoader,
    get_serverless_optimizer
)
from aws.optimization.service_mesh import (
    ServiceMesh,
    MeshConfig,
    get_service_mesh
)
from aws.optimization.database_per_service import (
    DatabaseAdapter,
    DynamoDBAdapter,
    PostgreSQLAdapter,
    ServiceDatabase,
    get_movement_database,
    get_trajectory_database,
    get_chat_database
)
from aws.optimization.observability import (
    DistributedTracer,
    MetricsCollector,
    StructuredLogger,
    ObservabilityManager,
    get_observability_manager
)

__all__ = [
    "ServerlessOptimizer",
    "ConnectionPool",
    "LazyLoader",
    "get_serverless_optimizer",
    "ServiceMesh",
    "MeshConfig",
    "get_service_mesh",
    "DatabaseAdapter",
    "DynamoDBAdapter",
    "PostgreSQLAdapter",
    "ServiceDatabase",
    "get_movement_database",
    "get_trajectory_database",
    "get_chat_database",
    "DistributedTracer",
    "MetricsCollector",
    "StructuredLogger",
    "ObservabilityManager",
    "get_observability_manager",
]















