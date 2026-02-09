"""
Event sourcing and CQRS implementation for Export IA.
"""

from .aggregates import (
    ExportAggregate,
    TaskAggregate,
    QualityAggregate,
    UserAggregate
)
from .events import (
    BaseEvent,
    ExportRequested,
    ExportCompleted,
    ExportFailed,
    TaskCreated,
    TaskUpdated,
    QualityValidated,
    UserAuthenticated
)
from .handlers import (
    EventHandler,
    ExportEventHandler,
    TaskEventHandler,
    QualityEventHandler,
    UserEventHandler
)
from .store import (
    EventStore,
    InMemoryEventStore,
    DatabaseEventStore
)
from .projections import (
    Projection,
    ExportProjection,
    TaskProjection,
    QualityProjection,
    UserProjection
)
from .commands import (
    Command,
    CreateExportCommand,
    UpdateTaskCommand,
    ValidateQualityCommand,
    AuthenticateUserCommand
)
from .queries import (
    Query,
    GetExportQuery,
    GetTaskQuery,
    GetQualityQuery,
    GetUserQuery
)

__all__ = [
    "ExportAggregate",
    "TaskAggregate",
    "QualityAggregate",
    "UserAggregate",
    "BaseEvent",
    "ExportRequested",
    "ExportCompleted",
    "ExportFailed",
    "TaskCreated",
    "TaskUpdated",
    "QualityValidated",
    "UserAuthenticated",
    "EventHandler",
    "ExportEventHandler",
    "TaskEventHandler",
    "QualityEventHandler",
    "UserEventHandler",
    "EventStore",
    "InMemoryEventStore",
    "DatabaseEventStore",
    "Projection",
    "ExportProjection",
    "TaskProjection",
    "QualityProjection",
    "UserProjection",
    "Command",
    "CreateExportCommand",
    "UpdateTaskCommand",
    "ValidateQualityCommand",
    "AuthenticateUserCommand",
    "Query",
    "GetExportQuery",
    "GetTaskQuery",
    "GetQualityQuery",
    "GetUserQuery"
]




