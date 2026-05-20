"""
Core module for Lovable Community

Provides database initialization, lifecycle management, caching, and core utilities.
"""

from .database import (
    get_db_engine,
    get_session_local,
    init_database,
    verify_database_connection,
    DatabaseManager
)

from .lifecycle import (
    lifespan,
    startup_handler,
    shutdown_handler,
    get_database_manager
)

from .unit_of_work import (
    UnitOfWork,
    unit_of_work
)

from .cache import (
    SimpleCache,
    get_cache,
    cached
)

from .performance_config import (
    configure_pytorch_performance,
    configure_transformers_performance,
    get_optimal_batch_size,
    configure_mixed_precision
)

from .async_operations import (
    AsyncProcessor,
    async_map,
    run_async
)

from .interfaces import (
    IChatRepository,
    IVoteRepository,
    IViewRepository,
    IRemixRepository,
    IRankingService,
    IValidator,
    IAIProcessor,
    IScoreManager,
    IIdentityService
)

from .dependency_container import (
    DependencyContainer,
    container,
    ServiceScope,
    ServiceRegistration,
    singleton,
    scoped,
    transient
)

from .container_helpers import (
    get_request_scope_id,
    create_container_dependency,
    create_type_based_dependency,
    setup_request_scope_middleware,
    register_database_session,
    get_chat_service_dependency,
    get_ranking_service_dependency,
    get_identity_service_dependency
)

__all__ = [
    # Database
    "get_db_engine",
    "get_session_local",
    "init_database",
    "verify_database_connection",
    "DatabaseManager",
    # Lifecycle
    "lifespan",
    "startup_handler",
    "shutdown_handler",
    "get_database_manager",
    # Unit of Work
    "UnitOfWork",
    "unit_of_work",
    # Cache
    "SimpleCache",
    "get_cache",
    "cached",
    # Performance Config
    "configure_pytorch_performance",
    "configure_transformers_performance",
    "get_optimal_batch_size",
    "configure_mixed_precision",
    # Async Operations
    "AsyncProcessor",
    "async_map",
    "run_async",
    # Interfaces
    "IChatRepository",
    "IVoteRepository",
    "IViewRepository",
    "IRemixRepository",
    "IRankingService",
    "IValidator",
    "IAIProcessor",
    "IScoreManager",
    "IIdentityService",
    # Dependency Container
    "DependencyContainer",
    "container",
    "ServiceScope",
    "ServiceRegistration",
    "singleton",
    "scoped",
    "transient",
    # Container Helpers
    "get_request_scope_id",
    "create_container_dependency",
    "create_type_based_dependency",
    "setup_request_scope_middleware",
    "register_database_session",
    "get_chat_service_dependency",
    "get_ranking_service_dependency",
    "get_identity_service_dependency",
]
