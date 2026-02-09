"""
Unified Infrastructure package for the ads feature.

This package consolidates all infrastructure functionality from the scattered implementations:
- db_service.py (basic database operations)
- optimized_db_service.py (production database with connection pooling)
- storage.py (basic file storage)
- optimized_storage.py (production storage with caching)
- scalable_api_patterns.py (cache management, rate limiting)

The new structure follows Clean Architecture principles with clear separation of concerns.
"""

try:
    from .database import (
        AdsRepository,
        CampaignRepository,
        GroupRepository,
        PerformanceRepository,
        AnalyticsRepository,
        OptimizationRepository,
        DatabaseManager,
        ConnectionPool,
        DatabaseConfig
    )
except Exception:  # pragma: no cover - optionalize heavy deps during tests
    AdsRepository = CampaignRepository = GroupRepository = PerformanceRepository = AnalyticsRepository = OptimizationRepository = object  # type: ignore
    DatabaseManager = ConnectionPool = DatabaseConfig = object  # type: ignore
try:
    from .storage import (
        StorageService,
        FileStorageManager,
        StorageConfig,
        StorageStrategy,
        LocalStorageStrategy,
        CloudStorageStrategy
    )
except Exception:  # pragma: no cover - optional in tests
    StorageService = FileStorageManager = StorageConfig = StorageStrategy = LocalStorageStrategy = CloudStorageStrategy = object  # type: ignore
try:
    from .cache import (
        CacheService,
        CacheManager,
        CacheStrategy,
        RedisCacheStrategy,
        MemoryCacheStrategy,
        CacheConfig
    )
except Exception:  # pragma: no cover - optional in tests
    CacheService = CacheManager = CacheStrategy = RedisCacheStrategy = MemoryCacheStrategy = CacheConfig = object  # type: ignore
try:
    from .external_services import (
        ExternalServiceManager,
        AIProviderService,
        AnalyticsService,
        NotificationService,
        ExternalServiceConfig
    )
except Exception:  # pragma: no cover - optional in tests
    ExternalServiceManager = AIProviderService = AnalyticsService = NotificationService = ExternalServiceConfig = object  # type: ignore
try:
    from .version_control import (
        VersionControlManager,
        VersionControlService,
        GitStatus,
        GitCommit,
        VersionInfo,
        ExperimentVersion
    )
except Exception:  # pragma: no cover - optional in tests
    VersionControlManager = VersionControlService = GitStatus = GitCommit = VersionInfo = ExperimentVersion = object  # type: ignore
try:
    from .project_management import (
        ProjectManager,
        ProjectInitializer,
        ProjectType,
        DatasetType,
        ProblemComplexity,
        DatasetInfo,
        ProblemDefinition,
        ProjectStructure,
        ProjectConfig
    )
except Exception:  # pragma: no cover - optional in tests
    ProjectManager = ProjectInitializer = ProjectType = DatasetType = ProblemComplexity = DatasetInfo = ProblemDefinition = ProjectStructure = ProjectConfig = object  # type: ignore
try:
    from .langchain_integration import (
        LangChainService,
        LangChainConfig,
        ContentAnalysis,
        ContentVariation,
        PerformanceMetrics
    )
except Exception:  # pragma: no cover - optional in tests
    LangChainService = LangChainConfig = ContentAnalysis = ContentVariation = PerformanceMetrics = object  # type: ignore
try:
    from .repositories import (
        AdsRepositoryImpl,
        CampaignRepositoryImpl,
        GroupRepositoryImpl,
        PerformanceRepositoryImpl,
        AnalyticsRepositoryImpl,
        OptimizationRepositoryImpl
    )
except Exception:  # pragma: no cover - optional in tests
    AdsRepositoryImpl = CampaignRepositoryImpl = GroupRepositoryImpl = PerformanceRepositoryImpl = AnalyticsRepositoryImpl = OptimizationRepositoryImpl = object  # type: ignore

__all__ = [
    # Database
    "AdsRepository",
    "CampaignRepository", 
    "GroupRepository",
    "PerformanceRepository",
    "AnalyticsRepository",
    "OptimizationRepository",
    "DatabaseManager",
    "ConnectionPool",
    "DatabaseConfig",
    
    # Storage
    "StorageService",
    "FileStorageManager",
    "StorageConfig",
    "StorageStrategy",
    "LocalStorageStrategy",
    "CloudStorageStrategy",
    
    # Cache
    "CacheService",
    "CacheManager",
    "CacheStrategy",
    "RedisCacheStrategy",
    "MemoryCacheStrategy",
    "CacheConfig",
    
                    # External Services
                "ExternalServiceManager",
                "AIProviderService",
                "AnalyticsService",
                "NotificationService",
                "ExternalServiceConfig",

                # Version Control
                "VersionControlManager",
                "VersionControlService",
                "GitStatus",
                "GitCommit",
                "VersionInfo",
                "ExperimentVersion",

                # Project Management
                "ProjectManager",
                "ProjectInitializer",
                "ProjectType",
                "DatasetType",
                "ProblemComplexity",
                "DatasetInfo",
                "ProblemDefinition",
                "ProjectStructure",
                "ProjectConfig",

                # LangChain Integration
                "LangChainService",
                "LangChainConfig",
                "ContentAnalysis",
                "ContentVariation",
                "PerformanceMetrics",
    
    # Repository Implementations
    "AdsRepositoryImpl",
    "CampaignRepositoryImpl",
    "GroupRepositoryImpl",
    "PerformanceRepositoryImpl",
    "AnalyticsRepositoryImpl",
    "OptimizationRepositoryImpl"
]
