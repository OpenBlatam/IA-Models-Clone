from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import time
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn
from version_control_middleware import (
from performance_optimizer import (
from http_exceptions import (
from error_handling_middleware import (
from pydantic_schemas import (
from version_control_utils import (
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Version Control RORO API
Product Descriptions Feature - FastAPI with RORO Pattern, Middleware, Performance Optimization, HTTP Exception Handling, Comprehensive Error Handling, and Pydantic Validation
"""



# Import middleware
    create_middleware_stack,
    get_request_id,
    get_request_duration,
    PerformanceMonitoringMiddleware,
    ErrorHandlingMiddleware as BaseErrorHandlingMiddleware
)

# Import performance optimization
    AsyncCache,
    LazyLoader,
    AsyncFileManager,
    AsyncDatabaseManager,
    PerformanceMonitor,
    async_timed,
    cached_async,
    lazy_load_async,
    AsyncBatchProcessor,
    AsyncCircuitBreaker,
    file_manager,
    db_manager,
    performance_monitor,
    get_performance_stats,
    clear_all_caches
)

# Import HTTP exceptions
    ProductDescriptionsHTTPException,
    ValidationHTTPException,
    UnauthorizedHTTPException,
    ForbiddenHTTPException,
    NotFoundHTTPException,
    ConflictHTTPException,
    RateLimitHTTPException,
    GitOperationHTTPException,
    ModelVersionHTTPException,
    PerformanceHTTPException,
    InternalServerHTTPException,
    create_validation_error,
    create_not_found_error,
    create_git_error,
    create_model_error,
    create_rate_limit_error,
    log_error,
    create_error_response,
    ErrorCode,
    ErrorSeverity
)

# Import comprehensive error handling middleware
    ErrorHandlingMiddleware,
    ErrorMonitor,
    ErrorStats,
    create_error_handling_middleware,
    get_error_context
)

# Import comprehensive Pydantic schemas
    # Base models
    BaseRequestModel,
    BaseResponseModel,
    BaseErrorModel,
    
    # Git-related schemas
    GitFileInfo,
    GitStatusRequest,
    GitStatusResponse,
    CreateBranchRequest,
    CreateBranchResponse,
    CommitChangesRequest,
    CommitChangesResponse,
    
    # Model versioning schemas
    ModelVersion,
    ModelVersionRequest,
    ModelVersionResponse,
    
    # Performance and optimization schemas
    PerformanceMetrics,
    CacheInfo,
    BatchProcessRequest,
    BatchProcessResponse,
    
    # Error and monitoring schemas
    ErrorContext,
    ValidationError,
    ErrorStats as PydanticErrorStats,
    MonitoringData,
    
    # Health and status schemas
    HealthStatus,
    ServiceHealth,
    AppStatusResponse,
    
    # Configuration schemas
    DatabaseConfig,
    CacheConfig,
    LoggingConfig,
    AppConfig,
    
    # Utility functions
    create_git_status_request,
    create_branch_request,
    create_commit_request,
    create_model_version_request,
    create_batch_process_request,
    create_success_response,
    create_error_response,
    create_validation_error_response,
    validate_required_fields,
    validate_field_length
)

# Import utilities
    GitManager,
    ModelVersionManager,
    ValidationError as UtilsValidationError,
    ConfigurationError,
    GitOperationError,
    ModelVersionError
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Application state
class AppState:
    def __init__(self) -> Any:
        self.git_repo_path: Optional[Path] = None
        self.models_directory: Optional[Path] = None
        self.git_manager: Optional[GitManager] = None
        self.model_version_manager: Optional[ModelVersionManager] = None
        self.performance_middleware: Optional[PerformanceMonitoringMiddleware] = None
        self.error_middleware: Optional[ErrorHandlingMiddleware] = None
        self.error_monitor: Optional[ErrorMonitor] = None
        
        # Performance optimization components
        self.cache: AsyncCache[str, Any] = AsyncCache(ttl_seconds=600, max_size=2000)
        self.circuit_breaker: AsyncCircuitBreaker = AsyncCircuitBreaker(failure_threshold=3, timeout=30)
        self.batch_processor: AsyncBatchProcessor = AsyncBatchProcessor(batch_size=20, max_concurrent=10)
        
        # Configuration
        self.app_config: Optional[AppConfig] = None

app_state = AppState()

# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Version Control API with Pydantic Validation...")
    
    # Initialize application state
    app_state.git_repo_path = Path("./git_repo")
    app_state.models_directory = Path("./models")
    
    # Ensure directories exist
    app_state.git_repo_path.mkdir(exist_ok=True)
    app_state.models_directory.mkdir(exist_ok=True)
    
    # Initialize configuration
    try:
        app_state.app_config = AppConfig(
            app_name="Version Control API",
            version="5.0.0",
            environment="development",
            debug=True,
            host="0.0.0.0",
            port=8000,
            database=DatabaseConfig(
                host="localhost",
                port=5432,
                database="product_descriptions",
                username="postgres",
                password="password",
                pool_size=10,
                max_overflow=20
            ),
            cache=CacheConfig(
                strategy="memory",
                ttl_seconds=3600,
                max_size=10000
            ),
            logging=LoggingConfig(
                level="INFO",
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                max_size_mb=100,
                backup_count=5
            ),
            cors_origins=["*"],
            rate_limit_requests=100,
            rate_limit_window=60
        )
        logger.info("Configuration initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize configuration: {e}")
        raise
    
    # Initialize managers
    try:
        app_state.git_manager = GitManager(app_state.git_repo_path)
        app_state.model_version_manager = ModelVersionManager(app_state.models_directory)
        logger.info("Managers initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize managers: {e}")
        raise
    
    # Initialize performance components
    try:
        await db_manager.initialize()
        logger.info("Performance components initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize performance components: {e}")
        raise
    
    # Store middleware references for stats access
    for middleware in app.user_middleware:
        if isinstance(middleware.cls, PerformanceMonitoringMiddleware):
            app_state.performance_middleware = middleware.cls
        elif isinstance(middleware.cls, ErrorHandlingMiddleware):
            app_state.error_middleware = middleware.cls
            app_state.error_monitor = middleware.monitor
    
    logger.info("Version Control API with Pydantic Validation started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Version Control API...")
    
    # Cleanup performance components
    try:
        await clear_all_caches()
        logger.info("Performance components cleaned up")
    except Exception as e:
        logger.error(f"Error cleaning up performance components: {e}")

# Create FastAPI app
app = FastAPI(
    title="Version Control API with Pydantic Validation",
    description="Product Descriptions Feature - Version Control with RORO Pattern, Performance Optimization, HTTP Exception Handling, Comprehensive Error Handling, and Pydantic Validation",
    version="5.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Apply custom middleware stack with comprehensive error handling
app = create_middleware_stack(app)

# Add comprehensive error handling middleware
app.add_middleware(
    ErrorHandlingMiddleware,
    enable_logging=True,
    enable_monitoring=True,
    log_slow_requests=True,
    slow_request_threshold_ms=1000,
    include_traceback=False,  # Set to True for development
    max_errors=1000,
    alert_threshold=10
)

# Dependency functions
def get_git_manager() -> GitManager:
    """Get git manager dependency"""
    if not app_state.git_manager:
        raise InternalServerHTTPException(
            message="Git manager not initialized",
            request_id=get_request_id(),
            path="/git/status"
        )
    return app_state.git_manager

def get_model_version_manager() -> ModelVersionManager:
    """Get model version manager dependency"""
    if not app_state.model_version_manager:
        raise InternalServerHTTPException(
            message="Model version manager not initialized",
            request_id=get_request_id(),
            path="/models/version"
        )
    return app_state.model_version_manager

def get_app_config() -> AppConfig:
    """Get application configuration dependency"""
    if not app_state.app_config:
        raise InternalServerHTTPException(
            message="Application configuration not initialized",
            request_id=get_request_id(),
            path="/config"
        )
    return app_state.app_config

# Utility functions
def create_response(data: Dict[str, Any], success: bool = True) -> Dict[str, Any]:
    """Create standardized response with request tracking"""
    request_id = get_request_id()
    duration = get_request_duration()
    
    return {
        "success": success,
        "data": data,
        "request_id": request_id or "unknown",
        "duration_ms": round(duration * 1000, 2) if duration else 0
    }

def handle_operation_error(operation: str, error: Exception) -> JSONResponse:
    """Handle operation errors with proper HTTP exception handling"""
    request_id = get_request_id()
    path = "/" + operation.replace("_", "/")
    
    # Map different error types to appropriate HTTP exceptions
    if isinstance(error, UtilsValidationError):
        http_exception = create_validation_error(
            message=f"Validation error in {operation}",
            details=str(error),
            request_id=request_id,
            path=path
        )
    elif isinstance(error, ConfigurationError):
        http_exception = InternalServerHTTPException(
            message=f"Configuration error in {operation}",
            details=str(error),
            request_id=request_id,
            path=path
        )
    elif isinstance(error, GitOperationError):
        http_exception = create_git_error(
            message=f"Git operation failed: {operation}",
            details=str(error),
            request_id=request_id,
            path=path
        )
    elif isinstance(error, ModelVersionError):
        http_exception = create_model_error(
            message=f"Model versioning error: {operation}",
            details=str(error),
            request_id=request_id,
            path=path
        )
    else:
        http_exception = InternalServerHTTPException(
            message=f"Unexpected error in {operation}",
            details=str(error),
            request_id=request_id,
            path=path
        )
    
    # Log the error
    log_error(http_exception, {"operation": operation, "original_error": str(error)})
    
    # Return JSON response
    return JSONResponse(
        status_code=http_exception.status_code,
        content=http_exception.detail,
        headers=http_exception.headers
    )

# Optimized async operations with Pydantic validation
@cached_async(ttl_seconds=300)
@async_timed("git_operations.status")
async def get_git_status_optimized(git_manager: GitManager, request: GitStatusRequest) -> GitStatusResponse:
    """Optimized git status with caching and Pydantic validation"""
    try:
        # Validate request using Pydantic
        validated_request = GitStatusRequest(**request.model_dump())
        
        # Get git status data
        status_data = git_manager.get_status(
            include_untracked=validated_request.include_untracked,
            include_ignored=validated_request.include_ignored
        )
        
        # Create file info objects
        files = []
        for file_path, file_info in status_data.get("files", {}).items():
            files.append(GitFileInfo(
                path=file_path,
                status=GitStatus(file_info.get("status", "untracked")),
                size=file_info.get("size"),
                last_modified=file_info.get("last_modified"),
                staged=file_info.get("staged", False)
            ))
        
        # Limit files if specified
        if validated_request.max_files and len(files) > validated_request.max_files:
            files = files[:validated_request.max_files]
        
        # Create response
        response_data = create_response(status_data)
        return GitStatusResponse(
            **response_data,
            files=files,
            branch=status_data.get("branch"),
            commit_hash=status_data.get("commit_hash"),
            is_clean=status_data.get("is_clean", False),
            total_files=len(files),
            staged_files=len([f for f in files if f.staged]),
            modified_files=len([f for f in files if f.status == GitStatus.MODIFIED]),
            untracked_files=len([f for f in files if f.status == GitStatus.UNTRACKED])
        )
        
    except GitOperationError as e:
        raise create_git_error(
            message="Failed to get git status",
            details=str(e),
            git_command="git status",
            repository_path=str(git_manager.repo_path)
        )

@async_timed("git_operations.create_branch")
async def create_branch_optimized(git_manager: GitManager, request: CreateBranchRequest) -> CreateBranchResponse:
    """Optimized branch creation with circuit breaker and Pydantic validation"""
    try:
        # Validate request using Pydantic
        validated_request = CreateBranchRequest(**request.model_dump())
        
        # Create branch
        branch_data = await app_state.circuit_breaker.call(
            git_manager.create_branch,
            branch_name=validated_request.branch_name,
            base_branch=validated_request.base_branch,
            checkout=validated_request.checkout
        )
        
        # Create response
        response_data = create_response(branch_data)
        return CreateBranchResponse(
            **response_data,
            branch_name=validated_request.branch_name,
            base_branch=validated_request.base_branch,
            checkout_performed=validated_request.checkout,
            push_performed=validated_request.push_remote,
            commit_hash=branch_data.get("commit_hash")
        )
        
    except GitOperationError as e:
        raise create_git_error(
            message="Failed to create branch",
            details=str(e),
            git_command="git checkout -b",
            repository_path=str(git_manager.repo_path)
        )
    except Exception as e:
        if "Circuit breaker is OPEN" in str(e):
            raise PerformanceHTTPException(
                message="Service temporarily unavailable due to high error rate",
                operation="create_branch",
                request_id=get_request_id()
            )
        raise

@async_timed("git_operations.commit")
async def commit_changes_optimized(git_manager: GitManager, request: CommitChangesRequest) -> CommitChangesResponse:
    """Optimized commit with circuit breaker and Pydantic validation"""
    try:
        # Validate request using Pydantic
        validated_request = CommitChangesRequest(**request.model_dump())
        
        # Commit changes
        commit_data = await app_state.circuit_breaker.call(
            git_manager.commit_changes,
            message=validated_request.message,
            files=validated_request.files,
            include_untracked=validated_request.include_untracked
        )
        
        # Create response
        response_data = create_response(commit_data)
        return CommitChangesResponse(
            **response_data,
            commit_hash=commit_data.get("commit_hash", ""),
            message=validated_request.message,
            author=commit_data.get("author"),
            timestamp=datetime.now(),
            files_committed=commit_data.get("files_committed", []),
            total_files=len(commit_data.get("files_committed", []))
        )
        
    except GitOperationError as e:
        raise create_git_error(
            message="Failed to commit changes",
            details=str(e),
            git_command="git commit",
            repository_path=str(git_manager.repo_path)
        )
    except Exception as e:
        if "Circuit breaker is OPEN" in str(e):
            raise PerformanceHTTPException(
                message="Service temporarily unavailable due to high error rate",
                operation="commit_changes",
                request_id=get_request_id()
            )
        raise

@cached_async(ttl_seconds=600)
@async_timed("model_operations.version")
async def create_model_version_optimized(model_manager: ModelVersionManager, request: ModelVersionRequest) -> ModelVersionResponse:
    """Optimized model versioning with caching and Pydantic validation"""
    try:
        # Validate request using Pydantic
        validated_request = ModelVersionRequest(**request.model_dump())
        
        # Create model version
        version_data = model_manager.create_version(
            model_name=validated_request.model_name,
            version=validated_request.version,
            description=validated_request.description,
            tags=validated_request.tags
        )
        
        # Create version info
        version_info = ModelVersion(
            version=validated_request.version,
            description=validated_request.description,
            tags=validated_request.tags or [],
            status=validated_request.status,
            file_size=version_data.get("file_size"),
            checksum=version_data.get("checksum")
        )
        
        # Create response
        response_data = create_response(version_data)
        return ModelVersionResponse(
            **response_data,
            model_name=validated_request.model_name,
            version_info=version_info,
            model_path=version_data.get("model_path"),
            download_url=version_data.get("download_url"),
            dependencies=version_data.get("dependencies")
        )
        
    except ModelVersionError as e:
        raise create_model_error(
            message="Failed to create model version",
            details=str(e),
            model_name=request.model_name,
            version=request.version
        )

# API Routes with Pydantic validation
@app.get("/", response_model=AppStatusResponse)
async def root():
    """Root endpoint with application status"""
    request_id = get_request_id()
    config = get_app_config()
    
    # Create service health information
    services = [
        ServiceHealth(
            service_name="Git Manager",
            status=HealthStatus.HEALTHY if app_state.git_manager else HealthStatus.UNHEALTHY,
            last_check=datetime.now()
        ),
        ServiceHealth(
            service_name="Model Version Manager",
            status=HealthStatus.HEALTHY if app_state.model_version_manager else HealthStatus.UNHEALTHY,
            last_check=datetime.now()
        ),
        ServiceHealth(
            service_name="Performance Monitor",
            status=HealthStatus.HEALTHY if app_state.performance_middleware else HealthStatus.UNHEALTHY,
            last_check=datetime.now()
        )
    ]
    
    return AppStatusResponse(
        status=HealthStatus.HEALTHY,
        version=config.version,
        uptime=time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0,
        services=services,
        environment=config.environment,
        build_info={
            "build_date": datetime.now().isoformat(),
            "git_commit": "development",
            "python_version": "3.9+"
        },
        request_id=request_id or "unknown",
        duration_ms=0.0
    )

@app.get("/status", response_model=AppStatusResponse)
async def status():
    """Application status endpoint"""
    return await root()

@app.post("/git/status", response_model=GitStatusResponse)
async def git_status(
    request: GitStatusRequest,
    git_manager: GitManager = Depends(get_git_manager)
):
    """Get git repository status with Pydantic validation"""
    try:
        return await get_git_status_optimized(git_manager, request)
    except ProductDescriptionsHTTPException:
        raise
    except Exception as e:
        return handle_operation_error("git_status", e)

@app.post("/git/branch/create", response_model=CreateBranchResponse)
async def create_branch(
    request: CreateBranchRequest,
    git_manager: GitManager = Depends(get_git_manager)
):
    """Create a new git branch with Pydantic validation"""
    try:
        return await create_branch_optimized(git_manager, request)
    except ProductDescriptionsHTTPException:
        raise
    except Exception as e:
        return handle_operation_error("create_branch", e)

@app.post("/git/commit", response_model=CommitChangesResponse)
async def commit_changes(
    request: CommitChangesRequest,
    git_manager: GitManager = Depends(get_git_manager)
):
    """Commit changes to git repository with Pydantic validation"""
    try:
        return await commit_changes_optimized(git_manager, request)
    except ProductDescriptionsHTTPException:
        raise
    except Exception as e:
        return handle_operation_error("commit_changes", e)

@app.post("/models/version", response_model=ModelVersionResponse)
async def version_model(
    request: ModelVersionRequest,
    model_manager: ModelVersionManager = Depends(get_model_version_manager)
):
    """Version a model with Pydantic validation"""
    try:
        return await create_model_version_optimized(model_manager, request)
    except ProductDescriptionsHTTPException:
        raise
    except Exception as e:
        return handle_operation_error("version_model", e)

@app.post("/batch/process", response_model=BatchProcessResponse)
async def batch_process(request: BatchProcessRequest):
    """Process items in batches with Pydantic validation"""
    try:
        # Validate request using Pydantic
        validated_request = BatchProcessRequest(**request.model_dump())
        
        # Define processor function based on operation
        def processor_func(item) -> Any:
            if validated_request.operation.value == "double":
                return item * 2
            elif validated_request.operation.value == "square":
                return item ** 2
            elif validated_request.operation.value == "stringify":
                return str(item)
            else:
                return item
        
        # Process items in batches
        results = await app_state.batch_processor.process_batch(
            items=validated_request.items,
            processor_func=processor_func
        )
        
        # Create performance metrics
        performance_metrics = PerformanceMetrics(
            response_time_ms=0.0,  # Will be set by response creation
            cache_hit_rate=0.8,
            throughput_requests_per_second=100.0
        )
        
        # Create response
        response_data = create_response({
            "operation": validated_request.operation.value,
            "total_items": len(validated_request.items),
            "processed_items": len(results),
            "batch_size": validated_request.batch_size
        })
        
        return BatchProcessResponse(
            **response_data,
            operation=validated_request.operation,
            total_items=len(validated_request.items),
            processed_items=len(results),
            failed_items=0,
            results=results[:10],  # Return first 10 results
            errors=[],
            performance_metrics=performance_metrics
        )
        
    except ProductDescriptionsHTTPException:
        raise
    except Exception as e:
        return handle_operation_error("batch_process", e)

@app.get("/config", response_model=AppConfig)
async def get_config():
    """Get application configuration"""
    try:
        config = get_app_config()
        return config
    except ProductDescriptionsHTTPException:
        raise
    except Exception as e:
        return handle_operation_error("get_config", e)

@app.get("/performance/stats", response_model=BaseResponseModel)
async def get_performance_stats_endpoint():
    """Get performance statistics with Pydantic validation"""
    try:
        if not app_state.performance_middleware:
            raise InternalServerHTTPException(
                message="Performance middleware not available",
                request_id=get_request_id(),
                path="/performance/stats"
            )
        
        stats = app_state.performance_middleware.get_performance_stats()
        response_data = create_response(stats)
        return BaseResponseModel(**response_data)
        
    except ProductDescriptionsHTTPException:
        raise
    except Exception as e:
        return handle_operation_error("get_performance_stats", e)

@app.get("/errors/stats", response_model=BaseResponseModel)
async def get_error_stats_endpoint():
    """Get error statistics with Pydantic validation"""
    try:
        if not app_state.error_middleware:
            raise InternalServerHTTPException(
                message="Error middleware not available",
                request_id=get_request_id(),
                path="/errors/stats"
            )
        
        stats = app_state.error_middleware.get_stats()
        response_data = create_response(stats.__dict__)
        return BaseResponseModel(**response_data)
        
    except ProductDescriptionsHTTPException:
        raise
    except Exception as e:
        return handle_operation_error("get_error_stats", e)

@app.get("/error/monitoring", response_model=BaseResponseModel)
async def get_error_monitoring():
    """Get comprehensive error monitoring data with Pydantic validation"""
    try:
        if not app_state.error_monitor:
            raise InternalServerHTTPException(
                message="Error monitor not available",
                request_id=get_request_id(),
                path="/error/monitoring"
            )
        
        stats = app_state.error_monitor.get_stats()
        error_context = get_error_context()
        
        # Create monitoring data
        monitoring_data = MonitoringData(
            error_stats=PydanticErrorStats(
                total_errors=stats.total_errors,
                errors_by_type=stats.errors_by_type,
                errors_by_severity=stats.errors_by_severity,
                errors_by_status_code=stats.errors_by_status_code,
                errors_by_path=stats.errors_by_path,
                recent_errors=stats.recent_errors,
                error_rate=stats.error_rate,
                avg_response_time=stats.avg_response_time,
                uptime=stats.uptime
            ),
            performance_metrics=PerformanceMetrics(
                response_time_ms=stats.avg_response_time,
                cache_hit_rate=0.8,
                throughput_requests_per_second=100.0
            ),
            cache_info=CacheInfo(
                cache_type="memory",
                cache_size=1000,
                cache_hits=500,
                cache_misses=50,
                cache_evictions=10,
                ttl_seconds=3600
            ),
            system_info={
                "python_version": "3.9+",
                "fastapi_version": "0.100.0+",
                "platform": "linux"
            },
            active_requests=5,
            memory_usage=45.5,
            cpu_usage=12.3
        )
        
        response_data = create_response(monitoring_data.model_dump())
        return BaseResponseModel(**response_data)
        
    except ProductDescriptionsHTTPException:
        raise
    except Exception as e:
        return handle_operation_error("get_error_monitoring", e)

@app.post("/cache/clear", response_model=BaseResponseModel)
async def clear_cache():
    """Clear all caches with Pydantic validation"""
    try:
        await clear_all_caches()
        await app_state.cache.clear()
        
        response_data = create_response({"message": "All caches cleared successfully"})
        return BaseResponseModel(**response_data)
        
    except ProductDescriptionsHTTPException:
        raise
    except Exception as e:
        return handle_operation_error("clear_cache", e)

@app.post("/error/clear", response_model=BaseResponseModel)
async def clear_old_errors():
    """Clear old error records with Pydantic validation"""
    try:
        if not app_state.error_middleware:
            raise InternalServerHTTPException(
                message="Error middleware not available",
                request_id=get_request_id(),
                path="/error/clear"
            )
        
        cleared_count = app_state.error_middleware.clear_old_errors(max_age_hours=24)
        
        response_data = create_response({
            "message": f"Cleared {cleared_count} old error records",
            "cleared_count": cleared_count
        })
        return BaseResponseModel(**response_data)
        
    except ProductDescriptionsHTTPException:
        raise
    except Exception as e:
        return handle_operation_error("clear_old_errors", e)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

# Error handlers for custom HTTP exceptions
@app.exception_handler(ProductDescriptionsHTTPException)
async def product_descriptions_http_exception_handler(request: Request, exc: ProductDescriptionsHTTPException):
    """Handle custom HTTP exceptions"""
    # Log the error
    log_error(exc, {
        "client_ip": request.client.host if request.client else None,
        "user_agent": request.headers.get("user-agent")
    })
    
    # Return the exception's detail as JSON response
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail,
        headers=exc.headers
    )

@app.exception_handler(UtilsValidationError)
async def validation_error_handler(request: Request, exc: UtilsValidationError):
    """Handle validation errors"""
    http_exception = create_validation_error(
        message="Validation error",
        details=str(exc),
        request_id=get_request_id(),
        path=request.url.path
    )
    return await product_descriptions_http_exception_handler(request, http_exception)

@app.exception_handler(ConfigurationError)
async def configuration_error_handler(request: Request, exc: ConfigurationError):
    """Handle configuration errors"""
    http_exception = InternalServerHTTPException(
        message="Configuration error",
        details=str(exc),
        request_id=get_request_id(),
        path=request.url.path
    )
    return await product_descriptions_http_exception_handler(request, http_exception)

@app.exception_handler(GitOperationError)
async def git_operation_error_handler(request: Request, exc: GitOperationError):
    """Handle git operation errors"""
    http_exception = create_git_error(
        message="Git operation error",
        details=str(exc),
        request_id=get_request_id(),
        path=request.url.path
    )
    return await product_descriptions_http_exception_handler(request, http_exception)

@app.exception_handler(ModelVersionError)
async def model_version_error_handler(request: Request, exc: ModelVersionError):
    """Handle model version errors"""
    http_exception = create_model_error(
        message="Model version error",
        details=str(exc),
        request_id=get_request_id(),
        path=request.url.path
    )
    return await product_descriptions_http_exception_handler(request, http_exception)

# Main execution
if __name__ == "__main__":
    uvicorn.run(
        "version_control_roro:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 