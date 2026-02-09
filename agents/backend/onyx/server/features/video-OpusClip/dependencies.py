"""
Improved Dependencies Module

Enhanced dependency injection with:
- Async dependency management
- Resource pooling and connection management
- Performance monitoring integration
- Error handling and fallback strategies
- Configuration management
- Security and authentication
"""

from __future__ import annotations
from typing import Any, Optional, Dict, List
from functools import lru_cache
from contextlib import asynccontextmanager
import asyncio
import time
import structlog
from fastapi import Depends, Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool, QueuePool

# Import processors and managers
from .processors import (
    VideoProcessor, VideoProcessorConfig,
    ViralVideoProcessor, ViralProcessorConfig,
    LangChainVideoProcessor, LangChainConfig,
    BatchVideoProcessor, BatchProcessorConfig
)
from .cache import CacheManager
from .monitoring import PerformanceMonitor, HealthChecker
from .models import ValidationResult

logger = structlog.get_logger("dependencies")

# Security
security = HTTPBearer(auto_error=False)

# =============================================================================
# CONFIGURATION
# =============================================================================

class DependencyConfig:
    """Configuration for dependency injection."""
    
    def __init__(self):
        self.video_processor_config = VideoProcessorConfig(
            max_workers=4,
            batch_size=5,
            enable_audit_logging=True,
            enable_performance_tracking=True
        )
        
        self.viral_processor_config = ViralProcessorConfig(
            max_variants=10,
            enable_langchain=True,
            enable_screen_division=True,
            enable_transitions=True,
            enable_effects=True
        )
        
        self.langchain_processor_config = LangChainConfig(
            model_name="gpt-4",
            enable_content_analysis=True,
            enable_engagement_analysis=True,
            enable_viral_analysis=True,
            enable_title_optimization=True,
            enable_caption_optimization=True,
            enable_timing_optimization=True,
            batch_size=5,
            max_retries=3,
            use_agents=True,
            use_memory=True
        )
        
        self.batch_processor_config = BatchProcessorConfig(
            max_workers=8,
            batch_size=10,
            enable_parallel_processing=True
        )

# Global configuration instance
config = DependencyConfig()

# =============================================================================
# DATABASE DEPENDENCIES
# =============================================================================

class DatabaseManager:
    """Database connection manager with connection pooling."""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = None
        self.session_factory = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize database connections."""
        if self._initialized:
            return
        
        try:
            # Create async engine with connection pooling
            self.engine = create_async_engine(
                self.database_url,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False
            )
            
            # Create session factory
            self.session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            self._initialized = True
            logger.info("Database manager initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize database manager", error=str(e))
            raise
    
    async def close(self):
        """Close database connections."""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connections closed")
    
    @asynccontextmanager
    async def get_session(self):
        """Get database session with automatic cleanup."""
        if not self._initialized:
            await self.initialize()
        
        session = self.session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

# Global database manager
db_manager = DatabaseManager("sqlite+aiosqlite:///./video_processing.db")

# =============================================================================
# PROCESSOR DEPENDENCIES
# =============================================================================

@lru_cache()
def get_video_processor() -> VideoProcessor:
    """Get video processor instance with caching."""
    try:
        processor = VideoProcessor(config.video_processor_config)
        logger.info("Video processor created successfully")
        return processor
    except Exception as e:
        logger.error("Failed to create video processor", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Video processor unavailable"
        )

@lru_cache()
def get_viral_processor() -> ViralVideoProcessor:
    """Get viral processor instance with caching."""
    try:
        processor = ViralVideoProcessor(config.viral_processor_config)
        logger.info("Viral processor created successfully")
        return processor
    except Exception as e:
        logger.error("Failed to create viral processor", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Viral processor unavailable"
        )

@lru_cache()
def get_langchain_processor() -> LangChainVideoProcessor:
    """Get LangChain processor instance with caching."""
    try:
        processor = LangChainVideoProcessor(config.langchain_processor_config)
        logger.info("LangChain processor created successfully")
        return processor
    except Exception as e:
        logger.error("Failed to create LangChain processor", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LangChain processor unavailable"
        )

@lru_cache()
def get_batch_processor() -> BatchVideoProcessor:
    """Get batch processor instance with caching."""
    try:
        processor = BatchVideoProcessor(config.batch_processor_config)
        logger.info("Batch processor created successfully")
        return processor
    except Exception as e:
        logger.error("Failed to create batch processor", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch processor unavailable"
        )

# =============================================================================
# CACHE DEPENDENCIES
# =============================================================================

class CacheDependency:
    """Cache dependency with fallback strategies."""
    
    def __init__(self):
        self._cache_manager: Optional[CacheManager] = None
        self._fallback_enabled = True
    
    async def get_cache_manager(self) -> CacheManager:
        """Get cache manager with fallback."""
        if self._cache_manager is None:
            try:
                self._cache_manager = CacheManager()
                await self._cache_manager.initialize()
                logger.info("Cache manager initialized successfully")
            except Exception as e:
                logger.warning("Failed to initialize cache manager", error=str(e))
                if not self._fallback_enabled:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Cache system unavailable"
                    )
                # Create a no-op cache manager as fallback
                self._cache_manager = NoOpCacheManager()
                logger.info("Using no-op cache manager as fallback")
        
        return self._cache_manager

# Global cache dependency
cache_dependency = CacheDependency()

async def get_cache_manager() -> CacheManager:
    """Get cache manager dependency."""
    return await cache_dependency.get_cache_manager()

# =============================================================================
# MONITORING DEPENDENCIES
# =============================================================================

class MonitoringDependency:
    """Monitoring dependency with performance tracking."""
    
    def __init__(self):
        self._performance_monitor: Optional[PerformanceMonitor] = None
        self._health_checker: Optional[HealthChecker] = None
    
    async def get_performance_monitor(self) -> PerformanceMonitor:
        """Get performance monitor."""
        if self._performance_monitor is None:
            self._performance_monitor = PerformanceMonitor()
            await self._performance_monitor.start()
            logger.info("Performance monitor initialized")
        return self._performance_monitor
    
    async def get_health_checker(self) -> HealthChecker:
        """Get health checker."""
        if self._health_checker is None:
            self._health_checker = HealthChecker()
            await self._health_checker.initialize()
            logger.info("Health checker initialized")
        return self._health_checker

# Global monitoring dependency
monitoring_dependency = MonitoringDependency()

async def get_performance_monitor() -> PerformanceMonitor:
    """Get performance monitor dependency."""
    return await monitoring_dependency.get_performance_monitor()

async def get_health_checker() -> HealthChecker:
    """Get health checker dependency."""
    return await monitoring_dependency.get_health_checker()

# =============================================================================
# DATABASE SESSION DEPENDENCY
# =============================================================================

async def get_db_session() -> AsyncSession:
    """Get database session dependency."""
    async with db_manager.get_session() as session:
        yield session

# =============================================================================
# AUTHENTICATION DEPENDENCIES
# =============================================================================

class AuthenticationManager:
    """Authentication manager with token validation."""
    
    def __init__(self):
        self.valid_tokens = {
            "admin": "admin_token_123",
            "user": "user_token_456",
            "api": "api_token_789"
        }
    
    async def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate authentication token."""
        for role, valid_token in self.valid_tokens.items():
            if token == valid_token:
                return {
                    "role": role,
                    "user_id": f"{role}_user",
                    "permissions": self._get_permissions(role)
                }
        return None
    
    def _get_permissions(self, role: str) -> List[str]:
        """Get permissions for role."""
        permissions = {
            "admin": ["read", "write", "delete", "admin"],
            "user": ["read", "write"],
            "api": ["read", "write"]
        }
        return permissions.get(role, ["read"])

# Global authentication manager
auth_manager = AuthenticationManager()

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[Dict[str, Any]]:
    """Get current authenticated user."""
    if not credentials:
        return None
    
    user_info = await auth_manager.validate_token(credentials.credentials)
    if not user_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    
    return user_info

async def require_auth(
    user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Require authentication for protected endpoints."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    return user

async def require_admin(
    user: Dict[str, Any] = Depends(require_auth)
) -> Dict[str, Any]:
    """Require admin privileges."""
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return user

# =============================================================================
# REQUEST CONTEXT DEPENDENCIES
# =============================================================================

def get_request_id(request: Request) -> str:
    """Get request ID from request state."""
    return getattr(request.state, 'request_id', 'unknown')

def get_client_ip(request: Request) -> str:
    """Get client IP address."""
    return request.client.host if request.client else 'unknown'

def get_user_agent(request: Request) -> str:
    """Get user agent from request headers."""
    return request.headers.get("user-agent", "unknown")

# =============================================================================
# VALIDATION DEPENDENCIES
# =============================================================================

class ValidationDependency:
    """Validation dependency with caching."""
    
    def __init__(self):
        self._validation_cache: Dict[str, ValidationResult] = {}
        self._cache_ttl = 300  # 5 minutes
    
    async def validate_video_request(self, request_data: Dict[str, Any]) -> ValidationResult:
        """Validate video request with caching."""
        cache_key = f"video_validation:{hash(str(request_data))}"
        
        # Check cache first
        if cache_key in self._validation_cache:
            return self._validation_cache[cache_key]
        
        # Perform validation
        validation_result = await self._perform_video_validation(request_data)
        
        # Cache result
        self._validation_cache[cache_key] = validation_result
        
        return validation_result
    
    async def _perform_video_validation(self, request_data: Dict[str, Any]) -> ValidationResult:
        """Perform actual video request validation."""
        errors = []
        warnings = []
        
        # Validate required fields
        if not request_data.get("youtube_url"):
            errors.append("YouTube URL is required")
        
        if not request_data.get("language"):
            errors.append("Language is required")
        
        # Validate URL format
        youtube_url = request_data.get("youtube_url", "")
        if youtube_url and not self._is_valid_youtube_url(youtube_url):
            errors.append("Invalid YouTube URL format")
        
        # Validate clip lengths
        min_length = request_data.get("min_clip_length", 15)
        max_length = request_data.get("max_clip_length", 60)
        
        if min_length > max_length:
            errors.append("Minimum clip length cannot be greater than maximum")
        
        if max_length > 600:
            warnings.append("Very long clip length may impact processing time")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _is_valid_youtube_url(self, url: str) -> bool:
        """Check if URL is a valid YouTube URL."""
        import re
        youtube_patterns = [
            r'https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+',
            r'https?://(?:www\.)?youtu\.be/[\w-]+',
            r'https?://(?:www\.)?youtube\.com/embed/[\w-]+'
        ]
        return any(re.match(pattern, url) for pattern in youtube_patterns)

# Global validation dependency
validation_dependency = ValidationDependency()

async def validate_video_request_data(request_data: Dict[str, Any]) -> ValidationResult:
    """Validate video request data dependency."""
    return await validation_dependency.validate_video_request(request_data)

# =============================================================================
# FALLBACK IMPLEMENTATIONS
# =============================================================================

class NoOpCacheManager:
    """No-operation cache manager for fallback scenarios."""
    
    async def get(self, key: str) -> Optional[Any]:
        """No-op get method."""
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """No-op set method."""
        pass
    
    async def delete(self, key: str) -> None:
        """No-op delete method."""
        pass
    
    async def clear(self) -> None:
        """No-op clear method."""
        pass
    
    async def initialize(self) -> None:
        """No-op initialize method."""
        pass
    
    async def close(self) -> None:
        """No-op close method."""
        pass

# =============================================================================
# DEPENDENCY HEALTH CHECK
# =============================================================================

class DependencyHealthChecker:
    """Check health of all dependencies."""
    
    async def check_all_dependencies(self) -> Dict[str, Any]:
        """Check health of all dependencies."""
        health_status = {
            "database": await self._check_database_health(),
            "cache": await self._check_cache_health(),
            "processors": await self._check_processors_health(),
            "monitoring": await self._check_monitoring_health()
        }
        
        overall_healthy = all(
            status.get("healthy", False) for status in health_status.values()
        )
        
        return {
            "overall_healthy": overall_healthy,
            "dependencies": health_status,
            "timestamp": time.time()
        }
    
    async def _check_database_health(self) -> Dict[str, Any]:
        """Check database health."""
        try:
            async with db_manager.get_session() as session:
                await session.execute("SELECT 1")
            return {"healthy": True, "message": "Database connection successful"}
        except Exception as e:
            return {"healthy": False, "message": f"Database error: {str(e)}"}
    
    async def _check_cache_health(self) -> Dict[str, Any]:
        """Check cache health."""
        try:
            cache_manager = await cache_dependency.get_cache_manager()
            await cache_manager.set("health_check", "ok", ttl=60)
            result = await cache_manager.get("health_check")
            return {"healthy": result == "ok", "message": "Cache system operational"}
        except Exception as e:
            return {"healthy": False, "message": f"Cache error: {str(e)}"}
    
    async def _check_processors_health(self) -> Dict[str, Any]:
        """Check processors health."""
        try:
            # Test processor creation
            video_processor = get_video_processor()
            viral_processor = get_viral_processor()
            langchain_processor = get_langchain_processor()
            batch_processor = get_batch_processor()
            
            return {"healthy": True, "message": "All processors available"}
        except Exception as e:
            return {"healthy": False, "message": f"Processor error: {str(e)}"}
    
    async def _check_monitoring_health(self) -> Dict[str, Any]:
        """Check monitoring health."""
        try:
            performance_monitor = await monitoring_dependency.get_performance_monitor()
            health_checker = await monitoring_dependency.get_health_checker()
            
            return {"healthy": True, "message": "Monitoring systems operational"}
        except Exception as e:
            return {"healthy": False, "message": f"Monitoring error: {str(e)}"}

# Global dependency health checker
dependency_health_checker = DependencyHealthChecker()

async def check_dependencies_health() -> Dict[str, Any]:
    """Check dependencies health dependency."""
    return await dependency_health_checker.check_all_dependencies()

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Configuration
    'DependencyConfig',
    'config',
    
    # Database
    'DatabaseManager',
    'db_manager',
    'get_db_session',
    
    # Processors
    'get_video_processor',
    'get_viral_processor',
    'get_langchain_processor',
    'get_batch_processor',
    
    # Cache
    'CacheDependency',
    'cache_dependency',
    'get_cache_manager',
    'NoOpCacheManager',
    
    # Monitoring
    'MonitoringDependency',
    'monitoring_dependency',
    'get_performance_monitor',
    'get_health_checker',
    
    # Authentication
    'AuthenticationManager',
    'auth_manager',
    'get_current_user',
    'require_auth',
    'require_admin',
    
    # Request context
    'get_request_id',
    'get_client_ip',
    'get_user_agent',
    
    # Validation
    'ValidationDependency',
    'validation_dependency',
    'validate_video_request_data',
    
    # Health checking
    'DependencyHealthChecker',
    'dependency_health_checker',
    'check_dependencies_health'
]
