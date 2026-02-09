#!/usr/bin/env python3
"""
FastAPI Integration for Video-OpusClip
FastAPI dependency injection integration
"""

from typing import Dict, List, Any, Optional, Type, Callable, Union
from functools import wraps
import inspect

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .dependency_container import DependencyContainer, ServiceProvider


class FastAPIDependencyContainer(DependencyContainer):
    """FastAPI-specific dependency container"""
    
    def __init__(self):
        super().__init__()
        self._fastapi_dependencies: Dict[str, Callable] = {}
        self._dependency_overrides: Dict[str, Callable] = {}
    
    def register_fastapi_dependency(
        self,
        name: str,
        dependency_func: Callable,
        override: bool = False
    ) -> 'FastAPIDependencyContainer':
        """Register a FastAPI dependency"""
        if name in self._fastapi_dependencies and not override:
            raise ValueError(f"FastAPI dependency '{name}' already registered")
        
        self._fastapi_dependencies[name] = dependency_func
        return self
    
    def override_dependency(
        self,
        name: str,
        dependency_func: Callable
    ) -> 'FastAPIDependencyContainer':
        """Override a FastAPI dependency for testing"""
        self._dependency_overrides[name] = dependency_func
        return self
    
    def get_fastapi_dependency(self, name: str) -> Optional[Callable]:
        """Get a FastAPI dependency"""
        if name in self._dependency_overrides:
            return self._dependency_overrides[name]
        return self._fastapi_dependencies.get(name)
    
    def get_all_fastapi_dependencies(self) -> Dict[str, Callable]:
        """Get all FastAPI dependencies"""
        dependencies = self._fastapi_dependencies.copy()
        dependencies.update(self._dependency_overrides)
        return dependencies


class FastAPIServiceProvider(ServiceProvider):
    """FastAPI-specific service provider"""
    
    def __init__(self):
        super().__init__()
        self._fastapi_dependencies: Dict[str, Callable] = {}
    
    def register_fastapi_dependency(
        self,
        name: str,
        dependency_func: Callable
    ) -> 'FastAPIServiceProvider':
        """Register a FastAPI dependency"""
        self._fastapi_dependencies[name] = dependency_func
        return self
    
    def get_fastapi_dependency(self, name: str) -> Optional[Callable]:
        """Get a FastAPI dependency"""
        return self._fastapi_dependencies.get(name)


class FastAPIDependencyResolver:
    """FastAPI dependency resolver"""
    
    def __init__(self, container: FastAPIDependencyContainer):
        self.container = container
    
    def resolve_dependency(self, dependency_type: Type) -> Any:
        """Resolve a dependency from the container"""
        return self.container.resolve(dependency_type)
    
    def create_dependency_function(self, dependency_type: Type) -> Callable:
        """Create a FastAPI dependency function"""
        def dependency_func():
            return self.container.resolve(dependency_type)
        return dependency_func
    
    def create_async_dependency_function(self, dependency_type: Type) -> Callable:
        """Create an async FastAPI dependency function"""
        async def dependency_func():
            return self.container.resolve(dependency_type)
        return dependency_func


def get_fastapi_dependencies(container: FastAPIDependencyContainer) -> Dict[str, Callable]:
    """
    Get FastAPI dependencies from container
    
    Args:
        container: FastAPI dependency container
        
    Returns:
        Dictionary of FastAPI dependencies
    """
    return container.get_all_fastapi_dependencies()


def inject_fastapi_dependencies(
    container: FastAPIDependencyContainer,
    *dependency_types: Type
) -> List[Callable]:
    """
    Create FastAPI dependency injection functions
    
    Args:
        container: FastAPI dependency container
        *dependency_types: Types to inject
        
    Returns:
        List of FastAPI dependency functions
    """
    resolver = FastAPIDependencyResolver(container)
    dependencies = []
    
    for dep_type in dependency_types:
        # Check if the dependency type has async methods
        if hasattr(dep_type, '__call__'):
            sig = inspect.signature(dep_type.__call__)
            if any(param.annotation == inspect.Parameter.empty for param in sig.parameters.values()):
                # Use async dependency function
                dependencies.append(resolver.create_async_dependency_function(dep_type))
            else:
                # Use sync dependency function
                dependencies.append(resolver.create_dependency_function(dep_type))
        else:
            # Default to sync dependency function
            dependencies.append(resolver.create_dependency_function(dep_type))
    
    return dependencies


def create_fastapi_dependency(
    container: FastAPIDependencyContainer,
    dependency_type: Type,
    async_func: bool = False
) -> Callable:
    """
    Create a single FastAPI dependency
    
    Args:
        container: FastAPI dependency container
        dependency_type: Type to inject
        async_func: Whether to create async function
        
    Returns:
        FastAPI dependency function
    """
    resolver = FastAPIDependencyResolver(container)
    
    if async_func:
        return resolver.create_async_dependency_function(dependency_type)
    else:
        return resolver.create_dependency_function(dependency_type)


def fastapi_dependency(
    container: FastAPIDependencyContainer,
    dependency_type: Type,
    async_func: bool = False
):
    """
    Decorator for FastAPI dependency injection
    
    Args:
        container: FastAPI dependency container
        dependency_type: Type to inject
        async_func: Whether to create async function
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        dependency_func = create_fastapi_dependency(container, dependency_type, async_func)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            dependency = dependency_func()
            return func(*args, dependency=dependency, **kwargs)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            dependency = await dependency_func()
            return await func(*args, dependency=dependency, **kwargs)
        
        if async_func:
            return async_wrapper
        else:
            return wrapper
    
    return decorator


def create_auth_dependency(container: FastAPIDependencyContainer) -> Callable:
    """
    Create authentication dependency for FastAPI
    
    Args:
        container: FastAPI dependency container
        
    Returns:
        Authentication dependency function
    """
    security = HTTPBearer()
    
    async def auth_dependency(credentials: HTTPAuthorizationCredentials = Depends(security)):
        try:
            # Resolve auth service from container
            auth_service = container.resolve("AuthService")
            
            # Validate token
            token = credentials.credentials
            user = auth_service.validate_token(token)
            
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            return user
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    return auth_dependency


def create_permission_dependency(
    container: FastAPIDependencyContainer,
    required_permission: str
) -> Callable:
    """
    Create permission dependency for FastAPI
    
    Args:
        container: FastAPI dependency container
        required_permission: Required permission
        
    Returns:
        Permission dependency function
    """
    auth_dependency = create_auth_dependency(container)
    
    async def permission_dependency(user=Depends(auth_dependency)):
        try:
            # Resolve permission service from container
            permission_service = container.resolve("PermissionService")
            
            # Check permission
            has_permission = permission_service.check_permission(user, required_permission)
            
            if not has_permission:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission '{required_permission}' required"
                )
            
            return user
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission check failed"
            )
    
    return permission_dependency


def create_role_dependency(
    container: FastAPIDependencyContainer,
    required_role: str
) -> Callable:
    """
    Create role dependency for FastAPI
    
    Args:
        container: FastAPI dependency container
        required_role: Required role
        
    Returns:
        Role dependency function
    """
    auth_dependency = create_auth_dependency(container)
    
    async def role_dependency(user=Depends(auth_dependency)):
        try:
            # Resolve role service from container
            role_service = container.resolve("RoleService")
            
            # Check role
            has_role = role_service.check_role(user, required_role)
            
            if not has_role:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Role '{required_role}' required"
                )
            
            return user
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Role check failed"
            )
    
    return role_dependency


def create_logging_dependency(container: FastAPIDependencyContainer) -> Callable:
    """
    Create logging dependency for FastAPI
    
    Args:
        container: FastAPI dependency container
        
    Returns:
        Logging dependency function
    """
    def logging_dependency():
        try:
            # Resolve logging service from container
            logging_service = container.resolve("LoggingService")
            return logging_service
        except Exception as e:
            # Return a default logger if service is not available
            import logging
            return logging.getLogger(__name__)
    
    return logging_dependency


def create_cache_dependency(container: FastAPIDependencyContainer) -> Callable:
    """
    Create cache dependency for FastAPI
    
    Args:
        container: FastAPI dependency container
        
    Returns:
        Cache dependency function
    """
    def cache_dependency():
        try:
            # Resolve cache service from container
            cache_service = container.resolve("CacheService")
            return cache_service
        except Exception as e:
            # Return a default cache if service is not available
            return {"type": "default_cache"}
    
    return cache_dependency


def create_database_dependency(container: FastAPIDependencyContainer) -> Callable:
    """
    Create database dependency for FastAPI
    
    Args:
        container: FastAPI dependency container
        
    Returns:
        Database dependency function
    """
    async def database_dependency():
        try:
            # Resolve database session from container
            session = container.resolve("DatabaseSession")
            return session
        except Exception as e:
            # Return a default session if service is not available
            return {"type": "default_session"}
    
    return database_dependency


def create_storage_dependency(container: FastAPIDependencyContainer) -> Callable:
    """
    Create storage dependency for FastAPI
    
    Args:
        container: FastAPI dependency container
        
    Returns:
        Storage dependency function
    """
    def storage_dependency():
        try:
            # Resolve storage service from container
            storage_service = container.resolve("StorageService")
            return storage_service
        except Exception as e:
            # Return a default storage if service is not available
            return {"type": "default_storage"}
    
    return storage_dependency


def create_monitoring_dependency(container: FastAPIDependencyContainer) -> Callable:
    """
    Create monitoring dependency for FastAPI
    
    Args:
        container: FastAPI dependency container
        
    Returns:
        Monitoring dependency function
    """
    def monitoring_dependency():
        try:
            # Resolve monitoring service from container
            monitoring_service = container.resolve("MonitoringService")
            return monitoring_service
        except Exception as e:
            # Return a default monitoring if service is not available
            return {"type": "default_monitoring"}
    
    return monitoring_dependency


def setup_fastapi_dependencies(container: FastAPIDependencyContainer) -> Dict[str, Callable]:
    """
    Setup all FastAPI dependencies
    
    Args:
        container: FastAPI dependency container
        
    Returns:
        Dictionary of FastAPI dependencies
    """
    dependencies = {}
    
    # Register common dependencies
    dependencies["auth"] = create_auth_dependency(container)
    dependencies["logging"] = create_logging_dependency(container)
    dependencies["cache"] = create_cache_dependency(container)
    dependencies["database"] = create_database_dependency(container)
    dependencies["storage"] = create_storage_dependency(container)
    dependencies["monitoring"] = create_monitoring_dependency(container)
    
    # Register service dependencies
    service_types = [
        "AuthService", "JWTService", "PasswordService",
        "PermissionService", "RoleService", "UserService",
        "LoggingService", "CacheService", "QueueService",
        "StorageService", "MonitoringService"
    ]
    
    for service_type in service_types:
        try:
            dependencies[service_type.lower()] = create_fastapi_dependency(container, service_type)
        except Exception:
            # Skip if service is not registered
            pass
    
    return dependencies


# Example usage
if __name__ == "__main__":
    # Example FastAPI dependency setup
    print("🚀 FastAPI Integration Example")
    
    # Create FastAPI container
    container = FastAPIDependencyContainer()
    
    # Register services (simplified for example)
    container.register_singleton("AuthService", factory=lambda: {"type": "auth_service"})
    container.register_singleton("LoggingService", factory=lambda: {"type": "logging_service"})
    container.register_singleton("CacheService", factory=lambda: {"type": "cache_service"})
    container.register_singleton("StorageService", factory=lambda: {"type": "storage_service"})
    container.register_singleton("MonitoringService", factory=lambda: {"type": "monitoring_service"})
    
    # Setup FastAPI dependencies
    dependencies = setup_fastapi_dependencies(container)
    
    print(f"✅ FastAPI dependencies created: {list(dependencies.keys())}")
    
    # Test dependency creation
    try:
        auth_dep = create_auth_dependency(container)
        print(f"✅ Auth dependency created: {auth_dep}")
        
        logging_dep = create_logging_dependency(container)
        print(f"✅ Logging dependency created: {logging_dep}")
        
        cache_dep = create_cache_dependency(container)
        print(f"✅ Cache dependency created: {cache_dep}")
        
    except Exception as e:
        print(f"❌ Dependency creation failed: {e}")
    
    # Test dependency injection
    try:
        auth_service = container.resolve("AuthService")
        print(f"✅ Auth service resolved: {auth_service}")
        
        logging_service = container.resolve("LoggingService")
        print(f"✅ Logging service resolved: {logging_service}")
        
    except Exception as e:
        print(f"❌ Service resolution failed: {e}")
    
    print("✅ FastAPI integration example completed!") 