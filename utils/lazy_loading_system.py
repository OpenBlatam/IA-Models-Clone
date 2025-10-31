from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import time
import logging
import weakref
import threading
from typing import Any, Optional, Dict, List, Callable, Awaitable, Set, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import gc
from pydantic import BaseModel, Field
            import sys
from typing import Any, List, Dict, Optional
"""
🔄 Advanced Lazy Loading System
===============================

Comprehensive lazy loading system with:
- Dependency management
- Circular dependency detection
- Performance optimization
- Resource pooling
- Memory management
- Load balancing
"""



logger = logging.getLogger(__name__)


class LoadState(Enum):
    """Resource load states"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"
    UNLOADING = "unloading"


class LoadPriority(Enum):
    """Load priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class LoadMetrics:
    """Load performance metrics"""
    load_count: int = 0
    unload_count: int = 0
    total_load_time: float = 0.0
    total_unload_time: float = 0.0
    average_load_time: float = 0.0
    average_unload_time: float = 0.0
    errors: int = 0
    memory_usage: int = 0
    
    def update_load(self, load_time: float, error: bool = False):
        """Update load metrics"""
        self.load_count += 1
        self.total_load_time += load_time
        self.average_load_time = self.total_load_time / self.load_count
        if error:
            self.errors += 1
    
    def update_unload(self, unload_time: float):
        """Update unload metrics"""
        self.unload_count += 1
        self.total_unload_time += unload_time
        self.average_unload_time = self.total_unload_time / self.unload_count


@dataclass
class ResourceInfo:
    """Resource information"""
    name: str
    loader_func: Callable
    dependencies: List[str] = field(default_factory=list)
    priority: LoadPriority = LoadPriority.NORMAL
    state: LoadState = LoadState.UNLOADED
    load_time: Optional[float] = None
    unload_time: Optional[float] = None
    memory_usage: int = 0
    access_count: int = 0
    last_access: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DependencyGraph:
    """
    Dependency graph for managing resource dependencies
    and detecting circular dependencies.
    """
    
    def __init__(self) -> Any:
        self.nodes: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_nodes: Dict[str, Set[str]] = defaultdict(set)
        self.node_info: Dict[str, ResourceInfo] = {}
        
    def add_resource(self, name: str, dependencies: List[str] = None):
        """Add resource to dependency graph"""
        dependencies = dependencies or []
        
        # Add node
        self.nodes[name] = set(dependencies)
        
        # Add reverse edges
        for dep in dependencies:
            self.reverse_nodes[dep].add(name)
        
        # Check for circular dependencies
        if self._has_circular_dependency(name):
            raise ValueError(f"Circular dependency detected for resource: {name}")
    
    def remove_resource(self, name: str):
        """Remove resource from dependency graph"""
        # Remove forward edges
        if name in self.nodes:
            del self.nodes[name]
        
        # Remove reverse edges
        for reverse_deps in self.reverse_nodes.values():
            reverse_deps.discard(name)
        
        # Clean up empty reverse nodes
        self.reverse_nodes = {
            k: v for k, v in self.reverse_nodes.items() if v
        }
        
        # Remove node info
        self.node_info.pop(name, None)
    
    def get_dependencies(self, name: str) -> Set[str]:
        """Get direct dependencies of a resource"""
        return self.nodes.get(name, set()).copy()
    
    def get_dependents(self, name: str) -> Set[str]:
        """Get resources that depend on this resource"""
        return self.reverse_nodes.get(name, set()).copy()
    
    def get_all_dependencies(self, name: str) -> Set[str]:
        """Get all dependencies (transitive) of a resource"""
        visited = set()
        deps = set()
        
        def dfs(node) -> Any:
            if node in visited:
                return
            visited.add(node)
            
            for dep in self.nodes.get(node, set()):
                deps.add(dep)
                dfs(dep)
        
        dfs(name)
        return deps
    
    def get_load_order(self, name: str) -> List[str]:
        """Get optimal load order for a resource and its dependencies"""
        all_deps = self.get_all_dependencies(name)
        all_deps.add(name)
        
        # Topological sort
        in_degree = defaultdict(int)
        for node in all_deps:
            for dep in self.nodes.get(node, set()):
                if dep in all_deps:
                    in_degree[node] += 1
        
        queue = deque([node for node in all_deps if in_degree[node] == 0])
        load_order = []
        
        while queue:
            node = queue.popleft()
            load_order.append(node)
            
            for dependent in self.reverse_nodes.get(node, set()):
                if dependent in all_deps:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)
        
        return load_order
    
    def _has_circular_dependency(self, start_node: str) -> bool:
        """Detect circular dependencies using DFS"""
        visited = set()
        rec_stack = set()
        
        def dfs(node) -> Any:
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self.nodes.get(node, set()):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        return dfs(start_node)


class ResourcePool:
    """
    Resource pool for managing loaded resources with memory limits
    and intelligent eviction.
    """
    
    def __init__(self, max_memory_mb: int = 1024, max_resources: int = 100):
        
    """__init__ function."""
self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.max_resources = max_resources
        self.resources: Dict[str, Any] = {}
        self.resource_info: Dict[str, ResourceInfo] = {}
        self.memory_usage = 0
        self.access_order = deque()
        
        # Thread safety
        self._lock = threading.RLock()
    
    def add_resource(self, name: str, resource: Any, info: ResourceInfo):
        """Add resource to pool"""
        with self._lock:
            # Estimate memory usage
            estimated_memory = self._estimate_memory_usage(resource)
            
            # Check memory limits
            if self.memory_usage + estimated_memory > self.max_memory_bytes:
                self._evict_resources(estimated_memory)
            
            # Add resource
            self.resources[name] = resource
            self.resource_info[name] = info
            self.memory_usage += estimated_memory
            
            # Update access order
            if name in self.access_order:
                self.access_order.remove(name)
            self.access_order.append(name)
            
            logger.debug(f"Added resource {name} to pool (memory: {estimated_memory} bytes)")
    
    def get_resource(self, name: str) -> Optional[Any]:
        """Get resource from pool"""
        with self._lock:
            if name in self.resources:
                # Update access order
                if name in self.access_order:
                    self.access_order.remove(name)
                self.access_order.append(name)
                
                # Update access count
                if name in self.resource_info:
                    self.resource_info[name].access_count += 1
                    self.resource_info[name].last_access = time.time()
                
                return self.resources[name]
            return None
    
    def remove_resource(self, name: str) -> Optional[Any]:
        """Remove resource from pool"""
        with self._lock:
            if name in self.resources:
                resource = self.resources.pop(name)
                info = self.resource_info.pop(name, None)
                
                if info:
                    self.memory_usage -= self._estimate_memory_usage(resource)
                
                if name in self.access_order:
                    self.access_order.remove(name)
                
                logger.debug(f"Removed resource {name} from pool")
                return resource
            return None
    
    def _evict_resources(self, required_memory: int):
        """Evict resources to free memory"""
        while (self.memory_usage + required_memory > self.max_memory_bytes and 
               len(self.resources) > 0):
            
            # Find least recently used resource
            if self.access_order:
                lru_name = self.access_order[0]
                self.remove_resource(lru_name)
            else:
                # Fallback: remove first resource
                first_name = next(iter(self.resources))
                self.remove_resource(first_name)
    
    def _estimate_memory_usage(self, resource: Any) -> int:
        """Estimate memory usage of a resource"""
        try:
            return sys.getsizeof(resource)
        except:
            # Fallback estimation
            return 1024  # 1KB default
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        with self._lock:
            return {
                "total_resources": len(self.resources),
                "max_resources": self.max_resources,
                "memory_usage_mb": self.memory_usage / (1024 * 1024),
                "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
                "memory_usage_percent": (self.memory_usage / self.max_memory_bytes) * 100
            }


class AdvancedLazyLoader:
    """
    Advanced lazy loader with dependency management, resource pooling,
    and intelligent loading strategies.
    """
    
    def __init__(self, max_memory_mb: int = 1024, max_resources: int = 100):
        
    """__init__ function."""
self.dependency_graph = DependencyGraph()
        self.resource_pool = ResourcePool(max_memory_mb, max_resources)
        self.loading_futures: Dict[str, asyncio.Future] = {}
        self.metrics = LoadMetrics()
        
        # Background tasks
        self.cleanup_task = None
        self.monitoring_task = None
        
        # Configuration
        self.auto_cleanup = True
        self.cleanup_interval = 300  # 5 minutes
        self.monitoring_interval = 60  # 1 minute
        
        # Thread safety
        self._lock = threading.RLock()
    
    async def initialize(self) -> Any:
        """Initialize lazy loader"""
        # Start background tasks
        if self.auto_cleanup:
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Advanced lazy loader initialized")
    
    async def cleanup(self) -> Any:
        """Cleanup lazy loader"""
        # Cancel background tasks
        if self.cleanup_task:
            self.cleanup_task.cancel()
        if self.monitoring_task:
            self.monitoring_task.cancel()
        
        # Wait for tasks to complete
        if self.cleanup_task:
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self.monitoring_task:
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Advanced lazy loader cleaned up")
    
    def register_resource(self, name: str, loader_func: Callable,
                         dependencies: List[str] = None,
                         priority: LoadPriority = LoadPriority.NORMAL,
                         metadata: Dict[str, Any] = None):
        """Register a resource for lazy loading"""
        with self._lock:
            # Create resource info
            info = ResourceInfo(
                name=name,
                loader_func=loader_func,
                dependencies=dependencies or [],
                priority=priority,
                metadata=metadata or {}
            )
            
            # Add to dependency graph
            self.dependency_graph.add_resource(name, dependencies)
            self.dependency_graph.node_info[name] = info
            
            logger.debug(f"Registered resource: {name}")
    
    async def load(self, name: str, force_reload: bool = False) -> Any:
        """Load a resource with dependency resolution"""
        start_time = time.time()
        
        # Check if already loaded
        if not force_reload:
            resource = self.resource_pool.get_resource(name)
            if resource is not None:
                return resource
        
        # Check if already loading
        if name in self.loading_futures:
            try:
                return await self.loading_futures[name]
            except Exception as e:
                logger.error(f"Error loading resource {name}: {e}")
                raise
        
        # Get load order
        try:
            load_order = self.dependency_graph.get_load_order(name)
        except ValueError as e:
            logger.error(f"Dependency error for {name}: {e}")
            raise
        
        # Load dependencies first
        for dep_name in load_order[:-1]:  # Exclude the target resource
            if dep_name != name:
                await self._load_dependency(dep_name)
        
        # Load the target resource
        return await self._load_resource(name, start_time)
    
    async def _load_dependency(self, name: str):
        """Load a dependency"""
        # Check if already loaded
        if self.resource_pool.get_resource(name) is not None:
            return
        
        # Check if already loading
        if name in self.loading_futures:
            await self.loading_futures[name]
            return
        
        # Load the dependency
        await self._load_resource(name, time.time())
    
    async def _load_resource(self, name: str, start_time: float) -> Any:
        """Actually load a resource"""
        # Create loading future
        future = asyncio.Future()
        self.loading_futures[name] = future
        
        try:
            # Get resource info
            info = self.dependency_graph.node_info.get(name)
            if not info:
                raise ValueError(f"Resource {name} not registered")
            
            # Update state
            info.state = LoadState.LOADING
            
            # Execute loader function
            if asyncio.iscoroutinefunction(info.loader_func):
                resource = await info.loader_func()
            else:
                # Run sync loader in thread pool
                loop = asyncio.get_event_loop()
                resource = await loop.run_in_executor(None, info.loader_func)
            
            # Add to resource pool
            info.state = LoadState.LOADED
            info.load_time = time.time() - start_time
            info.memory_usage = self.resource_pool._estimate_memory_usage(resource)
            
            self.resource_pool.add_resource(name, resource, info)
            
            # Update metrics
            self.metrics.update_load(info.load_time)
            
            # Set future result
            future.set_result(resource)
            
            logger.info(f"Loaded resource {name} in {info.load_time:.3f}s")
            return resource
            
        except Exception as e:
            # Update state and metrics
            if info:
                info.state = LoadState.ERROR
                info.error_message = str(e)
            
            self.metrics.update_load(time.time() - start_time, error=True)
            
            # Set future exception
            future.set_exception(e)
            
            logger.error(f"Failed to load resource {name}: {e}")
            raise
        
        finally:
            # Remove from loading futures
            self.loading_futures.pop(name, None)
    
    async def unload(self, name: str) -> bool:
        """Unload a resource"""
        start_time = time.time()
        
        with self._lock:
            # Check if resource is loaded
            resource = self.resource_pool.get_resource(name)
            if resource is None:
                return False
            
            # Update state
            info = self.dependency_graph.node_info.get(name)
            if info:
                info.state = LoadState.UNLOADING
            
            # Remove from pool
            self.resource_pool.remove_resource(name)
            
            # Update state
            if info:
                info.state = LoadState.UNLOADED
                info.unload_time = time.time() - start_time
            
            # Update metrics
            self.metrics.update_unload(time.time() - start_time)
            
            logger.info(f"Unloaded resource: {name}")
            return True
    
    async def unload_dependents(self, name: str):
        """Unload all resources that depend on the given resource"""
        dependents = self.dependency_graph.get_dependents(name)
        
        unload_tasks = []
        for dep_name in dependents:
            task = asyncio.create_task(self.unload(dep_name))
            unload_tasks.append(task)
        
        if unload_tasks:
            await asyncio.gather(*unload_tasks, return_exceptions=True)
            logger.info(f"Unloaded {len(unload_tasks)} dependent resources")
    
    def get_loaded_resources(self) -> List[str]:
        """Get list of loaded resources"""
        return list(self.resource_pool.resources.keys())
    
    def get_resource_info(self, name: str) -> Optional[ResourceInfo]:
        """Get information about a resource"""
        return self.dependency_graph.node_info.get(name)
    
    async def _cleanup_loop(self) -> Any:
        """Background loop for resource cleanup"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                # Find resources to cleanup (unused for a long time)
                current_time = time.time()
                cleanup_threshold = 1800  # 30 minutes
                
                resources_to_cleanup = []
                for name, info in self.dependency_graph.node_info.items():
                    if (info.state == LoadState.LOADED and 
                        info.last_access and 
                        current_time - info.last_access > cleanup_threshold):
                        resources_to_cleanup.append(name)
                
                # Cleanup resources
                for name in resources_to_cleanup:
                    await self.unload(name)
                
                if resources_to_cleanup:
                    logger.info(f"Cleaned up {len(resources_to_cleanup)} unused resources")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    async def _monitoring_loop(self) -> Any:
        """Background loop for monitoring"""
        while True:
            try:
                await asyncio.sleep(self.monitoring_interval)
                
                # Update memory usage
                pool_stats = self.resource_pool.get_stats()
                self.metrics.memory_usage = pool_stats["memory_usage_mb"]
                
                # Log statistics periodically
                if self.metrics.load_count % 10 == 0:  # Every 10 loads
                    logger.debug(f"Lazy loader stats: {self.get_stats()}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get lazy loader statistics"""
        return {
            "metrics": {
                "load_count": self.metrics.load_count,
                "unload_count": self.metrics.unload_count,
                "average_load_time": self.metrics.average_load_time,
                "average_unload_time": self.metrics.average_unload_time,
                "errors": self.metrics.errors,
                "memory_usage_mb": self.metrics.memory_usage
            },
            "pool_stats": self.resource_pool.get_stats(),
            "loaded_resources": len(self.get_loaded_resources()),
            "loading_resources": len(self.loading_futures),
            "registered_resources": len(self.dependency_graph.node_info)
        }


# Decorators for easy lazy loading
def lazy_load_resource(name: str, dependencies: List[str] = None,
                      priority: LoadPriority = LoadPriority.NORMAL):
    """Decorator for lazy loading resources"""
    def decorator(loader_func: Callable) -> Callable:
        # Store metadata in function
        loader_func._lazy_load_info = {
            "name": name,
            "dependencies": dependencies or [],
            "priority": priority
        }
        return loader_func
    return decorator


def lazy_load_dependencies(dependencies: List[str]):
    """Decorator for functions that require lazy-loaded dependencies"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # This would need access to a global lazy loader instance
            # For now, just return the function
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# Example usage
async def example_usage():
    """Example of how to use the advanced lazy loading system"""
    
    # Create lazy loader
    lazy_loader = AdvancedLazyLoader(max_memory_mb=512, max_resources=50)
    await lazy_loader.initialize()
    
    # Register resources
    def load_database_connection():
        
    """load_database_connection function."""
return {"type": "database", "connection": "active"}
    
    def load_cache_manager():
        
    """load_cache_manager function."""
return {"type": "cache", "manager": "active"}
    
    def load_config_service():
        
    """load_config_service function."""
return {"type": "config", "service": "active"}
    
    lazy_loader.register_resource(
        "database", load_database_connection,
        dependencies=[]
    )
    
    lazy_loader.register_resource(
        "cache", load_cache_manager,
        dependencies=["database"]
    )
    
    lazy_loader.register_resource(
        "config", load_config_service,
        dependencies=["database"]
    )
    
    # Load resources
    db = await lazy_loader.load("database")
    cache = await lazy_loader.load("cache")
    config = await lazy_loader.load("config")
    
    # Get statistics
    stats = lazy_loader.get_stats()
    print("Lazy Loader Stats:", stats)
    
    # Cleanup
    await lazy_loader.cleanup()


match __name__:
    case "__main__":
    asyncio.run(example_usage()) 