"""
Module Registry for Blaze AI

Manages all modules, their dependencies, lifecycle, and provides a central
interface for module discovery and management.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Set, Type, Union
from dataclasses import dataclass, field

from .base import BaseModule, ModuleConfig, ModuleType, ModulePriority, ModuleStatus, HealthStatus

logger = logging.getLogger(__name__)

# ============================================================================
# REGISTRY DATACLASSES
# ============================================================================

@dataclass
class ModuleInfo:
    """Information about a registered module."""
    module: BaseModule
    config: ModuleConfig
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    registration_time: float = field(default_factory=time.time)
    last_health_check: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.config.name,
            "type": self.config.module_type.name,
            "priority": self.config.priority.name,
            "status": self.module.status.name,
            "dependencies": list(self.dependencies),
            "dependents": list(self.dependents),
            "registration_time": self.registration_time,
            "last_health_check": self.last_health_check,
            "enabled": self.config.enabled,
            "auto_start": self.config.auto_start
        }

@dataclass
class RegistryStats:
    """Statistics about the module registry."""
    total_modules: int = 0
    active_modules: int = 0
    error_modules: int = 0
    shutdown_modules: int = 0
    module_types: Dict[str, int] = field(default_factory=dict)
    total_dependencies: int = 0
    circular_dependencies: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_modules": self.total_modules,
            "active_modules": self.active_modules,
            "error_modules": self.error_modules,
            "shutdown_modules": self.shutdown_modules,
            "module_types": self.module_types,
            "total_dependencies": self.total_dependencies,
            "circular_dependencies": self.circular_dependencies
        }

# ============================================================================
# MAIN MODULE REGISTRY
# ============================================================================

class ModuleRegistry:
    """Central registry for managing all Blaze AI modules."""
    
    def __init__(self):
        self.modules: Dict[str, ModuleInfo] = {}
        self.module_types: Dict[ModuleType, List[str]] = {}
        self.dependency_graph: Dict[str, Set[str]] = {}
        
        # Background tasks
        self._health_monitoring_task: Optional[asyncio.Task] = None
        self._dependency_check_task: Optional[asyncio.Task] = None
        
        # Registry state
        self._initialized = False
        self._shutdown = False
        self._lock = asyncio.Lock()
        
        # Health monitoring interval
        self.health_check_interval = 30.0  # 30 seconds
        
        # Start background tasks
        self._start_background_tasks()
    
    # ============================================================================
    # MODULE REGISTRATION AND MANAGEMENT
    # ============================================================================
    
    async def register_module(self, module: BaseModule) -> bool:
        """Register a module in the registry."""
        async with self._lock:
            try:
                module_name = module.config.name
                
                if module_name in self.modules:
                    logger.warning(f"Module {module_name} is already registered")
                    return False
                
                # Create module info
                module_info = ModuleInfo(
                    module=module,
                    config=module.config,
                    dependencies=module.config.dependencies.copy()
                )
                
                # Register module
                self.modules[module_name] = module_info
                
                # Update module type index
                module_type = module.config.module_type
                if module_type not in self.module_types:
                    self.module_types[module_type] = []
                self.module_types[module_type].append(module_name)
                
                # Update dependency graph
                self._update_dependency_graph(module_name, module_info.dependencies)
                
                # Check for circular dependencies
                if self._has_circular_dependencies():
                    logger.warning(f"Circular dependencies detected for module {module_name}")
                
                logger.info(f"Registered module: {module_name} ({module_type.name})")
                
                # Auto-start if configured
                if module.config.auto_start and not self._shutdown:
                    await self._start_module(module_name)
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to register module {module.config.name}: {e}")
                return False
    
    async def unregister_module(self, module_name: str) -> bool:
        """Unregister a module from the registry."""
        async with self._lock:
            try:
                if module_name not in self.modules:
                    logger.warning(f"Module {module_name} is not registered")
                    return False
                
                module_info = self.modules[module_name]
                
                # Shutdown module if active
                if module_info.module.status != ModuleStatus.SHUTDOWN:
                    await module_info.module.shutdown()
                
                # Remove from module type index
                module_type = module_info.config.module_type
                if module_type in self.module_types:
                    self.module_types[module_type].remove(module_name)
                    if not self.module_types[module_type]:
                        del self.module_types[module_type]
                
                # Update dependency graph
                self._remove_from_dependency_graph(module_name)
                
                # Remove module
                del self.modules[module_name]
                
                logger.info(f"Unregistered module: {module_name}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to unregister module {module_name}: {e}")
                return False
    
    def get_module(self, module_name: str) -> Optional[BaseModule]:
        """Get a module by name."""
        module_info = self.modules.get(module_name)
        return module_info.module if module_info else None
    
    def get_modules_by_type(self, module_type: ModuleType) -> List[BaseModule]:
        """Get all modules of a specific type."""
        module_names = self.module_types.get(module_type, [])
        return [self.modules[name].module for name in module_names if name in self.modules]
    
    def get_module_info(self, module_name: str) -> Optional[ModuleInfo]:
        """Get module information."""
        return self.modules.get(module_name)
    
    def list_modules(self) -> List[str]:
        """List all registered module names."""
        return list(self.modules.keys())
    
    def list_modules_by_type(self, module_type: ModuleType) -> List[str]:
        """List module names of a specific type."""
        return self.module_types.get(module_type, [])
    
    # ============================================================================
    # MODULE LIFECYCLE MANAGEMENT
    # ============================================================================
    
    async def start_module(self, module_name: str) -> bool:
        """Start a specific module."""
        async with self._lock:
            return await self._start_module(module_name)
    
    async def _start_module(self, module_name: str) -> bool:
        """Internal method to start a module."""
        try:
            if module_name not in self.modules:
                logger.error(f"Module {module_name} not found")
                return False
            
            module_info = self.modules[module_name]
            
            # Check dependencies
            if not await self._check_module_dependencies(module_name):
                logger.warning(f"Module {module_name} dependencies not ready")
                return False
            
            # Start module
            success = await module_info.module.initialize()
            
            if success:
                logger.info(f"Started module: {module_name}")
            else:
                logger.error(f"Failed to start module: {module_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error starting module {module_name}: {e}")
            return False
    
    async def stop_module(self, module_name: str) -> bool:
        """Stop a specific module."""
        async with self._lock:
            try:
                if module_name not in self.modules:
                    logger.error(f"Module {module_name} not found")
                    return False
                
                module_info = self.modules[module_name]
                success = await module_info.module.shutdown()
                
                if success:
                    logger.info(f"Stopped module: {module_name}")
                else:
                    logger.warning(f"Module {module_name} shutdown had issues")
                
                return success
                
            except Exception as e:
                logger.error(f"Error stopping module {module_name}: {e}")
                return False
    
    async def start_all_modules(self) -> Dict[str, bool]:
        """Start all enabled modules."""
        results = {}
        
        async with self._lock:
            for module_name in self.modules:
                if self.modules[module_name].config.enabled:
                    results[module_name] = await self._start_module(module_name)
        
        return results
    
    async def stop_all_modules(self) -> Dict[str, bool]:
        """Stop all modules."""
        results = {}
        
        async with self._lock:
            for module_name in self.modules:
                results[module_name] = await self.stop_module(module_name)
        
        return results
    
    # ============================================================================
    # DEPENDENCY MANAGEMENT
    # ============================================================================
    
    async def _check_module_dependencies(self, module_name: str) -> bool:
        """Check if all dependencies for a module are ready."""
        module_info = self.modules[module_name]
        
        for dep_name in module_info.dependencies:
            if dep_name not in self.modules:
                logger.warning(f"Dependency {dep_name} not found for module {module_name}")
                return False
            
            dep_module = self.modules[dep_name].module
            if not dep_module.is_ready():
                logger.debug(f"Dependency {dep_name} not ready for module {module_name}")
                return False
        
        return True
    
    def _update_dependency_graph(self, module_name: str, dependencies: Set[str]):
        """Update the dependency graph."""
        # Add module to graph
        if module_name not in self.dependency_graph:
            self.dependency_graph[module_name] = set()
        
        # Add dependencies
        for dep in dependencies:
            if dep not in self.dependency_graph:
                self.dependency_graph[dep] = set()
            self.dependency_graph[dep].add(module_name)
    
    def _remove_from_dependency_graph(self, module_name: str):
        """Remove a module from the dependency graph."""
        # Remove as dependent
        for deps in self.dependency_graph.values():
            deps.discard(module_name)
        
        # Remove module entry
        if module_name in self.dependency_graph:
            del self.dependency_graph[module_name]
    
    def _has_circular_dependencies(self) -> bool:
        """Check for circular dependencies using DFS."""
        visited = set()
        rec_stack = set()
        
        def has_cycle(node: str) -> bool:
            if node in rec_stack:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self.dependency_graph.get(node, set()):
                if has_cycle(neighbor):
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in self.dependency_graph:
            if has_cycle(node):
                return True
        
        return False
    
    def get_dependency_tree(self, module_name: str) -> Dict[str, Any]:
        """Get the dependency tree for a module."""
        if module_name not in self.modules:
            return {}
        
        def build_tree(name: str, visited: Set[str]) -> Dict[str, Any]:
            if name in visited:
                return {"name": name, "circular": True}
            
            visited.add(name)
            
            if name not in self.modules:
                return {"name": name, "error": "Module not found"}
            
            module_info = self.modules[name]
            dependencies = []
            
            for dep in module_info.dependencies:
                dep_tree = build_tree(dep, visited.copy())
                dependencies.append(dep_tree)
            
            return {
                "name": name,
                "type": module_info.config.module_type.name,
                "status": module_info.module.status.name,
                "dependencies": dependencies
            }
        
        return build_tree(module_name, set())
    
    # ============================================================================
    # HEALTH MONITORING
    # ============================================================================
    
    def _start_background_tasks(self):
        """Start background monitoring tasks."""
        # Health monitoring
        async def health_monitoring():
            while not self._shutdown:
                try:
                    await asyncio.sleep(self.health_check_interval)
                    await self._check_all_modules_health()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Health monitoring error: {e}")
                    await asyncio.sleep(5.0)
        
        self._health_monitoring_task = asyncio.create_task(health_monitoring())
        
        # Dependency checking
        async def dependency_checking():
            while not self._shutdown:
                try:
                    await asyncio.sleep(60.0)  # Check every minute
                    await self._check_dependencies()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Dependency checking error: {e}")
                    await asyncio.sleep(5.0)
        
        self._dependency_check_task = asyncio.create_task(dependency_checking())
    
    async def _check_all_modules_health(self):
        """Check health of all modules."""
        for module_name, module_info in self.modules.items():
            try:
                health_status = await module_info.module.get_health_status()
                module_info.last_health_check = time.time()
                
                # Log health issues
                if health_status.status == ModuleStatus.ERROR:
                    logger.warning(f"Module {module_name} health check failed: {health_status.message}")
                
            except Exception as e:
                logger.error(f"Error checking health of module {module_name}: {e}")
    
    async def _check_dependencies(self):
        """Check and update dependency status."""
        for module_name in self.modules:
            try:
                dependencies_ready = await self._check_module_dependencies(module_name)
                if dependencies_ready:
                    # Try to start module if it's not active
                    module_info = self.modules[module_name]
                    if (module_info.module.status == ModuleStatus.UNINITIALIZED and 
                        module_info.config.enabled and 
                        module_info.config.auto_start):
                        await self._start_module(module_name)
            except Exception as e:
                logger.error(f"Error checking dependencies for module {module_name}: {e}")
    
    # ============================================================================
    # REGISTRY OPERATIONS
    # ============================================================================
    
    async def initialize(self):
        """Initialize the module registry."""
        if self._initialized:
            return
        
        try:
            logger.info("Initializing Module Registry")
            
            # Start all enabled modules
            start_results = await self.start_all_modules()
            
            successful_starts = sum(1 for success in start_results.values() if success)
            total_modules = len(start_results)
            
            logger.info(f"Module Registry initialized: {successful_starts}/{total_modules} modules started")
            
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize Module Registry: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the module registry."""
        if self._shutdown:
            return
        
        try:
            logger.info("Shutting down Module Registry")
            
            self._shutdown = True
            
            # Stop all modules
            stop_results = await self.stop_all_modules()
            
            # Cancel background tasks
            if self._health_monitoring_task:
                self._health_monitoring_task.cancel()
            
            if self._dependency_check_task:
                self._dependency_check_task.cancel()
            
            successful_stops = sum(1 for success in stop_results.values() if success)
            total_modules = len(stop_results)
            
            logger.info(f"Module Registry shutdown completed: {successful_stops}/{total_modules} modules stopped")
            
        except Exception as e:
            logger.error(f"Error during Module Registry shutdown: {e}")
    
    # ============================================================================
    # STATISTICS AND REPORTING
    # ============================================================================
    
    def get_registry_stats(self) -> RegistryStats:
        """Get statistics about the registry."""
        stats = RegistryStats()
        
        # Count modules by status
        for module_info in self.modules.values():
            stats.total_modules += 1
            
            if module_info.module.status == ModuleStatus.ACTIVE:
                stats.active_modules += 1
            elif module_info.module.status == ModuleStatus.ERROR:
                stats.error_modules += 1
            elif module_info.module.status == ModuleStatus.SHUTDOWN:
                stats.shutdown_modules += 1
        
        # Count by module type
        for module_type, module_names in self.module_types.items():
            stats.module_types[module_type.name] = len(module_names)
        
        # Count dependencies
        stats.total_dependencies = sum(len(info.dependencies) for info in self.modules.values())
        
        # Check for circular dependencies
        stats.circular_dependencies = 1 if self._has_circular_dependencies() else 0
        
        return stats
    
    def get_registry_status(self) -> Dict[str, Any]:
        """Get comprehensive registry status."""
        stats = self.get_registry_stats()
        
        return {
            "initialized": self._initialized,
            "shutdown": self._shutdown,
            "stats": stats.to_dict(),
            "modules": {
                name: info.to_dict() for name, info in self.modules.items()
            },
            "module_types": {
                module_type.name: module_names 
                for module_type, module_names in self.module_types.items()
            }
        }
    
    # ============================================================================
    # CONTEXT MANAGER SUPPORT
    # ============================================================================
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()

# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_module_registry() -> ModuleRegistry:
    """Create a new module registry."""
    return ModuleRegistry()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

async def register_and_start_module(registry: ModuleRegistry, module: BaseModule) -> bool:
    """Register and start a module in one operation."""
    try:
        # Register module
        if not await registry.register_module(module):
            return False
        
        # Start module
        return await registry.start_module(module.config.name)
        
    except Exception as e:
        logger.error(f"Failed to register and start module {module.config.name}: {e}")
        return False

async def create_and_register_module(
    registry: ModuleRegistry,
    module_class: Type[BaseModule],
    config: ModuleConfig
) -> Optional[BaseModule]:
    """Create, register, and start a module."""
    try:
        # Create module
        module = module_class(config)
        
        # Register and start
        if await register_and_start_module(registry, module):
            return module
        else:
            return None
        
    except Exception as e:
        logger.error(f"Failed to create and register module: {e}")
        return None
