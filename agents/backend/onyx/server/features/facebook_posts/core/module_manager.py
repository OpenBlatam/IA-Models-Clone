#!/usr/bin/env python3
"""
Module Manager - Ultra-Modular Architecture v3.7
Manages loading, coordination, and lifecycle of all system modules
"""
import os
import sys
import json
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Any, Optional, Type, Callable
from datetime import datetime
import threading
import logging

from .base_system import BaseModule, ModuleConfig, ModuleState, ModulePriority, create_module_config

logger = logging.getLogger(__name__)

class ModuleManager:
    """
    Central manager for all system modules
    Handles module discovery, loading, dependency resolution, and coordination
    """
    
    def __init__(self, modules_path: str = "modules"):
        """Initialize module manager"""
        self.modules_path = Path(modules_path)
        self.modules: Dict[str, BaseModule] = {}
        self.module_classes: Dict[str, Type[BaseModule]] = {}
        self.module_configs: Dict[str, ModuleConfig] = {}
        
        # Module lifecycle management
        self._running_modules: List[str] = []
        self._stopped_modules: List[str] = []
        self._error_modules: List[str] = []
        
        # Event system
        self._global_event_callbacks: Dict[str, List[Callable]] = {}
        self._module_event_callbacks: Dict[str, Dict[str, List[Callable]]] = {}
        
        # Threading
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()
        
        # Statistics
        self._stats = {
            'total_modules': 0,
            'loaded_modules': 0,
            'running_modules': 0,
            'error_modules': 0,
            'start_time': datetime.now()
        }
        
        # Initialize
        self._discover_modules()
    
    def _discover_modules(self):
        """Discover available modules in the modules directory"""
        try:
            if not self.modules_path.exists():
                logger.warning(f"Modules directory not found: {self.modules_path}")
                return
            
            logger.info(f"Discovering modules in: {self.modules_path}")
            
            # Scan for Python files
            for py_file in self.modules_path.rglob("*.py"):
                if py_file.name.startswith("__"):
                    continue
                
                try:
                    # Import module
                    module_name = py_file.stem
                    module_path = str(py_file.parent)
                    
                    if module_path not in sys.path:
                        sys.path.insert(0, module_path)
                    
                    # Import the module
                    module = importlib.import_module(module_name)
                    
                    # Look for BaseModule subclasses
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, BaseModule) and 
                            obj != BaseModule):
                            
                            self.module_classes[module_name] = obj
                            logger.info(f"Discovered module class: {name} in {module_name}")
                            
                except Exception as e:
                    logger.warning(f"Error discovering module {py_file}: {e}")
            
            logger.info(f"Discovered {len(self.module_classes)} module classes")
            
        except Exception as e:
            logger.error(f"Error during module discovery: {e}")
    
    def load_module(self, module_name: str, config: Optional[ModuleConfig] = None) -> Optional[BaseModule]:
        """Load a specific module"""
        try:
            if module_name not in self.module_classes:
                logger.error(f"Module class not found: {module_name}")
                return None
            
            if module_name in self.modules:
                logger.warning(f"Module already loaded: {module_name}")
                return self.modules[module_name]
            
            # Create default config if none provided
            if config is None:
                config = create_module_config(
                    name=module_name,
                    version="1.0.0",
                    description=f"Module {module_name}",
                    auto_start=False
                )
            
            # Create module instance
            module_class = self.module_classes[module_name]
            module = module_class(config)
            
            # Store module
            self.modules[module_name] = module
            self.module_configs[module_name] = config
            
            # Set up event handling
            self._setup_module_events(module_name, module)
            
            # Update statistics
            self._stats['loaded_modules'] += 1
            self._stats['total_modules'] += 1
            
            logger.info(f"Module loaded successfully: {module_name}")
            return module
            
        except Exception as e:
            logger.error(f"Error loading module {module_name}: {e}")
            return None
    
    def load_modules_from_config(self, config_file: str) -> Dict[str, bool]:
        """Load multiple modules from configuration file"""
        results = {}
        
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            for module_name, module_config in config_data.get('modules', {}).items():
                try:
                    # Create module config
                    config = create_module_config(
                        name=module_name,
                        version=module_config.get('version', '1.0.0'),
                        description=module_config.get('description', f'Module {module_name}'),
                        priority=ModulePriority(module_config.get('priority', 50)),
                        enabled=module_config.get('enabled', True),
                        auto_start=module_config.get('auto_start', False),
                        dependencies=module_config.get('dependencies', []),
                        log_level=module_config.get('log_level', 'INFO')
                    )
                    
                    # Load module
                    module = self.load_module(module_name, config)
                    results[module_name] = module is not None
                    
                except Exception as e:
                    logger.error(f"Error loading module {module_name}: {e}")
                    results[module_name] = False
            
            return results
            
        except Exception as e:
            logger.error(f"Error loading modules from config: {e}")
            return {}
    
    def _setup_module_events(self, module_name: str, module: BaseModule):
        """Set up event handling for a module"""
        try:
            # Global event forwarding
            def forward_event(event: str, data: Any, source_module: BaseModule):
                if source_module.config.name == module_name:
                    self._trigger_global_event(event, data, module_name)
            
            module.add_event_callback("initialized", forward_event)
            module.add_event_callback("started", forward_event)
            module.add_event_callback("stopped", forward_event)
            module.add_event_callback("error", forward_event)
            module.add_event_callback("paused", forward_event)
            module.add_event_callback("resumed", forward_event)
            
            # Health monitoring
            def health_callback(health_data: Dict[str, Any], source_module: BaseModule):
                self._handle_module_health(module_name, health_data)
            
            module.add_health_callback(health_callback)
            
        except Exception as e:
            logger.error(f"Error setting up module events for {module_name}: {e}")
    
    def start_module(self, module_name: str) -> bool:
        """Start a specific module"""
        try:
            if module_name not in self.modules:
                logger.error(f"Module not loaded: {module_name}")
                return False
            
            module = self.modules[module_name]
            
            # Check dependencies
            if not self._check_module_dependencies(module):
                logger.error(f"Module dependencies not met: {module_name}")
                return False
            
            # Start module
            success = module.start()
            
            if success:
                if module_name not in self._running_modules:
                    self._running_modules.append(module_name)
                if module_name in self._stopped_modules:
                    self._stopped_modules.remove(module_name)
                if module_name in self._error_modules:
                    self._error_modules.remove(module_name)
                
                self._stats['running_modules'] += 1
                logger.info(f"Module started: {module_name}")
            else:
                if module_name not in self._error_modules:
                    self._error_modules.append(module_name)
                self._stats['error_modules'] += 1
                logger.error(f"Failed to start module: {module_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error starting module {module_name}: {e}")
            return False
    
    def start_all_modules(self) -> Dict[str, bool]:
        """Start all loaded modules"""
        results = {}
        
        # Sort modules by priority
        sorted_modules = sorted(
            self.modules.items(),
            key=lambda x: x[1].config.priority.value,
            reverse=True
        )
        
        for module_name, module in sorted_modules:
            if module.config.enabled:
                results[module_name] = self.start_module(module_name)
            else:
                logger.info(f"Module disabled, skipping: {module_name}")
                results[module_name] = False
        
        return results
    
    def stop_module(self, module_name: str) -> bool:
        """Stop a specific module"""
        try:
            if module_name not in self.modules:
                logger.error(f"Module not loaded: {module_name}")
                return False
            
            module = self.modules[module_name]
            success = module.stop()
            
            if success:
                if module_name in self._running_modules:
                    self._running_modules.remove(module_name)
                if module_name not in self._stopped_modules:
                    self._stopped_modules.append(module_name)
                
                self._stats['running_modules'] = max(0, self._stats['running_modules'] - 1)
                logger.info(f"Module stopped: {module_name}")
            else:
                logger.error(f"Failed to stop module: {module_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error stopping module {module_name}: {e}")
            return False
    
    def stop_all_modules(self) -> Dict[str, bool]:
        """Stop all running modules"""
        results = {}
        
        # Stop modules in reverse priority order
        sorted_modules = sorted(
            self.modules.items(),
            key=lambda x: x[1].config.priority.value
        )
        
        for module_name, module in sorted_modules:
            if module.status.state == ModuleState.RUNNING:
                results[module_name] = self.stop_module(module_name)
        
        return results
    
    def pause_module(self, module_name: str) -> bool:
        """Pause a specific module"""
        try:
            if module_name not in self.modules:
                return False
            
            module = self.modules[module_name]
            return module.pause()
            
        except Exception as e:
            logger.error(f"Error pausing module {module_name}: {e}")
            return False
    
    def resume_module(self, module_name: str) -> bool:
        """Resume a specific module"""
        try:
            if module_name not in self.modules:
                return False
            
            module = self.modules[module_name]
            return module.resume()
            
        except Exception as e:
            logger.error(f"Error resuming module {module_name}: {e}")
            return False
    
    def _check_module_dependencies(self, module: BaseModule) -> bool:
        """Check if module dependencies are satisfied"""
        for dep_name in module.config.dependencies:
            if dep_name not in self.modules:
                logger.error(f"Missing dependency: {dep_name} for {module.config.name}")
                return False
            
            dep_module = self.modules[dep_name]
            if dep_module.status.state != ModuleState.RUNNING:
                logger.error(f"Dependency not running: {dep_name} for {module.config.name}")
                return False
        
        return True
    
    def get_module(self, module_name: str) -> Optional[BaseModule]:
        """Get a specific module by name"""
        return self.modules.get(module_name)
    
    def get_modules_by_state(self, state: ModuleState) -> List[BaseModule]:
        """Get all modules in a specific state"""
        return [module for module in self.modules.values() if module.status.state == state]
    
    def get_modules_by_priority(self, priority: ModulePriority) -> List[BaseModule]:
        """Get all modules with a specific priority"""
        return [module for module in self.modules.values() if module.config.priority == priority]
    
    def get_module_status(self, module_name: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific module"""
        if module_name in self.modules:
            return self.modules[module_name].get_info()
        return None
    
    def get_all_modules_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all modules"""
        return {name: module.get_info() for name, module in self.modules.items()}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        return {
            'module_manager': {
                'total_modules': self._stats['total_modules'],
                'loaded_modules': self._stats['loaded_modules'],
                'running_modules': self._stats['running_modules'],
                'error_modules': self._stats['error_modules'],
                'start_time': self._stats['start_time'].isoformat(),
                'uptime': (datetime.now() - self._stats['start_time']).total_seconds()
            },
            'modules': self.get_all_modules_status(),
            'module_states': {
                'running': self._running_modules,
                'stopped': self._stopped_modules,
                'error': self._error_modules
            }
        }
    
    def add_global_event_callback(self, event: str, callback: Callable):
        """Add global event callback"""
        if event not in self._global_event_callbacks:
            self._global_event_callbacks[event] = []
        self._global_event_callbacks[event].append(callback)
    
    def add_module_event_callback(self, module_name: str, event: str, callback: Callable):
        """Add module-specific event callback"""
        if module_name not in self._module_event_callbacks:
            self._module_event_callbacks[module_name] = {}
        if event not in self._module_event_callbacks[module_name]:
            self._module_event_callbacks[module_name][event] = []
        self._module_event_callbacks[module_name][event].append(callback)
    
    def _trigger_global_event(self, event: str, data: Any, source_module: str):
        """Trigger global event callbacks"""
        if event in self._global_event_callbacks:
            for callback in self._global_event_callbacks[event]:
                try:
                    callback(event, data, source_module)
                except Exception as e:
                    logger.error(f"Error in global event callback: {e}")
    
    def _handle_module_health(self, module_name: str, health_data: Dict[str, Any]):
        """Handle module health updates"""
        try:
            # Update module health
            if module_name in self.modules:
                module = self.modules[module_name]
                module.update_health(health_data)
            
            # Trigger health events
            self._trigger_global_event("module_health", health_data, module_name)
            
        except Exception as e:
            logger.error(f"Error handling module health: {e}")
    
    def reload_module(self, module_name: str) -> bool:
        """Reload a specific module"""
        try:
            if module_name not in self.modules:
                return False
            
            # Stop module if running
            if self.modules[module_name].status.state == ModuleState.RUNNING:
                self.stop_module(module_name)
            
            # Remove old module
            old_module = self.modules.pop(module_name)
            old_module.cleanup()
            
            # Reload module class
            if module_name in self.module_classes:
                # Force reload of module
                module_path = str(self.modules_path)
                if module_path in sys.path:
                    sys.path.remove(module_path)
                
                # Re-import
                module = importlib.import_module(module_name)
                importlib.reload(module)
                
                # Update module class
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, BaseModule) and 
                        obj != BaseModule):
                        self.module_classes[module_name] = obj
                        break
            
            # Load new module
            config = self.module_configs.get(module_name)
            new_module = self.load_module(module_name, config)
            
            return new_module is not None
            
        except Exception as e:
            logger.error(f"Error reloading module {module_name}: {e}")
            return False
    
    def unload_module(self, module_name: str) -> bool:
        """Unload a specific module"""
        try:
            if module_name not in self.modules:
                return False
            
            # Stop module if running
            if self.modules[module_name].status.state == ModuleState.RUNNING:
                self.stop_module(module_name)
            
            # Cleanup and remove
            module = self.modules.pop(module_name)
            module.cleanup()
            
            # Update statistics
            self._stats['loaded_modules'] = max(0, self._stats['loaded_modules'] - 1)
            self._stats['total_modules'] = max(0, self._stats['total_modules'] - 1)
            
            logger.info(f"Module unloaded: {module_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading module {module_name}: {e}")
            return False
    
    def shutdown(self):
        """Shutdown all modules and cleanup"""
        try:
            logger.info("Shutting down module manager...")
            
            # Stop all modules
            self.stop_all_modules()
            
            # Cleanup all modules
            for module_name, module in list(self.modules.items()):
                try:
                    module.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up module {module_name}: {e}")
            
            # Clear modules
            self.modules.clear()
            self._running_modules.clear()
            self._stopped_modules.clear()
            self._error_modules.clear()
            
            # Set shutdown event
            self._shutdown_event.set()
            
            logger.info("Module manager shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()
